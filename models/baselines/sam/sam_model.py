#!/usr/bin/env python3
"""
SAM Fine-tuned Model for Medical Image Segmentation
基于Segment Anything Model的医学图像分割微调版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
import os

class SAMFineTuned(nn.Module):
    """
    SAM Fine-tuned for medical image segmentation
    冻结图像编码器，只微调mask解码器
    """
    
    def __init__(self, model_type="vit_b", checkpoint_path=None, freeze_encoder=True):
        super(SAMFineTuned, self).__init__()
        
        self.model_type = model_type
        self.freeze_encoder = freeze_encoder
        
        # 默认checkpoint路径
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path(model_type)
        
        # 加载SAM模型
        if os.path.exists(checkpoint_path):
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            print(f"✅ 加载SAM模型: {checkpoint_path}")
        else:
            print(f"⚠️ SAM checkpoint不存在: {checkpoint_path}")
            print("🔄 使用随机初始化的SAM模型")
            self.sam = sam_model_registry[model_type]()
        
        # 冻结图像编码器
        if freeze_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
            print("🔒 SAM图像编码器已冻结")
        
        # 设置为训练模式
        self.sam.train()
        
        # 添加适配层用于医学图像
        self.medical_adapter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),  # 单通道转RGB
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # 输出适配层
        self.output_adapter = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def _get_default_checkpoint_path(self, model_type):
        """获取默认的checkpoint路径"""
        checkpoint_paths = {
            "vit_b": "checkpoints/sam_vit_b_01ec64.pth",
            "vit_l": "checkpoints/sam_vit_l_0b3195.pth", 
            "vit_h": "checkpoints/sam_vit_h_4b8939.pth"
        }
        return checkpoint_paths.get(model_type, "checkpoints/sam_vit_b_01ec64.pth")
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 [B, 1, H, W]
        Returns:
            mask_logits: 分割mask logits [B, 1, H, W]
        """
        batch_size, _, height, width = x.shape
        
        # 转换为RGB格式
        x_rgb = self.medical_adapter(x)
        
        # 调整到SAM期望的输入尺寸 (1024x1024)
        x_resized = F.interpolate(x_rgb, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # SAM图像编码
        with torch.set_grad_enabled(not self.freeze_encoder):
            image_embeddings = self.sam.image_encoder(x_resized)
        
        # 生成默认的prompt embeddings (无特定prompt)
        sparse_embeddings = torch.empty((batch_size, 0, 256), device=x.device, dtype=x.dtype)
        dense_embeddings = self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, self.sam.prompt_encoder.image_embedding_size[0], 
            self.sam.prompt_encoder.image_embedding_size[1]
        )
        
        # SAM mask解码
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # 调整回原始尺寸
        masks = F.interpolate(
            low_res_masks,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        
        # 输出适配
        output = self.output_adapter(masks)
        
        return output
    
    def get_image_embeddings(self, x):
        """获取图像embeddings，用于prompt-based推理"""
        x_rgb = self.medical_adapter(x)
        x_resized = F.interpolate(x_rgb, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(x_resized)
        
        return image_embeddings
    
    def predict_with_points(self, x, points, labels):
        """
        使用点prompt进行预测
        Args:
            x: 输入图像 [B, 1, H, W]
            points: 点坐标 [B, N, 2]
            labels: 点标签 [B, N] (1为前景，0为背景)
        """
        batch_size, _, height, width = x.shape
        
        # 获取图像embeddings
        image_embeddings = self.get_image_embeddings(x)
        
        # 调整点坐标到1024尺寸
        points_1024 = points.clone()
        points_1024[:, :, 0] = points_1024[:, :, 0] * 1024 / width
        points_1024[:, :, 1] = points_1024[:, :, 1] * 1024 / height
        
        # 编码prompt
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(points_1024, labels),
            boxes=None,
            masks=None,
        )
        
        # 解码mask
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # 调整回原始尺寸
        masks = F.interpolate(
            low_res_masks,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        
        return masks, iou_predictions


class SAMWithPrompts(SAMFineTuned):
    """
    SAM with automatic prompt generation from ground truth
    自动从ground truth生成prompt的SAM版本
    """
    
    def __init__(self, model_type="vit_b", checkpoint_path=None, freeze_encoder=True, 
                 num_points=5, use_bbox=False):
        super().__init__(model_type, checkpoint_path, freeze_encoder)
        self.num_points = num_points
        self.use_bbox = use_bbox
    
    def extract_points_from_mask(self, mask, num_points=5):
        """
        从mask中提取前景和背景点
        Args:
            mask: ground truth mask [B, 1, H, W]
            num_points: 提取的点数
        Returns:
            points: 点坐标 [B, num_points*2, 2]
            labels: 点标签 [B, num_points*2]
        """
        batch_size, _, height, width = mask.shape
        points_list = []
        labels_list = []
        
        for b in range(batch_size):
            mask_b = mask[b, 0].cpu().numpy()
            
            # 前景点
            fg_coords = np.where(mask_b > 0.5)
            if len(fg_coords[0]) > 0:
                fg_indices = np.random.choice(len(fg_coords[0]), 
                                            min(num_points, len(fg_coords[0])), 
                                            replace=False)
                fg_points = np.stack([fg_coords[1][fg_indices], fg_coords[0][fg_indices]], axis=1)
            else:
                fg_points = np.array([[width//2, height//2]])  # 默认中心点
            
            # 背景点
            bg_coords = np.where(mask_b <= 0.5)
            if len(bg_coords[0]) > 0:
                bg_indices = np.random.choice(len(bg_coords[0]), 
                                            min(num_points, len(bg_coords[0])), 
                                            replace=False)
                bg_points = np.stack([bg_coords[1][bg_indices], bg_coords[0][bg_indices]], axis=1)
            else:
                bg_points = np.array([[0, 0]])  # 默认角落点
            
            # 合并点
            points = np.concatenate([fg_points, bg_points], axis=0)
            labels = np.concatenate([np.ones(len(fg_points)), np.zeros(len(bg_points))])
            
            points_list.append(torch.from_numpy(points).float())
            labels_list.append(torch.from_numpy(labels).float())
        
        # 转换为tensor
        max_points = max(len(p) for p in points_list)
        points_tensor = torch.zeros(batch_size, max_points, 2)
        labels_tensor = torch.zeros(batch_size, max_points)
        
        for b, (pts, lbls) in enumerate(zip(points_list, labels_list)):
            points_tensor[b, :len(pts)] = pts
            labels_tensor[b, :len(lbls)] = lbls
        
        return points_tensor.to(mask.device), labels_tensor.to(mask.device)
    
    def forward(self, x, gt_mask=None):
        """
        前向传播，使用ground truth生成prompt
        """
        if gt_mask is not None and self.training:
            # 训练时使用GT生成prompt
            points, labels = self.extract_points_from_mask(gt_mask, self.num_points)
            masks, iou_pred = self.predict_with_points(x, points, labels)
            return masks
        else:
            # 推理时使用无prompt模式
            return super().forward(x)


def create_sam_model(model_type="vit_b", checkpoint_path=None, freeze_encoder=True, 
                    use_prompts=True, **kwargs):
    """
    创建SAM模型的工厂函数
    """
    if use_prompts:
        return SAMWithPrompts(model_type, checkpoint_path, freeze_encoder, **kwargs)
    else:
        return SAMFineTuned(model_type, checkpoint_path, freeze_encoder)


if __name__ == "__main__":
    # 测试SAM模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SAMFineTuned(model_type="vit_b").to(device)
    
    # 测试输入
    x = torch.randn(2, 1, 256, 256).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    print("✅ SAM模型测试通过！")

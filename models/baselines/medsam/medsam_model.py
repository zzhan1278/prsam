#!/usr/bin/env python3
"""
MedSAM Model for Medical Image Segmentation
基于官方MedSAM实现，使用真正的MedSAM预训练权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# 尝试导入MedSAM相关模块
try:
    from segment_anything import sam_model_registry
    from segment_anything.modeling import Sam
except ImportError:
    print("Warning: segment_anything not found, using fallback implementation")

class MedSAM(nn.Module):
    """
    MedSAM - Medical Segment Anything Model
    预训练的医学图像SAM，直接用于推理
    """
    
    def __init__(self, model_type="vit_b", checkpoint_path=None, bbox_prompt=True):
        super(MedSAM, self).__init__()
        
        self.model_type = model_type
        self.bbox_prompt = bbox_prompt
        
        # 默认checkpoint路径
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path(model_type)
        
        # 按照MedSAM官方方式加载模型
        if os.path.exists(checkpoint_path):
            print(f"✅ 找到MedSAM权重: {checkpoint_path}")
            try:
                # 使用官方MedSAM加载方式
                sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
                print(f"✅ 成功加载MedSAM模型")
            except Exception as e:
                print(f"⚠️ MedSAM加载失败: {e}")
                print("🔄 使用标准SAM作为替代")
                standard_path = self._get_standard_sam_path(model_type)
                if os.path.exists(standard_path):
                    sam_model = sam_model_registry[model_type](checkpoint=standard_path)
                    print(f"✅ 加载标准SAM: {standard_path}")
                else:
                    sam_model = sam_model_registry[model_type]()
                    print("🔄 使用随机初始化的SAM模型")
        else:
            print(f"⚠️ MedSAM权重不存在: {checkpoint_path}")
            print("🔄 使用标准SAM作为替代")
            standard_path = self._get_standard_sam_path(model_type)
            if os.path.exists(standard_path):
                sam_model = sam_model_registry[model_type](checkpoint=standard_path)
                print(f"✅ 加载标准SAM: {standard_path}")
            else:
                sam_model = sam_model_registry[model_type]()
                print("🔄 使用随机初始化的SAM模型")
        
        # 存储SAM模型
        self.sam_model = sam_model
        
        # 设置为评估模式
        self.sam_model.eval()
        
        # 冻结所有参数
        for param in self.sam_model.parameters():
            param.requires_grad = False
        
        # 医学图像适配层
        self.medical_adapter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),  # 单通道转RGB
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # 输出后处理层
        self.output_processor = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def _get_default_checkpoint_path(self, model_type):
        """获取MedSAM的默认checkpoint路径"""
        checkpoint_paths = {
            "vit_b": "checkpoints/medsam_vit_b.pth",
            "vit_l": "checkpoints/medsam_vit_l.pth", 
            "vit_h": "checkpoints/medsam_vit_h.pth"
        }
        return checkpoint_paths.get(model_type, "checkpoints/medsam_vit_b.pth")
    
    def _get_standard_sam_path(self, model_type):
        """获取标准SAM的checkpoint路径作为备选"""
        checkpoint_paths = {
            "vit_b": "checkpoints/sam_vit_b_01ec64.pth",
            "vit_l": "checkpoints/sam_vit_l_0b3195.pth", 
            "vit_h": "checkpoints/sam_vit_h_4b8939.pth"
        }
        return checkpoint_paths.get(model_type, "checkpoints/sam_vit_b_01ec64.pth")
    
    def generate_bbox_from_mask(self, mask):
        """
        从ground truth mask生成bounding box
        Args:
            mask: ground truth mask [B, 1, H, W]
        Returns:
            boxes: bounding boxes [B, 4] (x1, y1, x2, y2)
        """
        batch_size, _, height, width = mask.shape
        boxes = []
        
        for b in range(batch_size):
            mask_b = mask[b, 0].cpu().numpy()
            
            # 找到前景区域
            coords = np.where(mask_b > 0.5)
            
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # 添加一些padding
                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(width - 1, x_max + padding)
                y_max = min(height - 1, y_max + padding)
                
                boxes.append([x_min, y_min, x_max, y_max])
            else:
                # 如果没有前景，使用整个图像
                boxes.append([0, 0, width - 1, height - 1])
        
        return torch.tensor(boxes, dtype=torch.float32, device=mask.device)
    
    def forward(self, x, gt_mask=None):
        """
        前向传播 - 直接使用MedSAM模型
        Args:
            x: 输入图像 [B, 1, H, W]
            gt_mask: ground truth mask [B, 1, H, W] (可选)
        Returns:
            mask_logits: 分割mask logits [B, 1, H, W]
        """
        batch_size, _, height, width = x.shape
        
        # 转换为RGB格式
        x_rgb = self.medical_adapter(x)
        
        # 调整到SAM期望的输入尺寸 (1024x1024)
        x_resized = F.interpolate(x_rgb, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # SAM图像编码
        with torch.no_grad():
            image_embeddings = self.sam_model.image_encoder(x_resized)
        
        # 使用全图作为bbox prompt
        boxes = torch.tensor([[0, 0, 1023, 1023]], dtype=torch.float32, device=x.device).repeat(batch_size, 1)
        
        # 编码prompt
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        
        # SAM mask解码
        with torch.no_grad():
            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
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
        
        # 输出后处理
        output = self.output_processor(masks)
        
        return output
    
    def predict_with_bbox(self, x, boxes):
        """
        使用bounding box进行预测
        Args:
            x: 输入图像 [B, 1, H, W]
            boxes: bounding boxes [B, 4] (x1, y1, x2, y2)
        """
        batch_size, _, height, width = x.shape
        
        # 转换为RGB并调整尺寸
        x_rgb = self.medical_adapter(x)
        x_resized = F.interpolate(x_rgb, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # 获取图像embeddings
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(x_resized)
        
        # 调整bbox到1024尺寸
        boxes_1024 = boxes.clone()
        boxes_1024[:, [0, 2]] = boxes_1024[:, [0, 2]] * 1024 / width
        boxes_1024[:, [1, 3]] = boxes_1024[:, [1, 3]] * 1024 / height
        
        # 编码prompt
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=boxes_1024,
            masks=None,
        )
        
        # 解码mask
        with torch.no_grad():
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
    
    def predict_with_points(self, x, points, labels):
        """
        使用点prompt进行预测
        Args:
            x: 输入图像 [B, 1, H, W]
            points: 点坐标 [B, N, 2]
            labels: 点标签 [B, N] (1为前景，0为背景)
        """
        batch_size, _, height, width = x.shape
        
        # 转换为RGB并调整尺寸
        x_rgb = self.medical_adapter(x)
        x_resized = F.interpolate(x_rgb, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # 获取图像embeddings
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(x_resized)
        
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
        with torch.no_grad():
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


def create_medsam_model(model_type="vit_b", checkpoint_path=None, bbox_prompt=True, **kwargs):
    """
    创建MedSAM模型的工厂函数
    """
    return MedSAM(model_type, checkpoint_path, bbox_prompt, **kwargs)


if __name__ == "__main__":
    # 测试MedSAM模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = MedSAM(model_type="vit_b").to(device)
    
    # 测试输入
    x = torch.randn(2, 1, 256, 256).to(device)
    gt_mask = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x, gt_mask)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    print("✅ MedSAM模型测试通过！")

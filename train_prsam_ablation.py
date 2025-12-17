#!/usr/bin/env python3
"""
PR-SAM Ablation Study Training Script
训练PR-SAM的各个消融模型
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.append(os.path.abspath('.'))

from data.int8_dataset import create_int8_multi_quality_dataloaders
from models.proposed.pgrsam.ablation_models import (
    PRSamBaseline, 
    PRSamNoCorrectionNetwork,
    PRSamPriorOnly,
    PRSamCBCTOnly
)
from models.proposed.pgrsam.pgr_sam_model import PRSamModel
from utils.metrics import dice_score, iou_score


def setup_logging(model_name):
    """设置日志系统"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"results/pr_sam_ablation/{model_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def get_model(model_name, device):
    """根据模型名称获取相应的消融模型"""
    
    # SAM checkpoint path
    sam_checkpoint = "F:/CT_guided_CBCT_segmentation/models/sam_weights/sam_vit_b_01ec64.pth"
    
    if model_name == "baseline":
        model = PRSamBaseline(
            sam_model_type="vit_b",
            sam_checkpoint_path=sam_checkpoint,
            device=device
        )
    elif model_name == "no_correction":
        model = PRSamNoCorrectionNetwork(
            sam_model_type="vit_b",
            sam_checkpoint_path=sam_checkpoint,
            device=device
        )
    elif model_name == "prior_only":
        model = PRSamPriorOnly(
            sam_model_type="vit_b",
            sam_checkpoint_path=sam_checkpoint,
            device=device
        )
    elif model_name == "cbct_only":
        model = PRSamCBCTOnly(
            sam_model_type="vit_b",
            sam_checkpoint_path=sam_checkpoint,
            device=device
        )
    elif model_name == "full":
        model = PRSamModel(
            sam_model_type="vit_b",
            sam_checkpoint_path=sam_checkpoint,
            device=device
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


def create_prior_features(masks):
    """从GT mask创建3通道先验特征"""
    import torch.nn.functional as F
    from scipy import ndimage
    import numpy as np
    
    batch_size = masks.shape[0]
    prior_features = []
    
    for i in range(batch_size):
        mask = masks[i, 0].cpu().numpy()  # (H, W)
        
        # Channel 1: Binary mask
        ch1 = mask.copy()
        
        # Channel 2: Signed distance transform
        if mask.sum() > 0:
            dist_pos = ndimage.distance_transform_edt(mask)
            dist_neg = ndimage.distance_transform_edt(1 - mask)
            ch2 = dist_pos - dist_neg
            ch2 = ch2 / (np.abs(ch2).max() + 1e-8)  # Normalize
        else:
            ch2 = np.zeros_like(mask)
        
        # Channel 3: Boundary
        kernel = np.ones((3, 3))
        dilated = ndimage.binary_dilation(mask, structure=kernel)
        eroded = ndimage.binary_erosion(mask, structure=kernel)
        ch3 = (dilated ^ eroded).astype(np.float32)
        
        # Stack channels
        prior = np.stack([ch1, ch2, ch3], axis=0)  # (3, H, W)
        prior_features.append(torch.from_numpy(prior))
    
    return torch.stack(prior_features).to(masks.device)


def get_bbox_from_mask(mask):
    """从mask提取bounding box"""
    if mask.sum() == 0:
        # 如果mask为空，返回整个图像的bbox
        h, w = mask.shape[-2:]
        return torch.tensor([0, 0, w-1, h-1], dtype=torch.float32)
    
    # 找到非零像素的位置
    nonzero = torch.nonzero(mask[0] > 0.5)
    if len(nonzero) == 0:
        h, w = mask.shape[-2:]
        return torch.tensor([0, 0, w-1, h-1], dtype=torch.float32)
    
    y_min, x_min = nonzero.min(dim=0).values
    y_max, x_max = nonzero.max(dim=0).values
    
    # 添加一些padding
    h, w = mask.shape[-2:]
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)
    
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)


def train_one_epoch(model, train_loader, optimizer, device, epoch, scaler=None, model_name=""):
    """训练一个epoch"""
    model.train()
    
    # 冻结SAM的encoder和decoder
    for name, param in model.named_parameters():
        if "sam_model" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    total_loss = 0
    dice_scores = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 创建prior features
        if model_name != "cbct_only":
            prior_features = create_prior_features(masks)
        else:
            prior_features = None
        
        # 获取bbox prompts
        bbox_prompts = []
        original_sizes = []
        for i in range(images.shape[0]):
            bbox = get_bbox_from_mask(masks[i:i+1])
            bbox_prompts.append(bbox)
            original_sizes.append(torch.tensor([256, 256]))  # 固定大小
        
        bbox_prompts = torch.stack(bbox_prompts).to(device)
        original_sizes = torch.stack(original_sizes).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(
                cbct_image=images,
                pct_atlas_image=prior_features,
                bbox_prompts_xyxy=bbox_prompts,
                original_image_size_hw=original_sizes,
                multimask_output=False
            )
            
            # 处理不同模型的输出
            if model_name == "full":
                # Full model返回两个mask
                pred_masks_fused = outputs[0]
                pred_masks_cbct = outputs[6]
                
                # 双分支损失
                loss_fused = compute_loss(pred_masks_fused, masks)
                loss_cbct = compute_loss(pred_masks_cbct, masks) if pred_masks_cbct is not None else 0
                loss = loss_fused + 0.5 * loss_cbct
                
                # 融合预测用于评估
                pred = (torch.sigmoid(pred_masks_fused) + torch.sigmoid(pred_masks_cbct)) / 2
            else:
                # 其他模型只有一个输出
                pred_masks = outputs[0]
                loss = compute_loss(pred_masks, masks)
                pred = torch.sigmoid(pred_masks)
            
            # 如果输出是多mask，取第一个
            if pred.dim() == 4 and pred.shape[1] > 1:
                pred = pred[:, 0:1]
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            dice = dice_score(pred, masks)
            dice_scores.append(dice.item())
            total_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = np.mean(dice_scores)
    
    return avg_loss, avg_dice


def compute_loss(pred_masks, gt_masks):
    """计算损失函数"""
    # BCE + Dice Loss
    bce_loss = nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_masks)
    
    pred_sigmoid = torch.sigmoid(pred_masks)
    smooth = 1e-5
    intersection = (pred_sigmoid * gt_masks).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
    dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
    dice_loss = dice_loss.mean()
    
    # 边界加权
    kernel = torch.ones(1, 1, 3, 3).to(gt_masks.device)
    dilated = torch.nn.functional.conv2d(gt_masks, kernel, padding=1) > 0
    eroded = torch.nn.functional.conv2d(gt_masks, kernel, padding=1) >= 9
    boundary = (dilated.float() - eroded.float()).abs()
    
    boundary_weight = 1 + boundary
    weighted_bce = (boundary_weight * nn.functional.binary_cross_entropy_with_logits(
        pred_masks, gt_masks, reduction='none'
    )).mean()
    
    return weighted_bce + dice_loss


def validate(model, val_loader, device, model_name=""):
    """验证模型"""
    model.eval()
    
    val_loss = 0
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 创建prior features
            if model_name != "cbct_only":
                prior_features = create_prior_features(masks)
            else:
                prior_features = None
            
            # 获取bbox prompts
            bbox_prompts = []
            original_sizes = []
            for i in range(images.shape[0]):
                bbox = get_bbox_from_mask(masks[i:i+1])
                bbox_prompts.append(bbox)
                original_sizes.append(torch.tensor([256, 256]))
            
            bbox_prompts = torch.stack(bbox_prompts).to(device)
            original_sizes = torch.stack(original_sizes).to(device)
            
            outputs = model(
                cbct_image=images,
                pct_atlas_image=prior_features,
                bbox_prompts_xyxy=bbox_prompts,
                original_image_size_hw=original_sizes,
                multimask_output=False
            )
            
            # 处理输出
            if model_name == "full":
                pred_masks_fused = outputs[0]
                pred_masks_cbct = outputs[6]
                loss_fused = compute_loss(pred_masks_fused, masks)
                loss_cbct = compute_loss(pred_masks_cbct, masks) if pred_masks_cbct is not None else 0
                loss = loss_fused + 0.5 * loss_cbct
                pred = (torch.sigmoid(pred_masks_fused) + torch.sigmoid(pred_masks_cbct)) / 2
            else:
                pred_masks = outputs[0]
                loss = compute_loss(pred_masks, masks)
                pred = torch.sigmoid(pred_masks)
            
            if pred.dim() == 4 and pred.shape[1] > 1:
                pred = pred[:, 0:1]
            
            val_loss += loss.item()
            dice_scores.append(dice_score(pred, masks).item())
            iou_scores.append(iou_score(pred, masks).item())
    
    avg_val_loss = val_loss / len(val_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    
    return avg_val_loss, avg_dice, avg_iou


def main():
    parser = argparse.ArgumentParser(description="PR-SAM Ablation Study Training")
    parser.add_argument('--model', type=str, required=True,
                       choices=['baseline', 'no_correction', 'prior_only', 'cbct_only', 'full'],
                       help='Ablation model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    log_file = setup_logging(args.model)
    logging.info(f"Starting PR-SAM Ablation Training: {args.model}")
    logging.info(f"Device: {device}")
    logging.info(f"Arguments: {args}")
    
    # Create dataloaders
    logging.info("Loading datasets...")
    base_dir = "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed"
    mask_dir = "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed/Masks"
    
    train_loader, val_loader, test_loaders = create_int8_multi_quality_dataloaders(
        base_dir=base_dir,
        mask_dir=mask_dir,
        qualities=['32', '64', '128', '256', '490'],
        batch_size=args.batch_size,
        val_split=0.1,
        test_split=0.1,
        num_workers=4,
        target_size=256,
        patient_level_split=True,
        load_subset='train_val'  # 只加载训练和验证集
    )
    
    # Create model
    logging.info(f"Creating model: {args.model}")
    model = get_model(args.model, device)
    
    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    best_dice = 0
    best_epoch = 0
    training_history = []
    
    save_dir = Path(f"results/pr_sam_ablation/{args.model}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scaler, args.model
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate(model, val_loader, device, args.model)
        
        # Learning rate scheduling
        scheduler.step(val_dice)
        
        # Log results
        logging.info(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': args
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            logging.info(f"New best model saved! Dice: {best_dice:.4f}")
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logging.info(f"\nTraining completed!")
    logging.info(f"Best Dice: {best_dice:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    main()

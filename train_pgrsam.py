#!/usr/bin/env python3
import os
import logging
from datetime import datetime
import json
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.proposed.pgrsam import PRSamModel
from data.int8_dataset import create_int8_multi_quality_dataloaders
from utils.metrics import (
    DiceBCELoss, dice_score, iou_score, hd95_score,
    accuracy_score, precision_score, recall_score, f1_score, specificity_score
)


def resize_to(target, tensor):
    """Resize tensor to target spatial size using bilinear, align_corners=False."""
    if tensor.shape[-2:] != target.shape[-2:]:
        return torch.nn.functional.interpolate(
            tensor, size=target.shape[-2:], mode='bilinear', align_corners=False
        )
    return tensor


def random_translate_batch(images: torch.Tensor, masks: torch.Tensor, max_shift: int = 16):
    """Randomly translate images and masks by the same (dy, dx). Keeps size via pad+crop.
    Args:
        images: [B, C, H, W]
        masks:  [B, C, H, W]
    Returns:
        images_t, masks_t with identical shifts applied
    """
    if max_shift <= 0:
        return images, masks
    b, c, h, w = images.shape
    device = images.device

    # Single random shift per batch to keep consistency
    dy = int(torch.randint(low=-max_shift, high=max_shift + 1, size=(1,), device=device).item())
    dx = int(torch.randint(low=-max_shift, high=max_shift + 1, size=(1,), device=device).item())

    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)

    # Pad (left, right, top, bottom)
    images_padded = torch.nn.functional.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
    masks_padded = torch.nn.functional.pad(masks, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)

    # Crop back to original size
    start_y = pad_top - dy
    start_x = pad_left - dx
    images_t = images_padded[:, :, start_y:start_y + h, start_x:start_x + w]
    masks_t = masks_padded[:, :, start_y:start_y + h, start_x:start_x + w]
    return images_t, masks_t


def _affine_grid_sample(image: torch.Tensor, mask: torch.Tensor, theta: torch.Tensor) -> tuple:
    """Apply affine warp defined by 2x3 theta to image (bilinear) and mask (nearest)."""
    b, c, h, w = image.shape
    grid = torch.nn.functional.affine_grid(theta, size=(b, c, h, w), align_corners=False)
    warped_img = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    warped_msk = torch.nn.functional.grid_sample(mask, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    return warped_img, warped_msk


def _elastic_displacement(b: int, h: int, w: int, device: torch.device, alpha: float = 0.02, kernel_size: int = 31) -> torch.Tensor:
    """Create smooth random displacement field in normalized coords [-1,1].
    alpha is max displacement ratio relative to image size."""
    # Random noise -> smooth via average pooling
    disp = torch.randn(b, 2, h, w, device=device)
    k = max(3, kernel_size // 2 * 2 + 1)
    disp = torch.nn.functional.avg_pool2d(disp, kernel_size=k, stride=1, padding=k // 2)
    # Normalize to [-1,1] then scale by alpha (relative to size) and convert to normalized grid units
    disp = disp / (disp.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
    # Convert pixel displacement to normalized grid: dx_norm = dx_px / ((w-1)/2)
    scale_x = alpha * w / ((w - 1) / 2.0)
    scale_y = alpha * h / ((h - 1) / 2.0)
    disp[:, 0, :, :] *= scale_x
    disp[:, 1, :, :] *= scale_y
    # To grid format [B,H,W,2]
    return disp.permute(0, 2, 3, 1).contiguous()


def random_deform_batch(images: torch.Tensor, masks: torch.Tensor,
                        max_rotate_deg: float = 3.0, max_scale: float = 0.02, max_shear: float = 0.02,
                        elastic_alpha: float = 0.02, elastic_kernel: int = 31, p_elastic: float = 0.5):
    """Apply small affine + optional elastic deformation to CBCT and GT mask (same transform), keep size.
    - images: bilinear
    - masks: nearest
    """
    b, c, h, w = images.shape
    device = images.device

    # Random small affine per-sample
    thetas = []
    for _ in range(b):
        rot = (torch.rand(1, device=device) * 2 - 1) * (max_rotate_deg * 3.14159265 / 180.0)
        scale = 1.0 + (torch.rand(1, device=device) * 2 - 1) * max_scale
        shear = (torch.rand(1, device=device) * 2 - 1) * max_shear
        cos_r = torch.cos(rot) * scale
        sin_r = torch.sin(rot) * scale
        # Simple shear on x
        sh = shear
        # 2x3 matrix
        theta = torch.tensor([[cos_r.item() + sh.item() * (-sin_r.item()), -sin_r.item(), 0.0],
                              [sin_r.item(), cos_r.item(), 0.0]], device=device, dtype=torch.float32)
        thetas.append(theta)
    theta_batch = torch.stack(thetas, dim=0)

    img_warp, msk_warp = _affine_grid_sample(images, masks, theta_batch)

    # Optional elastic
    if p_elastic > 0.0 and elastic_alpha > 0.0:
        do_elastic = torch.rand(1, device=device).item() < p_elastic
        if do_elastic:
            base_grid = torch.nn.functional.affine_grid(torch.eye(2, 3, device=device).unsqueeze(0).repeat(b, 1, 1),
                                                        size=(b, c, h, w), align_corners=False)
            disp = _elastic_displacement(b, h, w, device, alpha=elastic_alpha, kernel_size=elastic_kernel)
            grid = base_grid + disp
            img_warp = torch.nn.functional.grid_sample(img_warp, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            msk_warp = torch.nn.functional.grid_sample(msk_warp, grid, mode='nearest', padding_mode='zeros', align_corners=False)

    return img_warp, msk_warp


def setup_logging():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results/pr_sam', exist_ok=True)
    log_file = f'results/pr_sam/pr_sam_train_{ts}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file


def evaluate(model, loader, device, quality_name: str = ""):
    model.eval()
    metrics = {'dice': [], 'iou': [], 'hd95': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}
    with torch.no_grad():
        val_iter = tqdm(loader, desc=f"Val {quality_name}" if quality_name else "Val", leave=False)
        for batch in val_iter:
            images = batch['image'].to(device)  # [B,1,256,256] CBCT
            atlas = batch['atlas_prior'].to(device)   # [B,3,256,256] prior from CT mask
            masks = batch['mask'].to(device)
            # 我们的bbox在当前256x256空间内提取，需将original_hw传入同一坐标系
            original_hw = torch.tensor([[images.shape[-2], images.shape[-1]]], device=device, dtype=torch.float32).repeat(images.size(0), 1)

            # Build bbox from CT prior binary channel to avoid GT leakage
            with torch.no_grad():
                bboxes = []
                for i in range(masks.size(0)):
                    prior_bin = (atlas[i, 0] > 0.5)
                    ys, xs = torch.where(prior_bin)
                    if ys.numel() == 0:
                        bboxes.append(torch.tensor([0, 0, masks.shape[-1]-1, masks.shape[-2]-1], device=device, dtype=torch.float32))
                    else:
                        y1, y2 = ys.min().item(), ys.max().item()
                        x1, x2 = xs.min().item(), xs.max().item()
                        bboxes.append(torch.tensor([x1, y1, x2, y2], device=device, dtype=torch.float32))
                bbox_tensor = torch.stack(bboxes, dim=0)
            # For a first run, reuse CBCT as atlas proxy (later replace with real pCT loader)
            logits, _, *_ = model(images, atlas, bbox_prompts_xyxy=bbox_tensor, original_image_size_hw=original_hw, multimask_output=False, sample_ids=batch['idx'].to(device))
            # SAM decoder returns [B, 1, 256, 256] by default; ensure both sides same size & channel
            logits = resize_to(masks, logits)
            if logits.shape[1] != masks.shape[1]:
                logits = logits[:, :1]
            preds = torch.sigmoid(logits)

            metrics['dice'].append(float(dice_score(preds, masks)))
            metrics['iou'].append(float(iou_score(preds, masks)))
            metrics['hd95'].append(float(hd95_score(preds, masks)))
            metrics['accuracy'].append(float(accuracy_score(preds, masks)))
            metrics['precision'].append(float(precision_score(preds, masks)))
            metrics['recall'].append(float(recall_score(preds, masks)))
            metrics['f1'].append(float(f1_score(preds, masks)))
            metrics['specificity'].append(float(specificity_score(preds, masks)))
    out = {k: (sum(v)/len(v) if v else 0.0) for k, v in metrics.items()}
    return out


def main():
    log_file = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    base_dir = "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed"
    mask_dir = "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed/Masks"

    train_loader, val_loaders, _test_loaders = create_int8_multi_quality_dataloaders(
        base_dir=base_dir,
        mask_dir=mask_dir,
        qualities=['32', '64', '128', '256', '490'],
        batch_size=1,
        val_split=0.1,
        test_split=0.1,
        patient_level_split=True,
        num_workers=4,
        target_size=256,
        debug_max_patients=0
    )
    # Verbose dataset stats
    try:
        total_train = len(train_loader.dataset)
    except Exception:
        total_train = sum(1 for _ in train_loader)
    logging.info(f"Train samples (all qualities): {total_train}")
    for q, vloader in val_loaders.items():
        try:
            vsz = len(vloader.dataset)
        except Exception:
            vsz = sum(1 for _ in vloader)
        logging.info(f"Val samples quality {q}: {vsz}")

    # Init PR_SAM (SAM ViT-B checkpoint required)
    sam_ckpt = 'checkpoints/sam_vit_b_01ec64.pth'
    model = PRSamModel(
        sam_model_type='vit_b',
        sam_checkpoint_path=sam_ckpt,
        device=str(device)
    ).to(device)

    # Simple training loop on mixed-quality train loader
    # Boundary-weighted loss: amplify boundary pixels
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    epochs = 1
    best_overall_dice = 0.0
    history = []
    patience = 2
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        train_dice_sum = 0.0
        train_steps = 0
        for batch in train_pbar:
            images = batch['image'].to(device)
            atlas = batch['atlas_prior'].to(device)
            masks = batch['mask'].to(device)
            original_hw = torch.tensor([[images.shape[-2], images.shape[-1]]], device=device, dtype=torch.float32).repeat(images.size(0), 1)

            # Random translate + small deformation for CBCT and GT mask (keep prior unchanged)
            images, masks = random_translate_batch(images, masks, max_shift=8)
            images, masks = random_deform_batch(images, masks, max_rotate_deg=2.0, max_scale=0.01, max_shear=0.01, elastic_alpha=0.01, elastic_kernel=21, p_elastic=0.5)

            # bbox from CT prior (binary channel)
            with torch.no_grad():
                bboxes = []
                for i in range(masks.size(0)):
                    prior_bin = (atlas[i, 0] > 0.5)
                    ys, xs = torch.where(prior_bin)
                    if ys.numel() == 0:
                        bboxes.append(torch.tensor([0, 0, masks.shape[-1]-1, masks.shape[-2]-1], device=device, dtype=torch.float32))
                    else:
                        y1, y2 = ys.min().item(), ys.max().item()
                        x1, x2 = xs.min().item(), xs.max().item()
                        bboxes.append(torch.tensor([x1, y1, x2, y2], device=device, dtype=torch.float32))
                bbox_tensor = torch.stack(bboxes, dim=0)

            out = model(images, atlas, bbox_prompts_xyxy=bbox_tensor, original_image_size_hw=original_hw, multimask_output=False, sample_ids=batch['idx'].to(device))
            # PR-SAM+ returns fused logits and cbct-only logits at the end
            if isinstance(out, (list, tuple)) and len(out) >= 7:
                logits, logits_cbct = out[0], out[6]
            else:
                logits, logits_cbct = out[0], None
            logits = resize_to(masks, logits)
            if logits_cbct is not None:
                logits_cbct = resize_to(masks, logits_cbct)
            if logits.shape[1] != masks.shape[1]:
                logits = logits[:, :1]
            # Boundary weighting mask (ring)
            with torch.no_grad():
                m_np = (masks > 0.5).float()
                eroded = torch.nn.functional.max_pool2d(1 - m_np, kernel_size=3, stride=1, padding=1)
                eroded = 1 - eroded
                boundary = (m_np - (m_np * eroded)).abs()
                weight = 1.0 + 1.0 * boundary  # ×2 on boundary
            loss_main = criterion(logits, masks)
            loss_aux = 0.0
            if logits_cbct is not None:
                loss_aux = criterion(logits_cbct, masks)
            loss_raw = loss_main + 0.5 * loss_aux
            # Regularization placeholder (no FiLM now)
            loss = (loss_raw * weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            with torch.no_grad():
                if logits_cbct is not None:
                    preds = (torch.sigmoid(logits) + torch.sigmoid(logits_cbct)) / 2.0
                else:
                    preds = torch.sigmoid(logits)
                batch_dice_tensor = dice_score(preds, masks)
                batch_dice = float(batch_dice_tensor.detach().item() if hasattr(batch_dice_tensor, 'item') else batch_dice_tensor)
            train_dice_sum += batch_dice
            train_steps += 1
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{batch_dice:.4f}"})

        # Evaluate overall on merged val (average over qualities)
        overall = {'dice': [], 'iou': [], 'hd95': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}
        per_quality = {}
        for q, vloader in val_loaders.items():
            res = evaluate(model, vloader, device, quality_name=q)
            per_quality[q] = res
            for k in overall:
                overall[k].append(res[k])
        overall_mean = {k: (sum(v)/len(v) if v else 0.0) for k, v in overall.items()}

        train_dice_mean = (train_dice_sum / max(1, train_steps))
        logging.info(f"Epoch {epoch+1}/{epochs} done in {time.time()-epoch_start:.1f}s - loss: {running_loss:.4f} - train dice: {train_dice_mean:.4f} - overall dice: {overall_mean['dice']:.4f}")
        history.append({'epoch': epoch+1, 'loss': running_loss, 'train_dice': train_dice_mean, 'overall': overall_mean, 'per_quality': per_quality})

        if overall_mean['dice'] > best_overall_dice:
            best_overall_dice = overall_mean['dice']
            os.makedirs('results/pr_sam', exist_ok=True)
            torch.save(model.state_dict(), 'results/pr_sam/best_pr_sam.pt')
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step(overall_mean['dice'])
        if no_improve >= patience:
            logging.info("Early stopping triggered.")
            break
    # Save history
    os.makedirs('results/pr_sam', exist_ok=True)
    with open('results/pr_sam/pr_sam_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logging.info("Training finished.")


if __name__ == '__main__':
    main()



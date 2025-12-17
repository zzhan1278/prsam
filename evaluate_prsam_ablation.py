#!/usr/bin/env python3
"""
PR-SAM Ablation Study Evaluation Script
Evaluates trained ablation models on test set without test-time augmentation
"""
import os
import sys
import json
import torch
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))

from data.int8_dataset import create_int8_multi_quality_dataloaders
from models.proposed.pgrsam.ablation_study_models import (
    Ablation_Prior_Guided_Only,
    Ablation_No_Correction_Network,
    Ablation_No_CBCT_Adapter,
    PRSAM_Plus_Full
)
from utils.metrics import (
    dice_score, iou_score, hd95_score,
    accuracy_score, precision_score, recall_score, f1_score, specificity_score
)
from train_pgrsam import resize_to

# Configuration
CONFIG = {
    'sam_checkpoint': "F:/CT_guided_CBCT_segmentation/checkpoints/sam_vit_b_01ec64.pth",
    'data_base_dir': "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed",
    'data_mask_dir': "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed/Masks",
    'results_dir': "results/pr_sam_ablation_final"
}

def get_model(model_variant: str, device: torch.device):
    """Load ablation model variant"""
    model_map = {
        'prior_guided_only': Ablation_Prior_Guided_Only,
        'no_correction': Ablation_No_Correction_Network,
        'no_adapter': Ablation_No_CBCT_Adapter,
        'full': PRSAM_Plus_Full
    }
    
    if model_variant not in model_map:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    return model_map[model_variant](
        sam_checkpoint_path=CONFIG['sam_checkpoint'],
        device=str(device)
    ).to(device)

@torch.no_grad()
def evaluate_model(model, model_variant, test_loaders, device):
    """
    Evaluate model on test set WITHOUT test-time augmentation
    Following the same protocol as baseline models
    """
    model.eval()
    
    # Metrics to track
    metric_keys = ['dice', 'iou', 'hd95', 'accuracy', 'precision', 'recall', 'f1', 'specificity']
    overall_metrics = {k: [] for k in metric_keys}
    per_quality_results = {}
    
    for quality, loader in test_loaders.items():
        quality_metrics = {k: [] for k in metric_keys}
        
        pbar = tqdm(loader, desc=f"Testing on quality {quality}")
        
        for batch in pbar:
            # Get batch data
            images = batch['image'].to(device)
            atlas = batch['atlas_prior'].to(device)  # 3-channel prior features
            masks = batch['mask'].to(device)
            original_hw = torch.tensor([[images.shape[-2], images.shape[-1]]], 
                                      device=device, dtype=torch.float32).repeat(images.size(0), 1)
            
            # NO TEST-TIME AUGMENTATION - Direct inference only
            
            # Extract bbox from atlas prior (binary channel)
            bboxes = []
            for i in range(masks.size(0)):
                prior_bin = (atlas[i, 0] > 0.5)
                ys, xs = torch.where(prior_bin)
                if ys.numel() == 0:
                    bboxes.append(torch.tensor([0, 0, masks.shape[-1]-1, masks.shape[-2]-1], 
                                              device=device, dtype=torch.float32))
                else:
                    y1, y2 = ys.min().item(), ys.max().item()
                    x1, x2 = xs.min().item(), xs.max().item()
                    bboxes.append(torch.tensor([x1, y1, x2, y2], device=device, dtype=torch.float32))
            bbox_tensor = torch.stack(bboxes, dim=0)
            
            # Forward pass
            mask_logits_1, mask_logits_2 = model(images, atlas, bbox_tensor, original_hw)
            
            # Resize outputs to match mask size
            mask_logits_1 = resize_to(masks, mask_logits_1)
            if mask_logits_2 is not None:
                mask_logits_2 = resize_to(masks, mask_logits_2)
            
            # Ensure correct channel count
            if mask_logits_1.shape[1] != masks.shape[1]:
                mask_logits_1 = mask_logits_1[:, :1]
            if mask_logits_2 is not None and mask_logits_2.shape[1] != masks.shape[1]:
                mask_logits_2 = mask_logits_2[:, :1]
            
            # Get predictions
            if mask_logits_2 is not None:
                # Dual branch - average predictions
                preds = (torch.sigmoid(mask_logits_1) + torch.sigmoid(mask_logits_2)) / 2.0
            else:
                # Single branch
                preds = torch.sigmoid(mask_logits_1)
            
            # Calculate all metrics (handle both tensor and float returns)
            def safe_item(x):
                return x.item() if hasattr(x, 'item') else float(x)
            
            quality_metrics['dice'].append(safe_item(dice_score(preds, masks)))
            quality_metrics['iou'].append(safe_item(iou_score(preds, masks)))
            quality_metrics['hd95'].append(safe_item(hd95_score(preds, masks)))
            quality_metrics['accuracy'].append(safe_item(accuracy_score(preds, masks)))
            quality_metrics['precision'].append(safe_item(precision_score(preds, masks)))
            quality_metrics['recall'].append(safe_item(recall_score(preds, masks)))
            quality_metrics['f1'].append(safe_item(f1_score(preds, masks)))
            quality_metrics['specificity'].append(safe_item(specificity_score(preds, masks)))
            
            # Update progress bar
            pbar.set_postfix({'dice': f"{quality_metrics['dice'][-1]:.4f}"})
        
        # Calculate quality averages
        per_quality_results[quality] = {k: np.mean(v) for k, v in quality_metrics.items()}
        
        # Add to overall metrics
        for k in metric_keys:
            overall_metrics[k].append(per_quality_results[quality][k])
    
    # Calculate overall averages
    overall_avg = {k: np.mean(v) for k, v in overall_metrics.items()}
    
    return overall_avg, per_quality_results

def format_results_markdown(model_variant, overall, per_quality):
    """Format results as Markdown for documentation"""
    md_text = f"\n## PR-SAM Ablation: {model_variant}\n\n"
    
    # Overall results
    md_text += "Overall:\n\n"
    md_text += "| Dice | IoU | HD95 | Accuracy | Precision | Recall | F1 | Specificity |\n"
    md_text += "|------|-----|------|----------|-----------|--------|----|-------------|\n"
    md_text += f"| {overall['dice']:.4f} | {overall['iou']:.4f} | {overall['hd95']:.4f} | "
    md_text += f"{overall['accuracy']:.4f} | {overall['precision']:.4f} | "
    md_text += f"{overall['recall']:.4f} | {overall['f1']:.4f} | {overall['specificity']:.4f} |\n\n"
    
    # Per-quality results
    md_text += "Per-quality:\n\n"
    md_text += "| Quality | Dice | IoU | HD95 | Accuracy | Precision | Recall | F1 | Specificity |\n"
    md_text += "|---------|------|-----|------|----------|-----------|--------|----|-------------|\n"
    
    for quality in ['32', '64', '128', '256', '490']:
        if quality in per_quality:
            m = per_quality[quality]
            md_text += f"| {quality} | {m['dice']:.4f} | {m['iou']:.4f} | {m['hd95']:.4f} | "
            md_text += f"{m['accuracy']:.4f} | {m['precision']:.4f} | "
            md_text += f"{m['recall']:.4f} | {m['f1']:.4f} | {m['specificity']:.4f} |\n"
    
    return md_text

def main():
    parser = argparse.ArgumentParser(description="PR-SAM Ablation Study Evaluation")
    parser.add_argument('--variant', type=str, required=True,
                       choices=['prior_guided_only', 'no_correction', 'no_adapter', 'full'],
                       help='Model variant to evaluate')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--save_predictions', action='store_true', 
                       help='Save prediction masks for visualization')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"PR-SAM Ablation Evaluation: {args.variant}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.variant}")
    model = get_model(args.variant, device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = Path(CONFIG['results_dir']) / args.variant / 'best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or specify a valid checkpoint path.")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best validation Dice: {checkpoint.get('best_dice', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    # Load test data
    print("\nLoading test dataset...")
    _, _, test_loaders = create_int8_multi_quality_dataloaders(
        base_dir=CONFIG['data_base_dir'],
        mask_dir=CONFIG['data_mask_dir'],
        qualities=['32', '64', '128', '256', '490'],
        batch_size=1,  # Use batch_size=1 for accurate HD95 calculation
        val_split=0.1,
        test_split=0.1,
        patient_level_split=True,
        num_workers=4,
        target_size=256,
        load_subset='test_only'  # Only load test data
    )
    
    # Evaluate
    print(f"\nEvaluating {args.variant} on test set...")
    print("Note: NO test-time augmentation is applied (following baseline protocol)")
    
    overall_results, per_quality_results = evaluate_model(
        model, args.variant, test_loaders, device
    )
    
    # Print results
    print("\n" + "="*80)
    print(f"Test Results: {args.variant}")
    print("="*80)
    
    print("\nOverall Performance:")
    print(f"  Dice:        {overall_results['dice']:.4f}")
    print(f"  IoU:         {overall_results['iou']:.4f}")
    print(f"  HD95:        {overall_results['hd95']:.4f}")
    print(f"  Accuracy:    {overall_results['accuracy']:.4f}")
    print(f"  Precision:   {overall_results['precision']:.4f}")
    print(f"  Recall:      {overall_results['recall']:.4f}")
    print(f"  F1:          {overall_results['f1']:.4f}")
    print(f"  Specificity: {overall_results['specificity']:.4f}")
    
    print("\nPer-Quality Dice Scores:")
    for quality in ['32', '64', '128', '256', '490']:
        if quality in per_quality_results:
            print(f"  Quality {quality}: {per_quality_results[quality]['dice']:.4f}")
    
    # Save results
    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'model_variant': args.variant,
        'checkpoint': str(checkpoint_path),
        'protocol': 'pr_sam_ablation_test_only',
        'test_augmentation': False,
        'overall': overall_results,
        'per_quality': per_quality_results
    }
    
    save_dir = Path(CONFIG['results_dir']) / args.variant
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = save_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save markdown format
    md_text = format_results_markdown(args.variant, overall_results, per_quality_results)
    md_path = save_dir / 'test_results.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f"Markdown summary saved to: {md_path}")
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import os
import json
from datetime import datetime
from tqdm import tqdm
import torch

from models.proposed.pgrsam import PRSamModel
from data.int8_dataset import create_int8_multi_quality_dataloaders
from utils.metrics import (
    dice_score, iou_score, hd95_score,
    accuracy_score, precision_score, recall_score, f1_score, specificity_score
)


def resize_to(target, tensor):
    if tensor.shape[-2:] != target.shape[-2:]:
        return torch.nn.functional.interpolate(
            tensor, size=target.shape[-2:], mode='bilinear', align_corners=False
        )
    return tensor


@torch.no_grad()
def evaluate_model(model, val_loaders, device):
    model.eval()
    overall = {k: [] for k in ['dice', 'iou', 'hd95', 'accuracy', 'precision', 'recall', 'f1', 'specificity']}
    per_quality = {}

    for quality, loader in val_loaders.items():
        metrics = {k: [] for k in overall.keys()}
        pbar = tqdm(loader, desc=f"Eval {quality}")
        for batch in pbar:
            images = batch['image'].to(device)
            atlas_prior = batch['atlas_prior'].to(device)
            masks = batch['mask'].to(device)

            original_hw = torch.tensor(
                [[images.shape[-2], images.shape[-1]]], device=device, dtype=torch.float32
            ).repeat(images.size(0), 1)

            # bbox from CT prior (channel 0 assumed to be binary mask), avoid GT leakage
            bboxes = []
            for i in range(atlas_prior.size(0)):
                prior_bin = (atlas_prior[i, 0] > 0.5)
                ys, xs = torch.where(prior_bin)
                if ys.numel() == 0:
                    bboxes.append(torch.tensor([0, 0, masks.shape[-1]-1, masks.shape[-2]-1], device=device, dtype=torch.float32))
                else:
                    y1, y2 = ys.min().item(), ys.max().item()
                    x1, x2 = xs.min().item(), xs.max().item()
                    bboxes.append(torch.tensor([x1, y1, x2, y2], device=device, dtype=torch.float32))
            bbox_tensor = torch.stack(bboxes, dim=0)

            logits, *_ = model(
                images,
                atlas_prior,
                bbox_prompts_xyxy=bbox_tensor,
                original_image_size_hw=original_hw,
                multimask_output=False,
                sample_ids=batch['idx'].to(device)
            )
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

        per_quality[quality] = {k: (sum(v)/len(v) if v else 0.0) for k, v in metrics.items()}
        for k in overall:
            overall[k].append(per_quality[quality][k])

    overall_mean = {k: (sum(v)/len(v) if v else 0.0) for k, v in overall.items()}
    return overall_mean, per_quality


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed"
    mask_dir = "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed/Masks"

    # Only need val loaders; function returns (train_loader, val_loaders, test_loaders)
    _, val_loaders, _ = create_int8_multi_quality_dataloaders(
        base_dir=base_dir,
        mask_dir=mask_dir,
        qualities=['32', '64', '128', '256', '490'],
        batch_size=1,
        val_split=0.1,
        num_workers=4,
        target_size=256
    )

    sam_ckpt = 'checkpoints/sam_vit_b_01ec64.pth'
    model = PRSamModel(sam_model_type='vit_b', sam_checkpoint_path=sam_ckpt, device=str(device)).to(device)

    # Load best weights (renamed directory for PR_SAM)
    best_path = 'results/pr_sam/best_pr_sam.pt'
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state, strict=False)

    overall, per_quality = evaluate_model(model, val_loaders, device)

    os.makedirs('results/pr_sam', exist_ok=True)
    out = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'overall': overall,
        'per_quality': per_quality
    }
    with open('results/pr_sam/pr_sam_eval.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()



"""
Unified evaluation script for all baseline models.
Generates paper-ready results tables and visualizations.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import logging
import importlib
import argparse

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from data.dataset import CBCTDataset
from data.optimized_dataset import OptimizedCBCTDataset
from utils.metrics import DiceBCELoss, dice_score, iou_score, hd95_score, accuracy_score, precision_score, recall_score, f1_score, specificity_score

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_class(model_name, model_config):
    """Dynamically imports the model class from the corresponding module."""
    try:
        module_path = f"models.baselines.{model_name}.{model_name}_model"
        class_name = model_config['model_name']
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        return ModelClass
    except (ImportError, AttributeError) as e:
        logging.error(f"Could not import model '{model_name}': {e}")
        return None

def load_model(model_name, model_config, checkpoint_path, device):
    """Load a trained model from checkpoint"""
    ModelClass = get_model_class(model_name, model_config)
    if ModelClass is None:
        return None
    
    # Get model architecture parameters
    arch_args = model_config.get('architecture', {})
    arch_args['n_channels'] = 1
    arch_args['n_classes'] = 1
    
    # Create model
    model = ModelClass(**arch_args).to(device)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f"Loaded model {model_name} from {checkpoint_path}")
        return model
    else:
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return None

def evaluate_model_detailed(model, loader, device, model_name):
    """Detailed evaluation of a single model"""
    model.eval()
    
    all_dice = []
    all_iou = []
    all_hd95 = []
    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_specificity = []
    
    criterion = DiceBCELoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {model_name}", leave=False):
            images, true_masks = batch['image'].to(device), batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle deep supervision
            if isinstance(outputs, list):
                outputs = outputs[-1]
            
            # Calculate loss
            loss = criterion(outputs, true_masks)
            total_loss += loss.item()
            
            # Convert to predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Calculate metrics for each sample in batch
            batch_size = preds.shape[0]
            for i in range(batch_size):
                pred_slice = preds[i]
                target_slice = true_masks[i]
                
                dice = dice_score(pred_slice, target_slice).item()
                iou = iou_score(pred_slice, target_slice).item()
                hd95 = hd95_score(pred_slice, target_slice)
                accuracy = accuracy_score(pred_slice, target_slice).item()
                precision = precision_score(pred_slice, target_slice).item()
                recall = recall_score(pred_slice, target_slice).item()
                f1 = f1_score(pred_slice, target_slice).item()
                specificity = specificity_score(pred_slice, target_slice).item()
                
                all_dice.append(dice)
                all_iou.append(iou)
                all_hd95.append(hd95)
                all_accuracy.append(accuracy)
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)
                all_specificity.append(specificity)
    
    # Calculate statistics
    results = {
        'model': model_name,
        'dice_mean': np.mean(all_dice),
        'dice_std': np.std(all_dice),
        'iou_mean': np.mean(all_iou),
        'iou_std': np.std(all_iou),
        'hd95_mean': np.mean(all_hd95),
        'hd95_std': np.std(all_hd95),
        'accuracy_mean': np.mean(all_accuracy),
        'accuracy_std': np.std(all_accuracy),
        'precision_mean': np.mean(all_precision),
        'precision_std': np.std(all_precision),
        'recall_mean': np.mean(all_recall),
        'recall_std': np.std(all_recall),
        'f1_mean': np.mean(all_f1),
        'f1_std': np.std(all_f1),
        'specificity_mean': np.mean(all_specificity),
        'specificity_std': np.std(all_specificity),
        'loss_mean': total_loss / len(loader),
        'num_samples': len(all_dice)
    }
    
    return results, all_dice, all_iou, all_hd95

def get_model_info(model):
    """Get model parameter count and size"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb
    }

def save_results_table(all_results, output_dir):
    """Save results in multiple formats for paper"""
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for paper
    df = df[['model', 'dice_mean', 'dice_std', 'iou_mean', 'iou_std', 
             'hd95_mean', 'hd95_std', 'total_params', 'model_size_mb', 'num_samples']]
    
    # Sort by Dice score
    df = df.sort_values('dice_mean', ascending=False)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'baseline_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Save LaTeX table
    latex_path = os.path.join(output_dir, 'baseline_results.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline Model Comparison Results}\n")
        f.write("\\label{tab:baseline_results}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Model & Dice $\\uparrow$ & IoU $\\uparrow$ & HD95 $\\downarrow$ & Params (M) & Size (MB) \\\\\n")
        f.write("\\hline\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['model']} & "
                   f"{row['dice_mean']:.4f} $\\pm$ {row['dice_std']:.4f} & "
                   f"{row['iou_mean']:.4f} $\\pm$ {row['iou_std']:.4f} & "
                   f"{row['hd95_mean']:.2f} $\\pm$ {row['hd95_std']:.2f} & "
                   f"{row['total_params']/1e6:.1f} & "
                   f"{row['model_size_mb']:.1f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Save formatted text table
    txt_path = os.path.join(output_dir, 'baseline_results.txt')
    with open(txt_path, 'w') as f:
        f.write("BASELINE MODEL COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Model':<15} {'Dice':<15} {'IoU':<15} {'HD95':<15} {'Params(M)':<10} {'Size(MB)':<10}\n")
        f.write("-"*80 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['model']:<15} "
                   f"{row['dice_mean']:.4f}±{row['dice_std']:.4f}  "
                   f"{row['iou_mean']:.4f}±{row['iou_std']:.4f}  "
                   f"{row['hd95_mean']:.2f}±{row['hd95_std']:.2f}    "
                   f"{row['total_params']/1e6:.1f}       "
                   f"{row['model_size_mb']:.1f}\n")
    
    logging.info(f"Results saved to {output_dir}")
    return df

def create_comparison_plots(all_results, all_metrics, output_dir):
    """Create comparison plots for paper"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Bar plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = [r['model'] for r in all_results]
    dice_means = [r['dice_mean'] for r in all_results]
    dice_stds = [r['dice_std'] for r in all_results]
    iou_means = [r['iou_mean'] for r in all_results]
    iou_stds = [r['iou_std'] for r in all_results]
    hd95_means = [r['hd95_mean'] for r in all_results]
    hd95_stds = [r['hd95_std'] for r in all_results]
    
    # Dice scores
    axes[0].bar(models, dice_means, yerr=dice_stds, capsize=5)
    axes[0].set_title('Dice Score Comparison')
    axes[0].set_ylabel('Dice Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # IoU scores
    axes[1].bar(models, iou_means, yerr=iou_stds, capsize=5)
    axes[1].set_title('IoU Score Comparison')
    axes[1].set_ylabel('IoU Score')
    axes[1].tick_params(axis='x', rotation=45)
    
    # HD95 (lower is better)
    axes[2].bar(models, hd95_means, yerr=hd95_stds, capsize=5, color='coral')
    axes[2].set_title('HD95 Distance Comparison')
    axes[2].set_ylabel('HD95 (pixels)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plots for distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data for box plots
    dice_data = []
    iou_data = []
    hd95_data = []
    labels = []
    
    for model_name in models:
        if model_name in all_metrics:
            dice_data.extend(all_metrics[model_name]['dice'])
            iou_data.extend(all_metrics[model_name]['iou'])
            hd95_data.extend(all_metrics[model_name]['hd95'])
            labels.extend([model_name] * len(all_metrics[model_name]['dice']))
    
    # Create box plots
    df_box = pd.DataFrame({
        'model': labels,
        'dice': dice_data,
        'iou': iou_data,
        'hd95': hd95_data
    })
    
    sns.boxplot(data=df_box, x='model', y='dice', ax=axes[0])
    axes[0].set_title('Dice Score Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df_box, x='model', y='iou', ax=axes[1])
    axes[1].set_title('IoU Score Distribution')
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df_box, x='model', y='hd95', ax=axes[2])
    axes[2].set_title('HD95 Distance Distribution')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comparison plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate all baseline models")
    parser.add_argument('--use-optimized-dataset', action='store_true', 
                       help='Use optimized dataset for evaluation')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Setup
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    set_seed(42)
    
    project_root = os.path.abspath(os.path.dirname(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config_path = os.path.join(project_root, 'configs', 'baseline_models.yaml')
    with open(config_path, 'r') as f:
        master_config = yaml.safe_load(f)
    
    # Setup dataset
    data_dir = os.path.join(project_root, 'datasets', 'LiTS-preprocessed', 'CBCT_256')
    mask_dir = os.path.join(project_root, 'datasets', 'LiTS-preprocessed', 'Masks')
    
    if args.use_optimized_dataset:
        full_dataset = OptimizedCBCTDataset(data_dir=data_dir, mask_dir=mask_dir, 
                                           augment=False, load_to_memory=True)
    else:
        full_dataset = CBCTDataset(data_dir=data_dir, mask_dir=mask_dir, 
                                  augment=False, cache_size=40)
    
    # Same split as training (90/10)
    val_percent = 0.1
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    
    # Use validation set for evaluation
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    logging.info(f"Evaluation dataset: {len(val_set)} samples")
    
    # Evaluate each model
    all_results = []
    all_metrics = {}
    
    for model_name in master_config.keys():
        logging.info(f"\nEvaluating {model_name}...")
        
        # Check if model checkpoint exists
        checkpoint_path = os.path.join(project_root, 'results', 'baselines', 
                                     model_name, 'checkpoints', 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            logging.warning(f"Checkpoint not found for {model_name}: {checkpoint_path}")
            continue
        
        # Load model
        model = load_model(model_name, master_config[model_name], checkpoint_path, device)
        if model is None:
            continue
        
        # Get model info
        model_info = get_model_info(model)
        
        # Evaluate
        results, dice_scores, iou_scores, hd95_scores = evaluate_model_detailed(
            model, val_loader, device, model_name)
        
        # Add model info to results
        results.update(model_info)
        
        all_results.append(results)
        all_metrics[model_name] = {
            'dice': dice_scores,
            'iou': iou_scores,
            'hd95': hd95_scores
        }
        
        logging.info(f"{model_name} - Dice: {results['dice_mean']:.4f}±{results['dice_std']:.4f}, "
                    f"IoU: {results['iou_mean']:.4f}±{results['iou_std']:.4f}, "
                    f"HD95: {results['hd95_mean']:.2f}±{results['hd95_std']:.2f}")
    
    if not all_results:
        logging.error("No models found for evaluation!")
        return
    
    # Save detailed results
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump({
            'results': all_results,
            'raw_metrics': all_metrics
        }, f, indent=2)
    
    # Generate tables and plots
    df_results = save_results_table(all_results, output_dir)
    create_comparison_plots(all_results, all_metrics, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("\nTop 3 models by Dice score:")
    for i, (_, row) in enumerate(df_results.head(3).iterrows()):
        print(f"{i+1}. {row['model']}: {row['dice_mean']:.4f}±{row['dice_std']:.4f}")

if __name__ == "__main__":
    main()


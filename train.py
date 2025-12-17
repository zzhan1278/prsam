import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import logging
import torch.nn.functional as F
import yaml
import importlib
import json
import csv
import time
import random
import numpy as np

# Add project root to path to allow direct imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from data.dataset import CBCTDataset
from data.optimized_dataset import OptimizedCBCTDataset
from utils.metrics import DiceBCELoss, dice_score, iou_score, hd95_score


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Set all random seeds to {seed} for reproducibility")


def get_model_class(model_name, model_config):
    """Dynamically imports the model class from the corresponding module."""
    try:
        # Convention: for a model 'unet_plus_plus', the module is in
        # 'models/baselines/unet_plus_plus/unet_plus_plus_model.py'
        module_path = f"models.baselines.{model_name}.{model_name}_model"
        
        # The class name is read from the 'model_name' key in the config file
        class_name = model_config['model_name']

        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        logging.info(f"Successfully loaded model class '{class_name}' from '{module_path}'.")
        return ModelClass
    except (ImportError, AttributeError) as e:
        logging.error(f"Could not import model '{model_name}'. Please ensure the model file and class are correctly named and located according to convention.")
        logging.error(f"  - Searched for module: '{module_path}'")
        logging.error(f"  - Searched for class: '{class_name}'")
        logging.error(f"  - Details: {e}")
        return None

def calculate_loss(criterion, outputs, targets):
    """Calculates loss, handling both single and multiple outputs (deep supervision)."""
    if isinstance(outputs, list):
        # Initialize loss as a tensor on the same device as the targets
        loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        for i, output in enumerate(outputs):
            # The last output (most refined) could have a higher weight
            weight = 0.5 + (i + 1) / (2 * len(outputs)) 
            loss = loss + criterion(output, targets) * weight
        return loss
    else:
        # Standard single output
        return criterion(outputs, targets)

def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_dice = 0
    total_iou = 0
    total_hd95 = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            images, true_masks = batch['image'].to(device), batch['mask'].to(device)
            
            masks_pred_outputs = model(images)

            # Handle deep supervision output for loss calculation
            loss = calculate_loss(criterion, masks_pred_outputs, true_masks)
            total_loss += loss.item()

            # For metrics, only use the final output from deep supervision
            masks_pred = masks_pred_outputs
            if isinstance(masks_pred_outputs, list):
                masks_pred = masks_pred_outputs[-1]

            # For metrics, convert logits to probabilities and then to binary predictions
            probs = torch.sigmoid(masks_pred)
            preds = (probs > 0.5).float()
            
            total_dice += dice_score(preds, true_masks)
            total_iou += iou_score(preds, true_masks)
            total_hd95 += hd95_score(preds, true_masks)
    
    num_samples = len(loader)
    return (total_loss / num_samples, 
            total_dice / num_samples, 
            total_iou / num_samples, 
            total_hd95 / num_samples)


def train_model(model, device, train_loader, val_loader, epochs, lr, checkpoint_dir, early_stopping_patience, model_name):
    logging.info(f'''Starting training:
        Model:           {model_name}
        Epochs:          {epochs}
        Batch size:      {train_loader.batch_size}
        Learning rate:   {lr}
        Training size:   {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Device:          {device.type}
        Checkpoints:     {checkpoint_dir}
        Early Stopping:  {f'patience={early_stopping_patience}' if early_stopping_patience > 0 else 'Disabled'}
    ''')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss() # Our combined loss function
    
    best_val_dice = 0.0
    epochs_no_improve = 0
    
    # Training history for logging
    training_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_hd95': [],
        'learning_rate': [],
        'best_dice': []
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit='batch') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device)
                true_masks = batch['mask'].to(device)

                optimizer.zero_grad()
                
                masks_pred_outputs = model(images)

                loss = calculate_loss(criterion, masks_pred_outputs, true_masks)

                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        # --- Validation ---
        val_loss, val_dice, val_iou, val_hd95 = evaluate_model(model, val_loader, device, criterion)
        
        # Record training history
        current_train_loss = epoch_loss / len(train_loader)
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(current_train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_dice'].append(val_dice)
        training_history['val_iou'].append(val_iou)
        training_history['val_hd95'].append(val_hd95)
        training_history['learning_rate'].append(lr)
        training_history['best_dice'].append(best_val_dice)
        
        logging.info(f'Epoch {epoch + 1} --- Train Loss: {current_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val HD95: {val_hd95:.2f}')

        # --- Checkpointing & Early Stopping ---
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved: New best validation Dice score: {best_val_dice:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation Dice for {epochs_no_improve} consecutive epoch(s).")

        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
            logging.info(f"Best validation Dice score was: {best_val_dice:.4f}")
            break  # Exit the training loop
    
    # Save training history
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logging.info(f"Training history saved to {history_path}")
    
    # Save CSV for analysis
    csv_path = os.path.join(checkpoint_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou', 'val_hd95', 'learning_rate', 'best_dice'])
        for i in range(len(training_history['epoch'])):
            writer.writerow([
                training_history['epoch'][i],
                training_history['train_loss'][i],
                training_history['val_loss'][i],
                training_history['val_dice'][i],
                training_history['val_iou'][i],
                training_history['val_hd95'][i],
                training_history['learning_rate'][i],
                training_history['best_dice'][i]
            ])
    logging.info(f"Training metrics CSV saved to {csv_path}")
    
    return best_val_dice, training_history


def main():
    """
    Main function to orchestrate the training process.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Main training script for segmentation models.")
    parser.add_argument('--model', type=str, required=True, help="Name of the model to train (e.g., 'unet'). Must match a key in 'configs/baseline_models.yaml'.")
    parser.add_argument('--epochs', type=int, default=7, help='Number of training epochs.')
    parser.add_argument('--num-workers', type=int, default=10, help='Number of workers for data loading.')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Patience for early stopping. Set to 0 to disable.')
    parser.add_argument('--debug', action='store_true', help='Use a small subset of data for debugging.')
    parser.add_argument('--use-optimized-dataset', action='store_true', help='Use optimized dataset (filters empty masks, uses float16, loads to memory).')
    args = parser.parse_args()

    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # --- Set Random Seed for Reproducibility ---
    set_seed(42)

    # --- Load Master Config ---
    project_root = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(project_root, 'configs', 'baseline_models.yaml')
    try:
        with open(config_path, 'r') as f:
            master_config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        return

    model_name = args.model.lower()
    if model_name not in master_config:
        logging.error(f"Model '{model_name}' not found in configuration file '{config_path}'.")
        return
    
    model_config = master_config[model_name]
    
    # --- Override args with config values ---
    # The config file is the single source of truth for these hyperparameters
    lr = model_config['training']['learning_rate']
    batch_size = model_config['training']['batch_size']
    # Use epochs from config file, allow command line override only if explicitly provided
    epochs = model_config['training']['epochs']

    # --- Model Selection (Dynamic) ---
    ModelClass = get_model_class(model_name, model_config)
    if ModelClass is None:
        return
    
    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Get model-specific architecture arguments from config
        arch_args = model_config.get('architecture', {})
        # Ensure n_channels and n_classes are set for our task
        arch_args['n_channels'] = 1
        arch_args['n_classes'] = 1
        
        model = ModelClass(**arch_args).to(device)
        logging.info(f"Initialized model '{ModelClass.__name__}' with args: {arch_args}")

    except Exception as e:
        logging.error(f"Error initializing model class '{ModelClass.__name__}'.")
        logging.error(f"Details: {e}")
        return
    
    # --- Dataset and DataLoader ---
    data_dir = os.path.join(project_root, 'datasets', 'LiTS-preprocessed', 'CBCT_256')
    mask_dir = os.path.join(project_root, 'datasets', 'LiTS-preprocessed', 'Masks')
    
    # Choose dataset type based on argument
    if args.use_optimized_dataset:
        logging.info("Using optimized dataset (filters empty masks, uses float16, loads to memory)")
        full_dataset = OptimizedCBCTDataset(
            data_dir=data_dir, 
            mask_dir=mask_dir, 
            augment=False, 
            load_to_memory=True
        )
        
        # Print dataset info
        memory_info = full_dataset.get_memory_info()
        logging.info(f"Dataset loaded: {memory_info}")
        
        stats = full_dataset.get_statistics()
        logging.info(f"Dataset statistics: {stats}")
    else:
        logging.info("Using standard dataset with intelligent caching")
        # Use intelligent caching to leverage large memory efficiently
        # Cache ~40 volumes (out of ~131 total) to balance memory usage and performance
        full_dataset = CBCTDataset(data_dir=data_dir, mask_dir=mask_dir, augment=False, cache_size=40)
    
    # --- Train/Val Split ---
    val_percent = 0.1
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    
    if args.debug:
        logging.info("--- DEBUG MODE: Using a small subset of the data ---")
        train_set, _ = random_split(train_set, [int(len(train_set)*0.1), len(train_set) - int(len(train_set)*0.1)])
        val_set, _ = random_split(val_set, [int(len(val_set)*0.1), len(val_set) - int(len(val_set)*0.1)])

    # Optimized DataLoader settings
    if args.use_optimized_dataset:
        # For memory-loaded data, use fewer workers
        num_workers = 4
        logging.info("Using optimized DataLoader settings for memory-loaded data")
    else:
        # For disk-loaded data with caching, use more workers
        num_workers = args.num_workers
        logging.info("Using standard DataLoader settings with intelligent caching")
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # --- Checkpoint directory ---
    checkpoint_dir = os.path.join(project_root, 'results', 'baselines', model_name, 'checkpoints')

    # --- Start Training ---
    try:
        train_model(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            checkpoint_dir=checkpoint_dir,
            early_stopping_patience=args.early_stopping_patience,
            model_name=model_name
        )
        logging.info("Training finished!")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)


if __name__ == '__main__':
    main()
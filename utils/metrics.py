import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE + dice_loss

def dice_score(pred, targs):
    pred = (pred>0.5).float()
    return 2. * (pred*targs).sum() / ((pred+targs).sum() + 1e-8)

def iou_score(pred, targs):
    pred = (pred>0.5).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1e-8)

def hausdorff_distance_95(pred, target):
    """
    Calculate 95th percentile Hausdorff Distance (HD95) between prediction and target masks.
    
    Args:
        pred: Binary prediction tensor (H, W) or (B, 1, H, W)
        target: Binary target tensor (H, W) or (B, 1, H, W)
    
    Returns:
        HD95 distance in pixels
    """
    # Convert to numpy and ensure binary
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Handle batch dimension
    if pred.ndim == 4:  # (B, 1, H, W)
        pred = pred.squeeze()
        target = target.squeeze()
    elif pred.ndim == 3:  # (B, H, W)
        pred = pred.squeeze()
        target = target.squeeze()
    
    # Ensure 2D arrays
    if pred.ndim > 2:
        pred = pred[0] if pred.ndim == 3 else pred
        target = target[0] if target.ndim == 3 else target
    
    # Convert to binary
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    
    # If either mask is empty, return large distance
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return 100.0  # Large distance for empty masks
    
    # Get surface points
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return 100.0
    
    # Calculate distances from each surface point to the other surface
    distances_pred_to_target = []
    distances_target_to_pred = []
    
    for point in pred_surface:
        distances_pred_to_target.append(
            np.min(np.sqrt(np.sum((target_surface - point) ** 2, axis=1)))
        )
    
    for point in target_surface:
        distances_target_to_pred.append(
            np.min(np.sqrt(np.sum((pred_surface - point) ** 2, axis=1)))
        )
    
    # Combine all distances
    all_distances = distances_pred_to_target + distances_target_to_pred
    
    # Return 95th percentile
    return np.percentile(all_distances, 95)

def get_surface_points(binary_mask):
    """
    Extract surface points from binary mask.
    
    Args:
        binary_mask: 2D binary array
    
    Returns:
        Array of surface point coordinates
    """
    # Find contour points by looking for boundary pixels
    from scipy.ndimage import binary_erosion
    
    eroded = binary_erosion(binary_mask)
    boundary = binary_mask.astype(np.uint8) - eroded.astype(np.uint8)
    
    # Get coordinates of boundary points
    surface_coords = np.column_stack(np.where(boundary > 0))
    
    return surface_coords

def hd95_score(pred, target):
    """
    Wrapper function for HD95 calculation, compatible with training loop.
    
    Args:
        pred: Prediction tensor (B, 1, H, W) or (H, W)
        target: Target tensor (B, 1, H, W) or (H, W)
    
    Returns:
        Average HD95 score for batch
    """
    if pred.ndim == 4:  # Batch processing
        hd95_scores = []
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            pred_slice = pred[i, 0]  # Remove channel dim
            target_slice = target[i, 0]
            
            hd95 = hausdorff_distance_95(pred_slice, target_slice)
            hd95_scores.append(hd95)
        
        return np.mean(hd95_scores)
    else:
        return hausdorff_distance_95(pred, target)

def accuracy_score(pred, target):
    """
    Calculate pixel-level accuracy.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
    
    Returns:
        Accuracy score
    """
    pred = (pred > 0.5).float()
    target = target.float()
    correct = (pred == target).float()
    return correct.sum() / correct.numel()

def precision_score(pred, target):
    """
    Calculate precision (positive predictive value).
    
    Args:
        pred: Prediction tensor
        target: Target tensor
    
    Returns:
        Precision score
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    
    if predicted_positive == 0:
        return 0.0
    
    return true_positive / predicted_positive

def recall_score(pred, target):
    """
    Calculate recall (sensitivity/true positive rate).
    
    Args:
        pred: Prediction tensor
        target: Target tensor
    
    Returns:
        Recall score
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    
    if actual_positive == 0:
        return 0.0
    
    return true_positive / actual_positive

def f1_score(pred, target):
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    Args:
        pred: Prediction tensor
        target: Target tensor
    
    Returns:
        F1 score
    """
    precision = precision_score(pred, target)
    recall = recall_score(pred, target)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def specificity_score(pred, target):
    """
    Calculate specificity (true negative rate).
    
    Args:
        pred: Prediction tensor
        target: Target tensor
    
    Returns:
        Specificity score
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    true_negative = ((1 - pred) * (1 - target)).sum()
    actual_negative = (1 - target).sum()
    
    if actual_negative == 0:
        return 0.0
    
    return true_negative / actual_negative
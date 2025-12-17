import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from functools import lru_cache
import gc

class CBCTDataset(Dataset):
    """
    A PyTorch Dataset to load 2D slices from 3D CBCT/CT and Mask volumes.
    Uses intelligent caching to leverage large memory systems efficiently.
    """
    def __init__(self, data_dir, mask_dir, augment=False, transform=None, cache_size=50):
        """
        Args:
            data_dir (string): Directory with all the 3D data volumes.
            mask_dir (string): Directory with all the 3D mask volumes.
            augment (bool): Whether to apply data augmentation.
            transform (callable, optional): Optional transform to be applied on a sample.
            cache_size (int): Number of volumes to keep in memory cache (default: 50)
        """
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.transform = transform
        self.cache_size = cache_size
        
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        
        # We will store tuples of (file_index, slice_index)
        self.slice_map = []
        
        # Smart caching with LRU
        self._volume_cache = {}
        self._cache_access_order = []
        
        print("Loading dataset info and filtering empty slices...")
        total_slices = 0
        for idx, file_name in enumerate(self.data_files):
            # Assuming mask file names correspond to data file names
            data_path = os.path.join(self.data_dir, file_name)
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

            if os.path.exists(mask_path):
                # Load image headers just to get the number of slices
                data_obj = nib.load(data_path)
                mask_obj = nib.load(mask_path)
                
                # Use the minimum number of slices between the image and the mask to be safe
                num_slices = min(data_obj.shape[2], mask_obj.shape[2])
                total_slices += num_slices
                
                # Load mask volume to check which slices have liver
                mask_data = mask_obj.get_fdata()
                
                for slice_idx in range(num_slices):
                    # Check if this slice has any liver (non-zero pixels after merging classes)
                    mask_slice = mask_data[:, :, slice_idx]
                    mask_slice[mask_slice == 2] = 1  # Merge liver and tumor
                    
                    # Only include slices with liver present
                    if mask_slice.sum() > 0:
                        self.slice_map.append((idx, slice_idx))
                    
            else:
                print(f"Warning: Mask for {file_name} not found in {self.mask_dir}")
        
        filtered_slices = len(self.slice_map)
        filtered_ratio = (total_slices - filtered_slices) / total_slices * 100 if total_slices > 0 else 0
        print(f"Dataset initialized with {filtered_slices} valid slices from {len(self.data_files)} volumes")
        print(f"Filtered out {total_slices - filtered_slices} empty slices ({filtered_ratio:.1f}% of total)")
        print(f"Using intelligent caching (cache size: {cache_size} volumes)")

    def _load_volume(self, file_idx):
        """Load a volume pair (data + mask) with intelligent caching."""
        if file_idx in self._volume_cache:
            # Move to end of access order (most recently used)
            self._cache_access_order.remove(file_idx)
            self._cache_access_order.append(file_idx)
            return self._volume_cache[file_idx]
        
        # Load from disk
        data_path = os.path.join(self.data_dir, self.data_files[file_idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[file_idx])
        
        data_vol = nib.load(data_path).get_fdata(dtype=np.float32)
        mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)
        
        # Cache management - remove oldest if cache is full
        if len(self._volume_cache) >= self.cache_size:
            # Remove least recently used
            oldest_idx = self._cache_access_order.pop(0)
            del self._volume_cache[oldest_idx]
            gc.collect()  # Force garbage collection
        
        # Add to cache
        self._volume_cache[file_idx] = (data_vol, mask_vol)
        self._cache_access_order.append(file_idx)
        
        return data_vol, mask_vol

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        file_idx, slice_idx = self.slice_map[idx]
        
        # Use intelligent caching
        data_vol, mask_vol = self._load_volume(file_idx)
        
        # Extract the 2D slice
        # Data is (H, W, D), so we slice along the last dimension
        image = data_vol[:, :, slice_idx]
        mask = mask_vol[:, :, slice_idx]

        # --- TASK CHANGE: Merge liver (1) and tumor (2) into a single liver class (1) ---
        mask[mask == 2] = 1

        # --- Simple Augmentation ---
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                # Use np.fliplr and ensure array is C-contiguous
                image = np.ascontiguousarray(np.fliplr(image))
                mask = np.ascontiguousarray(np.fliplr(mask))

        # Add a channel dimension: (H, W) -> (1, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to float tensors. Loss function (BCE) expects float targets.
        sample = {
            'image': torch.from_numpy(image.copy()).float(), 
            'mask': torch.from_numpy(mask.copy()).float()
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_cache_stats(self):
        """Return cache statistics for monitoring."""
        return {
            'cached_volumes': len(self._volume_cache),
            'cache_size_limit': self.cache_size,
            'cache_hit_ratio': len(self._volume_cache) / len(self.data_files) if self.data_files else 0
        } 
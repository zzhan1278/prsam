import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from tqdm import tqdm
import gc

class OptimizedCBCTDataset(Dataset):
    """
    Optimized PyTorch Dataset that:
    1. Only keeps slices with non-empty masks (liver present)
    2. Uses float16 to reduce memory usage
    3. Loads all data into memory for maximum speed
    """
    def __init__(self, data_dir, mask_dir, augment=False, transform=None, load_to_memory=True):
        """
        Args:
            data_dir (string): Directory with all the 3D data volumes.
            mask_dir (string): Directory with all the 3D mask volumes.
            augment (bool): Whether to apply data augmentation.
            transform (callable, optional): Optional transform to be applied on a sample.
            load_to_memory (bool): Whether to load all data to memory (default: True)
        """
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.transform = transform
        self.load_to_memory = load_to_memory
        
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        
        # Store valid slices (only those with non-empty masks)
        self.valid_slices = []
        
        # If loading to memory, store all data here
        if self.load_to_memory:
            self.memory_data = []
            self._load_all_to_memory()
        else:
            # Store file mappings for on-demand loading
            self.slice_map = []
            self._build_slice_map()
        
        print(f"Optimized dataset initialized with {len(self.valid_slices)} valid slices (with liver)")
        if self.load_to_memory:
            print(f"All data loaded to memory using float16 precision")

    def _load_all_to_memory(self):
        """Load all valid slices (with non-empty masks) to memory"""
        print("Loading all data to memory (this may take a while)...")
        
        for idx, file_name in enumerate(tqdm(self.data_files, desc="Processing volumes")):
            data_path = os.path.join(self.data_dir, file_name)
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

            if os.path.exists(mask_path):
                # Load volumes
                data_vol = nib.load(data_path).get_fdata(dtype=np.float32)
                mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                num_slices = min(data_vol.shape[2], mask_vol.shape[2])
                
                # Process each slice
                for slice_idx in range(num_slices):
                    image_slice = data_vol[:, :, slice_idx]
                    mask_slice = mask_vol[:, :, slice_idx]
                    
                    # Merge liver (1) and tumor (2) into single liver class (1)
                    mask_slice[mask_slice == 2] = 1
                    
                    # Only keep slices with non-empty masks (liver present)
                    if mask_slice.sum() > 0:
                        # Convert to float16 to save memory
                        image_slice = image_slice.astype(np.float16)
                        mask_slice = mask_slice.astype(np.float16)
                        
                        # Add channel dimension
                        image_slice = np.expand_dims(image_slice, axis=0)
                        mask_slice = np.expand_dims(mask_slice, axis=0)
                        
                        # Store in memory
                        self.memory_data.append({
                            'image': image_slice,
                            'mask': mask_slice,
                            'file_idx': idx,
                            'slice_idx': slice_idx
                        })
                        
                        self.valid_slices.append((idx, slice_idx))
                
                # Force garbage collection after each volume
                del data_vol, mask_vol
                gc.collect()
            else:
                print(f"Warning: Mask for {file_name} not found in {self.mask_dir}")
        
        print(f"Memory usage: {len(self.memory_data)} slices loaded")

    def _build_slice_map(self):
        """Build slice map for on-demand loading (fallback option)"""
        print("Building slice map for on-demand loading...")
        
        for idx, file_name in enumerate(tqdm(self.data_files, desc="Scanning volumes")):
            data_path = os.path.join(self.data_dir, file_name)
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

            if os.path.exists(mask_path):
                # Load mask to check which slices have liver
                mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)
                
                for slice_idx in range(mask_vol.shape[2]):
                    mask_slice = mask_vol[:, :, slice_idx]
                    # Merge liver (1) and tumor (2)
                    mask_slice[mask_slice == 2] = 1
                    
                    # Only keep slices with non-empty masks
                    if mask_slice.sum() > 0:
                        self.slice_map.append((idx, slice_idx))
                        self.valid_slices.append((idx, slice_idx))
                
                del mask_vol
                gc.collect()

    def __len__(self):
        return len(self.valid_slices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.load_to_memory:
            # Load from memory
            data = self.memory_data[idx]
            image = data['image'].copy()
            mask = data['mask'].copy()
        else:
            # Load from disk (fallback)
            file_idx, slice_idx = self.slice_map[idx]
            
            data_path = os.path.join(self.data_dir, self.data_files[file_idx])
            mask_path = os.path.join(self.mask_dir, self.mask_files[file_idx])
            
            data_vol = nib.load(data_path).get_fdata(dtype=np.float32)
            mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)
            
            image = data_vol[:, :, slice_idx]
            mask = mask_vol[:, :, slice_idx]
            
            # Merge liver (1) and tumor (2) into single liver class (1)
            mask[mask == 2] = 1
            
            # Convert to float16 and add channel dimension
            image = np.expand_dims(image.astype(np.float16), axis=0)
            mask = np.expand_dims(mask.astype(np.float16), axis=0)

        # Simple Augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                mask = np.ascontiguousarray(np.fliplr(mask))

        # Convert to float32 tensors for training (PyTorch prefers float32)
        sample = {
            'image': torch.from_numpy(image).float(), 
            'mask': torch.from_numpy(mask).float()
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_memory_info(self):
        """Return memory usage information"""
        if self.load_to_memory:
            # Estimate memory usage
            sample_size = self.memory_data[0]['image'].nbytes + self.memory_data[0]['mask'].nbytes
            total_memory_mb = (sample_size * len(self.memory_data)) / (1024 * 1024)
            return {
                'loaded_to_memory': True,
                'total_slices': len(self.memory_data),
                'estimated_memory_mb': total_memory_mb,
                'data_type': 'float16'
            }
        else:
            return {
                'loaded_to_memory': False,
                'total_slices': len(self.valid_slices),
                'estimated_memory_mb': 0,
                'data_type': 'on_demand'
            }

    def get_statistics(self):
        """Return dataset statistics"""
        if not self.load_to_memory:
            return {"error": "Statistics only available when data is loaded to memory"}
        
        # Calculate statistics from memory data
        all_images = [data['image'] for data in self.memory_data]
        all_masks = [data['mask'] for data in self.memory_data]
        
        # Calculate basic stats
        total_pixels = sum(img.size for img in all_images)
        total_liver_pixels = sum(mask.sum() for mask in all_masks)
        
        return {
            'total_slices': len(self.memory_data),
            'total_pixels': total_pixels,
            'total_liver_pixels': int(total_liver_pixels),
            'liver_ratio': total_liver_pixels / total_pixels,
            'avg_liver_per_slice': total_liver_pixels / len(self.memory_data)
        } 
"""
Multi-Quality CBCT Dataset
支持多种质量CBCT数据的训练和验证

功能：
1. 训练时混合所有quality的数据
2. 验证时分别评估每种quality
3. 可视化时分别显示每种quality的结果
"""

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from tqdm import tqdm
import gc
from collections import defaultdict

class MultiQualityCBCTDataset(Dataset):
    """
    多质量CBCT数据集，支持：
    1. 混合训练所有quality数据
    2. 按quality分别验证和可视化
    """
    
    def __init__(self, base_dir, mask_dir, qualities=None, augment=False, transform=None, 
                 load_to_memory=True, target_size=256):
        """
        Args:
            base_dir: 基础数据目录 (e.g., 'datasets/LiTS-preprocessed')
            mask_dir: 掩码目录 (e.g., 'datasets/LiTS-preprocessed/Masks')
            qualities: 要使用的质量列表 (e.g., ['32', '64', '128', '256', '490'])
            augment: 是否使用数据增强
            transform: 可选的数据变换
            load_to_memory: 是否加载到内存
            target_size: 目标图像尺寸（所有图像会resize到这个尺寸）
        """
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.transform = transform
        self.load_to_memory = load_to_memory
        self.target_size = target_size
        
        # 默认使用所有可用的质量
        if qualities is None:
            qualities = ['32', '64', '128', '256', '490']
        self.qualities = qualities
        
        # 检查质量目录是否存在
        self.quality_dirs = {}
        for quality in qualities:
            quality_dir = os.path.join(base_dir, f'CBCT_{quality}')
            if os.path.exists(quality_dir):
                self.quality_dirs[quality] = quality_dir
            else:
                print(f"警告: 质量 {quality} 的目录不存在: {quality_dir}")
        
        print(f"发现 {len(self.quality_dirs)} 种质量的CBCT数据: {list(self.quality_dirs.keys())}")
        
        # 获取掩码文件列表
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        
        # 存储所有有效切片的信息
        self.valid_slices = []  # [(quality, file_idx, slice_idx), ...]
        self.quality_slice_counts = defaultdict(int)  # 每种质量的切片数量
        self.quality_file_mapping = {}  # 每种质量的文件映射
        
        # 如果加载到内存，存储所有数据
        if self.load_to_memory:
            self.memory_data = []
            self._load_all_to_memory()
        else:
            self._build_slice_map()
        
        # 打印统计信息
        total_slices = len(self.valid_slices)
        print(f"\n多质量数据集初始化完成:")
        print(f"  总有效切片: {total_slices:,}")
        for quality in sorted(self.quality_slice_counts.keys()):
            count = self.quality_slice_counts[quality]
            percentage = count / total_slices * 100 if total_slices > 0 else 0
            print(f"  质量 {quality}: {count:,} 切片 ({percentage:.1f}%)")
        
        if self.load_to_memory:
            print(f"  所有数据已加载到内存 (float16)")

    def _load_all_to_memory(self):
        """将所有有效切片加载到内存"""
        print("正在将多质量CBCT数据加载到内存...")
        
        for quality, quality_dir in tqdm(self.quality_dirs.items(), desc="处理质量"):
            # 获取该质量下的所有文件
            data_files = sorted([f for f in os.listdir(quality_dir) if f.endswith('.nii.gz')])
            self.quality_file_mapping[quality] = data_files
            
            for file_idx, file_name in enumerate(tqdm(data_files, desc=f"质量{quality}", leave=False)):
                data_path = os.path.join(quality_dir, file_name)
                
                # 找到对应的掩码文件
                mask_file = None
                for mask_name in self.mask_files:
                    if self._match_files(file_name, mask_name):
                        mask_file = mask_name
                        break
                
                if mask_file is None:
                    print(f"警告: 未找到 {file_name} 对应的掩码文件")
                    continue
                
                mask_path = os.path.join(self.mask_dir, mask_file)
                
                try:
                    # 加载数据和掩码
                    data_vol = nib.load(data_path).get_fdata(dtype=np.float32)
                    mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)
                    
                    num_slices = min(data_vol.shape[2], mask_vol.shape[2])
                    
                    # 处理每个切片
                    for slice_idx in range(num_slices):
                        image_slice = data_vol[:, :, slice_idx]
                        mask_slice = mask_vol[:, :, slice_idx]
                        
                        # 合并肝脏和肿瘤类别
                        mask_slice[mask_slice == 2] = 1
                        
                        # 只保留有肝脏的切片
                        if mask_slice.sum() > 0:
                            # Resize到目标尺寸
                            image_resized = self._resize_image(image_slice, self.target_size)
                            mask_resized = self._resize_image(mask_slice, self.target_size)
                            
                            # 转换为float16节省内存
                            image_resized = image_resized.astype(np.float16)
                            mask_resized = mask_resized.astype(np.float16)
                            
                            # 添加通道维度
                            image_resized = np.expand_dims(image_resized, axis=0)
                            mask_resized = np.expand_dims(mask_resized, axis=0)
                            
                            # 存储到内存
                            self.memory_data.append({
                                'image': image_resized,
                                'mask': mask_resized,
                                'quality': quality,
                                'file_idx': file_idx,
                                'slice_idx': slice_idx,
                                'original_size': image_slice.shape
                            })
                            
                            self.valid_slices.append((quality, file_idx, slice_idx))
                            self.quality_slice_counts[quality] += 1
                    
                    # 强制垃圾回收
                    del data_vol, mask_vol
                    gc.collect()
                    
                except Exception as e:
                    print(f"加载文件失败 {data_path}: {e}")
                    continue

    def _build_slice_map(self):
        """构建切片映射（不加载到内存的备选方案）"""
        print("构建多质量切片映射...")
        
        for quality, quality_dir in self.quality_dirs.items():
            data_files = sorted([f for f in os.listdir(quality_dir) if f.endswith('.nii.gz')])
            self.quality_file_mapping[quality] = data_files
            
            for file_idx, file_name in enumerate(data_files):
                # 找到对应的掩码文件
                mask_file = None
                for mask_name in self.mask_files:
                    if self._match_files(file_name, mask_name):
                        mask_file = mask_name
                        break
                
                if mask_file is None:
                    continue
                
                mask_path = os.path.join(self.mask_dir, mask_file)
                
                try:
                    # 只加载掩码来检查哪些切片有肝脏
                    mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)
                    
                    for slice_idx in range(mask_vol.shape[2]):
                        mask_slice = mask_vol[:, :, slice_idx]
                        mask_slice[mask_slice == 2] = 1
                        
                        if mask_slice.sum() > 0:
                            self.valid_slices.append((quality, file_idx, slice_idx))
                            self.quality_slice_counts[quality] += 1
                    
                    del mask_vol
                    gc.collect()
                    
                except Exception as e:
                    print(f"处理掩码失败 {mask_path}: {e}")
                    continue

    def _match_files(self, data_file, mask_file):
        """匹配数据文件和掩码文件"""
        # 匹配 cbct-X.nii.gz 和 mask-X.nii.gz
        data_base = data_file.replace('.nii.gz', '').replace('cbct-', '')
        mask_base = mask_file.replace('.nii.gz', '').replace('mask-', '')
        return data_base == mask_base

    def _resize_image(self, image, target_size):
        """将图像resize到目标尺寸"""
        from scipy.ndimage import zoom
        
        current_size = image.shape
        if current_size[0] == target_size and current_size[1] == target_size:
            return image
        
        # 计算缩放因子
        zoom_factors = (target_size / current_size[0], target_size / current_size[1])
        
        # 使用双线性插值进行resize
        resized = zoom(image, zoom_factors, order=1, mode='nearest')
        
        return resized

    def __len__(self):
        return len(self.valid_slices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.load_to_memory:
            # 从内存加载
            data = self.memory_data[idx]
            image = data['image'].copy()
            mask = data['mask'].copy()
            quality = data['quality']
        else:
            # 从磁盘加载（备选方案）
            quality, file_idx, slice_idx = self.valid_slices[idx]
            
            data_file = self.quality_file_mapping[quality][file_idx]
            data_path = os.path.join(self.quality_dirs[quality], data_file)
            
            # 找到对应的掩码文件
            mask_file = None
            for mask_name in self.mask_files:
                if self._match_files(data_file, mask_name):
                    mask_file = mask_name
                    break
            
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            # 加载数据
            data_vol = nib.load(data_path).get_fdata(dtype=np.float32)
            mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)
            
            image = data_vol[:, :, slice_idx]
            mask = mask_vol[:, :, slice_idx]
            
            # 合并类别
            mask[mask == 2] = 1
            
            # Resize
            image = self._resize_image(image, self.target_size)
            mask = self._resize_image(mask, self.target_size)
            
            # 添加通道维度
            image = np.expand_dims(image.astype(np.float16), axis=0)
            mask = np.expand_dims(mask.astype(np.float16), axis=0)

        # 数据增强
        if self.augment:
            if torch.rand(1) > 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                mask = np.ascontiguousarray(np.fliplr(mask))

        # 转换为tensor
        sample = {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).float(),
            'quality': quality,
            'idx': idx
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_quality_indices(self, quality):
        """获取指定质量的所有样本索引"""
        indices = []
        for i, (q, _, _) in enumerate(self.valid_slices):
            if q == quality:
                indices.append(i)
        return indices

    def get_quality_subset(self, quality):
        """获取指定质量的子数据集"""
        indices = self.get_quality_indices(quality)
        return torch.utils.data.Subset(self, indices)

    def get_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'total_slices': len(self.valid_slices),
            'qualities': list(self.quality_dirs.keys()),
            'quality_counts': dict(self.quality_slice_counts),
            'target_size': self.target_size,
            'loaded_to_memory': self.load_to_memory
        }
        
        if self.load_to_memory and self.memory_data:
            # 估算内存使用
            sample_size = self.memory_data[0]['image'].nbytes + self.memory_data[0]['mask'].nbytes
            total_memory_mb = (sample_size * len(self.memory_data)) / (1024 * 1024)
            stats['memory_usage_mb'] = total_memory_mb
        
        return stats

def create_multi_quality_dataloaders(base_dir, mask_dir, qualities=None, batch_size=8, 
                                   val_split=0.1, num_workers=4, target_size=256):
    """
    创建多质量数据加载器
    
    Returns:
        train_loader: 训练数据加载器（混合所有质量）
        val_loaders: 验证数据加载器字典 {quality: loader}
    """
    from torch.utils.data import DataLoader, random_split
    
    # 创建完整数据集
    full_dataset = MultiQualityCBCTDataset(
        base_dir=base_dir,
        mask_dir=mask_dir,
        qualities=qualities,
        augment=False,  # 验证时不使用增强
        load_to_memory=True,
        target_size=target_size
    )
    
    # 分割训练和验证集
    torch.manual_seed(42)  # 确保可重复性
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    
    # 创建训练数据加载器（混合所有质量）
    train_dataset.dataset.augment = True  # 训练时使用增强
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 为每种质量创建验证数据加载器
    val_loaders = {}
    for quality in full_dataset.qualities:
        # 获取该质量在验证集中的索引
        quality_val_indices = []
        for i, global_idx in enumerate(val_dataset.indices):
            sample_quality = full_dataset.valid_slices[global_idx][0]
            if sample_quality == quality:
                quality_val_indices.append(i)
        
        if quality_val_indices:
            quality_val_subset = torch.utils.data.Subset(val_dataset, quality_val_indices)
            val_loaders[quality] = DataLoader(
                quality_val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            print(f"质量 {quality} 验证集: {len(quality_val_indices)} 个样本")
    
    return train_loader, val_loaders

if __name__ == "__main__":
    # 测试多质量数据集
    base_dir = "datasets/LiTS-preprocessed"
    mask_dir = "datasets/LiTS-preprocessed/Masks"
    
    dataset = MultiQualityCBCTDataset(
        base_dir=base_dir,
        mask_dir=mask_dir,
        qualities=['32', '64', '128', '256'],  # 测试4种质量
        target_size=256
    )
    
    print(f"\n数据集统计: {dataset.get_statistics()}")
    
    # 测试样本
    sample = dataset[0]
    print(f"样本形状: 图像 {sample['image'].shape}, 掩码 {sample['mask'].shape}")
    print(f"样本质量: {sample['quality']}")

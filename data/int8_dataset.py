"""
Int8 Optimized Multi-Quality CBCT Dataset
使用int8数据类型大幅减少内存使用，提升训练和验证速度

内存优化：
- float16 -> int8: 内存减少50%
- 图像数据范围: [0, 255] (uint8)
- 掩码数据: {0, 1} (uint8)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import numpy as np
from tqdm import tqdm
import gc
from collections import defaultdict
import scipy.ndimage as ndi
import re

class Int8MultiQualityCBCTDataset(Dataset):
    """
    内存优化的多质量CBCT数据集
    使用int8/uint8数据类型，内存使用减少50%
    """
    
    def __init__(self, base_dir, mask_dir, qualities=None, augment=False, transform=None, 
                 load_to_memory=True, target_size=256, allowed_patient_ids=None):
        """
        Args:
            base_dir: 基础数据目录
            mask_dir: 掩码目录
            qualities: 要使用的质量列表
            augment: 是否使用数据增强
            transform: 可选的数据变换
            load_to_memory: 是否加载到内存
            target_size: 目标图像尺寸
        """
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.transform = transform
        self.load_to_memory = load_to_memory
        self.target_size = target_size
        self.allowed_patient_ids = set(allowed_patient_ids) if allowed_patient_ids is not None else None
        
        # 默认使用所有可用的质量
        if qualities is None:
            qualities = ['32', '64', '128', '256', '490']
        self.qualities = qualities
        
        # 检查质量目录是否存在
        self.quality_dirs = {}
        for quality in qualities:
            quality_dir = os.path.join(base_dir, f"CBCT_{quality}")
            if os.path.exists(quality_dir):
                self.quality_dirs[quality] = quality_dir
            else:
                print(f"警告: 质量 {quality} 的目录不存在: {quality_dir}")
        
        print(f"发现 {len(self.quality_dirs)} 种质量的CBCT数据: {list(self.quality_dirs.keys())}")
        
        # 获取掩码文件列表（CT masks，对齐到CBCT/CT）
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])

        # 获取CT文件列表并建立ID映射
        self.ct_dir = os.path.join(base_dir, "CT")
        self.ct_files = sorted([f for f in os.listdir(self.ct_dir) if f.endswith('.nii.gz')]) if os.path.exists(self.ct_dir) else []
        self.ctid_to_file = {}
        for f in self.ct_files:
            fid = self._extract_id(f)
            if fid is not None:
                self.ctid_to_file[fid] = f
        
        # 存储所有有效切片的信息
        self.valid_slices = []  # [(quality, file_idx, slice_idx), ...]
        self.quality_slice_counts = defaultdict(int)  # 每种质量的切片数量
        
        # 如果加载到内存，存储所有数据
        if self.load_to_memory:
            self.memory_data = []
            self._load_all_to_memory()
        else:
            self._build_slice_map()
        
        # 打印统计信息
        total_slices = len(self.valid_slices)
        print(f"\nInt8多质量数据集初始化完成:")
        print(f"  总有效切片: {total_slices:,}")
        for quality in sorted(self.quality_slice_counts.keys()):
            count = self.quality_slice_counts[quality]
            percentage = count / total_slices * 100 if total_slices > 0 else 0
            print(f"  质量 {quality}: {count:,} 切片 ({percentage:.1f}%)")
        
        if self.load_to_memory:
            # 估算内存使用
            if self.memory_data:
                sample_size = self.memory_data[0]['image'].nbytes + self.memory_data[0]['mask'].nbytes
                total_memory_mb = (sample_size * len(self.memory_data)) / (1024 * 1024)
                print(f"  内存使用: {total_memory_mb:.1f} MB (int8/uint8优化)")

    def _normalize_to_uint8(self, image):
        """将图像归一化到[0, 255]范围并转换为uint8"""
        # 假设输入图像已经在合理范围内，进行简单的归一化
        image_min = image.min()
        image_max = image.max()
        
        if image_max > image_min:
            # 归一化到[0, 1]
            image_norm = (image - image_min) / (image_max - image_min)
            # 转换到[0, 255]
            image_uint8 = (image_norm * 255).astype(np.uint8)
        else:
            # 如果图像是常数，设为128
            image_uint8 = np.full_like(image, 128, dtype=np.uint8)
        
        return image_uint8

    def _extract_id(self, filename):
        """提取文件名中的数字ID，统一用于CBCT/CT/Mask匹配。例如 cbct-0001.nii.gz -> 1; CT_0010.nii.gz -> 10."""
        name = os.path.basename(filename)
        m = re.findall(r"(\d+)", name)
        if not m:
            return None
        try:
            return int(m[-1])  # 使用最后一组数字更鲁棒
        except Exception:
            return None

    def _match_files(self, data_file, mask_file):
        """基于数字ID匹配CBCT与Mask"""
        did = self._extract_id(data_file)
        mid = self._extract_id(mask_file)
        return did is not None and mid is not None and did == mid

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

    def _load_all_to_memory(self):
        """将所有有效切片加载到内存（int8优化版本）"""
        print("正在将多质量CBCT数据加载到内存 (int8优化)...")
        
        for quality, quality_dir in tqdm(self.quality_dirs.items(), desc="处理质量"):
            # 获取该质量下的所有文件
            data_files = sorted([f for f in os.listdir(quality_dir) if f.endswith('.nii.gz')])
            
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
                # 匹配CT文件（基于数字ID）
                base_id = self._extract_id(file_name)
                # 如果限定了允许的病人ID，且当前不在其中，则跳过加载
                if self.allowed_patient_ids is not None and base_id not in self.allowed_patient_ids:
                    continue
                ct_file = self.ctid_to_file.get(base_id, None)
                if ct_file is None:
                    print(f"警告: 未找到 {file_name} 对应的CT文件 (id={base_id})")
                    continue
                ct_path = os.path.join(self.ct_dir, ct_file)
                
                try:
                    # 加载数据、CT与掩码
                    data_vol = nib.load(data_path).get_fdata(dtype=np.float32)
                    ct_vol = nib.load(ct_path).get_fdata(dtype=np.float32)
                    mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)
                    
                    num_slices = min(data_vol.shape[2], mask_vol.shape[2])
                    
                    # 处理每个切片
                    for slice_idx in range(num_slices):
                        image_slice = data_vol[:, :, slice_idx]
                        ct_slice = ct_vol[:, :, slice_idx]
                        mask_slice = mask_vol[:, :, slice_idx]
                        
                        # 合并肝脏和肿瘤类别
                        mask_slice[mask_slice == 2] = 1
                        
                        # 只保留有肝脏的切片
                        if mask_slice.sum() > 0:
                            # Resize到目标尺寸
                            image_resized = self._resize_image(image_slice, self.target_size)
                            atlas_resized = self._resize_image(ct_slice, self.target_size)
                            mask_resized = self._resize_image(mask_slice, self.target_size)
                            
                            # 转换为int8/uint8节省内存
                            image_uint8 = self._normalize_to_uint8(image_resized)
                            atlas_uint8 = self._normalize_to_uint8(atlas_resized)
                            mask_uint8 = (mask_resized > 0.5).astype(np.uint8)  # 二值化掩码
                            
                            # 添加通道维度
                            image_uint8 = np.expand_dims(image_uint8, axis=0)
                            atlas_uint8 = np.expand_dims(atlas_uint8, axis=0)
                            mask_uint8 = np.expand_dims(mask_uint8, axis=0)
                            
                            # 存储到内存
                            self.memory_data.append({
                                'image': image_uint8,
                                'atlas': atlas_uint8,
                                'mask': mask_uint8,
                                'quality': quality,
                                'file_idx': file_idx,
                                'slice_idx': slice_idx,
                                'original_size': image_slice.shape,
                                'patient_id': base_id
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
        """构建切片映射（非内存模式）"""
        # 简化版本，实际项目中可以实现按需加载
        pass

    def __len__(self):
        return len(self.valid_slices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.load_to_memory:
            # 从内存获取数据
            data = self.memory_data[idx]
            image = data['image'].copy()  # uint8
            atlas = data['atlas'].copy()  # uint8 (CT slice intensities, may be unused if using mask-derived prior)
            mask = data['mask'].copy()    # uint8
            quality = data['quality']
            patient_id = data.get('patient_id', -1)
        else:
            # 按需加载（简化版本）
            raise NotImplementedError("按需加载模式尚未实现")
        
        # 数据增强
        if self.augment:
            # 随机水平翻转
            if np.random.random() > 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                mask = np.ascontiguousarray(np.fliplr(mask))

        # 转换为tensor并归一化到[0, 1]
        # Build prior features from mask (3 channels: binary, signed distance, boundary)
        mask2d = mask[0].astype(np.uint8)
        dist_pos = ndi.distance_transform_edt(mask2d)
        dist_neg = ndi.distance_transform_edt(1 - mask2d)
        signed = dist_pos - dist_neg
        max_abs = np.max(np.abs(signed)) + 1e-6
        signed_norm = (signed / max_abs + 1.0) * 0.5  # to [0,1]
        eroded = ndi.binary_erosion(mask2d, iterations=1)
        boundary = np.logical_xor(mask2d.astype(bool), eroded).astype(np.float32)
        prior = np.stack([
            mask2d.astype(np.float32),
            signed_norm.astype(np.float32),
            boundary
        ], axis=0)

        sample = {
            'image': torch.from_numpy(image).float() / 255.0,   # uint8 -> [0,1]
            'atlas': torch.from_numpy(atlas).float() / 255.0,   # uint8 -> [0,1] (optional)
            'atlas_prior': torch.from_numpy(prior).float(),     # 3xHxW prior features from CT mask
            'mask': torch.from_numpy(mask).float(),             # uint8 -> float
            'quality': quality,
            'idx': torch.tensor(idx, dtype=torch.long),
            'original_size': data['original_size'],
            'patient_id': torch.tensor(patient_id, dtype=torch.long)
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
            'loaded_to_memory': self.load_to_memory,
            'data_type': 'uint8/int8 (optimized)'
        }
        
        if self.load_to_memory and self.memory_data:
            # 估算内存使用
            sample_size = self.memory_data[0]['image'].nbytes + self.memory_data[0]['mask'].nbytes
            total_memory_mb = (sample_size * len(self.memory_data)) / (1024 * 1024)
            stats['memory_usage_mb'] = total_memory_mb
        
        return stats

def create_int8_multi_quality_dataloaders(base_dir, mask_dir, qualities=None, batch_size=8, 
                                         val_split=0.1, num_workers=4, target_size=256,
                                         patient_level_split=False, test_split=0.1,
                                         load_subset='all', debug_max_patients: int = 0):
    """
    创建int8优化的多质量数据加载器
    
    Returns:
        train_loader: 训练数据加载器（混合所有质量）
        val_loaders: 验证数据加载器字典 {quality: loader}
    """
    
    # 如果只加载test子集（按病人级划分），先基于文件名确定测试病人ID集合
    if load_subset == 'test_only' and patient_level_split:
        import re
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        def _extract_id_local(filename):
            m = re.findall(r"(\d+)", os.path.basename(filename))
            if not m:
                return None
            try:
                return int(m[-1])
            except Exception:
                return None
        unique_ids = sorted(list({ _extract_id_local(f) for f in mask_files if _extract_id_local(f) is not None }))
        num_patients = len(unique_ids)
        n_test_pat = max(1, int(num_patients * test_split))
        n_val_pat = max(1, int(num_patients * val_split))
        n_train_pat = max(1, num_patients - n_val_pat - n_test_pat)
        test_patients = set(unique_ids[n_train_pat + n_val_pat:])
        # Debug: limit number of patients if requested
        if debug_max_patients and debug_max_patients > 0:
            test_patients = set(sorted(list(test_patients))[:debug_max_patients])
        full_dataset = Int8MultiQualityCBCTDataset(
            base_dir=base_dir,
            mask_dir=mask_dir,
            qualities=qualities,
            augment=False,
            load_to_memory=True,
            target_size=target_size,
            allowed_patient_ids=test_patients
        )
    else:
        # 创建完整数据集
        allowed_debug = None
        if debug_max_patients and debug_max_patients > 0:
            import re
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
            def _extract_id_local(filename):
                m = re.findall(r"(\d+)", os.path.basename(filename))
                if not m:
                    return None
                try:
                    return int(m[-1])
                except Exception:
                    return None
            unique_ids = sorted(list({ _extract_id_local(f) for f in mask_files if _extract_id_local(f) is not None }))
            allowed_debug = set(unique_ids[:debug_max_patients])
        full_dataset = Int8MultiQualityCBCTDataset(
            base_dir=base_dir,
            mask_dir=mask_dir,
            qualities=qualities,
            augment=False,  # 验证时不使用增强
            load_to_memory=True,
            target_size=target_size,
            allowed_patient_ids=allowed_debug
        )
    
    # 分割训练、验证与测试集
    torch.manual_seed(42)  # 确保可重复性
    if load_subset == 'test_only' and patient_level_split:
        # 仅构建测试集加载器（full_dataset已仅包含测试病人），不单独构建 'all'，避免重复计算
        train_loader = None
        val_loaders = {}
        test_loaders = {}
        # 分质量
        for quality in full_dataset.qualities:
            q_indices = full_dataset.get_quality_indices(quality)
            if q_indices:
                subset = torch.utils.data.Subset(full_dataset, q_indices)
                test_loaders[quality] = DataLoader(
                    subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )
        return train_loader, val_loaders, test_loaders

    if patient_level_split:
        # 基于patient_id做病人级划分
        # 收集所有patient_id
        patient_ids = []
        for i in range(len(full_dataset.memory_data)):
            patient_ids.append(full_dataset.memory_data[i]['patient_id'])
        unique_ids = sorted(list(set(patient_ids)))

        # 划分 patient -> train/val/test
        num_patients = len(unique_ids)
        n_test_pat = max(1, int(num_patients * test_split))
        n_val_pat = max(1, int(num_patients * val_split))
        n_train_pat = max(1, num_patients - n_val_pat - n_test_pat)

        # 固定顺序后切片
        train_patients = set(unique_ids[:n_train_pat])
        val_patients = set(unique_ids[n_train_pat:n_train_pat + n_val_pat])
        test_patients = set(unique_ids[n_train_pat + n_val_pat:])

        train_indices, val_indices, test_indices = [], [], []
        for idx in range(len(full_dataset.memory_data)):
            pid = full_dataset.memory_data[idx]['patient_id']
            if pid in train_patients:
                train_indices.append(idx)
            elif pid in val_patients:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    else:
        n_test = int(len(full_dataset) * test_split)
        n_val = int(len(full_dataset) * val_split)
        n_train = len(full_dataset) - n_val - n_test
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [n_train, n_val, n_test])
    
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
        # 获取该质量的所有索引
        quality_indices = full_dataset.get_quality_indices(quality)
        
        # 从验证集中筛选出该质量的样本
        val_quality_indices = []
        
        for val_idx in range(len(val_dataset)):
            original_idx = val_dataset.indices[val_idx]
            if original_idx in quality_indices:
                val_quality_indices.append(val_idx)
        
        if val_quality_indices:
            quality_val_dataset = torch.utils.data.Subset(val_dataset, val_quality_indices)
            val_loaders[quality] = DataLoader(
                quality_val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    
    # 创建测试集加载器（整体与分质量）
    test_loaders = {}
    # 总体测试loader
    test_loaders['all'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    # 分质量测试loader
    for quality in full_dataset.qualities:
        # 找到属于该质量且在test_dataset中的样本
        if isinstance(test_dataset, torch.utils.data.Subset):
            test_indices = test_dataset.indices
        else:
            test_indices = list(range(len(test_dataset)))

        q_indices = full_dataset.get_quality_indices(quality)
        test_quality_indices = []
        for i, orig_idx in enumerate(test_indices):
            if orig_idx in q_indices:
                test_quality_indices.append(i)
        if test_quality_indices:
            quality_test_subset = torch.utils.data.Subset(test_dataset, test_quality_indices)
            test_loaders[quality] = DataLoader(
                quality_test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

    return train_loader, val_loaders, test_loaders

if __name__ == "__main__":
    # 测试int8多质量数据集
    base_dir = "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed"
    mask_dir = "F:/CT_guided_CBCT_segmentation/datasets/LiTS-preprocessed/Masks"
    
    dataset = Int8MultiQualityCBCTDataset(
        base_dir=base_dir,
        mask_dir=mask_dir,
        qualities=['32', '64'],  # 测试2种质量
        target_size=256
    )
    
    print(f"\n数据集统计: {dataset.get_statistics()}")
    
    # 测试样本
    sample = dataset[0]
    print(f"样本形状: 图像 {sample['image'].shape}, 掩码 {sample['mask'].shape}")
    print(f"样本质量: {sample['quality']}")
    print(f"图像数据类型: {sample['image'].dtype}, 范围: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"掩码数据类型: {sample['mask'].dtype}, 范围: [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")

"""
Data exploration utilities for LiTS CBCT dataset
"""
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import ndimage
from skimage import measure
import warnings

from .data_preprocessing import resample_to_isotropic, convert_nifti_to_sitk

warnings.filterwarnings('ignore')

class LiTSDataExplorer:
    """
    LiTS CBCT数据集探索工具
    """
    
    def __init__(self, dataset_root=None):
        """
        初始化数据探索器
        
        Args:
            dataset_root: 数据集根目录, 如果为None, 则自动从当前工作目录推断.
        """
        if dataset_root is None:
            project_root = Path.cwd()
            dataset_root = project_root / "datasets" / "LiTS"
        
        self.dataset_root = Path(dataset_root)
        self.cbct_dirs = {
            '32': self.dataset_root / "TRAINCBCTSimulated" / "32",
            '64': self.dataset_root / "TRAINCBCTSimulated" / "64", 
            '128': self.dataset_root / "TRAINCBCTSimulated" / "128",
            '256': self.dataset_root / "TRAINCBCTSimulated" / "256",
            '490': self.dataset_root / "TRAINCBCTSimulated" / "490"
        }
        self.ct_dir = self.dataset_root / "TRAINCTAlignedToCBCT"
        self.mask_dir = self.dataset_root / "TRAINMasksAlignedToCBCT" / "masks"
        
        print(f"数据集根目录: {self.dataset_root}")
        print(f"CT目录: {self.ct_dir}")
        print(f"Mask目录: {self.mask_dir}")
        
    def scan_dataset_structure(self):
        """
        扫描数据集结构
        """
        print("=" * 60)
        print("数据集结构分析")
        print("=" * 60)
        
        # 检查各个目录是否存在
        print("\n1. 目录存在性检查:")
        print(f"   CT目录存在: {self.ct_dir.exists()}")
        print(f"   Mask目录存在: {self.mask_dir.exists()}")
        
        for resolution, cbct_dir in self.cbct_dirs.items():
            exists = cbct_dir.exists()
            print(f"   CBCT-{resolution}目录存在: {exists}")
            
        # 统计文件数量
        print("\n2. 文件数量统计:")
        
        # CT文件数量
        if self.ct_dir.exists():
            ct_files = list(self.ct_dir.glob("volume-*.nii"))
            print(f"   CT文件数量: {len(ct_files)}")
            
        # Mask文件数量
        if self.mask_dir.exists():
            mask_files = list(self.mask_dir.glob("*.nii"))
            print(f"   Mask文件数量: {len(mask_files)}")
            
        # CBCT文件数量
        cbct_counts = {}
        for resolution, cbct_dir in self.cbct_dirs.items():
            if cbct_dir.exists():
                cbct_files = list(cbct_dir.glob("REC-*.nii"))
                cbct_counts[resolution] = len(cbct_files)
                print(f"   CBCT-{resolution}文件数量: {len(cbct_files)}")
                
        return cbct_counts
    
    def load_sample_data(self, sample_id=0):
        """
        加载样本数据
        
        Args:
            sample_id: 样本ID
            
        Returns:
            dict: 包含CT、CBCT和mask的NIfTI图像对象字典
        """
        print(f"\n加载样本 {sample_id} 的数据...")
        
        data = {'id': sample_id, 'nifti': {}, 'numpy': {}}
        
        # 加载CT数据
        ct_path = self.ct_dir / f"volume-{sample_id}.nii"
        if ct_path.exists():
            nifti_img = nib.load(str(ct_path))
            data['nifti']['ct'] = nifti_img
            data['numpy']['ct'] = nifti_img.get_fdata()
            print(f"   CT数据已加载: {ct_path}")
        
        # 加载Mask数据
        mask_path = self.mask_dir / f"{sample_id}.nii"
        if mask_path.exists():
            nifti_img = nib.load(str(mask_path))
            data['nifti']['mask'] = nifti_img
            data['numpy']['mask'] = nifti_img.get_fdata()
            print(f"   Mask数据已加载: {mask_path}")

        # 加载不同分辨率的CBCT数据
        data['nifti']['cbct'] = {}
        data['numpy']['cbct'] = {}
        for resolution, cbct_dir in self.cbct_dirs.items():
            cbct_path = cbct_dir / f"REC-{sample_id}.nii"
            if cbct_path.exists():
                nifti_img = nib.load(str(cbct_path))
                data['nifti']['cbct'][resolution] = nifti_img
                data['numpy']['cbct'][resolution] = nifti_img.get_fdata()
                print(f"   CBCT-{resolution}数据已加载: {cbct_path}")
        
        return data
    
    def analyze_image_properties(self, data):
        """
        分析图像属性
        """
        print("\n" + "=" * 60)
        print("图像属性分析")
        print("=" * 60)
        
        if 'ct' in data['numpy']:
            img = data['numpy']['ct']
            header = data['nifti']['ct'].header
            print(f"\nCT图像分析:")
            print(f"  形状: {img.shape}")
            print(f"  体素间距 (mm): {header.get_zooms()[:3]}")
            print(f"  强度统计: 均值={img.mean():.2f}, 标准差={img.std():.2f}, 范围=[{img.min():.2f}, {img.max():.2f}]")

        for res, img in data['numpy'].get('cbct', {}).items():
            header = data['nifti']['cbct'][res].header
            print(f"\nCBCT-{res}图像分析:")
            print(f"  形状: {img.shape}")
            print(f"  体素间距 (mm): {header.get_zooms()[:3]}")
            print(f"  强度统计: 均值={img.mean():.2f}, 标准差={img.std():.2f}, 范围=[{img.min():.2f}, {img.max():.2f}]")
            
        if 'mask' in data['numpy']:
            img = data['numpy']['mask']
            print(f"\nMask分析:")
            print(f"  形状: {img.shape}")
            print(f"  唯一值: {np.unique(img)}")
            if len(np.unique(img)) > 1:
                foreground = img > 0
                print(f"  前景比例: {np.sum(foreground) / img.size * 100:.2f}%")
    
    def visualize_sample_corrected(self, data, slice_axis=2, slice_pct=0.5):
        """
        以正确的长宽比可视化样本数据
        """
        print(f"\n以正确的长宽比可视化样本 {data['id']} (切片轴: {slice_axis}, 位置: {slice_pct*100:.0f}%)...")
        
        images_to_show = {}
        if 'ct' in data['nifti']:
            images_to_show['CT'] = data['nifti']['ct']
        for res, img in data['nifti'].get('cbct', {}).items():
            images_to_show[f'CBCT-{res}'] = img
        if 'mask' in data['nifti']:
            images_to_show['Mask'] = data['nifti']['mask']
            
        num_images = len(images_to_show)
        if num_images == 0:
            print("没有可供可视化的图像。")
            return
            
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        if num_images == 1:
            axes = [axes]

        for i, (name, nifti_img) in enumerate(images_to_show.items()):
            img_data = nifti_img.get_fdata()
            zooms = nifti_img.header.get_zooms()
            
            slice_idx = int(img_data.shape[slice_axis] * slice_pct)
            
            if slice_axis == 0: # Sagittal
                slice_data = img_data[slice_idx, :, :]
                aspect = zooms[2] / zooms[1]
            elif slice_axis == 1: # Coronal
                slice_data = img_data[:, slice_idx, :]
                aspect = zooms[2] / zooms[0]
            else: # Axial
                slice_data = img_data[:, :, slice_idx]
                aspect = zooms[1] / zooms[0]
                
            ax = axes[i]
            cmap = 'jet' if name == 'Mask' else 'gray'
            ax.imshow(np.flipud(np.rot90(slice_data)), cmap=cmap, aspect=aspect)
            ax.set_title(f"{name}\nSlice: {slice_idx}, Aspect: {aspect:.2f}")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()

    def visualize_resampled(self, data, slice_axis=2, slice_pct=0.5):
        """
        可视化重采样到各向同性分辨率后的图像
        """
        print(f"\n可视化重采样后的样本 {data['id']} (1x1x1 mm)...")
        
        images_to_process = {}
        if 'ct' in data['nifti']:
            images_to_process['CT'] = (data['nifti']['ct'], sitk.sitkLinear)
        if 'mask' in data['nifti']:
            images_to_process['Mask'] = (data['nifti']['mask'], sitk.sitkNearestNeighbor)
        if '256' in data['nifti'].get('cbct', {}): # 选择一个CBCT分辨率进行演示
             images_to_process['CBCT-256'] = (data['nifti']['cbct']['256'], sitk.sitkLinear)

        num_images = len(images_to_process)
        if num_images == 0:
            print("没有可供重采样的图像。")
            return

        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        if num_images == 1:
            axes = [axes]
            
        for i, (name, (nifti_img, interpolator)) in enumerate(images_to_process.items()):
            # 转换为SimpleITK图像并重采样
            sitk_img = convert_nifti_to_sitk(nifti_img)
            resampled_sitk = resample_to_isotropic(sitk_img, interpolator=interpolator)
            resampled_np = sitk.GetArrayFromImage(resampled_sitk)

            slice_idx = int(resampled_np.shape[slice_axis] * slice_pct)
            
            if slice_axis == 0:
                slice_data = resampled_np[slice_idx, :, :]
            elif slice_axis == 1:
                slice_data = resampled_np[:, slice_idx, :]
            else:
                slice_data = resampled_np[:, :, slice_idx]

            ax = axes[i]
            cmap = 'jet' if name == 'Mask' else 'gray'
            ax.imshow(np.flipud(np.rot90(slice_data)), cmap=cmap, aspect=1.0)
            ax.set_title(f"{name} (Resampled)\nShape: {resampled_np.shape}")
            ax.axis('off')

        plt.tight_layout()
        plt.show() 
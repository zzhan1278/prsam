#!/usr/bin/env python
"""
Preprocesses the entire LiTS dataset.

This script performs the following steps for each sample:
1.  Resamples CT, CBCT, and Mask images to an isotropic resolution (1x1x1 mm).
2.  Applies intensity windowing to the CT image.
3.  Normalizes CT and CBCT images to the [0, 1] range.
4.  Saves the processed images to a new directory structure.
"""
import sys
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import warnings
import numpy as np
import argparse

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

try:
    from utils.data_preprocessing import resample_to_isotropic, window_ct, normalize, pad_or_crop_to_size, window_percentile
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you are running this script from the project root.")
    sys.exit(1)

warnings.filterwarnings('ignore')

def preprocess_and_save(input_path: Path, output_path: Path, image_type: str):
    """
    Loads an image, preprocesses it, and saves it to the output path.
    """
    try:
        # Load image directly using SimpleITK to preserve metadata correctly
        sitk_image = sitk.ReadImage(str(input_path))

        # Set interpolator based on image type
        is_mask = (image_type == 'mask')
        interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        
        # Resample to isotropic spacing (1x1x1 mm)
        resampled_image = resample_to_isotropic(sitk_image, interpolator=interpolator)

        # Define target size for XY plane, keeping depth the same
        # The depth will vary between patients, which is expected.
        target_size = (256, 256, -1) # -1 means keep original size for this dimension
        
        # Apply type-specific processing and padding
        if image_type == 'ct':
            # Pad with -1024 (air in HU) before windowing
            padded_image = pad_or_crop_to_size(resampled_image, target_size, constant_value=-1024.0)
            processed_image = window_ct(padded_image)
            processed_image = normalize(processed_image)
            # Cast to float32 for memory efficiency
            processed_image = sitk.Cast(processed_image, sitk.sitkFloat32)
        elif image_type == 'cbct':
            # Pad with 0.0, as percentile windowing will ignore non-positive values.
            padded_image = pad_or_crop_to_size(resampled_image, target_size, constant_value=0.0)
            # Apply adaptive windowing and normalization in one step.
            processed_image = window_percentile(padded_image, min_percentile=1.0, max_percentile=99.0)
            # Cast to float32 for memory efficiency
            processed_image = sitk.Cast(processed_image, sitk.sitkFloat32)
        else:  # mask
            # Pad with 0 (background label for mask)
            padded_image = pad_or_crop_to_size(resampled_image, target_size, constant_value=0)
            # Ensure mask remains an integer type after all transformations
            processed_image = sitk.Cast(padded_image, sitk.sitkUInt8)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the processed image
        sitk.WriteImage(processed_image, str(output_path))
        
    except FileNotFoundError:
        # This is expected if a sample is missing a specific modality
        pass
    except Exception as e:
        print(f"  [!] Error processing {input_path.name}: {e}")

def main():
    """
    Main function to run the dataset preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(description="Preprocess the LiTS dataset.")
    parser.add_argument('--sample_id', type=str, help="Process a single sample by its ID.")
    args = parser.parse_args()

    print("=" * 60)
    print("Starting LiTS Dataset Preprocessing")
    print("=" * 60)
    
    input_dir = project_root / "datasets" / "LiTS"
    output_dir = project_root / "datasets" / "LiTS-preprocessed"
    
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all sample IDs from the CT directory
    ct_dir = input_dir / "TRAINCTAlignedToCBCT"
    if not ct_dir.exists():
        print(f"Error: CT directory not found at {ct_dir}")
        return
    
    if args.sample_id:
        sample_ids = [args.sample_id]
        print(f"Processing single sample ID: {args.sample_id}")
    else:
        sample_files = sorted(list(ct_dir.glob("volume-*.nii")))
        sample_ids = [f.name.split('-')[1].split('.')[0] for f in sample_files]
        print(f"Found {len(sample_ids)} samples to process.")
    
    # Define CBCT resolutions
    cbct_resolutions = ['32', '64', '128', '256', '490']

    # Process each sample with a progress bar
    for sample_id in tqdm(sample_ids, desc="Processing Samples"):
        # Define paths for the current sample
        # CT
        ct_in_path = input_dir / "TRAINCTAlignedToCBCT" / f"volume-{sample_id}.nii"
        ct_out_path = output_dir / "CT" / f"volume-{sample_id}.nii.gz"
        preprocess_and_save(ct_in_path, ct_out_path, 'ct')
        
        # Mask
        mask_in_path = input_dir / "TRAINMasksAlignedToCBCT" / "masks" / f"{sample_id}.nii"
        mask_out_path = output_dir / "Masks" / f"mask-{sample_id}.nii.gz"
        preprocess_and_save(mask_in_path, mask_out_path, 'mask')
        
        # CBCTs
        for res in cbct_resolutions:
            cbct_in_path = input_dir / "TRAINCBCTSimulated" / res / f"REC-{sample_id}.nii"
            cbct_out_path = output_dir / f"CBCT_{res}" / f"cbct-{sample_id}.nii.gz"
            preprocess_and_save(cbct_in_path, cbct_out_path, 'cbct')
            
    print("\n" + "=" * 60)
    print("Dataset preprocessing complete!")
    print(f"Preprocessed data saved in: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    try:
        import SimpleITK
        from tqdm import tqdm
        import numpy
    except ImportError:
        print("Error: Missing required libraries.")
        print("Please install them by running:")
        print("pip install SimpleITK tqdm numpy")
        sys.exit(1)
    main() 
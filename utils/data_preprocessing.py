"""
Data preprocessing utilities for CT and CBCT images.
"""
import numpy as np
import SimpleITK as sitk
import nibabel as nib

def convert_nifti_to_sitk(nifti_image: nib.Nifti1Image):
    """Converts a Nibabel NIfTI image to a SimpleITK image."""
    data = nifti_image.get_fdata()
    affine = nifti_image.affine
    
    origin = affine[:3, 3]
    spacing = np.sqrt(np.sum(affine[:3, :3] * affine[:3, :3], axis=0))
    direction = affine[:3, :3] / spacing
    
    sitk_image = sitk.GetImageFromArray(np.transpose(data))
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetDirection(direction.flatten())
    
    return sitk_image

def resample_to_isotropic(
    image: sitk.Image,
    interpolator=sitk.sitkLinear,
    output_spacing=(1.0, 1.0, 1.0)
) -> sitk.Image:
    """
    Resamples a SimpleITK image to isotropic resolution.
    
    Args:
        image: The SimpleITK image to resample.
        interpolator: The interpolator to use (e.g., sitk.sitkLinear, sitk.sitkNearestNeighbor).
        output_spacing: The target isotropic spacing.
        
    Returns:
        The resampled SimpleITK image.
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / output_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / output_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / output_spacing[2])))
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    
    return resampler.Execute(image)

def window_ct(image: sitk.Image, window_center=40, window_width=400, window_min=None, window_max=None) -> sitk.Image:
    """
    Applies intensity windowing to a CT image.
    
    Can be defined by center/width or min/max.
    
    Args:
        image: The SimpleITK CT image.
        window_center: The center of the Hounsfield Unit (HU) window.
        window_width: The width of the HU window.
        window_min: The minimum value of the window. Overrides center/width.
        window_max: The maximum value of the window. Overrides center/width.
        
    Returns:
        The windowed image, with intensities scaled to [0, 255].
    """
    if window_min is None:
        min_val = window_center - window_width / 2
    else:
        min_val = window_min
        
    if window_max is None:
        max_val = window_center + window_width / 2
    else:
        max_val = window_max
    
    windowing_filter = sitk.IntensityWindowingImageFilter()
    windowing_filter.SetWindowMinimum(min_val)
    windowing_filter.SetWindowMaximum(max_val)
    windowing_filter.SetOutputMinimum(0.0)
    windowing_filter.SetOutputMaximum(255.0)
    
    return windowing_filter.Execute(image)

def window_percentile(image: sitk.Image, min_percentile=1.0, max_percentile=99.0, image_name="image") -> sitk.Image:
    """
    Applies intensity windowing to an image based on percentiles of non-zero values.
    
    Args:
        image: The SimpleITK image.
        min_percentile: The lower percentile for the window.
        max_percentile: The upper percentile for the window.
        image_name: An identifier for logging purposes.
        
    Returns:
        The windowed image, with intensities scaled to [0, 1].
    """
    # Convert to numpy array for processing. This creates a copy.
    np_image = sitk.GetArrayFromImage(image)
    
    # Calculate percentiles on non-zero pixels to avoid background skew
    # Use a small epsilon to handle potential floating point inaccuracies
    non_zero_pixels = np_image[np_image > np.finfo(float).eps]
    
    if non_zero_pixels.size == 0:
        # If no non-zero pixels, the image is likely all background, return as is.
        return image

    min_val, max_val = np.percentile(non_zero_pixels, [min_percentile, max_percentile])

    # Add a safety check in case min and max are the same (e.g., uniform image)
    if max_val <= min_val:
        max_val = min_val + 1e-6 # Add a small epsilon to avoid division by zero

    # --- Manual Windowing and Normalization using Numpy ---
    # 1. Clip the image to the calculated window
    np_image = np.clip(np_image, min_val, max_val)

    # 2. Normalize the clipped image to the [0, 1] range
    np_image = (np_image - min_val) / (max_val - min_val)

    # Convert back to SimpleITK image
    processed_image = sitk.GetImageFromArray(np_image.astype(np.float32))
    
    # IMPORTANT: Copy the metadata from the original image
    processed_image.SetOrigin(image.GetOrigin())
    processed_image.SetSpacing(image.GetSpacing())
    processed_image.SetDirection(image.GetDirection())
    
    return processed_image

def normalize(image: sitk.Image) -> sitk.Image:
    """
    Normalizes an image to the [0, 1] range.
    
    Args:
        image: The SimpleITK image.
        
    Returns:
        The normalized image.
    """
    return sitk.RescaleIntensity(image, 0.0, 1.0)

def pad_or_crop_to_size(image: sitk.Image, target_size=(256, 256, -1), constant_value=0.0):
    """
    Pads or crops a SimpleITK image to a target size from the center.
    The depth (z-axis) can be ignored by setting the corresponding target size to -1.
    """
    original_size = list(image.GetSize())
    target_size = list(target_size)

    # Replace -1 with original size to keep the original depth
    for i in range(len(target_size)):
        if target_size[i] == -1:
            target_size[i] = original_size[i]

    # First, pad the image if it's smaller than the target size
    pad_needed = any(orig < target for orig, target in zip(original_size, target_size))
    if pad_needed:
        lower_pad = [max(0, (target - orig) // 2) for orig, target in zip(original_size, target_size)]
        upper_pad = [max(0, target - orig - lower) for orig, target, lower in zip(original_size, target_size, lower_pad)]
        image = sitk.ConstantPad(image, lower_pad, upper_pad, constant_value)
    
    # Second, crop the image if it's larger than the target size
    current_size = image.GetSize()
    crop_needed = any(curr > target for curr, target in zip(current_size, target_size))
    if crop_needed:
        lower_crop = [max(0, (curr - target) // 2) for curr, target in zip(current_size, target_size)]
        upper_crop = [max(0, curr - target - lower) for curr, target, lower in zip(current_size, target_size, lower_crop)]
        image = sitk.Crop(image, lower_crop, upper_crop)
        
    return image 
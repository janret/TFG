import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Calculate Dice coefficient for evaluating segmentation quality.
    
    Args:
        y_true: Ground truth segmentation mask
        y_pred: Predicted segmentation mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient value between 0 and 1
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    """
    Combined loss function using both categorical crossentropy and Dice loss.
    
    Args:
        y_true: Ground truth segmentation mask
        y_pred: Predicted segmentation probabilities
    
    Returns:
        Combined loss value
    """
    # Categorical crossentropy
    ce_loss = K.categorical_crossentropy(y_true, y_pred)
    
    # Dice loss
    dice = dice_coefficient(y_true, y_pred)
    dice_loss = 1 - dice
    
    # Combine losses (equal weighting)
    return ce_loss + dice_loss

def calculate_volumes(label_mask, voxel_size=(2, 2, 2)):
    """
    Calculate volume (in mm³) for each tissue class.
    
    Args:
        label_mask: 3D numpy array of integer labels (0=background, 1=gray matter, 2=white matter, 3=csf)
        voxel_size: Tuple of voxel dimensions in mm
    
    Returns:
        Dictionary containing volume for each tissue class
    """
    # Calculate voxel volume in mm³
    voxel_volume = np.prod(voxel_size)
    
    # Count voxels for each class
    counts = np.bincount(label_mask.flatten(), minlength=4)
    
    # Calculate volumes
    return {
        'background_mm3': int(counts[0] * voxel_volume),
        'gray_matter_mm3': int(counts[1] * voxel_volume),
        'white_matter_mm3': int(counts[2] * voxel_volume),
        'csf_mm3': int(counts[3] * voxel_volume)
    }

def load_mgz_file(file_path):
    """
    Load an MGZ file and return its data array.
    
    Args:
        file_path: Path to the MGZ file
    
    Returns:
        Tuple of (data array, affine matrix)
    """
    img = nib.load(file_path)
    return img.get_fdata(), img.affine

def preprocess_scan(scan_data):
    """
    Preprocess MRI scan data.
    
    Args:
        scan_data: 3D numpy array of scan intensities
    
    Returns:
        Preprocessed scan data normalized to [0, 1]
    """
    # Handle NaN and Inf values
    scan_data = np.nan_to_num(scan_data)
    
    # Normalize to [0, 1]
    scan_min = scan_data.min()
    scan_max = scan_data.max()
    if scan_max > scan_min:
        scan_data = (scan_data - scan_min) / (scan_max - scan_min)
    
    return scan_data

def preprocess_segmentation(seg_data, n_classes=4):
    """
    Convert segmentation mask to one-hot encoding.
    
    Args:
        seg_data: 3D numpy array of integer labels
        n_classes: Number of segmentation classes
    
    Returns:
        One-hot encoded segmentation mask
    """
    # Initialize one-hot encoded array
    shape = (*seg_data.shape, n_classes)
    one_hot = np.zeros(shape, dtype=np.float32)
    
    # Fill in one-hot encoding
    for class_idx in range(n_classes):
        one_hot[..., class_idx] = (seg_data == class_idx)
    
    return one_hot

def save_segmentation(label_mask, reference_img_path, output_dir):
    """
    Save the segmentation mask as an MGZ file.
    
    Args:
        label_mask: 3D numpy array of integer labels
        reference_img_path: Path to the original image for affine and header
        output_dir: Base directory for saving outputs
    
    Returns:
        Path to the saved segmentation file
    """
    # Extract subject and timepoint info from path
    parts = reference_img_path.split(os.sep)
    subject_id = next(p for p in parts if p.startswith('sub-'))
    timepoint = os.path.basename(reference_img_path).replace('.mgz', '')
    
    # Create output filename and directory
    output_filename = f"{timepoint}_seg_hybrid.mgz"
    save_dir = os.path.join(output_dir, subject_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save segmented image
    out_path = os.path.join(save_dir, output_filename)
    ref_img = nib.load(reference_img_path)
    seg_img = nib.MGZImage(label_mask.astype(np.int16), ref_img.affine, ref_img.header)
    nib.save(seg_img, out_path)
    
    return out_path 
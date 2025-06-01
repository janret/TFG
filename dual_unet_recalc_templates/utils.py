import os
import numpy as np
import nibabel as nib

def majority_vote(masks):
    """Fusión 3D con preservación de dimensionalidad"""
    stacked = np.stack(masks, axis=0)
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked)

def compute_dice(one_hot, pred):
    """Cálculo Dice para múltiples clases"""
    dice_scores = []
    for c in range(one_hot.shape[-1]):
        gt = one_hot[..., c]
        pr = (pred == c).astype(np.float32)
        intersection = np.sum(gt * pr)
        union = np.sum(gt) + np.sum(pr)
        dice_scores.append(2*intersection/(union + 1e-8))
    return np.mean(dice_scores)

def save_segmentation(label_mask, affine, original_path, output_dir):
    """
    Save the segmentation mask as a NIfTI file following BIDS-style subdirectories.
    label_mask: 3D numpy array of integer labels.
    affine: affine transformation matrix.
    original_path: full path to the original image.
    output_dir: base directory for saving outputs.
    Returns the saved file path.
    """
    # Decode path if it's a bytes tensor
    if isinstance(original_path, bytes):
        original_path = original_path.decode('utf-8')
    
    # Extract BIDS parts from path
    parts = original_path.split(os.sep)
    bids_parts = [p for p in parts if p.startswith(('sub-', 'ses-'))]
    
    # Handle file extension - convert .mgz to .nii for output
    filename = os.path.basename(original_path)
    if '.mgz' in filename:
        filename = filename.replace('.mgz', '_dseg.nii')
    else:
        filename = filename.replace('.nii', '_seg.nii')
    
    # Create directory if it doesn't exist
    save_dir = os.path.join(output_dir, *bids_parts)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save segmented image
    out_path = os.path.join(save_dir, filename)
    
    # Create NIfTI image directly with segmentation mask and affine matrix
    segmentation_img = nib.Nifti1Image(label_mask.astype(np.int16), affine)
    
    # No need to access original header, just save with our data and affine
    nib.save(segmentation_img, out_path)
    return out_path

def calculate_volumes(label_mask, voxel_volume):
    """
    Calculate volume (in mm^3) for each tissue class.
    label_mask: 3D numpy array of integer labels (0=background, 1=gray matter, 2=white matter, 3=csf).
    voxel_volume: scalar volume of one voxel in mm^3.
    Returns a dict with named volumes for each tissue class.
    """
    counts = np.bincount(label_mask.flatten(), minlength=4)
    return {
        'background_mm3': int(counts[0] * voxel_volume),
        'gray_matter_mm3': int(counts[1] * voxel_volume),
        'white_matter_mm3': int(counts[2] * voxel_volume),
        'csf_mm3': int(counts[3] * voxel_volume)
    }
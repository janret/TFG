import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nibabel as nib

def load_mgz_file(file_path):
    """Load an MGZ file and return its data as a numpy array."""
    img = nib.load(file_path)
    return img.get_fdata()

def preprocess_scan(scan_data):
    """Preprocess the scan data."""
    # Normalize to [0, 1]
    scan_data = (scan_data - scan_data.min()) / (scan_data.max() - scan_data.min())
    return scan_data

def preprocess_segmentation(seg_data):
    """Preprocess the segmentation data."""
    # Convert to categorical
    return tf.keras.utils.to_categorical(seg_data, num_classes=4)

def dice_coefficient(y_true, y_pred):
    """Calculate Dice coefficient."""
    smooth = 1e-7
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Calculate Dice loss."""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combine categorical crossentropy with dice loss."""
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return ce_loss + dice

def compute_dice(one_hot_mask, pred_mask):
    """Calculate Dice coefficient for each class and return average"""
    # Convert tensors to numpy arrays if needed
    if isinstance(one_hot_mask, tf.Tensor):
        one_hot_mask = one_hot_mask.numpy()
    if isinstance(pred_mask, tf.Tensor):
        pred_mask = pred_mask.numpy()
        
    num_classes = one_hot_mask.shape[-1]
    dice_scores = []
    for c in range(num_classes):
        gt = one_hot_mask[..., c].astype(np.float32)
        pred = (pred_mask == c).astype(np.float32)
        intersection = np.sum(gt * pred)
        union = np.sum(gt) + np.sum(pred)
        dice = (2. * intersection) / union if union != 0 else 1.0
        dice_scores.append(dice)
    return np.mean(dice_scores)

def calculate_volumes(label_mask, voxel_size=(1, 1, 1)):
    """
    Calculate volume (in mm^3) for each tissue class.
    label_mask: 3D numpy array of integer labels (0=background, 1=gray matter, 2=white matter, 3=csf).
    voxel_size: tuple of voxel dimensions in mm.
    Returns a dict with named volumes for each tissue class.
    """
    voxel_volume = np.prod(voxel_size)
    counts = np.bincount(label_mask.flatten(), minlength=4)
    return {
        'background_mm3': int(counts[0] * voxel_volume),
        'gray_matter_mm3': int(counts[1] * voxel_volume),
        'white_matter_mm3': int(counts[2] * voxel_volume),
        'csf_mm3': int(counts[3] * voxel_volume)
    }

def registration_quality_loss(y_true, y_pred):
    """Custom loss component that considers segmentation consistency"""
    # Compute gradient of predictions to assess smoothness
    gradients = tf.image.sobel_edges(y_pred)
    gradient_magnitude = tf.reduce_mean(tf.square(gradients))
    
    # Regular categorical crossentropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Combine losses with weighting
    return cce + 0.1 * gradient_magnitude

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()
    return cm

def evaluate_segmentation(y_true, y_pred, output_dir=None):
    """Comprehensive segmentation evaluation"""
    # Convert predictions to class labels if needed
    if y_pred.shape[-1] > 1:  # one-hot encoded
        y_pred_classes = np.argmax(y_pred, axis=-1)
    else:
        y_pred_classes = y_pred
        
    if y_true.shape[-1] > 1:  # one-hot encoded
        y_true_classes = np.argmax(y_true, axis=-1)
    else:
        y_true_classes = y_true
    
    # Calculate Dice score
    dice = dice_coefficient(y_true, y_pred)
    
    # Calculate confusion matrix
    if output_dir:
        cm = plot_confusion_matrix(
            y_true_classes.flatten(), 
            y_pred_classes.flatten(),
            os.path.join(output_dir, 'confusion_matrix.png')
        )
    else:
        cm = confusion_matrix(y_true_classes.flatten(), y_pred_classes.flatten())
    
    return {
        'dice': dice,
        'confusion_matrix': cm
    }

def create_callbacks(checkpoint_path, monitor='val_dice_coefficient'):
    """Create standard set of training callbacks"""
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor=monitor,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=20,
            mode='max',
            restore_best_weights=True
        )
    ]
    return callbacks 
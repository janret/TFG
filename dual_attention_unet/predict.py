import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from DataLoader import BIDSDataLoader
from Utils import dice_coefficient, combined_loss, calculate_volumes

def save_segmentation(label_mask, reference_img_path, output_dir):
    """
    Save the segmentation mask as an MGZ file.
    label_mask: 3D numpy array of integer labels.
    reference_img_path: path to the original image for affine and header.
    output_dir: base directory for saving outputs.
    Returns the saved file path.
    """
    # Extract subject and timepoint info from path
    parts = reference_img_path.split(os.sep)
    subject_id = next(p for p in parts if p.startswith('sub-'))
    timepoint = os.path.basename(reference_img_path).replace('.mgz', '')
    
    # Create output filename and directory
    output_filename = f"{timepoint}_dseg.nii"
    save_dir = os.path.join(output_dir, subject_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save segmented image
    out_path = os.path.join(save_dir, output_filename)
    ref_img = nib.load(reference_img_path)
    seg_img = nib.Nifti1Image(label_mask.astype(np.int16), ref_img.affine, ref_img.header)
    nib.save(seg_img, out_path)
    
    return out_path

def main():
    parser = argparse.ArgumentParser(description='Hybrid 3D UNet segmentation for longitudinal data')
    parser.add_argument('--bids_root', required=True,
                      help='Root directory containing BIDS dataset')
    parser.add_argument('--model_path', required=True,
                      help='Path to trained hybrid model weights')
    parser.add_argument('--simple_unet_path', required=True,
                      help='Path to trained simple U-Net model for template segmentation')
    parser.add_argument('--output_dir', required=True,
                      help='Directory to save segmentations and volumes.tsv')
    parser.add_argument('--registered_dir', default=None,
                      help='Directory containing precomputed mean templates (optional)')
    parser.add_argument('--gpu', type=int, default=-1,
                      help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for prediction')
    parser.add_argument('--save_segmentations', action='store_true',
                      help='Whether to save segmentation masks (default: False)')
    args = parser.parse_args()

    # Configure GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"> Using {'CPU' if args.gpu == -1 else f'GPU {args.gpu}'}")

    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file {args.model_path} not found")
    if not os.path.exists(args.bids_root):
        raise NotADirectoryError(f"BIDS directory {args.bids_root} not found")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model(args.model_path, custom_objects={
        'dice_coefficient': dice_coefficient,
        'combined_loss': combined_loss
    })

    # Initialize data loader
    loader = BIDSDataLoader(
        bids_root=args.bids_root,
        trained_model_path=args.simple_unet_path,
        registered_dir=args.registered_dir,
        target_shape=(120, 120, 94),
        voxel_size=(2, 2, 2),
        n_classes=4
    )

    # Create dataset
    dataset = loader.get_dataset(batch_size=args.batch_size)

    results = []
    print("\nProcessing dataset...")

    for inputs, metadata in tqdm(dataset):
        # Generate prediction
        prediction = model.predict(inputs, verbose=0)
        
        # Process each sample in the batch
        for i in range(prediction.shape[0]):
            label_map = np.argmax(prediction[i], axis=-1)
            
            # Calculate volumes
            volumes = calculate_volumes(label_map, voxel_size=loader.voxel_size)
            original_path = metadata['original_path'][i].numpy().decode('utf-8')
            # Extract Label from original_path
            volumes['Label'] = os.path.basename(original_path).replace('_T1w_reg.mgz', '')
            results.append(volumes)
            
            # Save segmentation if requested
            if args.save_segmentations:
                save_segmentation(
                    label_map,
                    original_path,
                    args.output_dir
                )

    # Save volumes to TSV
    df = pd.DataFrame(results).sort_values('Label')
    tsv_file = os.path.join(args.output_dir, 'volumes.tsv')
    df.to_csv(tsv_file, sep='\t', index=False)
    print(f"\nDone. Results written to {tsv_file}")

if __name__ == "__main__":
    main() 
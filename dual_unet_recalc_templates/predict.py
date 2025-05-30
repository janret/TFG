import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from DataLoader import BIDSDataLoader
from utils import majority_vote, save_segmentation, calculate_volumes

def validate_paths(args):
    """Validate all input paths exist"""
    if not os.path.exists(args.bids_root):
        raise FileNotFoundError(f"BIDS root directory not found: {args.bids_root}")
    if not os.path.exists(args.simple_unet_path):
        raise FileNotFoundError(f"Simple U-Net model not found: {args.simple_unet_path}")
    if not os.path.exists(args.dual_unet_path):
        raise FileNotFoundError(f"Dual U-Net model not found: {args.dual_unet_path}")
    if args.registered_dir and not os.path.exists(args.registered_dir):
        raise FileNotFoundError(f"Registered directory not found: {args.registered_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Dual-input UNet segmentation for longitudinal BIDS datasets'
    )
    parser.add_argument(
        '--registered_dir', default=None,
        help='Directory containing precomputed mean templates (optional)'
    )
    parser.add_argument(
        '--bids_root', required=True,
        help='Root directory of the BIDS dataset'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Directory where segmentations and volumes.tsv will be saved'
    )
    parser.add_argument(
        '--simple_unet_path', required=True,
        help='Path to the trained Simple U-Net model (.h5)'
    )
    parser.add_argument(
        '--dual_unet_path', required=True,
        help='Path to the trained dual-input U-Net model (.h5)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for prediction'
    )
    parser.add_argument(
        '--iterations', type=int, default=3,
        help='Number of iterations for refinement (default=3)'
    )
    parser.add_argument(
        '--gpu', type=int, default=-1,
        help='GPU index to use; set to -1 for CPU'
    )
    parser.add_argument(
        '--save_segmentations', action='store_true',
        help='Whether to save segmentation masks (default: False)'
    )
    args = parser.parse_args()

    try:
        # Validate input paths
        validate_paths(args)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Configure GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.gpu >= 0 else '-1'
        print(f"Using {'CPU' if args.gpu < 0 else f'GPU {args.gpu}'} for computation")

        # Load model
        print("Loading dual U-Net model...")
        try:
            dual_unet = tf.keras.models.load_model(args.dual_unet_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load dual U-Net model: {str(e)}")

        for iteration in range(args.iterations):
            print(f"\n=== Iteration {iteration + 1}/{args.iterations} ===")

            # Initialize data loader
            loader = BIDSDataLoader(
                bids_root=args.bids_root,
                trained_model_path=args.simple_unet_path,
                registered_dir=args.registered_dir,
                n_classes=4,
                use_updated_templates=(iteration > 0)
            )

            dataset = loader.get_dataset(batch_size=args.batch_size)
            voxel_volume = np.prod(loader.voxel_size)
            all_preds = {}
            results = []

            # Process each batch
            print("Processing images...")
            for batch in tqdm(dataset):
                try:
                    images, templates, metadata = batch
                    predictions = dual_unet.predict([images, templates], verbose=0)

                    for i in range(images.shape[0]):
                        label_map = np.argmax(predictions[i], axis=-1)
                        original_path = metadata['original_path'][i].numpy().decode('utf-8')
                        
                        parts = original_path.split(os.sep)
                        subject = [p for p in parts if p.startswith('sub-')][0]

                        if iteration < args.iterations - 1:
                            if subject not in all_preds:
                                all_preds[subject] = []
                            all_preds[subject].append(label_map)

                        if iteration == args.iterations - 1:
                            volumes = calculate_volumes(label_map, voxel_volume)
                            # Extract Label in the format sub-XXX_ses-XX_run-X
                            parts = original_path.split(os.sep)
                            sub_ses_parts = [p for p in parts if p.startswith(('sub-', 'ses-'))]
                            run_part = original_path.split('_run-')[1].split('_')[0] if '_run-' in original_path else '1'
                            label = f"{sub_ses_parts[0]}_{sub_ses_parts[1]}_run-{run_part}"
                            volumes['Label'] = label
                            volumes['original_path'] = original_path
                            results.append(volumes)
                            if args.save_segmentations:
                                save_segmentation(
                                    label_map,
                                    metadata['affine'][i].numpy(),
                                    original_path,
                                    args.output_dir
                                )
                except Exception as e:
                    print(f"Warning: Error processing batch: {str(e)}")
                    continue

            # Update templates if not in final iteration
            if iteration < args.iterations - 1 and all_preds:
                print("Updating templates...")
                for sub, masks in all_preds.items():
                    if not masks:
                        continue

                    try:
                        fused_mask = majority_vote(np.stack(masks, axis=0))
                        sub_reg_dir = os.path.join(loader.registered_dir, sub)
                        os.makedirs(sub_reg_dir, exist_ok=True)
                        refined_path = os.path.join(sub_reg_dir, 'mean_refined.mgz')
                        
                        original_mean = nib.load(os.path.join(sub_reg_dir, 'mean.mgz'))
                        new_template = nib.Nifti1Image(
                            fused_mask.astype(np.int32),
                            original_mean.affine,
                            original_mean.header
                        )
                        nib.save(new_template, refined_path)
                        print(f"Updated template for subject {sub}")
                    except Exception as e:
                        print(f"Warning: Failed to update template for subject {sub}: {str(e)}")
                        continue

            # Save final results
            if iteration == args.iterations - 1 and results:
                print("\nSaving final results...")
                df = pd.DataFrame(results)
                # Drop original_path column and sort by Label
                df = df.drop(columns=['original_path']).sort_values('Label')
                output_file = os.path.join(args.output_dir, 'volumes.tsv')
                df.to_csv(output_file, sep='\t', index=False)
                print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
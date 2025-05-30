import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from DataLoader import BIDSDataLoader
from Utils import dice_coefficient, combined_loss, calculate_volumes, save_segmentation


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
            volumes['original_path'] = metadata['original_path'][i].numpy().decode('utf-8')
            results.append(volumes)
            
            # Save segmentation if requested
            if args.save_segmentations:
                save_segmentation(
                    label_map,
                    metadata['original_path'][i].numpy().decode('utf-8'),
                    args.output_dir
                )

    # Save volumes to TSV
    df = pd.DataFrame(results).sort_values('original_path')
    tsv_file = os.path.join(args.output_dir, 'volumes.tsv')
    df.to_csv(tsv_file, sep='\t', index=False)
    print(f"\nDone. Results written to {tsv_file}")


if __name__ == "__main__":
    main()
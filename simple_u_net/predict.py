import os
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from DataLoader import SimplePredictGenerator, BIDSPredictGenerator

def calculate_volumes(pred_mask, voxel_volume):
    counts = np.bincount(pred_mask.flatten(), minlength=4)
    return {
        'background_mm3': counts[0] * voxel_volume,
        'gray_matter_mm3': counts[1] * voxel_volume,
        'white_matter_mm3': counts[2] * voxel_volume,
        'csf_mm3': counts[3] * voxel_volume
    }

def save_prediction(pred_mask, meta, output_dir):
    original = meta['original_path']
    parts = original.split(os.sep)
    bids = [p for p in parts if p.startswith(('sub-','ses-'))]
    outpath = os.path.join(
        output_dir,
        *bids,
        os.path.basename(original).replace('_T1w.nii', '_seg.nii.gz')
    )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    nib.save(
        nib.Nifti1Image(pred_mask.astype(np.int16), meta['resampled_affine'], meta['resampled_header']),
        outpath
    )
    return outpath

def main():
    parser = argparse.ArgumentParser(description='Segmentation predictor')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--registered_dir', help='Use pre-registered images')
    group.add_argument('--use_template', action='store_true', help='Compute robust template before prediction')
    parser.add_argument('--input_dir', required=True, help='Input BIDS directory')
    parser.add_argument('--output_dir', required=True, help='Directory to save outputs')
    parser.add_argument('--model_path', required=True, help='Path to trained model (.h5)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prediction')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (-1 for CPU)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.gpu >= 0 else '-1'
    model = tf.keras.models.load_model(args.model_path)

    if args.registered_dir or args.use_template:
        datagen = BIDSPredictGenerator(
            base_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            registered_dir=args.registered_dir,
            use_template=args.use_template
        )
    else:
        datagen = SimplePredictGenerator(
            base_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    print(f"Processing {len(datagen)} scans...")

    for i in tqdm(range(len(datagen))):
        X, meta = datagen[i]
        preds = model.predict(X, verbose=0)
        for j in range(len(X)):
            mask = np.argmax(preds[j], axis=-1).astype(np.int16)
            vols = calculate_volumes(mask, datagen.voxel_volume)
            vols['original_path'] = os.path.basename(meta[j]['original_path'])
            results.append(vols)
            save_prediction(mask, meta[j], args.output_dir)

    df = pd.DataFrame(results).sort_values('original_path')
    tsv_path = os.path.join(args.output_dir, 'volumes.tsv')
    df.to_csv(tsv_path, sep='\t', index=False)
    print("Done. Volumes saved to volumes.tsv")

if __name__ == '__main__':
    main()

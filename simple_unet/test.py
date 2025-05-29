import os
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
from DataLoader import LongitudinalDataGenerator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model (.h5 file)')
    parser.add_argument('--test_data_dir', type=str, required=True,
                      help='Directory containing synthetic test data')
    parser.add_argument('--gpu', type=int, default=-1,
                      help='GPU ID to use (-1 for CPU)')
    return parser.parse_args()

def compute_dice(one_hot_mask, pred_mask):
    """Calculate Dice coefficient for each class and return average"""
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

def main():
    args = parse_arguments()
    
    # Configure device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"> Using {'CPU' if args.gpu == -1 else f'GPU {args.gpu}'}")

    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file {args.model_path} not found")
    if not os.path.exists(args.test_data_dir):
        raise NotADirectoryError(f"Data directory {args.test_data_dir} not found")

    # Load model
    model = tf.keras.models.load_model(args.model_path)
    print(f"\nLoaded model from {args.model_path}")

    # Get all test subjects
    test_subs = [d for d in os.listdir(args.test_data_dir) if d.startswith('sub-')]
    
    if not test_subs:
        raise ValueError("No test subjects found in the data directory")

    # Create test generator
    test_gen = LongitudinalDataGenerator(
        base_dir=args.test_data_dir,
        subjects=test_subs,
        batch_size=1,
        shuffle=False
    )

    # Evaluation metrics
    dice_gt_list = []
    dice_template_list = []

    print(f"\nEvaluating on {len(test_subs)} subjects ({len(test_gen)} samples)...")
    
    for idx in range(len(test_gen)):
        X_mri, y_true = test_gen[idx]
        y_pred = model.predict(X_mri, verbose=0)
        pred_mask = np.argmax(y_pred[0], axis=-1).astype(np.int32)

        # Calculate Dice against ground truth
        dice_gt_list.append(compute_dice(y_true[0], pred_mask))

        # Calculate Dice against template
        sub, tp_file = test_gen.samples[idx]
        tpl_path = os.path.join(args.test_data_dir, sub, f"{sub}_template_seg.mgz")
        
        if os.path.exists(tpl_path):
            tpl_data = nib.load(tpl_path).get_fdata().astype(np.int32)
            onehot_tpl = np.zeros_like(y_true[0])
            for c in range(y_true.shape[-1]):
                onehot_tpl[..., c] = (tpl_data == c).astype(np.float32)
            
            dice_template_list.append(compute_dice(onehot_tpl, pred_mask))
        else:
            print(f"Warning: Template not found at {tpl_path}")
            dice_template_list.append(0.0)

    # Calculate final metrics
    final_dice_gt = np.mean(dice_gt_list)
    final_dice_template = np.mean(dice_template_list)

    print("\nFinal Evaluation Results:")
    print(f"{'Metric':<25} | {'Value':>10}")
    print("-" * 38)
    print(f"{'Dice (Ground Truth)':<25} | {final_dice_gt:>10.4f}")
    print(f"{'Dice (Template)':<25} | {final_dice_template:>10.4f}")

if __name__ == "__main__":
    main()
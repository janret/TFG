import os
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
from DataLoader import LongitudinalDataGenerator
from Utils import dice_coefficient, combined_loss, compute_dice

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model (.h5 file)')
    parser.add_argument('--test_data_dir', type=str, required=True,
                      help='Directory containing synthetic test data')
    parser.add_argument('--gpu', type=int, default=-1,
                      help='GPU ID to use (-1 for CPU)')
    return parser.parse_args()

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
    model = tf.keras.models.load_model(args.model_path, custom_objects={
        'dice_coefficient': dice_coefficient,
        'combined_loss': combined_loss
    })
    print(f"\nLoaded model from {args.model_path}")

    # Get all test subjects
    test_subs = [d for d in os.listdir(args.test_data_dir) if d.startswith('sub-')]
    
    if not test_subs:
        raise ValueError("No test subjects found in the data directory")

    # Create test generator
    test_gen = LongitudinalDataGenerator(
        base_dir=args.test_data_dir,
        input_shape=(120, 120, 94),
        batch_size=1,
        n_classes=4,
        shuffle=False
    )

    # Evaluation metrics
    dice_gt_list = []
    dice_template_list = []

    print(f"\nEvaluating on {len(test_subs)} subjects ({len(test_gen)} samples)...")
    
    for idx in range(len(test_gen)):
        inputs, y_true = test_gen[idx]
        y_pred = model.predict({
            'mri_input': inputs['mri_input'],
            'template_input': inputs['template_input']
        }, verbose=0)
        
        # Convert predictions to class labels
        pred_mask = np.argmax(y_pred[0], axis=-1)

        # Calculate Dice against ground truth
        dice_gt_list.append(compute_dice(y_true[0], pred_mask))

        # Calculate Dice against template
        # Extract template segmentation from the template input (channels 1-4)
        template_input = inputs['template_input'].numpy()[0]  # Convert to numpy
        template_seg = np.argmax(template_input[..., 1:], axis=-1)
        onehot_tpl = np.zeros_like(y_true[0].numpy())  # Convert to numpy
        for c in range(y_true.shape[-1]):
            onehot_tpl[..., c] = (template_seg == c).astype(np.float32)
        
        dice_template_list.append(compute_dice(onehot_tpl, pred_mask))

    # Calculate final metrics
    final_dice_gt = np.mean(dice_gt_list)
    final_dice_template = np.mean(dice_template_list)

    print("\nFinal Evaluation Results:")
    print(f"{'Metric':<25} | {'Value':>10}")
    print("-" * 38)
    print(f"{'Dice (Ground Truth)':<25} | {final_dice_gt:>10.4f}")
    print(f"{'Dice (Template)':<25} | {final_dice_template:>10.4f}")

    # Save results to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'test_metrics.txt'), 'w') as f:
        f.write("Final Evaluation Results:\n")
        f.write(f"{'Metric':<25} | {'Value':>10}\n")
        f.write("-" * 38 + "\n")
        f.write(f"{'Dice (Ground Truth)':<25} | {final_dice_gt:>10.4f}\n")
        f.write(f"{'Dice (Template)':<25} | {final_dice_template:>10.4f}\n")

if __name__ == "__main__":
    main() 
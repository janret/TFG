import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from DataLoader import LongitudinalDataGenerator
import tensorflow as tf
from utils import majority_vote, compute_dice

def parse_arguments():
    parser = argparse.ArgumentParser(description='3D UNet evaluation with iterative template refinement')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Root directory containing synthetic data')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--trained_model', type=str, required=True,
                      help='Path to trained model weights')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Test batch size')
    parser.add_argument('--iterations', type=int, default=3,
                      help='Number of refinement iterations')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu >= 0 else ""
    
    # Load subjects
    subject_ids = [d for d in os.listdir(args.data_dir) if d.startswith('sub-')]
    test_subjects = subject_ids[int(0.95*len(subject_ids)):]  # Last 5% as test
    
    # Load model
    model = tf.keras.models.load_model(args.trained_model)
    
    history = {'dice_gt': [], 'dice_template': []}
    all_preds = []
    all_subjects = []

    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration+1}/{args.iterations} ===")
        
        # Configure generator
        test_gen = LongitudinalDataGenerator(
            base_dir=args.data_dir,
            subjects=test_subjects,
            batch_size=args.batch_size,
            use_updated_templates=(iteration > 0)
        )
        
        # Evaluation
        dice_scores = []
        template_scores = []
        
        for batch_idx in range(len(test_gen)):
            (mri, template), y_true = test_gen[batch_idx]
            y_pred = model.predict([mri, template], verbose=0)
            pred_mask = np.argmax(y_pred, axis=-1)
            
            # Store predictions
            batch_subjects = [s[0] for s in test_gen.samples[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]]
            all_preds.extend(pred_mask)
            all_subjects.extend(batch_subjects)
            
            # Calculate metrics
            dice_scores.append(compute_dice(y_true, pred_mask))
            template_scores.append(compute_dice(template, pred_mask))

        # Update templates
        if iteration < args.iterations - 1:
            print("Updating templates...")
            for sub in test_subjects:
                sub_masks = [all_preds[i] for i,s in enumerate(all_subjects) if s == sub]
                if len(sub_masks) == 0:
                    continue
                
                # Load original template as reference
                template_ref = nib.load(os.path.join(args.data_dir, sub, f"{sub}_template_seg.mgz"))
                fused_mask = majority_vote(np.array(sub_masks))
                
                # Save new template
                new_template = nib.Nifti1Image(
                    fused_mask.astype(np.int32),
                    template_ref.affine,
                    template_ref.header
                )
                nib.save(new_template, os.path.join(args.data_dir, sub, f"{sub}_template_refined.mgz"))

        # Record results
        dice_gt = np.mean(dice_scores)
        dice_template = np.mean(template_scores)

        history['dice_gt'].append(dice_gt)
        history['dice_template'].append(dice_template)

        print(f"Average Dice (Ground Truth): {dice_gt:.4f}")
        print(f"Average Dice (Template): {dice_template:.4f}")
    
    # Final results
    print("\nMETRIC EVOLUTION:")
    for i, (dice, tpl) in enumerate(zip(history['dice_gt'], history['dice_template'])):
        print(f"Iter {i+1}: Dice_GT={dice:.4f}, Dice_Template={tpl:.4f}")
    
    # Generate and save plot
    plt.figure(figsize=(10, 6))
    iterations = range(1, args.iterations + 1)
    plt.plot(iterations, history['dice_gt'], marker='o', linestyle='-', label='Dice vs Ground Truth')
    plt.plot(iterations, history['dice_template'], marker='s', linestyle='--', label='Dice vs Template')
    
    plt.title('Dice Score Evolution Through Iterations')
    plt.xlabel('Iteration Number')
    plt.ylabel('Dice Score')
    plt.xticks(iterations)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dice_evolution.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nPlot saved successfully at: {plot_path}")

if __name__ == "__main__":
    main()
import os
import argparse
import numpy as np
from DataLoader import LongitudinalDataGenerator
from Model import build_model
import tensorflow as tf

def parse_arguments():
    parser = argparse.ArgumentParser(description='3D UNet training and evaluation script')
    parser.add_argument('--test_data_dir', type=str, required=True,
                      help='Directory containing test data')
    parser.add_argument('--trained_model', type=str, default='./best_model.h5',
                      help='Path to load the trained model (default: ./best_model.h5)')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1,
                      help='Verbosity mode: 0=silent, 1=progress bar, 2=one line per epoch')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Test batch size (default: 1)')
    return parser.parse_args()

def compute_dice(one_hot_mask, pred_mask):
    """Compute Dice coefficient for each class and return the average"""
    num_classes = one_hot_mask.shape[-1]
    dice_scores = []
    
    for c in range(num_classes):
        gt = one_hot_mask[..., c].astype(np.float32)
        pred = (pred_mask == c).astype(np.float32)
        
        intersection = np.sum(gt * pred)
        union = np.sum(gt) + np.sum(pred)
        
        if union == 0:
            # Special case: no voxels in either mask
            dice = 1.0
        else:
            dice = (2. * intersection) / union
            
        dice_scores.append(dice)
    
    return np.mean(dice_scores)

def get_test_subjects(data_dir):
    """Get list of test subjects from the data directory"""
    if not os.path.exists(data_dir):
        raise ValueError(f"Test data directory does not exist: {data_dir}")
    
    # Assuming the directory structure has subject folders
    subjects = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    if not subjects:
        raise ValueError(f"No subject directories found in {data_dir}")
    
    return sorted(subjects)

def main():
    args = parse_arguments()
    
    # Validate test data directory
    if not os.path.exists(args.test_data_dir):
        raise ValueError(f"Test data directory does not exist: {args.test_data_dir}")
    
    # Validate model file
    if not os.path.exists(args.trained_model):
        raise ValueError(f"Trained model file does not exist: {args.trained_model}")
    
    # Configure GPU
    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU for testing")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu} for testing")

    # Fixed parameters
    image_size = (120, 120, 94)

    # Get test subjects from directory
    try:
        test_subjects = get_test_subjects(args.test_data_dir)
        print(f"Found {len(test_subjects)} test subjects")
    except Exception as e:
        print(f"Error getting test subjects: {str(e)}")
        return

    try:
        test_gen = LongitudinalDataGenerator(
            base_dir=args.test_data_dir,
            subjects=test_subjects,
            input_shape=image_size,
            batch_size=args.batch_size,
            n_classes=4
        )
    except Exception as e:
        print(f"Error creating data generator: {str(e)}")
        return

    # Evaluation phase
    print("\nLoading model for evaluation...")
    try:
        best_model = tf.keras.models.load_model(args.trained_model)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    dice_gt_list = []
    dice_template_list = []

    print("\nEvaluating on test set...")
    for i in range(len(test_gen)):
        try:
            (input_mri, input_template), y_true = test_gen[i]
            
            # Generate prediction
            y_pred = best_model.predict([input_mri, input_template], verbose=0)
            pred_mask = np.argmax(y_pred[0], axis=-1).astype(np.int32)
            
            # Calculate Dice scores
            dice_gt = compute_dice(y_true[0], pred_mask)
            dice_gt_list.append(dice_gt)
            
            dice_template = compute_dice(input_template[0], pred_mask)
            dice_template_list.append(dice_template)
            
            if args.verbose > 0:
                print(f"Processed subject {i+1}/{len(test_gen)}, "
                      f"Dice GT: {dice_gt:.4f}, Dice Template: {dice_template:.4f}")
                
        except Exception as e:
            print(f"Error processing subject {i}: {str(e)}")
            continue

    if not dice_gt_list:
        print("No subjects were successfully processed")
        return

    # Calculate final metrics
    final_dice_gt = np.mean(dice_gt_list)
    final_dice_template = np.mean(dice_template_list)

    print("\nFINAL RESULTS:")
    print(f"Average Dice (Ground Truth): {final_dice_gt:.4f}")
    print(f"Average Dice (Template): {final_dice_template:.4f}")
    print(f"Number of subjects successfully processed: {len(dice_gt_list)}/{len(test_gen)}")

if __name__ == "__main__":
    main()
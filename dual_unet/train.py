import os
import argparse
import numpy as np
from DataLoader import LongitudinalDataGenerator
from Model import build_model
import tensorflow as tf

def parse_arguments():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='3D UNet training script for synthetic longitudinal data')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Root directory containing synthetic data')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of training epochs')
    parser.add_argument('--final_model', type=str, 
                      default=os.path.join(current_dir, 'final_model.h5'),
                      help='Path to save final model weights')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1,
                      help='Verbosity mode: 0=silent, 1=progress bar, 2=one line per epoch')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Training batch size (default: 1)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # GPU configuration
    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU for training")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu} for training")

    # Fixed parameters
    image_size = (120, 120, 94)
    val_batch_size = 1  # Fixed validation batch size

    # Load subject data
    train_dir = os.path.join(args.data_dir, 'train')
    train_subjects = [d for d in os.listdir(train_dir) if d.startswith('sub-')]
    val_dir = os.path.join(args.data_dir, 'val')
    val_subjects = [d for d in os.listdir(val_dir) if d.startswith('sub-')]

    print(f"\nDataset splits:")
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")

    # Initialize data generators
    train_gen = LongitudinalDataGenerator(
        base_dir=train_dir,
        subjects=train_subjects,
        input_shape=image_size,
        batch_size=args.batch_size,
        n_classes=4
    )

    val_gen = LongitudinalDataGenerator(
        base_dir=val_dir,
        subjects=val_subjects,
        input_shape=image_size,
        batch_size=val_batch_size,
        n_classes=4
    )

    # Model configuration
    print("\nBuilding model...")
    model = build_model(
        input_shape_mri=(*image_size, 1),
        input_shape_template=(*image_size, 4),
        n_classes=4
    )

    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.OneHotMeanIoU(num_classes=4)],
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(current_dir, 'best_model.h5')

    # Training configuration
    print(f"\nStarting training for {args.epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        verbose=args.verbose,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path,
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            )
        ]
    )

    print("\nTraining completed.")
    print(f"- Best model saved to: {best_model_path}")
    print(f"- Final model saved to: {args.final_model}\n")
    model.save(args.final_model)

if __name__ == "__main__":
    main()
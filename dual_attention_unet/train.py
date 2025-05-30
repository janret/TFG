import os
import argparse
import numpy as np
from DataLoader import LongitudinalDataGenerator
from Model import build_model
import tensorflow as tf
from Utils import combined_loss

def parse_arguments():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, 'Models')
    
    parser = argparse.ArgumentParser(description='3D UNet training script for longitudinal data')
    parser.add_argument('--data_dir', type=str,
                      help='Root directory containing synthetic data')
    parser.add_argument('--gpu', type=int, default=-1,
                      help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--model_dir', type=str, default=default_model_dir,
                      help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Learning rate')
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

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)

    # Fixed parameters
    image_size = (120, 120, 94)
    n_classes = 4

    # Initialize data generators
    train_gen = LongitudinalDataGenerator(
        base_dir=os.path.join(args.data_dir, 'train'),
        input_shape=image_size,
        batch_size=args.batch_size,
        n_classes=n_classes,
        shuffle=True
    )

    val_gen = LongitudinalDataGenerator(
        base_dir=os.path.join(args.data_dir, 'val'),
        input_shape=image_size,
        batch_size=1,  # Fixed validation batch size
        n_classes=n_classes,
        shuffle=False
    )

    # Model configuration
    print("\nBuilding model...")
    model = build_model(
        input_shape_mri=(*image_size, 1),
        input_shape_template=(*image_size, 4),
        n_classes=n_classes
    )

    # Optimizer and compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    print("Compiling model...")
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[
            tf.keras.metrics.OneHotMeanIoU(num_classes=n_classes)
        ]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.model_dir, 'training_log.csv')
        )
    ]

    # Training
    print(f"\nStarting training for {args.epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nTraining completed. Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main() 
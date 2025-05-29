import os
import argparse
import sys
import tensorflow as tf
from DataLoader import LongitudinalDataGenerator
from Model import build_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Parent directory containing train/ and val/ subdirectories with synthetic data')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of training epochs')
    parser.add_argument('--final_model', type=str, default=None,
                      help='Path to save final model after training (defaults to script directory)')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1,
                      help='Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Get the directory where train.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set model paths relative to script directory
    best_model_path = os.path.join(script_dir, 'best_model.h5')
    if args.final_model is None:
        args.final_model = os.path.join(script_dir, 'final_model.h5')
    
    # Configure device
    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("> Using CPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"> Using GPU {args.gpu}")

    # Construct and validate data directories
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    if not os.path.exists(args.data_dir):
        sys.exit(f"Error: Parent directory {args.data_dir} does not exist")
    if not os.path.exists(train_dir):
        sys.exit(f"Error: Training directory {train_dir} does not exist")
    if not os.path.exists(val_dir):
        sys.exit(f"Error: Validation directory {val_dir} does not exist")

    # Model configuration
    image_size = (120, 120, 94)
    n_classes = 4

    # Get subjects for each dataset
    train_subs = [d for d in os.listdir(train_dir) if d.startswith('sub-')]
    val_subs = [d for d in os.listdir(val_dir) if d.startswith('sub-')]
    
    if not train_subs:
        sys.exit("Error: No training subjects found")
    if not val_subs:
        sys.exit("Error: No validation subjects found")

    # Create data generators
    train_gen = LongitudinalDataGenerator(
        base_dir=train_dir,
        subjects=train_subs,
        input_shape=image_size,
        batch_size=1,
        n_classes=n_classes
    )

    val_gen = LongitudinalDataGenerator(
        base_dir=val_dir,
        subjects=val_subs,
        input_shape=image_size,
        batch_size=1,
        n_classes=n_classes,
        shuffle=False
    )

    # Build and compile model
    model = build_model(
        input_shape=(*image_size, 1),
        n_classes=n_classes
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.OneHotMeanIoU(num_classes=4)],
    )

    # Training information
    print("\nStarting training...")
    print(f"- Training subjects: {len(train_subs)}")
    print(f"- Validation subjects: {len(val_subs)}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Best model will be automatically saved to: {best_model_path}")
    print(f"- Final model will be saved to: {args.final_model}\n")

    # Training callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=args.verbose
        )
    ]

    # Start training
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        verbose=args.verbose,
        callbacks=callbacks
    )

    # Save final model
    print("\nTraining completed..")
    model.save(args.final_model)
    print(f"Final model saved to: {args.final_model}")

if __name__ == "__main__":
    main()
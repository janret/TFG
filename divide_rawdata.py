import os
import shutil
import pandas as pd
from pathlib import Path
import argparse

def read_test_subjects(file_path):
    """Read test subjects from the txt file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def copy_subject_data(subject, source_dir, target_dir):
    """Copy a subject's data from source to target directory."""
    source_path = os.path.join(source_dir, subject)
    target_path = os.path.join(target_dir, subject)
    
    if os.path.exists(source_path):
        print(f"Copying {subject} data...")
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    else:
        print(f"Warning: {subject} directory not found in source")

def process_tsv_files(source_dir, target_dir, test_subjects):
    """Process and copy relevant TSV files."""
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Look for TSV files in the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.tsv'):
                source_file = os.path.join(root, file)
                try:
                    # Read the TSV file
                    df = pd.read_csv(source_file, sep='\t')
                    
                    # Check if the DataFrame has a subject-related column
                    subject_cols = [col for col in df.columns if 'sub' in col.lower()]
                    
                    if subject_cols:
                        # Filter rows for test subjects
                        filtered_df = df[df[subject_cols[0]].isin(test_subjects)]
                        
                        if not filtered_df.empty:
                            # Create the same directory structure in target
                            rel_path = os.path.relpath(root, source_dir)
                            target_subdir = os.path.join(target_dir, rel_path)
                            os.makedirs(target_subdir, exist_ok=True)
                            
                            # Save the filtered TSV
                            target_file = os.path.join(target_subdir, file)
                            filtered_df.to_csv(target_file, sep='\t', index=False)
                            print(f"Processed TSV file: {file}")
                
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Divide raw data into train and test sets.')
    parser.add_argument('--test-subjects-file', '-t', required=True,
                        help='Path to the file containing test subject IDs')
    parser.add_argument('--source-dir', '-s', required=True,
                        help='Source directory containing the raw data')
    parser.add_argument('--target-dir', '-o', required=True,
                        help='Target directory where test data will be copied')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Enable verbose output if requested
    if args.verbose:
        print(f"Reading test subjects from: {args.test_subjects_file}")
        print(f"Source directory: {args.source_dir}")
        print(f"Target directory: {args.target_dir}")
    
    # Read test subjects
    test_subjects = read_test_subjects(args.test_subjects_file)
    print(f"Found {len(test_subjects)} test subjects")
    
    # Create target directory
    os.makedirs(args.target_dir, exist_ok=True)
    
    # Copy subject data
    for subject in test_subjects:
        copy_subject_data(subject, args.source_dir, args.target_dir)
    
    # Process TSV files
    process_tsv_files(args.source_dir, args.target_dir, test_subjects)

if __name__ == "__main__":
    main()

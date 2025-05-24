#!/usr/bin/env python3
"""
Neuroimaging Data Processing Pipeline
Combines volumetric brain data from SynthSeg with clinical metadata in BIDS format
"""

import os
import pandas as pd
import argparse

def load_synthseg_volumes(derivatives_dir: str) -> pd.DataFrame:
    """Load and combine SynthSeg volumetric outputs
    
    Args:
        derivatives_dir: Path to derivatives directory containing SynthSeg CSV files
        
    Returns:
        Combined DataFrame with all volumetric measurements
    """
    volume_dfs = []
    
    for root, _, files in os.walk(derivatives_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, index_col=0)
                df = df.reset_index().rename(columns={'index': 'Label'})
                df['Label'] = df['Label'].str.replace('_T1w.nii', '', regex=False)
                volume_dfs.append(df)
    
    return pd.concat(volume_dfs, ignore_index=True).sort_values(by='Label')

def merge_with_metadata(rawdata_dir: str, derivatives_dir: str, volumes_df: pd.DataFrame) -> pd.DataFrame:
    """Merge volumetric data with clinical metadata
    
    Args:
        rawdata_dir: Path to BIDS rawdata directory
        derivatives_dir: Path to output directory
        volumes_df: DataFrame containing volumetric measurements
        
    Returns:
        Combined DataFrame with clinical and volumetric data
    """
    metadata_path = os.path.join(rawdata_dir, 'participants_time.tsv')
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    
    merged_df = pd.merge(metadata_df, volumes_df, on='Label', how='inner')
    
    output_path = os.path.join(derivatives_dir, 'participants.tsv')
    merged_df.to_csv(output_path, sep='\t', index=False)
    
    return merged_df

def create_baseline_dataset(derivatives_dir: str, full_dataset: pd.DataFrame) -> pd.DataFrame:
    """Create baseline dataset for cross-sectional analysis
    
    Args:
        derivatives_dir: Path to output directory
        full_dataset: Combined DataFrame with all timepoints
        
    Returns:
        DataFrame containing only baseline measurements
    """
    baseline_df = full_dataset[full_dataset['time (years) from baseline'] == 0].copy()
    
    # Process run numbers
    baseline_df['run'] = baseline_df['Label'].str.extract(r'run-(\d+)').astype(int)
    baseline_df = baseline_df.sort_values(by=['Subject', 'run'])
    baseline_df = baseline_df.drop_duplicates(subset=['Subject'], keep='first')
    baseline_df = baseline_df.drop(columns=['run'])
    
    # Save output
    output_path = os.path.join(derivatives_dir, 'participants_baseline.tsv')
    baseline_df.to_csv(output_path, sep='\t', index=False)
    
    return baseline_df

def main():
    """Main processing pipeline"""
    parser = argparse.ArgumentParser(description='Process neuroimaging volumes and clinical data')
    parser.add_argument(
        '--rawdata', 
        required=True,
        help='Path to BIDS rawdata directory containing participants_time.tsv'
    )
    parser.add_argument(
        '--derivatives',
        required=True,
        help='Path to derivatives directory containing SynthSeg outputs'
    )
    
    args = parser.parse_args()
    
    # 1. Load volumetric data
    print("Loading SynthSeg volumes...")
    volumes_df = load_synthseg_volumes(args.derivatives)
    print(f"Loaded {len(volumes_df)} volumetric measurements")
    
    # 2. Merge with clinical data
    print("\nMerging with clinical metadata...")
    merged_df = merge_with_metadata(args.rawdata, args.derivatives, volumes_df)
    print(f"Combined dataset contains {len(merged_df)} records")
    print("Sample data:")
    print(merged_df.head())
    
    # 3. Create baseline dataset
    print("\nGenerating baseline dataset...")
    baseline_df = create_baseline_dataset(args.derivatives, merged_df)
    print(f"Baseline dataset contains {len(baseline_df)} subjects")
    print("Sample baseline data:")
    print(baseline_df.head())

if __name__ == "__main__":
    main()
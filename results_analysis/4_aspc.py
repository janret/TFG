#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_aspc(run1_data: pd.DataFrame, run2_data: pd.DataFrame, volume_col: str) -> float:
    """
    Calculate ASPC (Annualized Symmetric Percentage Change) between run1 and run2.
    
    Args:
        run1_data: DataFrame with first run measurements
        run2_data: DataFrame with second run measurements
        volume_col: Name of the volume column to analyze
    
    Returns:
        float: Mean ASPC value
    """
    # Calculate time difference in years
    time_diff = (pd.to_datetime(run2_data['session_date']) - pd.to_datetime(run1_data['session_date'])).dt.days / 365.25
    
    # Calculate ASPC
    aspc = 2 * (run2_data[volume_col].values - run1_data[volume_col].values) / \
           (run2_data[volume_col].values + run1_data[volume_col].values) * 100 / time_diff
    return np.mean(aspc)


def process_single_file(input_file: str) -> tuple:
    """
    Process a single participants file and calculate ASPC between run1 and run2.
    
    Args:
        input_file (str): Path to participants.tsv file containing both runs
    
    Returns:
        tuple: (results list, number of subjects)
    """
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    
    # Split data into run1 and run2
    run1_data = df[df['session_id'].str.contains('run-1')]
    run2_data = df[df['session_id'].str.contains('run-2')]
    
    # Ensure we have matching participants in both runs
    common_participants = set(run1_data['participant_id']) & set(run2_data['participant_id'])
    run1_data = run1_data[run1_data['participant_id'].isin(common_participants)]
    run2_data = run2_data[run2_data['participant_id'].isin(common_participants)]
    
    # Sort both dataframes by participant_id to ensure alignment
    run1_data = run1_data.sort_values('participant_id')
    run2_data = run2_data.sort_values('participant_id')
    
    # Volumes to analyze
    volumes = ['gray_matter_mm3', 'white_matter_mm3', 'csf_mm3']
    
    # Calculate ASPC for each volume
    results = []
    for vol in volumes:
        aspc = calculate_aspc(run1_data, run2_data, vol)
        results.append({
            'File': Path(input_file).stem,
            'Volume': vol.replace('_mm3', ''),
            'ASPC': aspc,
            'N_subjects': len(common_participants)
        })
    
    return results, len(common_participants)


def process_volumes(input_files: list, output_dir: str):
    """
    Process multiple participants files and calculate ASPC between run1 and run2 for each.
    
    Args:
        input_files (list): List of paths to participants.tsv files
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all files
    all_results = []
    for input_file in input_files:
        results, n_subjects = process_single_file(input_file)
        all_results.extend(results)
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(output_path / 'aspc_results.tsv', sep='\t', index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Volume', y='ASPC', hue='File')
    plt.title('Annualized Symmetric Percentage Change by Volume Type')
    plt.ylabel('ASPC (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path / 'aspc_plot.png', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Calculate ASPC between run1 and run2 for multiple participants files')
    parser.add_argument('-i', '--inputs', required=True, nargs='+',
                      help='One or more participants.tsv files to analyze')
    parser.add_argument('-o', '--output', required=True,
                      help='Output directory for results')
    
    args = parser.parse_args()
    process_volumes(args.inputs, args.output)


if __name__ == '__main__':
    exit(main()) 
#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_apc(df: pd.DataFrame, volume_col: str) -> pd.DataFrame:
    """
    Calculate Annual Percentage Change (APC) for a given volume column.
    
    Args:
        df: DataFrame with the participants data
        volume_col: Name of the volume column to analyze
    
    Returns:
        DataFrame with APC results
    """
    # Sort by participant and date
    df = df.sort_values(['participant_id', 'session_date'])
    
    results = []
    
    # Process each participant
    for participant in df['participant_id'].unique():
        participant_data = df[df['participant_id'] == participant].copy()
        
        if len(participant_data) < 2:
            continue
            
        # Calculate time differences in years
        participant_data['date'] = pd.to_datetime(participant_data['session_date'])
        time_diff = (participant_data['date'].max() - participant_data['date'].min()).days / 365.25
        
        # Calculate volume change
        initial_volume = participant_data.iloc[0][volume_col]
        final_volume = participant_data.iloc[-1][volume_col]
        
        # Calculate APC
        apc = ((final_volume / initial_volume) ** (1 / time_diff) - 1) * 100
        
        results.append({
            'participant_id': participant,
            'initial_date': participant_data['date'].min(),
            'final_date': participant_data['date'].max(),
            'time_diff_years': time_diff,
            'initial_volume': initial_volume,
            'final_volume': final_volume,
            'apc': apc
        })
    
    return pd.DataFrame(results)


def create_visualizations(results: pd.DataFrame, output_dir: Path, volume_name: str):
    """
    Create and save visualizations for APC analysis.
    
    Args:
        results: DataFrame with APC results
        output_dir: Directory to save the plots
        volume_name: Name of the volume being analyzed
    """
    # APC Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results, x='apc', bins=20)
    plt.title(f'Distribution of Annual Percentage Change - {volume_name}')
    plt.xlabel('Annual Percentage Change (%)')
    plt.ylabel('Count')
    plt.savefig(output_dir / f'apc_distribution_{volume_name.lower()}.png')
    plt.close()
    
    # Time vs APC Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results, x='time_diff_years', y='apc')
    plt.title(f'Time Difference vs APC - {volume_name}')
    plt.xlabel('Time Difference (years)')
    plt.ylabel('Annual Percentage Change (%)')
    plt.savefig(output_dir / f'time_vs_apc_{volume_name.lower()}.png')
    plt.close()


def process_volumes(input_file: str, output_dir: str):
    """
    Process volume data and calculate APC for each brain structure.
    
    Args:
        input_file: Path to the participants.tsv file
        output_dir: Directory to save output files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    
    # Process each volume type
    volumes = {
        'White Matter': 'white_matter_mm3',
        'Gray Matter': 'gray_matter_mm3',
        'CSF': 'csf_mm3'
    }
    
    all_results = []
    
    for volume_name, volume_col in volumes.items():
        # Calculate APC
        results = calculate_apc(df, volume_col)
        results['volume_type'] = volume_name
        all_results.append(results)
        
        # Create visualizations
        create_visualizations(results, output_path, volume_name)
    
    # Combine all results and save
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv(output_path / 'apc_results.tsv', sep='\t', index=False)
    
    # Create summary statistics
    summary = final_results.groupby('volume_type').agg({
        'apc': ['mean', 'std', 'min', 'max', 'count']
    }).round(3)
    
    summary.to_csv(output_path / 'apc_summary.tsv', sep='\t')
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Calculate Annual Percentage Change (APC) from participants data')
    parser.add_argument('-i', '--input', required=True, help='Input participants.tsv file')
    parser.add_argument('-o', '--output', required=True, help='Output directory for results and plots')
    
    args = parser.parse_args()
    process_volumes(args.input, args.output)


if __name__ == '__main__':
    main() 
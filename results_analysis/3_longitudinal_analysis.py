#!/usr/bin/env python3

import argparse
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path


def perform_lme_analysis(input_file: str, output_dir: str):
    """
    Perform Linear Mixed Effects analysis on longitudinal brain volume data.
    
    Args:
        input_file (str): Path to the participants.tsv file containing all timepoints
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    
    # List of volumes to analyze
    volumes = {
        'WM': 'white_matter_mm3',
        'GM': 'gray_matter_mm3',
        'CSF': 'csf_mm3'
    }
    
    # Store results for each volume type
    all_results = {}
    
    for vol_name, vol_column in volumes.items():
        # Prepare data for analysis
        model_df = df.assign(
            Intercept=1,
            AD=(df['Group'] == 'AD').astype(int),
            Male=(df['M/F'] == 'M').astype(int),
            ICV=df['TIV'],
            VOL=df[vol_column]
        ).rename(columns={
            'time (years) from baseline': 'time',
            'Age': 'age',
        })
        
        # Define the formula
        formula = """
        VOL ~
            time + 
            AD + 
            AD:time +
            Male +
            age +
            ICV
        """
        
        # Fit the mixed model
        mixed_model = smf.mixedlm(
            formula,
            data=model_df,
            groups=model_df['Subject'],
            re_formula="~1 + time"
        ).fit()
        
        # Store results
        all_results[vol_name] = {
            'summary': mixed_model.summary().as_text(),
            'pvalues': mixed_model.pvalues.round(8),
            'params': mixed_model.params.round(3),
            'conf_int': mixed_model.conf_int().round(3)
        }
        
        # Save individual results
        with open(output_path / f'{vol_name.lower()}_model_summary.txt', 'w') as f:
            f.write(all_results[vol_name]['summary'])
            f.write('\n\nP-values:\n')
            f.write(all_results[vol_name]['pvalues'].to_string())
            f.write('\n\nParameters:\n')
            f.write(all_results[vol_name]['params'].to_string())
            f.write('\n\nConfidence Intervals:\n')
            f.write(all_results[vol_name]['conf_int'].to_string())
    
    # Save combined results
    combined_results = pd.DataFrame({
        vol_name: {
            'Intercept_coef': results['params']['Intercept'],
            'Intercept_pval': results['pvalues']['Intercept'],
            'time_coef': results['params']['time'],
            'time_pval': results['pvalues']['time'],
            'AD_coef': results['params']['AD'],
            'AD_pval': results['pvalues']['AD'],
            'AD:time_coef': results['params']['AD:time'],
            'AD:time_pval': results['pvalues']['AD:time'],
            'Male_coef': results['params']['Male'],
            'Male_pval': results['pvalues']['Male'],
            'age_coef': results['params']['age'],
            'age_pval': results['pvalues']['age'],
            'ICV_coef': results['params']['ICV'],
            'ICV_pval': results['pvalues']['ICV']
        }
        for vol_name, results in all_results.items()
    }).T

    combined_results.to_csv(output_path / 'combined_results.tsv', sep='\t')


def main():
    parser = argparse.ArgumentParser(description='Perform longitudinal analysis on brain volume data using Linear Mixed Effects models')
    parser.add_argument('-i', '--input', required=True, help='Input participants.tsv file containing all timepoints')
    parser.add_argument('-o', '--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    perform_lme_analysis(args.input, args.output)


if __name__ == '__main__':
    exit(main()) 
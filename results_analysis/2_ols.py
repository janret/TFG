#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def perform_ols_analysis(input_file: str, output_dir: str):
    """
    Perform OLS analysis on participants data.
    
    Args:
        input_file (str): Path to the participants_baseline.tsv file
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    
    # Prepare data for analysis
    volumes = ['gray_matter_mm3', 'white_matter_mm3', 'csf_mm3']
    results = []
    
    # Perform OLS for each volume type
    for vol in volumes:
        # Create the model
        X = sm.add_constant(df['age'])
        y = df[vol]
        model = sm.OLS(y, X).fit()
        
        # Store results
        results.append({
            'volume': vol,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'p_value': model.f_pvalue,
            'age_coef': model.params['age'],
            'age_p_value': model.pvalues['age'],
            'intercept': model.params['const'],
            'intercept_p_value': model.pvalues['const']
        })
        
        # Create scatter plot with regression line
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x='age', y=vol)
        plt.title(f'Age vs {vol}')
        plt.savefig(output_path / f'{vol}_regression.png')
        plt.close()
    
    # Save statistical results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'ols_results.tsv', sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Perform OLS analysis on brain volume data')
    parser.add_argument('-i', '--input', required=True, help='Input participants_baseline.tsv file containing volume data')
    parser.add_argument('-o', '--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    perform_ols_analysis(args.input, args.output)


if __name__ == '__main__':
    exit(main()) 
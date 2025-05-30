# TFG Project

This repository contains the code for the TFG project, focusing on neuroimaging data processing and analysis using deep learning and traditional methods. Below you'll find instructions on how to set up the development environment and run the code.

## Repository Structure
```
TFG/
├── simple_unet/           # Basic U-Net implementation for segmentation
│   ├── Model.py           # U-Net model architecture
│   ├── DataLoader.py      # Data loading and preprocessing
│   ├── train.py          # Training script
│   ├── test.py           # Testing script
│   └── predict.py        # Prediction script
├── dual_unet/            # Alternative U-Net implementation
│   ├── Model.py          # U-Net model architecture
│   ├── DataLoader.py     # Data loading and preprocessing
│   ├── Utils.py         # Utility functions
│   ├── train.py         # Training script
│   ├── test.py          # Testing script
│   └── predict.py       # Prediction script
├── dual_attention_unet/   # Advanced U-Net with dual attention mechanism
│   ├── Model.py           # Dual attention U-Net architecture
│   ├── DataLoader.py      # Data loading and preprocessing
│   ├── Utils.py          # Utility functions
│   ├── train.py          # Training script
│   ├── test.py           # Testing script
│   └── predict.py        # Prediction script
├── results_analysis/     # Scripts for analyzing segmentation results
│   ├── 1_merge_participants_volumes.py  # Merge participant data with volume measurements
│   ├── 2_ols.py                        # Ordinary Least Squares analysis
│   ├── 3_longitudinal_analysis.py      # Longitudinal data analysis
│   ├── 4_aspc.py                       # Annualized Symmetric Percentage Change analysis
│   └── 5_apc.py                        # Annual Percentage Change analysis
├── create_synthetic_data.py          # Generate synthetic data for training
├── from_miriad_to_bids.py           # Convert MIRIAD format to BIDS
├── run_samseg_long.py               # Run longitudinal SAMSEG processing
├── synthseg_generate_participants_file.py  # Generate participant information
├── visualitza_tps.sh                # Visualize temporal processing steps
├── divide_rawdata.py                # Split data into train/test sets
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Prerequisites

### FreeSurfer
- FreeSurfer version 7.4.1 is required
- Installation instructions:
  1. Download FreeSurfer 7.4.1 from [FreeSurfer website](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)
  2. Set up FreeSurfer environment variables in your `.bashrc` or `.bash_profile`:
     ```bash
     export FREESURFER_HOME=/path/to/freesurfer/7.4.1
     source $FREESURFER_HOME/SetUpFreeSurfer.sh
     ```
  3. Make sure you have a valid FreeSurfer license file installed

### Python Environment Setup

1. Create a new virtual environment:
```bash
python -m venv tfg_env
```

2. Activate the virtual environment:
- On Linux/Mac:
  ```bash
  source tfg_env/bin/activate
  ```
- On Windows:
  ```bash
  .\tfg_env\Scripts\activate
  ```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Required Packages

The project uses several key packages including:
- TensorFlow (2.19.0)
- PyTorch (2.7.0)
- NumPy (2.1.3)
- Matplotlib (3.10.3)
- Pandas (2.2.3)
- Scikit-learn (1.6.1)
- SimpleITK (2.5.0)
- Nibabel (5.3.2)
- TorchIO (0.20.8)
- SciPy (1.15.3)
- Einops (0.8.1)
- Seaborn (0.13.2)
- Statsmodels (0.14.4)

Additional dependencies include various utilities for data processing, visualization, and deep learning. For a complete list of dependencies and their versions, see `requirements.txt`.

## Scripts and Usage

### 1. Synthetic Data Generation (`create_synthetic_data.py`)
Generate synthetic neuroimaging data for testing and validation.

**Features**:
- Creates multiple synthetic timepoints from single MRI scans
- Applies random affine and elastic deformations
- Supports train/validation/test splitting

**Usage**:
```bash
python create_synthetic_data.py \
  --input_dir /path/to/samseg/data \
  --synthetic_dir /path/to/output \
  --num_transforms 15 \
  --voxel_size 1.0 \
  --split
```

### 2. MIRIAD to BIDS Conversion (`from_miriad_to_bids.py`)
Convert neuroimaging data from MIRIAD format to BIDS format.

**Features**:
- Organizes data according to BIDS specification
- Generates required metadata files
- Creates longitudinal time information
- Expects participant information in TSV format with columns:
  - Label (format: sub-XXX_ses-XX_run-X)
  - Subject
  - M/F
  - Age
  - Group
  - time(days) from baseline
  - time (years) from baseline

**Usage**:
```bash
python from_miriad_to_bids.py \
  --input_dir /path/to/miriad/data \
  --output_dir /path/to/bids/output \
  --csv_file /path/to/metadata.csv
```

### 3. Longitudinal Segmentation (`run_samseg_long.py`)
Process longitudinal brain MRI data using FreeSurfer's SAMSEG algorithm.

**Features**:
- Creates subject-specific templates
- Runs longitudinal SAMSEG processing
- Generates volume measurements

**Usage**:
```bash
python run_samseg_long.py \
  --input_dir /path/to/bids/data \
  --output_dir /path/to/samseg/output \
  --inverted_dir /path/to/inverted/output \
  --threads 4
```

### 4. Participant File Generation (`synthseg_generate_participants_file.py`)
Generate participant information files combining volumetric and clinical data.

**Features**:
- Combines SynthSeg volumetric outputs with clinical metadata
- Creates baseline datasets for cross-sectional analysis
- Generates standardized TSV files

**Usage**:
```bash
python synthseg_generate_participants_file.py \
  --rawdata /path/to/bids/rawdata \
  --derivatives /path/to/synthseg/output
```

### 5. Temporal Processing Visualization (`visualitza_tps.sh`)
Visualize temporal processing steps using FreeSurfer's FreeView tool.

**Features**:
- Interactive visualization of longitudinal scans
- Side-by-side comparison of timepoints
- Segmentation overlay support

**Usage**:
```bash
./visualitza_tps.sh -d /path/to/subject -t 15
```

### 6. Data Division (`divide_rawdata.py`)
Split data into training and test sets.

**Features**:
- Preserves directory structure
- Processes metadata files
- Maintains data organization

**Usage**:
```bash
python divide_rawdata.py \
  --test-subjects-file /path/to/test_subjects.txt \
  --source-dir /path/to/rawdata \
  --target-dir /path/to/rawdata_test
```

### 7. Simple U-Net Model (`simple_unet/`)
Deep learning model for brain MRI segmentation.

**Components**:
- Model architecture (`Model.py`)
- Data loading utilities (`DataLoader.py`)
- Training and evaluation scripts

**Output Format**:
- Segmentation masks (.nii.gz format)
- Volume measurements (volumes.tsv) containing:
  - Label column (format: sub-XXX_ses-XX_run-X)
  - Volume measurements (gray_matter_mm3, white_matter_mm3, csf_mm3)

**Usage**:
```bash
# Training
python simple_unet/train.py \
  --data_dir /path/to/training/data \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 0.001

# Prediction
python simple_unet/predict.py \
  --input_image /path/to/input.nii \
  --output_mask /path/to/output.nii \
  --model_path /path/to/trained/model.h5
```

### 8. Dual U-Net Model (`dual_unet/`)
Alternative U-Net implementation for brain MRI segmentation.

**Components**:
- Model architecture (`Model.py`)
- Data loading utilities (`DataLoader.py`)
- Utility functions (`Utils.py`)
- Training and evaluation scripts

**Output Format**:
- Segmentation masks (.nii.gz format)
- Volume measurements (volumes.tsv) containing:
  - Label column (format: sub-XXX_ses-XX_run-X)
  - Volume measurements (gray_matter_mm3, white_matter_mm3, csf_mm3)

**Usage**:
```bash
# Training
python dual_unet/train.py \
  --data_dir /path/to/training/data \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 0.001

# Prediction
python dual_unet/predict.py \
  --input_image /path/to/input.nii \
  --output_mask /path/to/output.nii \
  --model_path /path/to/trained/model.h5
```

### 9. Dual Attention U-Net Model (`dual_attention_unet/`)
Advanced U-Net architecture incorporating dual attention mechanisms for improved segmentation accuracy.

**Components**:
- Enhanced model architecture with attention mechanisms (`Model.py`)
- Data loading and augmentation utilities (`DataLoader.py`)
- Utility functions for attention computation and visualization (`Utils.py`)
- Training and evaluation scripts

**Output Format**:
- Segmentation masks (.mgz format)
- Volume measurements (volumes.tsv) containing:
  - Label column (format: sub-XXX_ses-XX_run-X)
  - Volume measurements (gray_matter_mm3, white_matter_mm3, csf_mm3)

**Usage**:
```bash
# Training
python dual_attention_unet/train.py \
  --data_dir /path/to/training/data \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 0.001

# Prediction
python dual_attention_unet/predict.py \
  --input_image /path/to/input.nii \
  --output_mask /path/to/output.nii \
  --model_path /path/to/trained/model.h5
```

### 10. Results Analysis Scripts (`results_analysis/`)
Collection of scripts for analyzing segmentation results and performing statistical analyses.

#### 10.1 Merge Participants Volumes (`1_merge_participants_volumes.py`)
Combines participant information with volume measurements from segmentation models.

**Features**:
- Merges participant metadata with volume measurements
- Creates baseline and longitudinal datasets
- Handles volume measurements from all U-Net variants (simple, dual, and dual attention)
- Calculates Total Intracranial Volume (TIV)

**Input Format Requirements**:
- Participants file (TSV) must contain:
  - Label column (format: sub-XXX_ses-XX_run-X)
  - Demographic and clinical data columns
- Volumes file (TSV) must contain:
  - Label column (matching format)
  - Volume measurements (gray_matter_mm3, white_matter_mm3, csf_mm3)

**Usage**:
```bash
python 1_merge_participants_volumes.py \
  -p /path/to/participants.tsv \
  -v /path/to/volumes.tsv \
  -o /path/to/output_dir
```

**Outputs**:
- `participants.tsv`: Complete dataset with volumes and metadata
- `participants_baseline.tsv`: Filtered dataset containing only baseline timepoints

#### 10.2 Ordinary Least Squares Analysis (`2_ols.py`)
Performs OLS regression analysis on brain volume data.

**Features**:
- Statistical analysis of volume changes
- Generates regression plots and statistics
- Outputs comprehensive statistical reports

**Usage**:
```bash
python 2_ols.py \
  -i /path/to/participants_baseline.tsv \
  -o /path/to/output_dir
```

#### 10.3 Longitudinal Analysis (`3_longitudinal_analysis.py`)
Analyzes longitudinal changes in brain volumes using linear mixed effects models.

**Features**:
- Linear mixed effects modeling
- Multiple timepoint analysis
- Visualization of longitudinal trends

**Usage**:
```bash
python 3_longitudinal_analysis.py \
  -i /path/to/participants.tsv \
  -o /path/to/output_dir
```

#### 10.4 Annualized Symmetric Percentage Change (`4_aspc.py`)
Calculates ASPC between different timepoints for brain volume measurements.

**Features**:
- Handles multiple input files
- Compares run1 vs run2 measurements
- Generates visualizations and statistics

**Usage**:
```bash
python 4_aspc.py \
  -i /path/to/participants1.tsv /path/to/participants2.tsv \
  -o /path/to/output_dir
```

#### 10.5 Annual Percentage Change (`5_apc.py`)
Calculates APC for longitudinal brain volume changes.

**Features**:
- Single file analysis
- Distribution plots of APC values
- Time vs APC scatter plots
- Summary statistics generation

**Usage**:
```bash
python 5_apc.py \
  -i /path/to/participants.tsv \
  -o /path/to/output_dir
```

**Outputs**:
- APC distribution plots for each volume type
- Time vs APC scatter plots
- Detailed results in TSV format
- Summary statistics

## Troubleshooting

1. Verify FreeSurfer 7.4.1 installation:
   ```bash
   echo $FREESURFER_HOME
   which freeview
   ```

2. Check Python environment:
   ```bash
   python --version
   pip list
   ```

3. Common issues:
   - FreeSurfer license missing
   - Incorrect environment variables
   - Missing dependencies
   - Insufficient disk space for synthetic data

For specific issues, check error messages or create an issue in the repository.
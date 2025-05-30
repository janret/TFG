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

## Data Flow and File Formats

### Input Data Format
- Raw data should be in MIRIAD format
- Can be converted to BIDS format using `from_miriad_to_bids.py`
- Participant information should be in TSV format with columns:
  - Label (format: sub-XXX_ses-XX_run-X)
  - Subject
  - M/F
  - Age
  - Group
  - time(days) from baseline
  - time (years) from baseline

### Output Data Format
All segmentation models (simple_unet, dual_unet, dual_attention_unet) generate:
- Segmentation masks (.nii.gz or .mgz format)
- Volume measurements (volumes.tsv) containing:
  - Label column (format: sub-XXX_ses-XX_run-X)
  - Volume measurements (gray_matter_mm3, white_matter_mm3, csf_mm3)

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

## Pipeline Components

### 1. Data Preparation

#### Converting MIRIAD to BIDS (`from_miriad_to_bids.py`)
Converts neuroimaging data from MIRIAD format to BIDS format.

**Usage**:
```bash
python from_miriad_to_bids.py \
  --input_dir /path/to/miriad/data \
  --output_dir /path/to/bids/output \
  --csv_file /path/to/metadata.csv
```

#### Synthetic Data Generation (`create_synthetic_data.py`)
Generate synthetic neuroimaging data for testing and validation.

**Usage**:
```bash
python create_synthetic_data.py \
  --input_dir /path/to/samseg/data \
  --synthetic_dir /path/to/output \
  --num_transforms 15 \
  --voxel_size 1.0 \
  --split
```

### 2. Data Processing

#### SAMSEG Processing (`run_samseg_long.py`)
Process longitudinal brain MRI data using FreeSurfer's SAMSEG algorithm.

**Usage**:
```bash
python run_samseg_long.py \
  --input_dir /path/to/bids/data \
  --output_dir /path/to/samseg/output \
  --inverted_dir /path/to/inverted/output \
  --threads 4
```

### 3. Segmentation Models

#### Simple U-Net (`simple_unet/`)
Basic U-Net implementation for brain MRI segmentation.

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

#### Dual U-Net (`dual_unet/`)
Alternative U-Net implementation for brain MRI segmentation.

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

#### Dual Attention U-Net (`dual_attention_unet/`)
Advanced U-Net architecture with dual attention mechanisms.

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

### 4. Results Analysis

#### Merge Participants Volumes (`results_analysis/1_merge_participants_volumes.py`)
Combines participant information with volume measurements from segmentation models.

**Input Requirements**:
- Participants file (TSV):
  - Label column (format: sub-XXX_ses-XX_run-X)
  - Demographic and clinical data columns
- Volumes file (TSV):
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
- `participants_baseline.tsv`: Filtered dataset with baseline timepoints

#### Statistical Analysis Scripts

1. **Ordinary Least Squares (`2_ols.py`)**
   ```bash
   python 2_ols.py \
     -i /path/to/participants_baseline.tsv \
     -o /path/to/output_dir
   ```

2. **Longitudinal Analysis (`3_longitudinal_analysis.py`)**
   ```bash
   python 3_longitudinal_analysis.py \
     -i /path/to/participants.tsv \
     -o /path/to/output_dir
   ```

3. **ASPC Analysis (`4_aspc.py`)**
   ```bash
   python 4_aspc.py \
     -i /path/to/participants1.tsv /path/to/participants2.tsv \
     -o /path/to/output_dir
   ```

4. **APC Analysis (`5_apc.py`)**
   ```bash
   python 5_apc.py \
     -i /path/to/participants.tsv \
     -o /path/to/output_dir
   ```

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
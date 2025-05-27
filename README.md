# TFG Project

This repository contains the code for the TFG project, focusing on neuroimaging data processing and analysis using deep learning and traditional methods. Below you'll find instructions on how to set up the development environment and run the code.

## Repository Structure
```
TFG/
├── simple_u_net/           # U-Net implementation for segmentation
│   ├── Model.py           # U-Net model architecture
│   ├── DataLoader.py      # Data loading and preprocessing
│   ├── train.py          # Training script
│   ├── test.py           # Testing script
│   ├── predict.py        # Prediction script
│   └── best_model.h5     # Pre-trained model weights
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
- TensorFlow (2.13.0)
- NumPy (1.24.3)
- Matplotlib (3.7.1)
- Pandas (2.0.3)
- Scikit-learn (1.3.0)
- SimpleITK (2.2.1)
- Nibabel (5.1.0)
- SciPy (1.11.2)

For a complete list of dependencies and their versions, see `requirements.txt`.

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

### 7. U-Net Segmentation Model (`simple_u_net/`)
Deep learning model for brain MRI segmentation.

**Components**:
- Model architecture (`Model.py`)
- Data loading utilities (`DataLoader.py`)
- Training and evaluation scripts
- Pre-trained weights

**Usage**:
```bash
# Training
python simple_u_net/train.py \
  --data_dir /path/to/training/data \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 0.001

# Prediction
python simple_u_net/predict.py \
  --input_image /path/to/input.nii \
  --output_mask /path/to/output.nii \
  --model_path simple_u_net/best_model.h5
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

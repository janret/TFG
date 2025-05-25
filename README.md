# TFG Project

This repository contains the code for the TFG project, focusing on neuroimaging data processing and analysis. Below you'll find instructions on how to set up the development environment and run the code.

## Prerequisites

### FreeSurfer
- FreeSurfer version 7.4.1 is required
- Installation instructions:
  1. Download FreeSurfer 7.4.1 from [FreeSurfer website](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)
  2. Set up FreeSurfer environment variables in your `.bashrc` or `.bash_profile`:
     ```bash
     export FREESURFER_HOME=/path/to/freesurfer
     source $FREESURFER_HOME/SetUpFreeSurfer.sh
     ```
  3. Make sure you have a valid FreeSurfer license file installed

### Python Environment Setup

1. Create a new virtual environment:
```bash
python -m venv tf
```

2. Activate the virtual environment:
- On Linux/Mac:
  ```bash
  source tf/bin/activate
  ```
- On Windows:
  ```bash
  .\tf\Scripts\activate
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
- TorchIO (0.20.8)

For a complete list of dependencies and their versions, see `requirements.txt`.

## Scripts and Usage

### 1. Synthetic Data Generation
Script: `create_synthetic_data.py`
- **Purpose**: Generates synthetic neuroimaging data for testing and validation by applying transformations to existing MRI scans
- **Key Features**:
  - Generates multiple synthetic timepoints from a single MRI scan
  - Applies random affine and elastic deformations
  - Supports data splitting into train/validation/test sets
  - Processes both MRI images and their segmentations
- **Required Arguments**:
  - `--input_dir`: Directory containing original SAMSEG data
  - `--synthetic_dir`: Output directory for synthetic data
- **Optional Arguments**:
  - `--preprocessed_dir`: Directory for preprocessed data (default: /tmp/preprocessed)
  - `--num_transforms`: Number of synthetic timepoints to generate (default: 10)
  - `--voxel_size`: Target voxel size for preprocessing
  - `--split`: Enable train/val/test splitting (80/15/5 split)
- **Example Usage**:
  ```bash
  python create_synthetic_data.py \
    --input_dir /path/to/samseg/data \
    --synthetic_dir /path/to/output \
    --num_transforms 15 \
    --voxel_size 1.0 \
    --split
  ```

### 2. MIRIAD to BIDS Conversion
Script: `from_miriad_to_bids.py`
- **Purpose**: Converts neuroimaging data from MIRIAD format to BIDS (Brain Imaging Data Structure) format
- **Key Features**:
  - Organizes data according to BIDS specification
  - Generates required metadata files
  - Creates longitudinal time information
  - Sets up derivatives directory structure
- **Required Arguments**:
  - `--input_dir`: Source directory of MIRIAD dataset
  - `--output_dir`: Root directory for output (rawdata and derivatives)
  - `--csv_file`: Path to metadata CSV file
- **Output Structure**:
  ```
  output_dir/
  ├── rawdata/
  │   ├── sub-<id>/
  │   │   ├── ses-<session>/
  │   │   │   └── anat/
  │   │   │       └── sub-<id>_ses-<session>_run-<run>_T1w.nii
  │   │   └── time.tsv
  │   ├── participants.tsv
  │   ├── participants_time.tsv
  │   └── dataset_description.json
  └── derivatives/
      └── synthseg/
  ```
- **Example Usage**:
  ```bash
  python from_miriad_to_bids.py \
    --input_dir /path/to/miriad/data \
    --output_dir /path/to/bids/output \
    --csv_file /path/to/metadata.csv
  ```

### 3. Longitudinal Segmentation
Script: `run_samseg_long.py`
- **Purpose**: Implements longitudinal segmentation using FreeSurfer's SAMSEG algorithm
- **Key Features**:
  - Creates robust templates for each subject
  - Runs longitudinal SAMSEG processing
  - Inverts segmentations back to original space
  - Processes statistical measures
- **Steps**:
  1. Template Creation: Generates robust mean template from all timepoints
  2. SAMSEG Processing: Runs longitudinal segmentation
  3. Inversion: Maps segmentations back to original space
  4. Stats Processing: Extracts and compiles volume measurements
- **Required Environment**:
  - FreeSurfer 7.4.1
  - Properly set FREESURFER_HOME
  - Valid FreeSurfer license
- **Arguments**:
  - `--input_dir`: Directory with BIDS-formatted input data
  - `--output_dir`: Directory for SAMSEG output
  - `--inverted_dir`: Directory for inverted segmentations
  - `--threads`: Number of threads for processing (default: 1)
- **Example Usage**:
  ```bash
  python run_samseg_long.py \
    --input_dir /path/to/bids/data \
    --output_dir /path/to/samseg/output \
    --inverted_dir /path/to/inverted/output \
    --threads 4
  ```

### 4. Participant File Generation
Script: `synthseg_generate_participants_file.py`
- **Purpose**: Generates participant information files for SynthSeg processing
- **Usage**:
  ```bash
  python synthseg_generate_participants_file.py [options]
  ```
- **Note**: Additional documentation needed for specific options and parameters

### 5. Temporal Processing Visualization
Script: `visualitza_tps.sh`
- **Purpose**: Shell script for visualizing temporal processing steps
- **Features**:
  - Visualizes longitudinal changes
  - Supports multiple visualization modes
- **Usage**:
  ```bash
  ./visualitza_tps.sh [parameters]
  ```
- **Note**: Additional documentation needed for specific parameters and visualization modes

## Note

- Ensure Python and FreeSurfer 7.4.1 are properly installed and configured
- Always activate your virtual environment before running the scripts
- Make sure all required environment variables are set
- For detailed options and parameters for each script, run with `-h` or `--help` flag

## Troubleshooting

If you encounter any issues:
1. Verify FreeSurfer 7.4.1 is properly installed and configured
2. Ensure all environment variables are set correctly
3. Check that your virtual environment is activated
4. Verify all dependencies are installed correctly

For specific script issues, refer to the error messages or contact the repository maintainers.
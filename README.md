# TFG Project

This repository contains the code for the TFG project, focusing on neuroimaging data processing and analysis. Below you'll find instructions on how to set up the development environment and run the code.

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
├── create_synthetic_data.py
├── from_miriad_to_bids.py
├── run_samseg_long.py
├── synthseg_generate_participants_file.py
├── visualitza_tps.sh
├── divide_rawdata.py
├── requirements.txt
└── README.md
```

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
- Nibabel (5.3.2)
- SciPy (1.15.3)

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
- **Purpose**: Generates and processes participant information files by combining volumetric brain data from SynthSeg with clinical metadata in BIDS format
- **Key Features**:
  - Loads and combines SynthSeg volumetric outputs
  - Merges volumetric data with clinical metadata
  - Creates baseline datasets for cross-sectional analysis
  - Handles multiple timepoints and run numbers
  - Generates standardized TSV files
- **Required Arguments**:
  - `--rawdata`: Path to BIDS rawdata directory containing participants_time.tsv
  - `--derivatives`: Path to derivatives directory containing SynthSeg outputs
- **Output Files**:
  1. `participants.tsv`: Combined dataset with all timepoints
     - Contains clinical metadata
     - Includes volumetric measurements
     - Preserves temporal information
  2. `participants_baseline.tsv`: Dataset with only baseline measurements
     - Filtered for time = 0
     - One entry per subject (first run)
     - Useful for cross-sectional analysis
- **Example Usage**:
  ```bash
  python synthseg_generate_participants_file.py \
    --rawdata /path/to/bids/rawdata \
    --derivatives /path/to/synthseg/output
  ```
- **Processing Steps**:
  1. Loads all SynthSeg CSV files from derivatives directory
  2. Combines volumetric measurements into a single dataset
  3. Merges with clinical metadata from participants_time.tsv
  4. Creates a separate baseline dataset
  5. Saves both complete and baseline datasets

### 5. Temporal Processing Visualization
Script: `visualitza_tps.sh`
- **Purpose**: Shell script for visualizing temporal processing steps using FreeSurfer's FreeView tool
- **Key Features**:
  - Interactive visualization of longitudinal brain scans
  - Side-by-side comparison of timepoints
  - Overlay segmentation maps
  - Coronal view layout
  - Template comparison
- **Required Arguments**:
  - `-d <subject_directory>`: Directory containing the subject's timepoint data
  - `-t <num_timepoints>`: Number of timepoints to visualize
- **Expected File Structure**:
  ```
  subject_directory/
  ├── sub-001_template.mgz           # Template MRI
  ├── sub-001_template_seg.mgz       # Template segmentation
  ├── sub-001_tp001.mgz             # Timepoint 1 MRI
  ├── sub-001_tp001_seg.mgz         # Timepoint 1 segmentation
  ├── sub-001_tp002.mgz             # Timepoint 2 MRI
  ├── sub-001_tp002_seg.mgz         # Timepoint 2 segmentation
  └── ...
  ```
- **Visualization Features**:
  - Two-panel layout (coronal view)
  - Template image display
  - Segmentation overlay with 50% opacity
  - Color-coded segmentation maps
  - Interactive navigation
- **Example Usage**:
  ```bash
  # Basic usage
  ./visualitza_tps.sh -d /path/to/subject -t 15

  # Display help
  ./visualitza_tps.sh -h
  ```
- **Requirements**:
  - FreeSurfer installation
  - FreeView tool available in PATH
  - X11 display server (for GUI)
- **Controls in FreeView**:
  - Mouse wheel: Zoom in/out
  - Left click + drag: Pan
  - Right click + drag: Adjust contrast
  - Middle click + drag: Navigate through slices

### 6. Data Division for Testing
Script: `divide_rawdata.py`
- **Purpose**: Divides raw data into training and test sets by copying specified test subjects to a separate directory
- **Key Features**:
  - Copies subject data while preserving directory structure
  - Processes and filters TSV files to include only test subjects
  - Maintains data organization and metadata
  - Verbose output option for debugging
- **Required Arguments**:
  - `-t, --test-subjects-file`: Path to file containing test subject IDs
  - `-s, --source-dir`: Source directory containing the raw data
  - `-o, --target-dir`: Target directory where test data will be copied
- **Optional Arguments**:
  - `-v, --verbose`: Enable verbose output for detailed processing information
- **Example Usage**:
  ```bash
  python divide_rawdata.py \
    --test-subjects-file /path/to/test_subjects.txt \
    --source-dir /path/to/rawdata \
    --target-dir /path/to/rawdata_test \
    --verbose
  ```
- **Input File Format**:
  - `test_subjects.txt`: Text file with one subject ID per line
- **Processing Steps**:
  1. Reads list of test subjects from input file
  2. Creates target directory structure
  3. Copies subject directories for test subjects
  4. Processes and filters TSV files to include only test subjects
  5. Maintains all metadata and file organization

### 7. U-Net Segmentation Model
Directory: `simple_u_net/`
- **Purpose**: Implements a U-Net architecture for brain MRI segmentation
- **Components**:
  - `Model.py`: U-Net model architecture implementation
  - `DataLoader.py`: Data loading and preprocessing utilities
  - `train.py`: Training script with configuration options
  - `test.py`: Model evaluation script
  - `predict.py`: Inference script for new images
  - `best_model.h5`: Pre-trained model weights
- **Key Features**:
  - Custom U-Net implementation for 3D brain MRI
  - Efficient data loading and preprocessing
  - Training with various loss functions
  - Model evaluation metrics
  - Easy-to-use prediction interface
- **Example Usage**:
  ```bash
  # Training
  python simple_u_net/train.py \
    --data_dir /path/to/training/data \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 0.001

  # Testing
  python simple_u_net/test.py \
    --model_path simple_u_net/best_model.h5 \
    --test_data /path/to/test/data

  # Prediction
  python simple_u_net/predict.py \
    --input_image /path/to/input.nii \
    --output_mask /path/to/output.nii \
    --model_path simple_u_net/best_model.h5
  ```

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
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
- Purpose: Generates synthetic neuroimaging data for testing and validation
- Usage:
  ```bash
  python create_synthetic_data.py [options]
  ```

### 2. MIRIAD to BIDS Conversion
Script: `from_miriad_to_bids.py`
- Purpose: Converts neuroimaging data from MIRIAD format to BIDS (Brain Imaging Data Structure) format
- Usage:
  ```bash
  python from_miriad_to_bids.py [input_dir] [output_dir]
  ```

### 3. Longitudinal Segmentation
Script: `run_samseg_long.py`
- Purpose: Implements longitudinal segmentation using SAMSEG algorithm
- Requirements: FreeSurfer 7.4.1
- Usage:
  ```bash
  python run_samseg_long.py [options]
  ```

### 4. Participant File Generation
Script: `synthseg_generate_participants_file.py`
- Purpose: Generates participant information files for SynthSeg processing
- Usage:
  ```bash
  python synthseg_generate_participants_file.py [options]
  ```

### 5. Temporal Processing Visualization
Script: `visualitza_tps.sh`
- Purpose: Shell script for visualizing temporal processing steps
- Usage:
  ```bash
  ./visualitza_tps.sh [parameters]
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
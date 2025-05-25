# Medical Image Processing and Analysis Tools

This repository contains a collection of Python scripts and tools for medical image processing, focusing on neuroimaging data manipulation, synthetic data generation, and segmentation analysis.

## Scripts Overview

### 1. Synthetic Data Generation
- `create_synthetic_data.py`: Generates synthetic neuroimaging data for testing and validation purposes.

### 2. Data Format Conversion
- `from_miriad_to_bids.py`: Converts neuroimaging data from MIRIAD format to BIDS (Brain Imaging Data Structure) format.

### 3. Segmentation Tools
- `run_samseg_long.py`: Implements longitudinal segmentation using SAMSEG algorithm.
- `synthseg_generate_participants_file.py`: Generates participant information files for SynthSeg processing.

### 4. Visualization
- `visualitza_tps.sh`: Shell script for visualizing temporal processing steps.

## Requirements

- Python 3.x
- Neuroimaging processing libraries (specific requirements for each script)
- Shell environment for visualization scripts

## Usage

### Synthetic Data Generation
```bash
python create_synthetic_data.py [options]
```

### BIDS Conversion
```bash
python from_miriad_to_bids.py [input_dir] [output_dir]
```

### Running Segmentation
```bash
python run_samseg_long.py [options]
```

### Generating Participant Files
```bash
python synthseg_generate_participants_file.py [options]
```

### Visualization
```bash
./visualitza_tps.sh [parameters]
```

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
```

2. Install required dependencies (requirements.txt will be provided separately)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Contact

[Your contact information]
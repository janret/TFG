import os
import shutil
import pandas as pd
import argparse
from bids import BIDSLayout
import json

# Configure command line arguments
parser = argparse.ArgumentParser(description='Process MIRIAD dataset into BIDS format')
parser.add_argument('--input_dir', required=True, help='Source directory of MIRIAD dataset')
parser.add_argument('--output_dir', required=True, help='Root directory for output (rawdata and derivatives)')
parser.add_argument('--csv_file', required=True, help='Path to metadata CSV file')
args = parser.parse_args()

# Define dynamic paths
rawdata_dir = os.path.join(args.output_dir, 'rawdata')
derivatives_dir = os.path.join(args.output_dir, 'derivatives/synthseg')
aux_dir = os.path.join(args.output_dir, 'aux')

# -------------------------------------------
# Step 1: Copy .nii files to auxiliary folder
# -------------------------------------------
os.makedirs(aux_dir, exist_ok=True)

# Copy .nii files from input_dir to aux_dir
for root, dirs, files in os.walk(args.input_dir):
    for file in files:
        if file.endswith('.nii'):
            src = os.path.join(root, file)
            dst = os.path.join(aux_dir, file)
            shutil.copy(src, dst)
            print(f'Copied: {file}')

# -------------------------------------------
# Step 2: Organize into BIDS structure
# -------------------------------------------
os.makedirs(rawdata_dir, exist_ok=True)
group_df = pd.DataFrame(columns=['Label', 'Group'])

for file in os.listdir(aux_dir):
    if file.endswith('.nii'):
        parts = file.split('_')
        subject = parts[1]
        group = parts[2]
        session = parts[4]
        run = parts[6].split('.')[0]

        # Create folder structure
        subject_path = os.path.join(rawdata_dir, f'sub-{subject}')
        session_path = os.path.join(subject_path, f'ses-{session}', 'anat')
        os.makedirs(session_path, exist_ok=True)

        # New filename
        new_name = f'sub-{subject}_ses-{session}_run-{run}_T1w.nii'
        dest_path = os.path.join(session_path, new_name)
        
        # Move file
        shutil.copy(os.path.join(aux_dir, file), dest_path)
        
        # Update dataframe
        label = f'sub-{subject}_ses-{session}_run-{run}'
        group_df = pd.concat([group_df, pd.DataFrame({'Label': [label], 'Group': [group]})], ignore_index=True)

# -------------------------------------------
# Step 3: Process CSV metadata
# -------------------------------------------
df = pd.read_csv(args.csv_file, sep=',')
df['Label'] = df['Label'].apply(lambda x: f"sub-{x.split('_')[1]}_ses-{x.split('_')[2].zfill(2)}_run-{x.split('_')[4]}")
df['Subject'] = df['Subject'].apply(lambda x: f"sub-{x.split('_')[1]}")
df = df.drop(columns=['Project', 'Date', 'Type', 'Scanner', 'Scans']).sort_values('Label')

# Merge with group data
merged_df = pd.merge(df, group_df, on='Label').sort_values('Label')
merged_df.to_csv(os.path.join(rawdata_dir, 'participants.tsv'), sep='\t', index=False)

# -------------------------------------------
# Step 4: Create longitudinal time files
# -------------------------------------------
for subject, data in merged_df.groupby('Subject'):
    subject_id = subject.split('-')[1]
    subject_path = os.path.join(rawdata_dir, f'sub-{subject_id}')
    
    # Calculate time from baseline
    data = data.copy()
    data['time(years) from baseline'] = data['Age'] - data['Age'].iloc[0]
    data['time(days) from baseline'] = data['time(years) from baseline'] * 365.25
    
    # Save time.tsv
    output_path = os.path.join(subject_path, 'time.tsv')
    data[['Label', 'time(days) from baseline', 'time(years) from baseline']].to_csv(output_path, sep='\t', index=False)

# -------------------------------------------
# Step 5: Generate participants_time.tsv
# -------------------------------------------
participants_path = os.path.join(rawdata_dir, 'participants.tsv')
df = pd.read_csv(participants_path, delimiter='\t')

time_days_list = []
time_years_list = []

subjects = [s for s in os.listdir(rawdata_dir) if s.startswith('sub-')]
subjects.sort()

for subject in subjects:
    time_file = os.path.join(rawdata_dir, subject, 'time.tsv')
    
    if os.path.exists(time_file):
        time_df = pd.read_csv(time_file, delimiter='\t')
        
        merged_df = df[df['Subject'] == subject].merge(time_df, on='Label', how='left')
        
        time_days_list.extend(merged_df['time(days) from baseline'].tolist())
        time_years_list.extend(merged_df['time(years) from baseline'].tolist())

df['time(days) from baseline'] = time_days_list
df['time(years) from baseline'] = time_years_list

participants_time_path = os.path.join(rawdata_dir, 'participants_time.tsv')
df.to_csv(participants_time_path, sep='\t', index=False)
print(f"\nFitxer temporal creat: {participants_time_path}")
print("Mostrem les primeres files:")
print(df.head())

# -------------------------------------------
# Step 6: Create dataset_description.json
# -------------------------------------------
dataset_description = {
    "Name": "MIRIAD - Multiple Time Point Alzheimer's MR Imaging Dataset",
    "BIDSVersion": "1.8.0",
    "License": "Restricted - MIRIAD Data Use Agreement",
    "Authors": [
        "Malone IB", 
        "Cash D", 
        "Ridgway GR", 
        "Macmanus DG", 
        "Ourselin S", 
        "Fox NC", 
        "Schott JM"
    ],
    "Acknowledgements": "Data used in the preparation of this dataset were obtained from the MIRIAD database (http://miriad.drc.ion.ucl.ac.uk). The MIRIAD dataset is made available through the support of the UK Alzheimer's Society (Grant RF116). The original data collection was funded through an unrestricted educational grant from GlaxoSmithKline (Grant 6GKC).",
    "Funding": [
        "UK Alzheimer's Society (Grant RF116)",
        "GlaxoSmithKline (Grant 6GKC)"
    ],
    "ReferencesAndLinks": [
        "http://miriad.drc.ion.ucl.ac.uk",
        "https://doi.org/10.1016/j.neuroimage.2012.12.044"
    ],
    "DatasetDOI": "10.1016/j.neuroimage.2012.12.044",
    "DataUseAgreement": [
        "Users shall respect restrictions of access to sensitive data and will not attempt to identify individuals.",
        "Redistribution of these data to third parties is not permitted without prior agreement.",
        "Whilst every effort will be made to ensure data quality, users employ these data at their own risk.",
        "Users must acknowledge the dataset when publicly presenting findings or algorithms derived from it.",
        "Publications benefiting from these data must reference: Malone IB, et al. Neuroimage. 2012 Dec 28;70C:33-36. doi:10.1016/j.neuroimage.2012.12.044.",
        "Publications must include the provided acknowledgement statement.",
        "Users must send copies of accepted manuscripts to drc-miriad@ucl.ac.uk."
    ]
}

with open(os.path.join(rawdata_dir, 'dataset_description.json'), 'w') as f:
    json.dump(dataset_description, f, indent=4)

# -------------------------------------------
# Step 7: Prepare derivatives structure
# -------------------------------------------
shutil.copytree(rawdata_dir, derivatives_dir, dirs_exist_ok=True)

# Remove NIfTI files from derivatives
for root, dirs, files in os.walk(derivatives_dir):
    for file in files:
        if file.endswith(('.nii', '.nii.gz')):
            os.remove(os.path.join(root, file))

# -------------------------------------------
# Step 8: Generate path files
# -------------------------------------------
# Generate rawdata paths list
with open(os.path.join(aux_dir, 'rawdata_dirs.txt'), 'w') as f:
    for root, dirs, files in os.walk(rawdata_dir):
        for file in files:
            if file.endswith('.nii'):
                f.write(os.path.join(root, file) + '\n')

# Generate derivatives paths
with open(os.path.join(aux_dir, 'output_seg_dirs.txt'), 'w') as f:
    with open(os.path.join(aux_dir, 'rawdata_dirs.txt'), 'r') as orig:
        for line in orig:
            new_line = line.replace('rawdata', 'derivatives/synthseg').replace('T1w.nii', 'T1w_dseg.nii')
            f.write(new_line)

# -------------------------------------------
# Final cleanup
# -------------------------------------------
shutil.rmtree(aux_dir)
print("Processing completed successfully!")
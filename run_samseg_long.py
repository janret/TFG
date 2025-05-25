import os
import subprocess
import argparse
import pandas as pd

def create_templates(input_dir, output_dir):
    """
    Step 1: Create robust templates using mri_robust_template for each subject.
    Generates registered images and LTA files for longitudinal processing.
    """
    os.makedirs(output_dir, exist_ok=True)

    for subject in os.listdir(input_dir):
        if not subject.startswith("sub-"):
            continue

        subj_dir = os.path.join(input_dir, subject)
        if not os.path.isdir(subj_dir):
            continue

        # Collect all T1w images for the subject
        image_list = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(subj_dir)
            for f in files if f.endswith("T1w.nii")
        ])

        if not image_list:
            print(f"⚠️ {subject}: No T1w images found, skipping")
            continue

        # Create output directories
        subj_output = os.path.join(output_dir, subject)
        lta_dir = os.path.join(subj_output, "lta")
        os.makedirs(lta_dir, exist_ok=True)

        # Generate template and registration paths
        template_path = os.path.join(subj_output, "mean.mgz")
        registered_images = []
        lta_files = []

        for image_path in image_list:
            base_name = os.path.basename(image_path)
            registered_name = base_name.replace("T1w.nii", "T1w_reg.mgz")
            lta_name = base_name.replace("T1w.nii", "to_template.lta")
            
            registered_images.append(os.path.join(subj_output, registered_name))
            lta_files.append(os.path.join(lta_dir, lta_name))

        # Build mri_robust_template command
        cmd = [
            "mri_robust_template",
            "--mov", *image_list,
            "--template", template_path,
            "--satit",
            "--mapmov", *registered_images,
            "--lta", *lta_files,
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"{subject}: Template created successfully")
        except subprocess.CalledProcessError as e:
            print(f"{subject}: Template creation failed - {e}")

def run_samseg(output_dir, num_threads):
    """
    Step 2: Run longitudinal SAMSEG processing using run_samseg_long.
    Processes registered images to generate longitudinal segmentations.
    """
    for subject in os.listdir(output_dir):
        subj_dir = os.path.join(output_dir, subject)
        if not os.path.isdir(subj_dir) or not subject.startswith("sub-"):
            continue

        # Check for existing processing
        if all([os.path.exists(os.path.join(subj_dir, d)) 
              for d in ["base", "latentAtlases"]]):
            print(f"⏩ {subject}: Already processed, skipping SAMSEG")
            continue

        # Collect registered images
        registered_images = sorted([
            os.path.join(subj_dir, f)
            for f in os.listdir(subj_dir)
            if f.endswith("T1w_reg.mgz")
        ])

        if not registered_images:
            print(f"{subject}: No registered images found, skipping SAMSEG")
            continue

        # Build timepoint arguments
        timepoint_args = []
        for img in registered_images:
            timepoint_args.extend(["--timepoint", img])

        # Execute SAMSEG longitudinal
        cmd = [
            "run_samseg_long",
            *timepoint_args,
            "--output", subj_dir,
            "--threads", str(num_threads)
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"{subject}: SAMSEG longitudinal completed")
        except subprocess.CalledProcessError as e:
            print(f"{subject}: SAMSEG failed - {e}")

def invert_segmentations(input_dir, output_dir, inverted_dir):
    """
    Step 3: Invert segmentations to original space using mri_vol2vol.
    Matches segmentation outputs with original BIDS structure.
    """
    os.makedirs(inverted_dir, exist_ok=True)

    for subject in os.listdir(output_dir):
        if not subject.startswith("sub-"):
            continue

        # Path setup
        subj_raw = os.path.join(input_dir, subject)
        subj_output = os.path.join(output_dir, subject)
        lta_dir = os.path.join(subj_output, "lta")

        if not os.path.exists(lta_dir):
            print(f"{subject}: LTA directory missing, skipping inversion")
            continue

        # Collect original images and LTAs
        original_images = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(subj_raw)
            for f in files if f.endswith("T1w.nii")
        ])

        lta_files = sorted([
            os.path.join(lta_dir, f) 
            for f in os.listdir(lta_dir) if f.endswith(".lta")
        ])

        # Get timepoint folders
        tp_folders = sorted(
            [d for d in os.listdir(subj_output) if d.startswith("tp")],
            key=lambda x: int(x[2:])
        )

        # Validation
        if len(original_images) != len(lta_files) or len(original_images) != len(tp_folders):
            print(f"{subject}: Data mismatch, cannot invert")
            continue

        # Process each timepoint
        for t1_path, lta_path, tp_folder in zip(original_images, lta_files, tp_folders):
            seg_path = os.path.join(subj_output, tp_folder, "seg.mgz")
            if not os.path.exists(seg_path):
                print(f"{subject} {tp_folder}: Missing seg.mgz, skipping")
                continue

            # Create BIDS-compatible output path
            filename = os.path.basename(t1_path)
            session = next((p.split("_")[1] for p in filename.split("_") if p.startswith("ses-")), "")
            output_path = os.path.join(
                inverted_dir,
                subject,
                session,
                "anat",
                filename.replace("T1w.nii", "T1w_seg.nii.gz")
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Inversion command
            cmd = [
                "mri_vol2vol",
                "--mov", t1_path,
                "--targ", seg_path,
                "--lta", lta_path,
                "--inv",
                "--nearest",
                "--o", output_path
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"{subject} {tp_folder}: Inversion successful")
            except subprocess.CalledProcessError as e:
                print(f"{subject} {tp_folder}: Inversion failed - {e}")

def process_stats_measures(input_dir, output_dir):
    """
    Step 4: Process .stats files and update participants TSV
    """
    try:
        # Automatic BIDS-based paths
        original_tsv = os.path.join(input_dir, "participants_time.tsv")
        output_tsv = os.path.join(output_dir, "participants.tsv")
        
        df = pd.read_csv(original_tsv, sep='\t')
        
        def get_stats_measures(row):
            measures = {}
            try:
                label_parts = row['Label'].split('_')
                subject, session, run = label_parts[0], label_parts[1], label_parts[2]
                stats_dir = os.path.join(output_dir, subject, session)
                file_prefix = f"{subject}_{session}_{run}"

                # Process statistics
                for stats_type in ['sbtiv', 'samseg']:
                    stats_path = os.path.join(stats_dir, f"{file_prefix}_{stats_type}.stats")
                    if os.path.exists(stats_path):
                        with open(stats_path, "r") as f:
                            for line in f:
                                if line.startswith("# Measure"):
                                    parts = [p.strip() for p in line.split(",")]
                                    measure_name = parts[0].replace("# Measure ", "")
                                    measures[measure_name] = float(parts[1])

            except Exception as e:
                print(f"Error processing {row['Label']}: {str(e)}")
            return pd.Series(measures)

        # Add measures to DataFrame
        df = df.join(df.apply(get_stats_measures, axis=1))
        
        # Preserve original column order
        original_columns = df.columns.tolist()[:7]
        new_columns = sorted([col for col in df.columns if col not in original_columns])
        
        # Save result
        df[original_columns + new_columns].to_csv(output_tsv, sep='\t', index=False, float_format='%.6f')
        print(f"Measures TSV generated: {output_tsv}")
        
        return output_tsv

    except Exception as e:
        print(f"Error processing statistics: {str(e)}")
        return None

def generate_baseline_tsv(measures_tsv, output_dir):
    """
    Step 5: Generate baseline TSV
    """
    try:
        baseline_tsv = os.path.join(output_dir, "participants_baseline.tsv")
        df = pd.read_csv(measures_tsv, sep='\t')
        
        # Filter and clean data
        df_filtered = df[df['time (years) from baseline'] == 0].copy()
        df_filtered['run'] = df_filtered['Label'].str.extract(r'run-(\d+)').astype(int)
        df_filtered = df_filtered.sort_values(by=['Subject', 'run']).drop_duplicates(subset=['Subject'], keep='first')
        df_filtered.drop(columns=['run'], inplace=True)
        
        df_filtered.to_csv(baseline_tsv, sep='\t', index=False)
        print(f"Baseline TSV generated: {baseline_tsv}")

    except Exception as e:
        print(f"Error generating baseline: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Complete SAMSEG Longitudinal Pipeline")
    parser.add_argument("--input", required=True, help="Input BIDS directory (rawdata)")
    parser.add_argument("--output", required=True, help="Main output directory")
    parser.add_argument("--threads", type=int, required=True, help="Number of parallel threads")
    parser.add_argument("--inverted", required=True, help="Directory for inverted segmentations")
    
    args = parser.parse_args()

    # Pipeline execution
    print("\n=== STEP 1/5: Creating templates ===")
    create_templates(args.input, args.output)
    
    print("\n=== STEP 2/5: Running longitudinal SAMSEG ===")
    run_samseg(args.output, args.threads)
    
    print("\n=== STEP 3/5: Inverting segmentations ===")
    invert_segmentations(args.input, args.output, args.inverted)
    
    print("\n=== STEP 4/5: Processing quantitative measures ===")
    measures_tsv = process_stats_measures(args.input, args.output)
    
    if measures_tsv:
        print("\n=== STEP 5/5: Generating baseline dataset ===")
        generate_baseline_tsv(measures_tsv, args.output)

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
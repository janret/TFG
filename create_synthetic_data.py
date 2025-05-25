import os
import subprocess
import argparse
import numpy as np
import nibabel as nib
import torchio as tio
from tqdm import tqdm

# Preprocessing functions
def regroup_segmentation_labels(input_path, output_path):
    """Convert segmentation to 4 classes (GM, WM, CSF, BG)"""
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Label definitions
    gm = [3,10,11,12,13,16,17,18,26,28,42,49,50,51,52,53,54,58,60,8,47]
    wm = [2,7,41,46,77,85]
    csf = [4,5,14,15,24,31,43,44,63,259]
    bg = [0,30,62,80,165,258]
    
    new_data = np.zeros_like(data, dtype=np.uint8)
    new_data[np.isin(data, gm)] = 1
    new_data[np.isin(data, wm)] = 2
    new_data[np.isin(data, csf)] = 3
    new_data[np.isin(data, bg)] = 0
    
    nib.save(nib.Nifti1Image(new_data, img.affine), output_path)

def process_image(input_path, output_path, voxel_size):
    """Apply mri_convert with specified parameters"""
    cmd = [
        'mri_convert',
        '--voxsize', str(voxel_size), str(voxel_size), str(voxel_size),
        '-rt', 'nearest',
        input_path,
        output_path
    ]
    subprocess.run(cmd, check=True)

def convert_segmentation(input_path, output_path, regroup, voxel_size):
    """Full segmentation processing pipeline"""
    if regroup:
        temp_path = '/tmp/temp_seg.mgz'
        regroup_segmentation_labels(input_path, temp_path)
        process_image(temp_path, output_path, voxel_size)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        process_image(input_path, output_path, voxel_size)

def replicate_structure_with_processing(original_dir, output_dir, regroup, voxel_size):
    """Create parallel directory structure with processed files"""
    for root, dirs, files in tqdm(os.walk(original_dir), desc='Processing'):
        current_dir = os.path.basename(root)
        parent_dir = os.path.dirname(root)
        
        if os.path.basename(parent_dir).startswith('sub-') and not current_dir.startswith('tp'):
            dirs[:] = []
            continue
        
        relative_path = os.path.relpath(root, original_dir)
        new_root = os.path.join(output_dir, relative_path)
        os.makedirs(new_root, exist_ok=True)
        
        for file in files:
            if not file.endswith('.mgz'):
                continue
                
            original_file = os.path.join(root, file)
            new_file = os.path.join(new_root, file)
            
            if current_dir.startswith('tp'):
                if file == 'seg.mgz':
                    convert_segmentation(original_file, new_file, regroup, voxel_size)
            else:
                if file.endswith('seg.mgz'):
                    convert_segmentation(original_file, new_file, regroup, voxel_size)
                else:
                    process_image(original_file, new_file, voxel_size)

# Synthetic data generation functions
def apply_transformations(synthetic_sub_id, mri_path, seg_path, output_dir, num_transforms=10):
    """Generate synthetic transformations using TorchIO"""
    try:
        mri_img = nib.load(mri_path)
        original_mri = mri_img.get_fdata()
        seg_img = nib.load(seg_path)
        original_seg = seg_img.get_fdata()
    except Exception as e:
        print(f"Error loading {mri_path} or {seg_path}: {e}")
        return

    # Save templates
    mri_template_path = os.path.join(output_dir, f"{synthetic_sub_id}_template.mgz")
    seg_template_path = os.path.join(output_dir, f"{synthetic_sub_id}_template_seg.mgz")
    nib.save(nib.MGHImage(original_mri, mri_img.affine), mri_template_path)
    nib.save(nib.MGHImage(original_seg, seg_img.affine), seg_template_path)

    # Create TorchIO subject
    subject = tio.Subject(
        mri=tio.ScalarImage(tensor=original_mri[np.newaxis].astype(np.float32), affine=mri_img.affine),
        seg=tio.LabelMap(tensor=original_seg[np.newaxis].astype(np.int16), affine=seg_img.affine)
    )

    # Define transformations
    transform = tio.Compose([
        tio.RandomAffine(degrees=(-3, 3), image_interpolation='bspline'),
        tio.RandomElasticDeformation(
            num_control_points=8,
            max_displacement=14,
            locked_borders=2,
            image_interpolation='bspline'
        )
    ])

    # Generate transformations
    for i in range(1, num_transforms + 1):
        transformed = transform(subject)
        mri_trans = transformed['mri'].data[0].numpy()
        seg_trans = transformed['seg'].data[0].numpy().astype(np.int16)

        mri_out_path = os.path.join(output_dir, f"{synthetic_sub_id}_tp{i:03d}.mgz")
        seg_out_path = os.path.join(output_dir, f"{synthetic_sub_id}_tp{i:03d}_seg.mgz")

        nib.save(nib.MGHImage(mri_trans, mri_img.affine), mri_out_path)
        nib.save(nib.MGHImage(seg_trans, seg_img.affine), seg_out_path)

def process_all_data(samseg_dir, output_root, num_transforms=10, split=False):
    """Process all subjects to generate synthetic data"""
    synthetic_id = 1
    original_subs = [sub for sub in os.listdir(samseg_dir) if sub.startswith('sub-')]

    if split:
        np.random.seed(42)
        np.random.shuffle(original_subs)
        n_total = len(original_subs)
        n_train = int(0.8 * n_total)
        n_val = int(0.15 * n_total)
        groups = {
            'train': original_subs[:n_train],
            'val': original_subs[n_train:n_train+n_val],
            'test': original_subs[n_train+n_val:]
        }
        
        # Create splits directory and save subject lists
        splits_dir = os.path.join(output_root, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        for group_name, group_subs in groups.items():
            list_path = os.path.join(splits_dir, f'{group_name}_subjects.txt')
            with open(list_path, 'w') as f:
                f.write('\n'.join(group_subs))
            print(f"Saved {len(group_subs)} {group_name} subjects list to {list_path}")
            
    else:
        groups = {'all': original_subs}

    for group, subs in groups.items():
        if split:
            group_dir = os.path.join(output_root, group)
            os.makedirs(group_dir, exist_ok=True)
        else:
            group_dir = output_root

        for sub in tqdm(subs, desc=f"Processing {group if split else 'all'} subjects"):
            sub_dir = os.path.join(samseg_dir, sub)
            
            mri_files = sorted([f for f in os.listdir(sub_dir) 
                              if f.endswith('.mgz') and f != 'mean.mgz'
                              and os.path.isfile(os.path.join(sub_dir, f))])
            tp_folders = sorted([d for d in os.listdir(sub_dir) 
                              if d.startswith('tp')], 
                             key=lambda x: int(x[2:]))

            if len(mri_files) != len(tp_folders):
                print(f"❌ {sub}: MRIs ({len(mri_files)}) ≠ Timepoints ({len(tp_folders)})")
                continue

            for mri_file, tp_folder in zip(mri_files, tp_folders):
                synthetic_sub_id = f"sub-{synthetic_id:03d}"
                output_dir = os.path.join(group_dir, synthetic_sub_id)
                os.makedirs(output_dir, exist_ok=True)

                mri_path = os.path.join(sub_dir, mri_file)
                seg_path = os.path.join(sub_dir, tp_folder, 'seg.mgz')

                if not os.path.exists(seg_path):
                    print(f"❌ Segmentation not found: {seg_path}")
                    continue

                apply_transformations(synthetic_sub_id, mri_path, seg_path, output_dir, num_transforms)
                synthetic_id += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MRI processing and synthetic data generation pipeline')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with original SAMSEG data')
    parser.add_argument('--preprocessed_dir', type=str, default='/tmp/preprocessed',
                        help='Output directory for preprocessed data')
    parser.add_argument('--synthetic_dir', type=str, required=True,
                        help='Output directory for synthetic data')
    parser.add_argument('--voxel_size', type=int, choices=[1,2], default=1,
                        help='Voxel size for resampling (1 or 2 mm)')
    parser.add_argument('--num_transforms', type=int, default=10,
                        help='Number of synthetic transformations per image')
    parser.add_argument('--regroup', action='store_true',
                        help='Enable label regrouping into 4 classes')
    parser.add_argument('--split', action='store_true',
                        help='Split subjects into train/val/test groups')
    args = parser.parse_args()

    # Step 1: Preprocessing
    print("Starting preprocessing...")
    replicate_structure_with_processing(
        args.input_dir,
        args.preprocessed_dir,
        args.regroup,
        args.voxel_size
    )
    
    # Step 2: Synthetic data generation
    print("\nStarting synthetic data generation...")
    process_all_data(
        args.preprocessed_dir,
        args.synthetic_dir,
        args.num_transforms,
        args.split
    )
    
    print(f"\nProcessing complete! Preprocessed data at: {args.preprocessed_dir}")
    print(f"Synthetic data generated at: {args.synthetic_dir}")
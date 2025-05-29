import os
import subprocess
import tempfile
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom
from Utils import load_mgz_file, preprocess_scan, preprocess_segmentation

class LongitudinalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_dir, input_shape=(120, 120, 94), batch_size=1, n_classes=4, shuffle=True):
        """
        Initialize the data generator.
        
        Args:
            base_dir: Directory containing subject folders
            input_shape: Shape of input images (H, W, D)
            batch_size: Batch size for training
            n_classes: Number of segmentation classes
            shuffle: Whether to shuffle the data between epochs
        """
        self.base_dir = base_dir
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        
        # Get all valid subject-timepoint pairs
        self.subject_timepoints = self._get_subject_timepoints()
        self.indexes = np.arange(len(self.subject_timepoints))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _get_subject_timepoints(self):
        """Get all valid subject-timepoint pairs with their corresponding files."""
        pairs = []
        subjects = [d for d in os.listdir(self.base_dir) if d.startswith('sub-')]
        
        for subject in subjects:
            subject_dir = os.path.join(self.base_dir, subject)
            if not os.path.isdir(subject_dir):
                continue
                
            # Get template files
            template_path = os.path.join(subject_dir, f"{subject}_template.mgz")
            template_seg_path = os.path.join(subject_dir, f"{subject}_template_seg.mgz")
            
            if not (os.path.exists(template_path) and os.path.exists(template_seg_path)):
                continue
            
            # Get all timepoints
            timepoints = [f for f in os.listdir(subject_dir) 
                         if f.endswith('.mgz') and 'tp' in f and not f.endswith('_seg.mgz')]
            
            for tp_file in timepoints:
                tp_base = tp_file[:-4]  # Remove .mgz
                mri_path = os.path.join(subject_dir, tp_file)
                seg_path = os.path.join(subject_dir, f"{tp_base}_seg.mgz")
                
                if os.path.exists(seg_path):
                    pairs.append({
                        'subject': subject,
                        'timepoint': tp_base,
                        'mri_path': mri_path,
                        'seg_path': seg_path,
                        'template_path': template_path,
                        'template_seg_path': template_seg_path
                    })
        
        return pairs

    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.subject_timepoints) / self.batch_size))

    def __load_mgz(self, path, dtype=np.float32):
        """Load and process MGZ file."""
        img = nib.load(path)
        data = img.get_fdata().astype(dtype)
        return np.nan_to_num(data)

    def __normalize(self, volume):
        """Normalize intensity values to [0, 1]."""
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            return (volume - v_min) / (v_max - v_min)
        return volume

    def __to_onehot(self, mask):
        """Convert label mask to one-hot encoding."""
        onehot = np.zeros((*self.input_shape, self.n_classes), dtype=np.float32)
        for class_idx in range(self.n_classes):
            onehot[..., class_idx] = (mask == class_idx).astype(np.float32)
        return onehot

    def __getitem__(self, idx):
        """Get batch of data."""
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.subject_timepoints[i] for i in batch_indexes]
        
        # Initialize batch arrays
        batch_mri = np.zeros((len(batch_samples), *self.input_shape, 1), dtype=np.float32)
        batch_template = np.zeros((len(batch_samples), *self.input_shape, self.n_classes), dtype=np.float32)
        batch_y = np.zeros((len(batch_samples), *self.input_shape, self.n_classes), dtype=np.float32)
        
        for i, sample in enumerate(batch_samples):
            # Load and normalize MRI
            mri_data = self.__load_mgz(sample['mri_path'])
            mri_data = self.__normalize(mri_data)
            batch_mri[i, ..., 0] = mri_data
            
            # Load and process template
            template_data = self.__load_mgz(sample['template_path'])
            template_data = self.__normalize(template_data)
            template_seg = self.__load_mgz(sample['template_seg_path'], dtype=np.int32)
            
            # Create template input (intensity + segmentation)
            batch_template[i, ..., 0] = template_data
            for c in range(1, self.n_classes):
                batch_template[i, ..., c] = (template_seg == c).astype(np.float32)
            
            # Load and process target segmentation
            seg_data = self.__load_mgz(sample['seg_path'], dtype=np.int32)
            batch_y[i] = self.__to_onehot(seg_data)
        
        return (
            {
                'mri_input': batch_mri,
                'template_input': batch_template
            },
            batch_y
        )

    def on_epoch_end(self):
        """Called at the end of every epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

class BIDSDataLoader:
    def __init__(
        self,
        bids_root,
        trained_model_path,
        subjects=None,
        target_shape=(120, 120, 94),
        voxel_size=(2, 2, 2),
        registered_dir=None,
        n_classes=4
    ):
        self.bids_root = bids_root
        self.target_shape = target_shape
        self.voxel_size = voxel_size
        self.registered_dir = registered_dir or os.path.join(bids_root, "derivatives", "registered")
        self.n_classes = n_classes

        # Get subjects from bids_root first
        self.subjects = subjects or self._get_subjects()
        print(f"Found {len(self.subjects)} subjects in BIDS directory")

        os.makedirs(self.registered_dir, exist_ok=True)
        self.simple_unet = tf.keras.models.load_model(trained_model_path, compile=False)
        self.mean_masks = {}
        self._prepare_subjects()

    def _get_subjects(self):
        """Get list of valid subjects from BIDS root directory"""
        return sorted([
            d for d in os.listdir(self.bids_root)
            if d.startswith('sub-') and os.path.isdir(os.path.join(self.bids_root, d))
        ])

    def _run_cmd(self, cmd):
        subprocess.check_call(cmd)

    def _resample_to_target(self, arr, current_shape):
        factors = [t / c for t, c in zip(self.target_shape, current_shape)]
        return zoom(arr, factors, order=1)

    def _process_mean_and_registrations(self, sub):
        """Generate mean template and register all images to mean space using mapmov."""
        sub_reg_dir = os.path.join(self.registered_dir, sub)
        os.makedirs(sub_reg_dir, exist_ok=True)

        # Collect all T1w paths
        anat_paths = []
        sessions = []
        runs = []
        subj_dir = os.path.join(self.bids_root, sub)
        for ses in os.listdir(subj_dir):
            anat_dir = os.path.join(subj_dir, ses, 'anat')
            if os.path.isdir(anat_dir):
                for fname in os.listdir(anat_dir):
                    if '_T1w.nii' in fname:
                        anat_paths.append(os.path.join(anat_dir, fname))
                        sessions.append(ses)
                        runs.append(fname.split('_run-')[1].split('_')[0] if '_run-' in fname else '01')

        if not anat_paths:
            return None

        # Template base: 'mean' to produce mean.mgz
        tpl_base = os.path.join(sub_reg_dir, 'mean')

        # Prepare mapmov and lta outputs
        regs = []
        ltas = []
        for i, (ses, run) in enumerate(zip(sessions, runs)):
            regs.append(os.path.join(sub_reg_dir, f"{sub}_ses-{ses}_run-{run}_reg.mgz"))
            ltas.append(os.path.join(sub_reg_dir, f"mean_{i}.lta"))

        # Run mri_robust_template with mapmov
        cmd = [
            'mri_robust_template',
            '--mov', *anat_paths,
            '--template', tpl_base,
            '--satit',
            '--mapmov', *regs,
            '--lta', *ltas
        ]
        self._run_cmd(cmd)

        return tpl_base + ".mgz"

    def _prepare_subjects(self):
        for sub in self.subjects:
            sub_reg_dir = os.path.join(self.registered_dir, sub)
            mean_path = os.path.join(sub_reg_dir, 'mean.mgz')

            # Check if subject exists in registered_dir and has mean template
            if os.path.exists(mean_path):
                print(f"Using existing mean for {sub}")
            else:
                print(f"Generating mean and registrations for {sub}")
                mean_path = self._process_mean_and_registrations(sub)
                if not mean_path:
                    print(f"Warning: Could not process subject {sub}, skipping")
                    continue

            # Convert, normalize and segment template
            with tempfile.NamedTemporaryFile(suffix='.mgz') as tmp:
                self._run_cmd([
                    'mri_convert',
                    '--voxsize', str(self.voxel_size[0]), str(self.voxel_size[1]), str(self.voxel_size[2]),
                    mean_path, tmp.name
                ])
                img = nib.load(tmp.name)
                arr = img.get_fdata().astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min())
                arr = self._resample_to_target(arr, img.shape)

                x = arr[np.newaxis, ..., np.newaxis]
                pred = self.simple_unet.predict(x, verbose=0)[0]
                labels = np.argmax(pred, axis=-1)
                self.mean_masks[sub] = np.eye(self.n_classes)[labels].astype(np.uint8)

    def get_dataset(self, batch_size=1):
        def generator():
            for sub in self.subjects:  # Iterate only over subjects from bids_root
                sub_reg_dir = os.path.join(self.registered_dir, sub)
                if not os.path.exists(sub_reg_dir):
                    print(f"Warning: Subject {sub} not found in registered directory, skipping")
                    continue
                    
                for fname in os.listdir(sub_reg_dir):
                    if not fname.endswith('_reg.mgz'):
                        continue
                    reg_path = os.path.join(sub_reg_dir, fname)
                    with tempfile.NamedTemporaryFile(suffix='.mgz') as tmp:
                        self._run_cmd([
                            'mri_convert',
                            '--voxsize', str(self.voxel_size[0]), str(self.voxel_size[1]), str(self.voxel_size[2]),
                            reg_path, tmp.name
                        ])
                        img = nib.load(tmp.name)
                        arr = img.get_fdata().astype(np.float32)

                    arr = (arr - arr.min()) / (arr.max() - arr.min())
                    arr = self._resample_to_target(arr, img.shape)

                    meta = {
                        'original_path': reg_path,
                        'affine': img.affine,
                    }
                    yield {
                        'mri_input': arr[..., np.newaxis],
                        'template_input': self.mean_masks[sub]
                    }, meta

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    'mri_input': tf.TensorSpec((*self.target_shape, 1), tf.float32),
                    'template_input': tf.TensorSpec((*self.target_shape, self.n_classes), tf.uint8)
                },
                {
                    'original_path': tf.TensorSpec((), tf.string),
                    'affine': tf.TensorSpec((4, 4), tf.float32)
                }
            )
        ).batch(batch_size) 
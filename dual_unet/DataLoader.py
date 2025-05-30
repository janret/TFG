import os
import subprocess
import tempfile
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom

class LongitudinalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_dir, subjects, batch_size=1, input_shape=(120, 120, 94), n_classes=4, shuffle=True):
        self.base_dir = base_dir
        self.subjects = subjects
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.shuffle = shuffle
        
        self.samples = []
        for subject_id in subjects:
            subject_dir = os.path.join(base_dir, subject_id)
            timepoints = [f for f in os.listdir(subject_dir) 
                         if f.endswith(".mgz") and "tp" in f and "seg" not in f]
            for tp_file in timepoints:
                self.samples.append((subject_id, tp_file))
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __load_nifti(self, path, dtype=np.float32):
        """Load and process NIfTI file"""
        img = nib.load(path)
        data = img.get_fdata().astype(dtype)
        return np.nan_to_num(data)

    def __to_onehot(self, mask):
        """Convert 3D mask to one-hot encoding"""
        onehot = np.zeros((*self.input_shape, self.n_classes), dtype=np.float32)
        for class_idx in range(self.n_classes):
            onehot[..., class_idx] = (mask == class_idx).astype(np.float32)
        return onehot

    def __normalize_mri(self, volume):
        """Normalize intensity values to [0, 1] without resizing"""
        v_min, v_max = np.min(volume), np.max(volume)
        if v_max > v_min:
            return (volume - v_min) / (v_max - v_min)
        return np.zeros_like(volume)

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        # Initialize arrays (without resizing)
        X_mri = np.empty((len(batch_samples), *self.input_shape, 1), dtype=np.float32)
        X_template = np.empty((len(batch_samples), *self.input_shape, self.n_classes), dtype=np.float32)
        y = np.empty((len(batch_samples), *self.input_shape, self.n_classes), dtype=np.float32)

        for i, (subject_id, tp_file) in enumerate(batch_samples):
            # Load template segmentation (one-hot encoded)
            template_path = os.path.join(self.base_dir, subject_id, f"{subject_id}_template_seg.mgz")
            template = self.__load_nifti(template_path, dtype=np.int32)
            X_template[i] = self.__to_onehot(template)  # Shape (Height, Width, Depth, Channels)

            # Load and normalize MRI
            mri_path = os.path.join(self.base_dir, subject_id, tp_file)
            mri_volume = self.__load_nifti(mri_path, dtype=np.float32)
            normalized_mri = self.__normalize_mri(mri_volume)[..., np.newaxis]  # Add channel dimension
            X_mri[i] = normalized_mri

            # Load segmentation (one-hot encoded)
            seg_path = mri_path.replace(".mgz", "_seg.mgz")
            segmentation = self.__load_nifti(seg_path, dtype=np.int32)
            y[i] = self.__to_onehot(segmentation)

        return (X_mri, X_template), y
    

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
        """
        Initialize the BIDSDataLoader.
        
        Args:
            bids_root: Root directory containing BIDS dataset
            trained_model_path: Path to trained simple U-Net model for template segmentation
            subjects: Optional list of specific subjects to process (must exist in bids_root)
            target_shape: Target shape for all images (H, W, D)
            voxel_size: Target voxel size in mm
            registered_dir: Optional directory containing precomputed mean templates
            n_classes: Number of segmentation classes
        """
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
        subjects = [
            d for d in os.listdir(self.bids_root)
            if d.startswith('sub-') and os.path.isdir(os.path.join(self.bids_root, d))
        ]
        if not subjects:
            raise ValueError(f"No subjects found in BIDS root directory: {self.bids_root}")
        return sorted(subjects)

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
            print(f"Warning: No T1w images found for subject {sub}")
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
        try:
            self._run_cmd(cmd)
            return tpl_base + ".mgz"
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate mean template for subject {sub}: {str(e)}")
            return None

    def _prepare_subjects(self):
        """Prepare mean templates and segmentations for all subjects."""
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
            try:
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
            except Exception as e:
                print(f"Warning: Failed to process mean template for subject {sub}: {str(e)}")
                continue

    def get_dataset(self, batch_size=1):
        """
        Create a dataset generator for the subjects.
        Only processes subjects that exist in both BIDS root and registered directory.
        """
        def generator():
            for sub in self.subjects:  # Iterate only over subjects from bids_root
                sub_reg_dir = os.path.join(self.registered_dir, sub)
                if not os.path.exists(sub_reg_dir):
                    print(f"Warning: Subject {sub} not found in registered directory, skipping")
                    continue
                
                if sub not in self.mean_masks:
                    print(f"Warning: No mean template mask for subject {sub}, skipping")
                    continue
                    
                for fname in os.listdir(sub_reg_dir):
                    if not fname.endswith('_reg.mgz'):
                        continue
                    reg_path = os.path.join(sub_reg_dir, fname)
                    try:
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
                    except Exception as e:
                        print(f"Warning: Failed to process image {reg_path}: {str(e)}")
                        continue

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
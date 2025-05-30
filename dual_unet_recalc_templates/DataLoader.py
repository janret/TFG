import os
import subprocess
import tempfile
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom

class LongitudinalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_dir, subjects, batch_size=1, input_shape=(120,120,94), 
                 n_classes=4, shuffle=True, use_updated_templates=False):
        self.base_dir = base_dir
        self.subjects = subjects
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.use_updated_templates = use_updated_templates
        
        # Muestras: (subject_id, timepoint_file)
        self.samples = []
        for sub in subjects:
            sub_dir = os.path.join(base_dir, sub)
            self.samples.extend([
                (sub, f) for f in os.listdir(sub_dir) 
                if f.endswith(".mgz") and "tp" in f and "seg" not in f
            ])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __load_volume(self, path):
        img = nib.load(path)
        data = np.nan_to_num(img.get_fdata())
        return data.astype(np.float32) if "seg" not in path else data.astype(np.int32)

    def __getitem__(self, idx):
        batch_samples = self.samples[idx*self.batch_size : (idx+1)*self.batch_size]
        
        X_mri = np.empty((len(batch_samples), *self.input_shape, 1))
        X_template = np.empty((len(batch_samples), *self.input_shape, self.n_classes))
        y = np.empty((len(batch_samples), *self.input_shape, self.n_classes))

        for i, (sub_id, tp_file) in enumerate(batch_samples):
            # Cargar template
            if self.use_updated_templates:
                template_path = os.path.join(self.base_dir, sub_id, f"{sub_id}_template_refined.mgz")
            else:
                template_path = os.path.join(self.base_dir, sub_id, f"{sub_id}_template_seg.mgz")
                
            template = self.__load_volume(template_path)
            assert template.ndim == 3, f"Invalid template shape: {template.shape}"
            X_template[i] = self.__to_onehot(template)

            # Cargar MRI
            mri_path = os.path.join(self.base_dir, sub_id, tp_file)
            mri = self.__load_volume(mri_path)[..., np.newaxis]
            X_mri[i] = (mri - np.min(mri)) / (np.max(mri) - np.min(mri) + 1e-8)

            # Cargar ground truth
            seg_path = mri_path.replace(".mgz", "_seg.mgz")
            y[i] = self.__to_onehot(self.__load_volume(seg_path))

        return (X_mri, X_template), y

    def __to_onehot(self, mask):
        onehot = np.zeros((*self.input_shape, self.n_classes))
        for c in range(self.n_classes):
            onehot[..., c] = (mask == c).astype(np.float32)
        return onehot

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)
    

class BIDSDataLoader:
    def __init__(
        self,
        bids_root,
        trained_model_path,
        subjects=None,
        target_shape=(120, 120, 94),
        voxel_size=(2, 2, 2),
        registered_dir=None,
        n_classes=4,
        use_updated_templates=False

    ):
        self.use_updated_templates = use_updated_templates
        self.bids_root = bids_root
        
        # Get subjects from bids_root first
        self.bids_subjects = self._get_subjects()
        print(f"Found {len(self.bids_subjects)} subjects in BIDS directory")
        
        # If subjects parameter is provided, validate they exist in bids_root
        if subjects:
            invalid_subjects = [s for s in subjects if s not in self.bids_subjects]
            if invalid_subjects:
                raise ValueError(f"The following subjects were not found in BIDS root: {invalid_subjects}")
            self.subjects = sorted(subjects)
        else:
            self.subjects = self.bids_subjects
            
        self.target_shape = target_shape
        self.voxel_size = voxel_size
        self.registered_dir = registered_dir or os.path.join(bids_root, "derivatives", "registered")
        self.n_classes = n_classes

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
            if self.use_updated_templates:
                mean_path = os.path.join(sub_reg_dir, 'mean_refined.mgz')
                if not os.path.exists(mean_path):
                    mean_path = os.path.join(sub_reg_dir, 'mean.mgz')
            else:
                mean_path = os.path.join(sub_reg_dir, 'mean.mgz')

            if os.path.exists(mean_path):
                print(f"Using {'refined' if self.use_updated_templates else 'mean'} template for {sub}")
            else:
                mean_path = self._process_mean_and_registrations(sub)
                if not mean_path:
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
            for sub in self.subjects:  # Only iterate over subjects from bids_root
                if sub not in self.bids_subjects:
                    print(f"Warning: Subject {sub} not found in BIDS directory, skipping")
                    continue
                    
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
                    yield arr[..., np.newaxis], self.mean_masks[sub], meta

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec((*self.target_shape, 1), tf.float32),
                tf.TensorSpec((*self.target_shape, self.n_classes), tf.uint8),
                {
                    'original_path': tf.TensorSpec((), tf.string),
                    'affine': tf.TensorSpec((4, 4), tf.float32),
                }
            )
        ).batch(batch_size)
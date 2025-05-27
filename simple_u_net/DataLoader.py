import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import tempfile
import subprocess
from collections import defaultdict

class LongitudinalDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for longitudinal training data.
    Loads .mgz timepoints and corresponding segmentation masks.
    """
    def __init__(self, base_dir, subjects, batch_size=1, input_shape=(120, 120, 94), n_classes=4, shuffle=True):
        self.base_dir = base_dir
        self.subjects = subjects
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.samples = []
        for sub in subjects:
            sub_dir = os.path.join(base_dir, sub)
            timepoints = [f for f in os.listdir(sub_dir) if f.endswith('.mgz') and 'tp' in f and 'seg' not in f]
            for tp in timepoints:
                self.samples.append((sub, tp))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

    def _load_nifti(self, path, dtype=np.float32):
        img = nib.load(path)
        data = img.get_fdata().astype(dtype)
        return np.nan_to_num(data)

    def _to_onehot(self, mask):
        onehot = np.zeros((*self.input_shape, self.n_classes), dtype=np.float32)
        for c in range(self.n_classes):
            onehot[..., c] = (mask == c).astype(np.float32)
        return onehot

    def _normalize_mri(self, volume):
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            return (volume - vmin) / (vmax - vmin)
        return np.zeros_like(volume)

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.empty((len(batch), *self.input_shape, 1), dtype=np.float32)
        y = np.empty((len(batch), *self.input_shape, self.n_classes), dtype=np.float32)
        for i, (sub, tp_file) in enumerate(batch):
            mri_path = os.path.join(self.base_dir, sub, tp_file)
            mri = self._load_nifti(mri_path)
            X[i] = self._normalize_mri(mri)[..., np.newaxis]
            seg_path = mri_path.replace('.mgz', '_seg.mgz')
            seg = self._load_nifti(seg_path, dtype=np.int32)
            y[i] = self._to_onehot(seg)
        return X, y

class SimplePredictGenerator(tf.keras.utils.Sequence):
    """
    Simple BIDS predictor: loads T1w images, resamples and preprocesses
    without registration or template creation. Saves intermediate steps
    to output_dir/preprocessed for traceability.
    """
    def __init__(self, base_dir, output_dir, batch_size=1, target_shape=(120, 120, 94)):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.preproc_dir = os.path.join(output_dir, 'preprocessed')
        os.makedirs(self.preproc_dir, exist_ok=True)
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.voxel_size = (2.0, 2.0, 2.0)
        self.voxel_volume = np.prod(self.voxel_size)
        self.samples = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith('_T1w.nii') or f.endswith('_T1w.nii.gz'):
                    self.samples.append(os.path.join(root, f))
        if not self.samples:
            raise ValueError(f"No T1w files found in {base_dir}")

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def _resample_image(self, input_path):
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            subprocess.run([
                'mri_convert', input_path, tmp.name,
                '--voxsize', '2', '2', '2'
            ], check=True)
            img = nib.load(tmp.name)
            data = img.get_fdata().astype(np.float32)
            aff = img.affine
            hdr = img.header
        rel_path = os.path.relpath(input_path, self.base_dir)
        out_path = os.path.join(self.preproc_dir, rel_path.replace('_T1w.nii', '_T1w_resampled.nii.gz'))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nib.save(nib.Nifti1Image(data, aff, hdr), out_path)
        return data, aff, hdr

    def _pad_or_crop(self, data):
        pads = []
        for i in range(3):
            diff = self.target_shape[i] - data.shape[i]
            pads.append((max(diff//2, 0), max(diff-diff//2, 0)))
        if any(pad[0] or pad[1] for pad in pads):
            data = np.pad(data, pads, mode='constant')
        return data[tuple(slice(0, self.target_shape[i]) for i in range(3))]

    def _preprocess(self, data):
        arr = np.nan_to_num(data)
        mn, mx = arr.min(), arr.max()
        arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
        arr = self._pad_or_crop(arr)
        return arr[..., np.newaxis]

    def __getitem__(self, idx):
        batch_paths = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, meta = [], []
        for path in batch_paths:
            data, aff, hdr = self._resample_image(path)
            proc = self._preprocess(data)
            rel_path = os.path.relpath(path, self.base_dir)
            preproc_path = os.path.join(self.preproc_dir, rel_path.replace('_T1w.nii', '_T1w_preproc.nii.gz'))
            os.makedirs(os.path.dirname(preproc_path), exist_ok=True)
            nib.save(nib.Nifti1Image(proc[..., 0], aff, hdr), preproc_path)
            X.append(proc)
            meta.append({'original_path': path, 'resampled_affine': aff, 'resampled_header': hdr})
        return np.array(X), meta

class BIDSPredictGenerator(SimplePredictGenerator):
    """
    BIDS predictor that optionally uses pre-registered images or computes a
    robust template and registers images before prediction.
    """
    def __init__(self, base_dir, output_dir, batch_size=1,
                 target_shape=(120, 120, 94), registered_dir=None, use_template=False):
        # ensure inherited preproc_dir exists
        super().__init__(base_dir, output_dir, batch_size, target_shape)
        if registered_dir and use_template:
            raise ValueError('Cannot use both --registered_dir and --use_template')
        self.registered_dir = registered_dir
        self.use_template = use_template
        if registered_dir:
            self.samples = self._load_registered(registered_dir)
        elif use_template:
            reg_dir = os.path.join(output_dir, 'registered')
            os.makedirs(reg_dir, exist_ok=True)
            self.samples = self._process_longitudinal_data()
        # else, samples already set by super()

    def _load_registered(self, reg_dir):
        samples = []
        for root, _, files in os.walk(reg_dir):
            for f in files:
                if f.endswith('_T1w_reg.mgz'):
                    orig = f.replace('_T1w_reg.mgz', '_T1w.nii')
                    samples.append({'original_path': os.path.join(root, orig), 'registered_path': os.path.join(root, f)})
        return samples

    def _process_longitudinal_data(self):
        files_by_subj = defaultdict(list)
        for root, _, files in os.walk(self.base_dir):
            for f in files:
                if f.endswith(('_T1w.nii', '_T1w.nii.gz')):
                    try:
                        subj = next(p for p in root.split(os.sep) if p.startswith('sub-'))
                        files_by_subj[subj].append(os.path.join(root, f))
                    except StopIteration:
                        continue
        samples = []
        for subj, files in files_by_subj.items():
            subj_reg = os.path.join(self.output_dir, 'registered', subj)
            lta_dir = os.path.join(subj_reg, 'lta')
            os.makedirs(lta_dir, exist_ok=True)
            template = os.path.join(subj_reg, 'mean.mgz')
            regs, ltas = [], []
            for path in files:
                bn = os.path.basename(path)
                regs.append(os.path.join(subj_reg, bn.replace('_T1w.nii', '_T1w_reg.mgz')))
                ltas.append(os.path.join(lta_dir, bn.replace('_T1w.nii', '_to_template.lta')))
            self._run_robust_template(files, template, regs, ltas)
            for orig, reg in zip(files, regs):
                samples.append({'original_path': orig, 'registered_path': reg})
        return samples

    def _run_robust_template(self, inputs, template, regs, ltas):
        cmd = [
            'mri_robust_template', '--mov', *inputs,
            '--template', template, '--satit',
            '--mapmov', *regs, '--lta', *ltas
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, meta = [], []
        for s in batch:
            data, aff, hdr = self._resample_image(s['registered_path'])
            proc = self._preprocess(data)
            X.append(proc)
            meta.append({'original_path': s['original_path'], 'resampled_affine': aff, 'resampled_header': hdr})
        return np.array(X), meta

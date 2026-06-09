import os
import glob
import numpy as np
import nibabel as nib
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter


def generate_gmm_labels(img_data, mask_data, n_bg_clusters, n_liver_clusters):
    """Fits GMMs and returns the label map."""
    label_map = np.zeros_like(img_data, dtype=np.int32)

    # Background
    bg_voxels = img_data[~mask_data].reshape(-1, 1)
    if len(bg_voxels) > 0:
        bg_sample = bg_voxels[::5] if len(bg_voxels) > 1000000 else bg_voxels
        gmm_bg = GaussianMixture(n_components=n_bg_clusters, random_state=42, n_init=3)
        bg_labels = gmm_bg.fit(bg_sample)
        # Predict full and map
        label_map[~mask_data] = gmm_bg.predict(bg_voxels) + 1

        # Liver
    liver_voxels = img_data[mask_data].reshape(-1, 1)
    if len(liver_voxels) > 0:
        gmm_liver = GaussianMixture(n_components=n_liver_clusters, random_state=42, n_init=3)
        liver_labels = gmm_liver.fit(liver_voxels)
        label_map[mask_data] = gmm_liver.predict(liver_voxels) + 1 + n_bg_clusters

    return label_map


# --- EXPERIMENTAL SWEEP CONFIGURATION ---
# Format: (Background Components, Liver Components)
cluster_experiments = [
    (2, 1),
    (2, 2),  # baseline / minimal
    (4, 1),  # decent bg, coarse liver
    (4, 2),  # ← recommended sweet spot
    (4, 3),  # fine-grained liver
]

DATA_ROOT = 'new_dataset/CT'

for patient_dir in glob.glob(os.path.join(DATA_ROOT, '*')):
    img_file = os.path.join(patient_dir, 'image.nii.gz')
    mask_file = os.path.join(patient_dir, 'truth.nii.gz')

    if os.path.exists(img_file) and os.path.exists(mask_file):
        print(f"Processing patient: {os.path.basename(patient_dir)}")
        img_nii = nib.load(img_file)
        mask_nii = nib.load(mask_file)

        img_data = gaussian_filter(img_nii.get_fdata(), sigma=0.5)
        mask_data = mask_nii.get_fdata().astype(bool)

        for n_bg, n_liver in cluster_experiments:
            total_c = n_bg + n_liver
            out_file = os.path.join(patient_dir, f'labels_gmm_{total_c}c_2.nii.gz')

            # Skip if already generated to save time
            if not os.path.exists(out_file):
                label_map = generate_gmm_labels(img_data, mask_data, n_bg, n_liver)
                label_nii = nib.Nifti1Image(label_map, img_nii.affine, img_nii.header)
                nib.save(label_nii, out_file)
                print(f"  -> Saved {total_c}c")
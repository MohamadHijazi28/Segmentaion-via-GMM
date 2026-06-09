# chaos_utils.py
import os
import glob
import random
import numpy as np
import pydicom
import cv2
from PIL import Image
import nibabel as nib

from utils.utils import (
    NiftiReadWrite, InferenceUtils, resize_image,
    ParamsReadWrite, Metrics, Postprocessing
)


# ── 1. DICOM + PNG mask reader ───────────────────────────────────────────────

def load_dicom_volume(dicom_dir):
    """
    Read a folder of DICOM slices and return a 3D numpy array (H, W, D)
    sorted by InstanceNumber (slice position).
    """
    dcm_files = sorted(
        glob.glob(os.path.join(dicom_dir, '*.dcm')),
        key=lambda f: pydicom.dcmread(f).InstanceNumber
    )
    slices = [pydicom.dcmread(f).pixel_array.astype(np.float32) for f in dcm_files]
    return np.stack(slices, axis=-1)   # (H, W, D)


def load_ground_truth_volume(ground_dir):
    """
    Read PNG masks from Ground/ folder.
    CHAOS CT masks: pixel value 255 = liver, 0 = background.
    Returns integer volume (H, W, D) with values 0 or 1.
    """
    png_files = sorted(glob.glob(os.path.join(ground_dir, '*.png')))
    masks = []
    for f in png_files:
        mask = np.array(Image.open(f).convert('L'))   # grayscale
        mask = (mask > 127).astype(np.int16)          # 255 → 1, 0 → 0
        masks.append(mask)
    return np.stack(masks, axis=-1)   # (H, W, D)


# ── 2. Data split ────────────────────────────────────────────────────────────

def split_chaos_ct(ct_train_root, num_train=15, num_valid=5, seed=42):
    """
    Split the 20 CHAOS CT training cases into train / val / test.
    test is empty — CT test set has no public labels.

    ct_train_root: path to  CT/Train_Sets/CT/
    """
    case_ids = sorted([
        d for d in os.listdir(ct_train_root)
        if os.path.isdir(os.path.join(ct_train_root, d))
    ])
    random.seed(seed)
    random.shuffle(case_ids)

    train_lst = case_ids[:num_train]
    valid_lst = case_ids[num_train:num_train + num_valid]
    test_lst  = case_ids[num_train + num_valid:]

    print(f"Train: {len(train_lst)}, Valid: {len(valid_lst)}, Test: {len(test_lst)}")
    return train_lst, valid_lst, test_lst


# ── 3. Main data loader (mirrors your load_data) ─────────────────────────────

def load_chaos_ct_data(ct_train_root, train_lst, valid_lst, test_lst,
                       target_size=(256, 256)):
    """
    Load CHAOS CT data. Processes each 3D volume slice-by-slice,
    exactly as your existing pipeline does.

    Returns:
        train_label_maps : list of 2D label arrays  (H, W, 1)  int16
        train_images_meta: [train_images, [], []]   (affines/headers are empty
                           since DICOM metadata isn't used downstream)
        valid_data       : (valid_images, valid_label_maps)
    """
    train_images, train_label_maps = [], []
    valid_images, valid_label_maps = [], []

    all_ids = train_lst + valid_lst + test_lst

    for case_id in all_ids:
        case_dir   = os.path.join(ct_train_root, str(case_id))
        dicom_dir  = os.path.join(case_dir, 'DICOM_anon')
        ground_dir = os.path.join(case_dir, 'Ground')

        # load full 3D volume  (H, W, D)
        volume = load_dicom_volume(dicom_dir)

        # iterate over slices
        for s in range(volume.shape[-1]):
            img_slice  = volume[..., s]                        # (H, W)
            img_resized = cv2.resize(img_slice, target_size,
                                     interpolation=cv2.INTER_LINEAR)
            img_resized = img_resized[..., np.newaxis]         # (H, W, 1)

            if case_id in train_lst or case_id in valid_lst:
                mask_vol   = load_ground_truth_volume(ground_dir)
                mask_slice = mask_vol[..., s]                  # (H, W)
                mask_resized = cv2.resize(
                    mask_slice.astype(np.float32), target_size,
                    interpolation=cv2.INTER_NEAREST             # ← nearest for labels!
                ).astype(np.int16)
                mask_resized = mask_resized[..., np.newaxis]   # (H, W, 1)

                if case_id in train_lst:
                    train_images.append(img_resized)
                    train_label_maps.append(mask_resized)
                else:
                    valid_images.append(img_resized)
                    valid_label_maps.append(mask_resized)

    print(f"Train slices: {len(train_images)}, Valid slices: {len(valid_images)}")
    valid_data = (valid_images, valid_label_maps)
    train_images_meta = [train_images, [], []]   # no affines/headers needed
    return train_label_maps, train_images_meta, valid_data


# ── 4. Label mapping for CHAOS CT ────────────────────────────────────────────

def get_chaos_ct_labels_out():
    """
    CHAOS CT: binary task — background (0) → 0, liver (1) → 1.
    Returns labels_in and labels_out in the same format your pipeline expects.
    """
    labels_in  = np.array([0, 1])
    labels_out = {0: 0, 1: 1}
    return labels_in, labels_out
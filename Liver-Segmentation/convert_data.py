# # # # import os
# # # # import pydicom
# # # # import nibabel as nib
# # # # import numpy as np
# # # # from PIL import Image
# # # #
# # # #
# # # # def process_chaos_dataset(base_dataset_path, output_base_path, modality="CT"):
# # # #     """
# # # #     Loops through the CHAOS dataset folder structure and converts 2D DICOM/PNGs to 3D NIfTIs.
# # # #     """
# # # #     # Define the path to the specific modality (e.g., CHAOS/Train_Sets/CT)
# # # #     modality_dir = os.path.join(base_dataset_path, "Train_Sets", modality)
# # # #
# # # #     if not os.path.exists(modality_dir):
# # # #         print(f"Error: Could not find directory {modality_dir}")
# # # #         return
# # # #
# # # #     # Get a list of all case folders (1, 2, 3...) and sort them numerically
# # # #     case_folders = [f for f in os.listdir(modality_dir) if os.path.isdir(os.path.join(modality_dir, f))]
# # # #     case_folders.sort(key=lambda x: int(x) if x.isdigit() else x)
# # # #
# # # #     for case_id in case_folders:
# # # #         print(f"\n--- Processing {modality} Case: {case_id} ---")
# # # #
# # # #         # Setup input paths
# # # #         case_path = os.path.join(modality_dir, case_id)
# # # #         dicom_dir = os.path.join(case_path, "DICOM_anon")
# # # #         ground_dir = os.path.join(case_path, "Ground")
# # # #
# # # #         # Setup output paths
# # # #         output_case_dir = os.path.join(output_base_path, modality, case_id)
# # # #         os.makedirs(output_case_dir, exist_ok=True)
# # # #
# # # #         if not os.path.exists(dicom_dir) or not os.path.exists(ground_dir):
# # # #             print(f"Skipping Case {case_id}: Missing DICOM_anon or Ground folder.")
# # # #             continue
# # # #
# # # #         # --- 1. Process DICOMs ---
# # # #         # Read all files that don't start with a dot (ignoring hidden OS files)
# # # #         dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if not f.startswith('.')]
# # # #         slices = []
# # # #         for f in dicom_files:
# # # #             try:
# # # #                 ds = pydicom.dcmread(f)
# # # #                 slices.append(ds)
# # # #             except Exception:
# # # #                 pass  # Skip files that aren't valid DICOMs
# # # #
# # # #         # CRITICAL: Sort by their spatial position in the body
# # # #         slices.sort(key=lambda x: int(x.InstanceNumber))
# # # #
# # # #         # Stack into 3D array (X, Y, Z)
# # # #         image_3d = np.stack([s.pixel_array for s in slices], axis=-1)
# # # #
# # # #         # --- 2. Process PNG Masks ---
# # # #         # Find all PNG files and sort them alphabetically (CHAOS names them sequentially)
# # # #         png_files = sorted([f for f in os.listdir(ground_dir) if f.endswith('.png')])
# # # #
# # # #         if len(slices) != len(png_files):
# # # #             print(f"Warning for Case {case_id}: Mismatch! {len(slices)} DICOMs vs {len(png_files)} Masks.")
# # # #
# # # #         mask_slices = []
# # # #         for png in png_files:
# # # #             png_path = os.path.join(ground_dir, png)
# # # #             # Load, convert to grayscale, and append
# # # #             mask_2d = np.array(Image.open(png_path).convert('L'))
# # # #             mask_slices.append(mask_2d)
# # # #
# # # #         mask_3d = np.stack(mask_slices, axis=-1)
# # # #
# # # #         # --- 3. Save as NIfTI ---
# # # #         affine = np.eye(4)  # Dummy affine matrix is fine for basic pixel clustering
# # # #
# # # #         image_nii = nib.Nifti1Image(image_3d.astype(np.float32), affine)
# # # #         mask_nii = nib.Nifti1Image(mask_3d.astype(np.uint8), affine)
# # # #
# # # #         nib.save(image_nii, os.path.join(output_case_dir, "image.nii.gz"))
# # # #         nib.save(mask_nii, os.path.join(output_case_dir, "mask.nii.gz"))
# # # #
# # # #         print(f"Saved successfully to {output_case_dir}")
# # # #
# # # #
# # # # # --- Execute the Script ---
# # # # # Change these paths if your main CHAOS folder is located elsewhere
# # # # DATASET_FOLDER = "CHAOS"
# # # # OUTPUT_FOLDER = "DataSet"
# # # #
# # # # print("Starting CT Data Conversion...")
# # # # process_chaos_dataset(DATASET_FOLDER, OUTPUT_FOLDER, modality="CT")
# # #
# # # import os
# # # import pydicom
# # # import nibabel as nib
# # # import numpy as np
# # # from PIL import Image
# # #
# # #
# # # def process_chaos_mri_dataset(base_dataset_path, output_base_path):
# # #     """
# # #     Loops through the CHAOS MRI dataset and converts 2D DICOM/PNGs to 3D NIfTIs.
# # #     Handles the T1DUAL and T2SPIR sequence subfolders.
# # #     """
# # #     modality = "MR"
# # #     modality_dir = os.path.join(base_dataset_path, "Train_Sets", modality)
# # #
# # #     if not os.path.exists(modality_dir):
# # #         print(f"Error: Could not find directory {modality_dir}")
# # #         return
# # #
# # #     # Get a list of all case folders (1, 2, 3, 5...)
# # #     case_folders = [f for f in os.listdir(modality_dir) if os.path.isdir(os.path.join(modality_dir, f))]
# # #     case_folders.sort(key=lambda x: int(x) if x.isdigit() else x)
# # #
# # #     # The two sequences provided in the CHAOS MRI dataset
# # #     sequences = ["T1DUAL", "T2SPIR"]
# # #
# # #     for case_id in case_folders:
# # #         print(f"\n--- Processing MR Case: {case_id} ---")
# # #         case_path = os.path.join(modality_dir, case_id)
# # #
# # #         for seq in sequences:
# # #             seq_path = os.path.join(case_path, seq)
# # #             if not os.path.exists(seq_path):
# # #                 continue  # Skip if sequence doesn't exist for some reason
# # #
# # #             print(f"  -> Extracting {seq} sequence...")
# # #
# # #             # 1. Setup Input Paths (Handling the T1 vs T2 difference)
# # #             if seq == "T1DUAL":
# # #                 # T1DUAL splits DICOMs into InPhase and OutPhase. We will use InPhase.
# # #                 dicom_dir = os.path.join(seq_path, "DICOM_anon", "InPhase")
# # #             else:
# # #                 # T2SPIR has them right inside DICOM_anon
# # #                 dicom_dir = os.path.join(seq_path, "DICOM_anon")
# # #
# # #             ground_dir = os.path.join(seq_path, "Ground")
# # #
# # #             # 2. Setup Output Paths
# # #             # We save them in DataSet/MR/{case_id}/{Sequence}/
# # #             output_seq_dir = os.path.join(output_base_path, modality, case_id, seq)
# # #             os.makedirs(output_seq_dir, exist_ok=True)
# # #
# # #             if not os.path.exists(dicom_dir) or not os.path.exists(ground_dir):
# # #                 print(f"     Skipping {seq}: Missing DICOM or Ground folder.")
# # #                 continue
# # #
# # #             # --- Process DICOMs ---
# # #             dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if not f.startswith('.')]
# # #             slices = []
# # #             for f in dicom_files:
# # #                 try:
# # #                     ds = pydicom.dcmread(f)
# # #                     slices.append(ds)
# # #                 except Exception:
# # #                     pass
# # #
# # #             # Sort by physical position
# # #             slices.sort(key=lambda x: int(x.InstanceNumber))
# # #             image_3d = np.stack([s.pixel_array for s in slices], axis=-1)
# # #
# # #             # --- Process PNG Masks ---
# # #             png_files = sorted([f for f in os.listdir(ground_dir) if f.endswith('.png')])
# # #
# # #             if len(slices) != len(png_files):
# # #                 print(f"     Warning: Mismatch! {len(slices)} DICOMs vs {len(png_files)} Masks.")
# # #
# # #             mask_slices = []
# # #             for png in png_files:
# # #                 png_path = os.path.join(ground_dir, png)
# # #                 mask_2d = np.array(Image.open(png_path).convert('L'))
# # #                 mask_slices.append(mask_2d)
# # #
# # #             mask_3d = np.stack(mask_slices, axis=-1)
# # #
# # #             # --- Save as NIfTI ---
# # #             affine = np.eye(4)
# # #             image_nii = nib.Nifti1Image(image_3d.astype(np.float32), affine)
# # #             mask_nii = nib.Nifti1Image(mask_3d.astype(np.uint8), affine)
# # #
# # #             nib.save(image_nii, os.path.join(output_seq_dir, "image.nii.gz"))
# # #             nib.save(mask_nii, os.path.join(output_seq_dir, "mask.nii.gz"))
# # #
# # #             print(f"     Saved to {output_seq_dir}")
# # #
# # #
# # # # --- Execute the Script ---
# # # DATASET_FOLDER = "CHAOS"
# # # OUTPUT_FOLDER = "DataSet"
# # #
# # # print("Starting MRI Data Conversion...")
# # # process_chaos_mri_dataset(DATASET_FOLDER, OUTPUT_FOLDER)
# # # print("\nAll MRI cases converted successfully!")
# #
# #
# # import os
# # import cv2
# # import sys
# # import shutil
# # import numpy as np
# # import glob
# # import pydicom
# # from PIL import Image, ImageOps
# # import traceback
# #
# #
# # class CTImageMaskDatasetGenerator:
# #     def __init__(self, input_dir="./CT", output_dir="./Liver-master/"):
# #         self.input_dir = input_dir
# #
# #         self.output_dir = output_dir
# #         if os.path.exists(self.output_dir):
# #             shutil.rmtree(self.output_dir)
# #         if not os.path.exists(self.output_dir):
# #             os.makedirs(self.output_dir)
# #
# #         self.output_images_dir = self.output_dir + "/images"
# #         self.output_masks_dir = self.output_dir + "/masks"
# #
# #         os.makedirs(self.output_images_dir)
# #         os.makedirs(self.output_masks_dir)
# #
# #         # normalization parameter
# #         self.normalize = 28
# #
# #         # sharpening parameters
# #         self.brightness = 1.4
# #         self.contrast = 4.0
# #
# #     def generate(self):
# #         subdirs = os.listdir(self.input_dir)
# #         subdir_index = 100
# #         for subdir in subdirs:
# #             subdir_path = os.path.join(self.input_dir, subdir)
# #             subdir_index += 1
# #             self.create_image_files(subdir_index, input_dir=subdir_path, output_dir=self.output_images_dir)
# #             self.create_mask_files(subdir_index, input_dir=subdir_path, output_dir=self.output_masks_dir)
# #
# #     def create_image_files(self, subdir_index, input_dir="./CT/1", output_dir="./Liver-master/images/"):
# #         pattern = input_dir + "/DICOM_anon/*.dcm"
# #         dcm_files = glob.glob(pattern)
# #         dcm_files = sorted(dcm_files)
# #         index = 1000
# #         for dcm_file in dcm_files:
# #             file = pydicom.dcmread(dcm_file)
# #             img = file.pixel_array / self.normalize
# #             img = self.sharpen(img)
# #
# #             image = Image.fromarray(img)
# #             image = image.convert("RGB")
# #             index += 1
# #             filename = str(subdir_index) + "_" + str(index) + ".png"
# #             output_filepath = os.path.join(output_dir, filename)
# #             image.save(output_filepath)
# #             print("--- Saved {}".format(output_filepath))
# #
# #     def create_mask_files(self, subdir_index, input_dir="./CT/1", output_dir="./Liver-master/masks/"):
# #         pattern = input_dir + "/Ground/*.png"
# #         mask_files = glob.glob(pattern)
# #         mask_files = sorted(mask_files)
# #         index = 1000
# #         for mask_file in mask_files:
# #             image = Image.open(mask_file)
# #             image = image.convert("L")
# #             index += 1
# #             filename = str(subdir_index) + "_" + str(index) + ".png"
# #             output_filepath = os.path.join(output_dir, filename)
# #             image.save(output_filepath)
# #             print("--- Saved {}".format(output_filepath))
# #
# #     def sharpen(self, image):
# #         base = np.zeros(image.shape, image.dtype)
# #         image = cv2.addWeighted(image, self.contrast, base, 0, self.brightness)
# #         return image
# #
# #
# # if __name__ == "__main__":
# #     try:
# #         input_dir = "CHAOS/Train_Sets/CT"
# #         output_dir = "CT-Liver-master"
# #
# #         generator = CTImageMaskDatasetGenerator(input_dir=input_dir, output_dir=output_dir)
# #         generator.generate()
# #
# #     except:
# #         traceback.print_exc()
#
#
#
#
# # convert_chaos_ct.py
# import os
# import pydicom
# import nibabel as nib
# import numpy as np
# from PIL import Image
#
# def process_chaos_ct(base_dataset_path, output_base_path):
#     modality_dir = os.path.join(base_dataset_path, "Train_Sets", "CT")
#     if not os.path.exists(modality_dir):
#         print("CT folder not found")
#         return
#     case_folders = [f for f in os.listdir(modality_dir) if os.path.isdir(os.path.join(modality_dir, f))]
#     case_folders.sort(key=lambda x: int(x) if x.isdigit() else x)
#
#     for case_id in case_folders:
#         case_path = os.path.join(modality_dir, case_id)
#         dicom_dir = os.path.join(case_path, "DICOM_anon")
#         ground_dir = os.path.join(case_path, "Ground")
#
#         output_case_dir = os.path.join(output_base_path, "CT", case_id)
#         os.makedirs(output_case_dir, exist_ok=True)
#
#         # Process DICOMs
#         dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if not f.startswith('.')]
#         slices = []
#         for f in dicom_files:
#             try:
#                 ds = pydicom.dcmread(f)
#                 slices.append(ds)
#             except:
#                 pass
#         slices.sort(key=lambda x: int(x.InstanceNumber))
#         image_3d = np.stack([s.pixel_array for s in slices], axis=-1).astype(np.float32)
#
#         # Process PNG masks (binary liver)
#         png_files = sorted([f for f in os.listdir(ground_dir) if f.endswith('.png')])
#         mask_slices = []
#         for png in png_files:
#             mask_2d = np.array(Image.open(os.path.join(ground_dir, png)).convert('L'))
#             mask_slices.append(mask_2d)
#         mask_3d = np.stack(mask_slices, axis=-1).astype(np.uint8)
#
#         affine = np.eye(4)  # dummy affine
#         nib.save(nib.Nifti1Image(image_3d, affine), os.path.join(output_case_dir, "image.nii.gz"))
#         nib.save(nib.Nifti1Image(mask_3d, affine), os.path.join(output_case_dir, "mask.nii.gz"))
#         print(f"Saved CT case {case_id}")
#
# if __name__ == "__main__":
#     process_chaos_ct("CHAOS", "DataSet")

import os
import glob
import pydicom
import numpy as np
import nibabel as nib
from PIL import Image


def load_dicom_series(dicom_dir):
    """
    Loads a DICOM series from a directory, sorts it by physical spatial location,
    and extracts the voxel spacing.
    """
    dicom_files = glob.glob(os.path.join(dicom_dir, '*.dcm'))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    # Read all DICOM headers
    slices = [pydicom.dcmread(f) for f in dicom_files]

    # CRITICAL: Sort slices by their spatial position (Instance Number or Image Position Patient)
    # This prevents the 3D volume from being scrambled if the OS read the files out of order.
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        # Fallback if ImagePositionPatient is missing
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # Stack into a 3D numpy array (Height, Width, Depth)
    image_3d = np.stack([s.pixel_array for s in slices], axis=-1).astype(np.float32)

    # Convert to Hounsfield Units (HU) if Rescale Intercept/Slope are present (Standard for CT)
    if hasattr(slices[0], 'RescaleIntercept') and hasattr(slices[0], 'RescaleSlope'):
        intercept = slices[0].RescaleIntercept
        slope = slices[0].RescaleSlope
        image_3d = image_3d * slope + intercept

    # Extract pixel spacing (X, Y) and slice thickness (Z)
    dx, dy = slices[0].PixelSpacing
    dz = slices[0].SliceThickness

    return image_3d, (dx, dy, dz)


def load_png_masks(mask_dir, expected_depth):
    """
    Loads PNG masks, stacks them, and binarizes the liver class.
    """
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    if not mask_files:
        raise FileNotFoundError(f"No PNG masks found in {mask_dir}")

    if len(mask_files) != expected_depth:
        print(f"WARNING: Mask count ({len(mask_files)}) does not match DICOM count ({expected_depth}).")

    masks = [np.array(Image.open(f)) for f in mask_files]
    mask_3d = np.stack(masks, axis=-1)

    # In CHAOS, liver is usually 63 or 255. We binarize it: Liver = 1, Background = 0.
    binary_mask = (mask_3d > 0).astype(np.uint8)
    return binary_mask


def process_chaos_dataset(base_raw_dir, base_out_dir, modality="CT"):
    """
    Crawls the CHAOS dataset directory and converts patient records to NIfTI.
    """
    # Assuming standard CHAOS structure: Train_Sets/CT/1, Train_Sets/CT/2, etc.
    modality_dir = os.path.join(base_raw_dir, modality)
    patient_folders = sorted([f for f in os.listdir(modality_dir) if os.path.isdir(os.path.join(modality_dir, f))])

    print(f"Found {len(patient_folders)} patients for modality {modality}.")

    for patient_id in patient_folders:
        print(f"Processing Patient {patient_id}...")
        patient_raw_dir = os.path.join(modality_dir, patient_id, 'T2SPIR')

        # Paths specific to the CHAOS Zenodo dump
        dicom_dir = os.path.join(patient_raw_dir, 'DICOM_anon')
        mask_dir = os.path.join(patient_raw_dir, 'Ground')

        try:
            # 1. Load Data
            img_vol, spacing = load_dicom_series(dicom_dir)
            mask_vol = load_png_masks(mask_dir, expected_depth=img_vol.shape[2])

            # 2. Modality-Specific Preprocessing
            if modality == "CT":
                # Soft Tissue Windowing: Clips bones (high HU) and air/lungs (low HU)
                # This makes it MUCH easier for the subsequent GMM to cluster the liver.
                img_vol = np.clip(img_vol, -100, 250)

                # Global min-max normalization to [0, 1] for Neural Networks / GMM
            img_vol = (img_vol - np.min(img_vol)) / (np.max(img_vol) - np.min(img_vol) + 1e-8)

            # 3. Create Affine Matrix from DICOM spacing
            # The negative signs on dx and dy are standard transformations from DICOM to NIfTI coordinate space
            affine = np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])

            # 4. Save NIfTI files
            patient_out_dir = os.path.join(base_out_dir, modality, patient_id)
            os.makedirs(patient_out_dir, exist_ok=True)

            img_nii = nib.Nifti1Image(img_vol, affine)
            mask_nii = nib.Nifti1Image(mask_vol, affine)

            nib.save(img_nii, os.path.join(patient_out_dir, 'image.nii.gz'))
            nib.save(mask_nii, os.path.join(patient_out_dir, 'truth.nii.gz'))

        except Exception as e:
            print(f"FAILED on Patient {patient_id}: {e}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Point this to where you extracted the CHAOS ZIP file
    RAW_DATA_ROOT = 'CHAOS/Train_Sets'
    # Point this to where you want the NIfTI files saved
    OUTPUT_ROOT = 'new_dataset'

    # Process CT Data
    print("--- Starting CT Conversion ---")
    process_chaos_dataset(RAW_DATA_ROOT, OUTPUT_ROOT, modality="MR")

    print("\nData preparation complete. Files are ready for GMM processing.")
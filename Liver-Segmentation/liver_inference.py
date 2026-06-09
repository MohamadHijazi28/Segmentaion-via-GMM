from utils.utils import *
from utils.eval_utils import *
import os
import json
import numpy as np
from skimage.transform import resize
import glob

# --- 1. CONFIGURATION ---
# POINT THIS TO YOUR MRI DATASET (e.g., T1-DUAL or T2-SPIR)
mri_test_data_path = 'new_dataset/MR/'

base_models_dir = 'new_experiments_chaos_liver/'
nb_classes = 2
classes = [0, 1]
out_dirname = 'testing_results_mri_cross_modality'
extract_single_component = True
plot_test = False

# The sweep you trained on CT
cluster_experiments = [3, 4, 5, 6, 7]


def predict_3d_volume(unet_model, vol_3d, target_size=(256, 256)):
    """Predicts a 3D volume slice-by-slice."""
    z_depth = vol_3d.shape[-1]
    pred_3d = np.zeros((*target_size, z_depth), dtype=np.float32)

    for z in range(z_depth):
        slice_2d = vol_3d[:, :, z]
        slice_resized = resize(slice_2d, target_size, order=1, preserve_range=True, anti_aliasing=True)
        slice_norm = ne.utils.minmax_norm(slice_resized)
        slice_input = np.expand_dims(slice_norm, axis=(0, -1))

        pred = unet_model.predict(slice_input, verbose=0)
        pred_3d[:, :, z] = Postprocessing.postprocess_binary(pred)

    return pred_3d


# --- 2. MAIN INFERENCE LOOP ---
for n_clusters_total in cluster_experiments:
    print(f"\n===========================================================")
    print(f"--- Running ZERO-SHOT MRI Inference (Trained on {n_clusters_total}c CT) ---")
    print(f"===========================================================")

    # Search dynamically for all matching GMM configurations (e.g. labels_gmm_3c, labels_gmm_11c_1)
    pattern = os.path.join(base_models_dir, f"labels_gmm_{n_clusters_total}c*")
    matching_dirs = glob.glob(pattern)
    
    if not matching_dirs:
        print(f"No directories found matching pattern: {pattern}")
        continue

    for exp_dir in matching_dirs:
        # Find runs inside the directory (folders like '1', '2', etc.)
        runs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d)) and d.isdigit()]
        for run in runs:
            model_dir = os.path.join(exp_dir, run)
            config_path = os.path.join(model_dir, 'config.json')
            if not os.path.exists(config_path):
                continue
                
            print(f"\n-> Running inference for model: {model_dir}")
            
            # Resolve mapping file
            mapping_path = f'./config/chaos_liver/mapping_{n_clusters_total}c.csv'
            if not os.path.exists(mapping_path):
                # Strip suffixes (e.g., '11c_1' -> '11c')
                base_name = os.path.basename(exp_dir).replace('labels_gmm_', '')
                clean_c = base_name.split('_')[0]
                mapping_path = f'./config/chaos_liver/mapping_{clean_c}.csv'

            with open(config_path, 'r') as f:
                config = json.load(f)

            # --- THE BIG CHANGE: Ignore CT splits, load all MRI patients ---
            # Find all patient folders in the MRI directory
            mri_patient_dirs = sorted(glob.glob(os.path.join(mri_test_data_path, '*')))
            test_lst = [os.path.basename(d) for d in mri_patient_dirs if os.path.isdir(d)]

            if len(test_lst) == 0:
                print(f"WARNING: No MRI patients found in {mri_test_data_path}")
                continue

            # Load mapping
            # Retrieve labels_in from GMM configuration total clusters
            # For mapping_path resolution, extract clean cluster count
            base_name = os.path.basename(exp_dir).replace('labels_gmm_', '')
            try:
                # E.g. '11c_1' -> '11'
                clean_c_val = int(base_name.split('_')[0].replace('c', ''))
            except:
                clean_c_val = n_clusters_total
            labels_in = np.arange(0, clean_c_val + 1)
            labels_out = get_labels_out(labels_in, mapping_path)
            in_shape = tuple(config['in_shape'])

            # Load the CT-trained model
            unet_model, model_path = InferenceUtils.get_model(
                model_dir, in_shape, in_shape, labels_in, labels_out, nb_classes, config
            )

            inference_writer = InferenceWriter(os.path.join(model_dir, 'Output'), out_dirname)

            # --- 3. Evaluate on MRI ---
            print(f"Found {len(test_lst)} MRI cases to evaluate.")

            for case_id in test_lst:
                print(f"  Predicting MRI Patient {case_id}...")
                case_dir = os.path.join(mri_test_data_path, case_id)

                image_path = os.path.join(case_dir, 'image.nii.gz')
                mask_path = os.path.join(case_dir, 'truth.nii.gz')

                orig_img, affine, header = NiftiReadWrite.read_nifti_img_meta(image_path)
                truth_mask, _, _ = NiftiReadWrite.read_nifti_img_meta(mask_path)
                truth_mask = np.squeeze(truth_mask)

                # Predict
                pred_labels = predict_3d_volume(unet_model, orig_img, target_size=in_shape)

                # Resize to original MRI dimensions
                orig_h, orig_w, orig_z = orig_img.shape
                pred_resized = resize(
                    pred_labels, (orig_h, orig_w, orig_z),
                    order=0, preserve_range=True, anti_aliasing=False
                ).astype(np.int32)

                # Single Connected Component
                if extract_single_component:
                    pred_resized = Postprocessing.extract_one_connected_component_multiclass(pred_resized)

                # Save
                inference_writer.save_result(
                    case_id, orig_img, pred_resized, truth_mask, affine, header
                )

            # --- 4. EVALUATION METRICS ---
            print(f"Calculating Final Cross-Modality Metrics for {n_clusters_total}c (Model: {model_dir})...")
            metrics_without_rescaling = [Metrics.dice, Metrics.IoU]

            eval_dir = os.path.join(model_dir, 'Output', out_dirname)
            pathes = glob.glob(os.path.join(eval_dir, '*/'))

            pred_scores_single, pred_scores_multiclass = evaluate_all(
                pathes, metrics_without_rescaling, [],
                truth_filename='truth.nii.gz', result_filename='prediction.nii.gz', classes=classes
            )

            nn_folder_name = os.path.basename(model_dir)
            filename = f'eval_MRI_CrossModality_{n_clusters_total}c_{nn_folder_name}.xlsx'
            output_path = os.path.join(eval_dir, filename)

            classes_mapping = {0: 'background', 1: 'liver'}
            write_to_excel(pred_scores_multiclass, pred_scores_single, output_path, classes, classes_mapping)

            print(f"--- Finished zero-shot MRI inference for {n_clusters_total}c (Model: {model_dir}) ---")
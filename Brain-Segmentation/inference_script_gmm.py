from utils.utils import *
from utils.eval_utils import *
import os
import json
import numpy as np
from skimage.transform import resize

# --- Common parameters ---
data_path = 'gmm-all-labels'
base_models_dir = 'experiments_all_labels-gmm'
nb_classes = 2
classes = [0, 1]
out_dirname = 'testing-dataset-fsm'
extract_single_component = True
plot_test = False
brain_cluster_list = [3, 4, 5, 8, 12, 16]

# --- Loop over 8 labels ---
for i, n_clusters_brain in enumerate(brain_cluster_list):
    print(f"\n--- Running inference for label set {i + 1} ---")

    model_dir = os.path.join(base_models_dir, f"labels_gmm_{n_clusters_brain}c", '1')
    mapping_path = f'config/brain_2D_synthstrip/mapping{i + 1}.csv'

    # Load model config
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    # Load train/test split
    split_dir = os.path.join(model_dir, 'data_split')
    train_lst = set(ParamsReadWrite.list_load(os.path.join(split_dir, 'training_ids.txt')))
    valid_lst = set(ParamsReadWrite.list_load(os.path.join(split_dir, 'validation_ids.txt')))
    test_lst = set(ParamsReadWrite.list_load(os.path.join(split_dir, 'test_ids.txt')))

    # Load train label maps to get label info
    train_label_maps, _, _ = load_data(
        data_path, '', train_lst, valid_lst, test_lst,
        truth_filename=f"labels_gmm_{n_clusters_brain}c.nii.gz", input_filename='image.nii.gz'
    )
    in_shape = (256, 256)  # match the trained model
    # Load test images
    test_images, test_masks, affines, headers, test_lst, origin_test_images = load_conform_test_masks(
        data_path, split_dir, in_shape, test_ids_filename='test_ids.txt',
        mask_filename='mask.nii.gz'
    )

    flattened_label_maps = [item for subarray in train_label_maps for item in subarray.flatten()]
    labels_in = np.unique(flattened_label_maps)
    labels_out = get_labels_out(labels_in, mapping_path)

    # Load the trained model
    unet_model, model_path = InferenceUtils.get_model(
        model_dir, in_shape, in_shape, labels_in, labels_out, nb_classes, config
    )
    # Inference
    inference_writer = InferenceWriter(os.path.join(model_dir, 'Output'), out_dirname)

    for j in range(len(test_images)):
        # Step 1: Predict
        prediction = unet_model.predict(test_images[j])  # (1, 256, 256, C)

        # Step 2: Postprocess to get 2D label map
        pred_labels = Postprocessing.postprocess_binary(prediction)  # (256,256)
        if extract_single_component:
            pred_labels = Postprocessing.extract_one_connected_component_multiclass(pred_labels)

        # Step 3: Resize prediction to original image size
        orig_h, orig_w = origin_test_images[j].shape[:2]
        pred_resized = resize(
            pred_labels, (orig_h, orig_w),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.int32)  # keep integer labels

        # Add channel dimension
        # pred_resized = np.expand_dims(pred_resized, axis=-1)  # -> (H, W, 1)# Step 4: Optional single-component extraction

        # if extract_single_component:
        #    pred_resized = Postprocessing.extract_one_connected_component_multiclass(pred_resized)

        # Step 5: Prepare image and mask for plotting
        test_image = np.squeeze(test_images[j], 0)
        test_image = np.squeeze(test_image, -1)

        truth_labels = test_masks[j]
        if truth_labels.shape[-1] == 1:
            truth_labels = np.squeeze(truth_labels, -1)

        if plot_test:
            plot_multiclass_results(test_image, pred_resized, truth_labels)

        # Step 6: Save
        inference_writer.save_result(
            test_lst[j], origin_test_images[j], pred_resized, truth_labels, affines[j], headers[j]
        )

    inference_writer.save_inference_parameters(
        data_path, split_dir, mapping_path, in_shape,
        extract_single_component, model_path, in_shape
    )

    # --- Evaluation ---
    metrics_without_rescaling = [Metrics.dice, Metrics.IoU]
    metrics_with_rescaling = [Metrics.hausdorff, Metrics.assd, Metrics.hausdorff_robust]
    eval_dir = os.path.join(model_dir, 'Output', out_dirname)

    pathes = glob.glob(os.path.join(eval_dir, '*/'))
    pred_scores_single, pred_scores_multiclass = evaluate_all(
        pathes, metrics_without_rescaling, metrics_with_rescaling,
        truth_filename='truth.nii.gz', result_filename='prediction.nii.gz', classes=classes
    )

    # Save evaluation to Excel
    nn_folder_name = os.path.basename(model_dir)
    filename = f'eval_LabelSet{i}_{nn_folder_name}.xlsx'
    output_path = os.path.join(eval_dir, filename)
    classes_mapping = {0: 'background', 1: 'brain'}
    write_to_excel(pred_scores_multiclass, pred_scores_single, output_path, classes, classes_mapping)

    print(f"--- Finished inference for label set {i} ---")
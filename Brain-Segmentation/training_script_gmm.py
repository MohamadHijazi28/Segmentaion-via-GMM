from utils.utils import *
import os
import numpy as np
import tensorflow as tf

# --- Common parameters ---
data_path = 'gmm-all-labels/'
data_substring = ''
num_train = 100
num_valid = 25
input_filename = 'image.nii.gz'
in_shape = (256, 256)
pad_to_block_size = False
batch_size = 8
steps_per_epoch = 500
origin_shape = in_shape
seed = 612385
loss_name = "soft_dice_monai"
num_epochs = 20
visualize_labels = False
train_val_txt_file_path = 'gmm-all-labels/scans_all.txt'
# Parameters of the synthesis model
labels_to_image_params = {
    "aff_rotate": 45,
    "aff_scale": 0.3,
    "aff_shear": 0.1,
    "aff_shift": 40,
    "crop_prob": 1,
    "slice_prob": 1
}

# Unet parameters
unet_params = {
    "batch_norm": -1,
    "conv_size": 3,
    "feat_mult": 2,
    "nb_conv_per_level": 2,
    "nb_levels": 5
}

# Base experiments directory
#base_experiments_dir = "experiments_fsm_labels-gmm/"
base_experiments_dir = "experiments_all_labels-gmm/"
brain_cluster_list = [3, 4, 5, 8, 12, 16]
for i, n_clusters_brain in enumerate(brain_cluster_list[4:], start=4):
    print(f"\n--- Training model for label set {i} ---")

    truth_filename = f"labels_gmm_{n_clusters_brain}c.nii.gz"
    mapping_path = f'./config/brain_2D_synthstrip/mapping{i + 1}.csv'

    # Create a separate folder for each experiment
    experiments_dir = os.path.join(base_experiments_dir, f"labels_gmm_{n_clusters_brain}c")
    os.makedirs(experiments_dir, exist_ok=True)

    # --- Prepare model directory and save config ---
    model_dir = get_create_model_dir(experiments_dir)
    out_path = os.path.join(model_dir, 'config.json')
    ParamsReadWrite.write_config(out_path, data_path, truth_filename, input_filename, in_shape,
                                 mapping_path, batch_size, steps_per_epoch, loss_name,
                                 num_epochs, labels_to_image_params, unet_params)

    # --- Split data ---
    train_lst, valid_lst, test_lst = split_data_from_txt(data_path, data_substring, num_train, num_valid,
                                                         train_val_txt_file_path)
    ParamsReadWrite.save_split_data(model_dir, train_lst, valid_lst, test_lst)

    # --- Load data ---
    train_label_maps, train_images_meta, valid_data = load_data(
        data_path, data_substring, train_lst, valid_lst, test_lst,
        truth_filename, input_filename=input_filename
    )

    labels_in = np.unique(train_label_maps)
    print(f'unique labels are: {labels_in}')

    # --- Get labels_out for mapping ---
    labels_out = get_labels_out(labels_in, mapping_path)
    classes = np.unique(list(labels_out.values()))
    print('labels in ', labels_in)
    print('labels out ', labels_out)

    # --- Build model ---
    model, out, unet_model = LabelsToImageUnet.get_model(
        origin_shape, in_shape, labels_in, labels_out, len(classes),
        labels_to_image_params, unet_params
    )

    gen = Generators.synth_unet_gen(train_label_maps, batch_size=batch_size)
    tf.random.set_seed(seed)

    add_model_loss(model, loss_name, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # --- Validation callback data ---
    validation_callback_data = {
        'validation_data': valid_data[0],
        'validation_truth': valid_data[1],
        'metrics_path': os.path.join(model_dir, 'evaluation.csv'),
        'in_shape': in_shape,
        'labels_out': labels_out,
        'unet_model': unet_model,
        'model': model
    }

    # --- Train ---
    model.fit(
        gen,
        initial_epoch=0,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        workers=1,
        max_queue_size=20,
        callbacks=get_callbacks(
            model_dir,
            learning_rate_drop=0.5,
            learning_rate_epochs=None,
            learning_rate_patience=20,
            early_stopping_patience=None,
            reduce_plateau_with_restarts=False,
            save_best_only=False,
            gamma_dyn=None,
            validation_callback_data=validation_callback_data
        )
    )

    print(f"--- Finished training model for label set {i} ---")
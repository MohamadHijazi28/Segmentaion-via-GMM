from utils.utils import *
import os
import numpy as np
import tensorflow as tf
import csv

# --- 1. CONFIGURATION ---
data_path = 'new_dataset/CT/'
data_substring = ''
num_train = 16  # Adjust based on how many CT volumes you have in new_dataset/CT
num_valid = 4
input_filename = 'image.nii.gz'
in_shape = (256, 256)
batch_size = 8
steps_per_epoch = 500
seed = 612385
loss_name = "soft_dice_monai"
num_epochs = 50
train_val_txt_file_path = 'new_dataset/scans_all_ct.txt'  # Make sure this file exists and lists your patient IDs

# --- Abdominal Synthesis Parameters ---
# Tuned for abdominal breathing and organ shifting
labels_to_image_params = {
    "aff_rotate": 25,
    "aff_scale": 0.25,
    "aff_shear": 0.2,
    "aff_shift": 50,
    "crop_prob": 1,
    "slice_prob": 1
}

# Unet parameters (Same as your Brain project)
unet_params = {
    "batch_norm": -1,
    "conv_size": 3,
    "feat_mult": 2,
    "nb_conv_per_level": 2,
    "nb_levels": 5
}

base_experiments_dir = "new_experiments_chaos_liver/"

# The GMM sweeps to train
cluster_experiments = [
    (2, 1),  # 3c (2 BG, 1 Liver)
    (2, 2),  # 4c (2 BG, 2 Liver)
    (4, 1),  # 5c (4 BG, 1 Liver)
    (4, 2),  # 6c (4 BG, 2 Liver)
    (4, 3),  # 7c (4 BG, 3 Liver)
]


# --- Helper Function: Auto-Generate Mapping CSVs ---
def create_mapping_csv(csv_path, n_bg, n_liver):
    """Automatically creates the mapping: 0 to n_bg -> 0 (Background), the rest -> 1 (Liver)"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'mapping'])
        writer.writerow([0, 0])  # True background is always 0

        # Background GMM labels
        for i in range(1, n_bg + 1):
            writer.writerow([i, 0])

        # Liver GMM labels
        for j in range(n_bg + 1, n_bg + n_liver + 1):
            writer.writerow([j, 1])
    print(f"Created mapping file: {csv_path}")


# --- 2. MAIN TRAINING LOOP ---
for n_bg, n_liver in cluster_experiments:
    n_clusters_total = n_bg + n_liver

    print(f"\n===================================================================")
    print(f"--- Training Model for CHAOS Liver | Label Set: {n_clusters_total}c ({n_bg} BG, {n_liver} Liver) ---")
    print(f"===================================================================")

    truth_filename = f"labels_gmm_{n_clusters_total}c_2.nii.gz"
    mapping_path = f'./config/chaos_liver/mapping_{n_clusters_total}c.csv'

    # 1. Auto-generate the mapping CSV for this specific sweep
    create_mapping_csv(mapping_path, n_bg, n_liver)

    # 2. Setup Experiment Directory
    experiments_dir = os.path.join(base_experiments_dir, f"labels_gmm_{n_clusters_total}c")
    os.makedirs(experiments_dir, exist_ok=True)
    model_dir = get_create_model_dir(experiments_dir)

    # Save Configuration
    out_path = os.path.join(model_dir, 'config.json')
    ParamsReadWrite.write_config(out_path, data_path, truth_filename, input_filename, in_shape,
                                 mapping_path, batch_size, steps_per_epoch, loss_name,
                                 num_epochs, labels_to_image_params, unet_params)

    # 3. Split Data
    train_lst, valid_lst, test_lst = split_data_from_txt(data_path, data_substring, num_train, num_valid,
                                                         train_val_txt_file_path)
    ParamsReadWrite.save_split_data(model_dir, train_lst, valid_lst, test_lst)

    # 4. Load Data
    train_label_maps, train_images_meta, valid_data = load_data(
        data_path, data_substring, train_lst, valid_lst, test_lst,
        truth_filename, input_filename=input_filename
    )


    # --- NEW CODE: Unpack 3D volumes into 2D slices ---
    def unpack_3d_to_2d(volume_list):
        slices_2d = []
        for vol in volume_list:
            # 1. Squeeze out any dummy dimensions (e.g., changes (256, 256, 1, Z) to (256, 256, Z))
            vol_sq = np.squeeze(vol)

            # 2. Extract the 2D slices safely
            if vol_sq.ndim == 2:
                # It is already a single 2D slice
                slices_2d.append(vol_sq)
            elif vol_sq.ndim == 3:
                # Check if depth is the first axis (Z, 256, 256) or the last (256, 256, Z)
                if vol_sq.shape[0] != 256 and vol_sq.shape[1] == 256:
                    # Shape is (Z, X, Y)
                    for z in range(vol_sq.shape[0]):
                        slices_2d.append(vol_sq[z, :, :])
                else:
                    # Standard NIfTI orientation (X, Y, Z)
                    for z in range(vol_sq.shape[-1]):
                        slices_2d.append(vol_sq[:, :, z])
            else:
                print(f"Skipping volume with unexpected shape: {vol.shape}")

        return slices_2d


    print(f"Original 3D training volumes: {len(train_label_maps)}")

    # Flatten train data
    train_label_maps = unpack_3d_to_2d(train_label_maps)
    print(f"Unpacked 2D training slices: {len(train_label_maps)}")

    # Flatten validation data so the validation callback doesn't crash!
    valid_data_x = unpack_3d_to_2d(valid_data[0])
    valid_data_y = unpack_3d_to_2d(valid_data[1])
    valid_data = (valid_data_x, valid_data_y)
    # --------------------------------------------------

    #labels_in = np.unique(train_label_maps)
    labels_in = np.unique(np.concatenate([np.unique(vol) for vol in train_label_maps]))
    print(f'Unique labels found in dataset for this sweep: {labels_in}')

    # 5. Load Mapping
    labels_out = get_labels_out(labels_in, mapping_path)
    classes = np.unique(list(labels_out.values()))

    # 6. Build Synthesis Model & U-Net
    model, out, unet_model = LabelsToImageUnet.get_model(
        in_shape, in_shape, labels_in, labels_out, len(classes),
        labels_to_image_params, unet_params
    )

    gen = Generators.synth_unet_gen(train_label_maps, batch_size=batch_size)
    tf.random.set_seed(seed)

    add_model_loss(model, loss_name, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # --- NEW CODE: Load weights to resume training ---
    from utils.utils import ModelReadWrite

    initial_epoch = 0
    last_model_path = ModelReadWrite.get_last_model_path(model_dir)

    if last_model_path and os.path.exists(last_model_path):
        print(f"\n>>> Resuming training from checkpoint: {last_model_path} <<<\n")
        model.load_weights(last_model_path)
        # Dynamically extract epoch number from filename (e.g. 'epoch_020_model.hdf5')
        try:
            basename = os.path.basename(last_model_path)
            epoch_str = basename.split('_')[1]
            initial_epoch = int(epoch_str)
            print(f"Resuming training from epoch {initial_epoch}...")
        except Exception as e:
            print(f"WARNING: Could not parse epoch from filename, starting from 0: {e}")
    else:
        print("\n>>> WARNING: No checkpoint found! Starting from scratch. <<<\n")
    # -------------------------------------------------

    # 7. Validation Callbacks
    validation_callback_data = {
        'validation_data': valid_data[0],
        'validation_truth': valid_data[1],
        'metrics_path': os.path.join(model_dir, 'evaluation.csv'),
        'in_shape': in_shape,
        'labels_out': labels_out,
        'unet_model': unet_model,
        'model': model
    }

    # 8. Train the Model
    model.fit(
        gen,
        initial_epoch=initial_epoch,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        workers=1,
        max_queue_size=20,
        callbacks=get_callbacks(
            model_dir,
            learning_rate_drop=0.5,
            learning_rate_epochs=None,
            learning_rate_patience=10,
            early_stopping_patience=None,
            reduce_plateau_with_restarts=False,
            save_best_only=False,
            gamma_dyn=None,
            validation_callback_data=validation_callback_data
        )
    )

    print(f"--- Finished training sweep for {n_clusters_total}c ---\n")
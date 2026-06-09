from utils.utils import *


if __name__ == "__main__":
    """
    Training script. For each experiment, a new folder is created with new index to avoid override.
    """
    data_path = 'synthstrip_data_v1.5_2d'
    data_substring = 'fsm_t1'
    num_train = 15
    num_valid = 5
    truth_filename = 'labels.nii.gz'
    input_filename = 'image.nii.gz'
    in_shape = (256,) * 2
    pad_to_block_size = False
    mapping_path = './config/brain_2D_synthstrip/mapping.csv'#path to mapping file to map structures to brain or background
    experiments_dir = "experiments/"
    batch_size = 8
    steps_per_epoch = 500
    origin_shape = in_shape
    seed = 612385
    loss_name = "soft_dice_monai"
    num_epochs = 5
    classes = [0, 1]
    visualize_labels = False #set to True if you want to visualize labels

    #parameters of the synthesis model
    labels_to_image_params = {
        "aff_rotate": 45,
        "aff_scale": 0.3,
        "aff_shear": 0.1,
        "aff_shift": 40,
        "crop_prob": 1,
        "slice_prob": 1
    }
    #unet parameters
    unet_params = {
        "batch_norm": -1,
        "conv_size": 3,
        "feat_mult": 2,
        "nb_conv_per_level": 2,
        "nb_levels": 5
    }

    model_dir = get_create_model_dir(experiments_dir)
    out_path = os.path.join(model_dir, 'config.json')
    ParamsReadWrite.write_config(out_path, data_path, truth_filename, input_filename, in_shape, mapping_path, batch_size,
                 steps_per_epoch, loss_name,
                 num_epochs, labels_to_image_params, unet_params)
    train_lst, valid_lst, test_lst = split_data(data_path, data_substring, num_train, num_valid)
    ParamsReadWrite.save_split_data(model_dir, train_lst, valid_lst, test_lst)

    # read data from 2D synthstrip dataset
    train_label_maps, train_images_meta, valid_data, = load_data(
        data_path, data_substring, train_lst, valid_lst, test_lst, truth_filename, input_filename=input_filename)
    # extract unique label values
    labels_in = np.unique(train_label_maps)

    print('unique labels are:' + str(labels_in))

    # visualize training label maps
    if visualize_labels:
        num_vis = 10
        ne.plot.slices(train_label_maps[0:num_vis], cmaps=['turbo'])

#******************Training flow***********************************
    labels_out = get_labels_out(labels_in, mapping_path)
    classes = np.unique(list(labels_out.values()))
    print('labels in ', labels_in)
    print('labels out ', labels_out)
    model, out, unet_model = LabelsToImageUnet.get_model(origin_shape, in_shape, labels_in, labels_out, len(classes),
                                                         labels_to_image_params, unet_params)
    gen = Generators.synth_unet_gen(
        train_label_maps,
        batch_size=batch_size,
    )
    tf.random.set_seed(seed)

    add_model_loss(model, loss_name, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    validation_callback_data = {}
    validation_callback_data['validation_data'] = valid_data[0]
    validation_callback_data['validation_truth'] = valid_data[1]
    validation_callback_data['metrics_path'] = os.path.join(model_dir, 'evaluation.csv')
    validation_callback_data['in_shape'] = in_shape
    validation_callback_data['labels_out'] = labels_out
    validation_callback_data['unet_model'] = unet_model
    validation_callback_data['model'] = model

    model.fit(
        gen,
        initial_epoch=0,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        workers=1,
        max_queue_size=20,
        callbacks=get_callbacks(model_dir,
                                learning_rate_drop=0.5,
                                learning_rate_epochs=None,
                                learning_rate_patience=20,
                                early_stopping_patience=None,
                                reduce_plateau_with_restarts=False,
                                save_best_only=False,
                                gamma_dyn=None,
                                validation_callback_data=validation_callback_data))

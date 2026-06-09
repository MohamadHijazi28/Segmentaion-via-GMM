from utils.utils import *
from utils.eval_utils import *
import os

if __name__ == '__main__':
    """
    Inference script. If output location is not specified, it will be saved in the model_dir path under the folders 
    output.test
    """
    data_path = 'synthstrip_data_v1.5_2d'
    model_dir = 'experiments/6/'
    nb_classes = 2 #number of classes, 2 for binary prediction (one-hot encoding result)
    classes = [0, 1]
    mapping_path = 'config/brain_2D_synthstrip/mapping.csv'  #path to mapping file to map structures to brain or background
    out_dirname = 'fsm_results'
    extract_single_component = True #should a single connected component be extracted following segmentation
    plot_test = False #should plot test results for debugging purposes?

    #*************Inference*******************
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)


    split_dir = os.path.join(model_dir, 'data_split')
    train_lst = set(ParamsReadWrite.list_load(os.path.join(split_dir, 'training_ids.txt')))
    valid_lst = set(ParamsReadWrite.list_load(os.path.join(split_dir, 'validation_ids.txt')))
    test_lst = set(ParamsReadWrite.list_load(os.path.join(split_dir, 'test_ids.txt')))

    train_label_maps, _, _ = load_data(data_path, '', train_lst, valid_lst, test_lst)

    in_shape = train_label_maps[0].shape
    if in_shape[-1] == 1: #if the last dimension is 1, remove it
        in_shape = in_shape[:-1]
    test_images, test_label_maps, affines, headers, test_lst, origin_test_images = load_conform_test_data(data_path,
                                                                                                          split_dir,
                                                                                                          in_shape)
    flattened_label_maps = [item for subarray in train_label_maps for item in subarray.flatten()]
    labels_in = np.unique(flattened_label_maps)
    labels_out = get_labels_out(labels_in, mapping_path)
    unet_model, model_path = InferenceUtils.get_model(model_dir, in_shape, in_shape, labels_in, labels_out, nb_classes, config)

    predictions = []
    inference_writer = InferenceWriter(os.path.join(model_dir, 'Output'), out_dirname)
    for i in range(0,len(test_images)):
        prediction = unet_model.predict(test_images[i])
        truth_labels = map_values(test_label_maps[i], labels_out)
        if len(in_shape) == 2: #perform squeeze only in case of 2D data
            truth_labels = np.squeeze(truth_labels, -1)
        test_image = np.squeeze(test_images[i], 0)
        test_image = np.squeeze(test_image, -1)

        prediction = Postprocessing.postprocess_binary(prediction)
        if extract_single_component is True:
            prediction = Postprocessing.extract_one_connected_component_multiclass(prediction)
        if plot_test:
            plot_multiclass_results(test_image, prediction, truth_labels)

        inference_writer.save_result(test_lst[i], origin_test_images[i], prediction, truth_labels, affines[i],
                                     headers[i])

    inference_writer.save_inference_parameters(data_path, split_dir, mapping_path, in_shape,
                                               extract_single_component, model_path, in_shape)

    #*************Inference evaluation*******************
    """
    Perform evaluation on data with the specified evaluation functions  
    """
    #   Evaluation functions
    metrics_without_rescaling = [Metrics.dice, Metrics.IoU]
    metrics_with_rescaling = [Metrics.hausdorff, Metrics.assd, Metrics.hausdorff_robust]
    eval_dir = os.path.join(model_dir, 'Output', out_dirname)

    pathes = [_ for _ in (glob.glob(os.path.join(eval_dir, '*/')))]
    pred_scores_single, pred_scores_multiclass = evaluate_all(pathes, metrics_without_rescaling,
                                                              metrics_with_rescaling, truth_filename='truth.nii.gz',
                                                              result_filename='prediction.nii.gz', classes=classes)

    #   Save evaluation to excel file.
    nn_folder_name = os.path.basename(model_dir)
    filename = 'eval_' + nn_folder_name + '.xlsx'
    output_path = os.path.join(eval_dir, filename)
    classes_mapping = {0: 'background', 1: 'brain'}
    write_to_excel(pred_scores_multiclass, pred_scores_single, output_path, classes, classes_mapping)



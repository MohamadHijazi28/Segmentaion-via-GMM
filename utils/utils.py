
import random
import glob
import os
import json

import cv2
import numpy as np
import nibabel as nib
import tensorflow as tf
from skimage.transform import resize

import neurite.tf.models as ne_models
import neurite as ne
import pandas as pd
from keras.src.callbacks import Callback
from scipy import ndimage
from scipy.ndimage.measurements import label
from neurite.tf.models import labels_to_image_new


def split_data(data_path, substring, num_train, num_valid):
    """
    Split data to train/validation/test
    :param data_path: path to data
    :param substring: substring to search for in the dataset
    :param num_train: number of training examples
    :param num_valid: number of validation examples
    :return: direcotry names for train, validation and test data
    """
    label_maps = []
    images = []
    test_indices = []
    pathes = glob.glob(os.path.join(data_path, '*' + substring + '*'))
    dirs_lst = [os.path.basename(path) for path in pathes]

    random.shuffle(dirs_lst)
    print('retrived ' + str(len(dirs_lst)) + ' directories')

    train_dirs = dirs_lst[0:num_train]
    valid_dirs = dirs_lst[num_train:num_train + num_valid]
    test_dirs = dirs_lst[num_train + num_valid:]

    return train_dirs, valid_dirs, test_dirs


def split_data_from_txt(data_path, substring, num_train, num_valid, txt_file):
    all_dirs = [os.path.basename(p) for p in glob.glob(os.path.join(data_path, '*' + substring + '*')) if os.path.isdir(p)]

    # Read names from txt file
    with open(txt_file, "r") as f:
        cases = [line.strip() for line in f if line.strip()]

    print(f"Retrieved {len(all_dirs)} total directories in {data_path}")
    print(f"Retrieved {len(cases)} cases from txt file")

    train_dirs = cases[:num_train]
    valid_dirs = cases[num_train:num_train + num_valid]

    reserved = set(train_dirs) | set(valid_dirs)
    test_dirs = [d for d in all_dirs if d not in reserved]

    print(f"Train: {len(train_dirs)}, Valid: {len(valid_dirs)}, Test: {len(test_dirs)}")

    return train_dirs, valid_dirs, test_dirs

def resize_image(img, target_size=(256, 256), is_label=False):
    """Resize 2D or 3D image slices to (256,256,1)."""
    # Use NEAREST for labels to prevent decimals, LINEAR for images
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR

    if img.ndim == 3:
        # assume shape (H, W, D) -> take slices along last axis
        resized_slices = []
        for i in range(img.shape[-1]):
            slice_2d = img[..., i]
            resized = cv2.resize(slice_2d, target_size, interpolation=interp)
            resized_slices.append(resized[..., np.newaxis])  # keep channel
        return np.stack(resized_slices, axis=-1)  # shape (256,256,D)
    else:
        # 2D image
        resized = cv2.resize(img, target_size, interpolation=interp)
        return resized[..., np.newaxis]  # shape (256,256,1)


#class to read and write nifti data
class NiftiReadWrite:
    @staticmethod
    def load_nifti(in_file):
     #   print("Reading: {0}".format(in_file))
        image = nib.load(os.path.abspath(in_file))
        return image

    @staticmethod
    def read_nifti_labels(path):
        nib_img = NiftiReadWrite.load_nifti(path)
        affineData = nib_img.affine
        hdr = nib_img.header
        np_img = nib_img.get_fdata()

        return np_img, affineData, hdr

    @staticmethod
    def read_nifti_img_meta(path):
        nib_img = NiftiReadWrite.load_nifti(path)
        affineData = nib_img.affine
        hdr = nib_img.header
        np_img = nib_img.get_fdata()

        return np_img, affineData, hdr

    @staticmethod
    def save_nifti_with_metadata(np_array, affine, header, out_path):
        nifti_prediction = nib.nifti1.Nifti1Image(np_array, affine, header)
        nib.save(nifti_prediction, out_path)

    @staticmethod
    def save_nifti(data, path, dtype=float):
        img = nib.Nifti1Image(data, np.eye(4), dtype)
    #    nifti = NiftiReadWrite.get_image(data)
        nib.save(img, path)

def load_data(data_path, substring, train_lst, valid_lst, test_lst, truth_filename='labels.nii.gz',
              input_filename='image.nii.gz', target_size=(256, 256)):
    """
    Load data based on existing train/validation/test split
    :param data_path: path to data
    :param substring: substring to search for in the dataset
    :param num_train: number of training examples
    :return:
    """
    label_maps = []
    train_images = []
    train_label_maps = []
    valid_images = []
    valid_label_maps = []
    headers = []
    affines = []

    if '%' in data_path:
        data_paths = data_path.split('%')
    else:
        data_paths = [data_path]
    if '%' in substring:
        substrings = substring.split('%')
    else:
        substrings = [substring]
    dirs = []
    for data_path in data_paths:
        for substr in substrings:
            substr_dirs = glob.glob(os.path.join(data_path, substr + '*'))
            dirs = dirs + substr_dirs

    for case_dir in dirs:
        case_id = os.path.basename(case_dir)

        if case_id not in train_lst and case_id not in valid_lst and case_id not in test_lst:
            continue

        images_path = os.path.join(case_dir, input_filename)
        image, affine, header = NiftiReadWrite.read_nifti_img_meta(images_path)
        headers.append(header)
        affines.append(affine)

        image = resize_image(image, target_size)

        if case_id in train_lst:
            truth_path = os.path.join(case_dir, truth_filename)
            truth, _, _ = NiftiReadWrite.read_nifti_img_meta(truth_path)
            truth = resize_image(np.int16(truth), target_size,is_label=True)
            train_images.append(image)
            train_label_maps.append(np.int16(truth))

        elif case_id in valid_lst:
            truth_path = os.path.join(case_dir, truth_filename)
            truth, _, _ = NiftiReadWrite.read_nifti_img_meta(truth_path)
            truth = resize_image(np.int16(truth), target_size, is_label=True)
            valid_images.append(image)
            valid_label_maps.append(np.int16(truth))

        elif case_id in test_lst:
            # test set has no labels
            pass
        # truth_path = os.path.join(case_dir, truth_filename)
        # case_id = os.path.basename(case_dir)
        # if (case_id not in train_lst) and (case_id not in valid_lst) and (case_id not in test_lst):
        #     continue
        # truth, affine, header = NiftiReadWrite.read_nifti_img_meta(truth_path)
        # label_maps.append(np.int16(truth))
        # images_path = os.path.join(case_dir, input_filename)
        # image, affine, header = NiftiReadWrite.read_nifti_img_meta(images_path)
        # headers.append(header)
        # affines.append(affine)
        #
        # if case_id in train_lst:
        #     train_images.append(image)
        #     train_label_maps.append(np.int16(truth))
        # if case_id in valid_lst:
        #     valid_images.append(image)
        #     valid_label_maps.append(np.int16(truth))

    valid_data = (valid_images, valid_label_maps)
    train_images_meta = [train_images, affines, headers]
    return train_label_maps, train_images_meta, valid_data

class LabelsToImageUnet:
    #class for the unified synthesis and unet model
    @staticmethod
    def get_model(origin_shape, in_shape, labels_in, labels_out, nb_classes, labels_to_image_params, unet_params):
        """
        Get unified model of labels_to_image_new and unet
        :param in_shape: input shape
        :param labels_in:
        :param labels_out: mapped values
        :param unet_type: type of unet used. Can be either "unet_neurite" or "unet_neuron"
        :param anisotropic_levels: in case of unet_type=='anisotropic', how many levels in the z dimension
        should refrain from pooling
        :return: model, out, unet_model
        """
        if len(origin_shape) > len(in_shape):
            origin_shape = origin_shape[:-1]
        print('length of labels in that are passed to the model is: ' + str(len(labels_in)))
        gen_arg = dict(
            in_shape = origin_shape,
            out_shape=in_shape, #the output shape of the gen model should be the input shape of the unet network
            labels_in=labels_in,
            labels_out=labels_out,
            return_map=True,
            one_hot=True,
            aff_shift=labels_to_image_params['aff_shift'],
            aff_rotate=labels_to_image_params['aff_rotate'],
            aff_scale=labels_to_image_params['aff_scale'],
            aff_shear=labels_to_image_params['aff_shear'],
            slice_prob=labels_to_image_params['slice_prob'],
            crop_prob=labels_to_image_params['crop_prob']
        )
        gen_model = labels_to_image_new(**gen_arg, id=1)
        gen_image, gen_map = gen_model.outputs

        gen_model.summary()

        unet_model = ne_models.unet(
            input_shape=(*in_shape, 1),
            nb_features=nb_classes,
            nb_levels=unet_params['nb_levels'],
            nb_labels=nb_classes,
            conv_size=unet_params['conv_size'],
            feat_mult=unet_params['feat_mult'],
            use_logp=False,
            nb_conv_per_level=unet_params['nb_conv_per_level'],
            batch_norm=unet_params['batch_norm']
        )

        unet_model.summary()

        pred = unet_model(gen_image)
        out = (gen_map, pred)

        model = tf.keras.Model(gen_model.inputs[0], out)
        model.summary()

        return model, out, unet_model

def get_labels_out(labels_in, mapping_path):
    """
    Load labels mapping and output labels out given labels_in
    :param labels_in: input labels set
    :param mapping_path: mapping of labels file path
    :return: labels_out
    """
    mapping = {}
    ids_pd = pd.read_csv(mapping_path, encoding="unicode_escape", index_col='label')
    mapping_all = ids_pd.to_dict()['mapping']
    print(mapping_all)
    for label in labels_in:
        mapping[label] = mapping_all[label]

    return mapping


class Loss:

    def soft_dice(self, a, b):
        """
        Soft Dice loss implementation
        """
        dim = len(a.shape) - 2
        space = list(range(1, dim + 1))
        top = 2 * tf.reduce_sum(a * b, axis=space)
        bot = tf.reduce_sum(a ** 2, axis=space) + tf.reduce_sum(b ** 2, axis=space)

        out = tf.divide(top, bot + 1e-6)
        return -tf.reduce_mean(out)

    def soft_dice_monai(self, a, b):
        """
        Soft Dice loss implementation
        Added epsilon in denominator and 1-loss instead of -loss
        """

        dim = len(a.shape) - 2
        space = list(range(1, dim + 1))
        top = 2 * tf.reduce_sum(a * b, axis=space)
        bot = tf.reduce_sum(a ** 2, axis=space) + tf.reduce_sum(b ** 2, axis=space)

        out = tf.divide(top + tf.keras.backend.epsilon(), bot + tf.keras.backend.epsilon())
        return 1 - tf.reduce_mean(out)

    def dice_CE_loss(self, y_true, y_pred):
        """
        Dice and cross entropy loss
        Args:
            gt:ground truth mask
            pred: network prediction

        Returns: unified Dice and cross entropy loss

        """
        return self.soft_dice_monai(y_true, y_pred) + tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def add_model_loss(model, loss_name, loss_params):
    """
    Add loss to the model according to loss name
    """
    if loss_name == 'soft_dice':
        model.add_loss(Loss().soft_dice(*loss_params))
    elif loss_name == 'soft_dice_monai':
        model.add_loss(Loss().soft_dice_monai(*loss_params))
    elif loss_name == 'dice_CE':
        model.add_loss(Loss().dice_CE_loss(*loss_params))

class ModelReadWrite:
    @staticmethod
    def get_last_model_path(model_dir):
        model_names = glob.glob(os.path.join(model_dir, '*.hdf5'))
        if not model_names:
            model_names = glob.glob(model_dir + '*.hdf5')
        if (len(model_names) == 0):
            print('error loading any model with prefix ' + model_dir)

        return sorted(model_names, key=os.path.getmtime)[-1]

    @staticmethod
    def get_best_model_path(model_dir, min_epoch=0, dice_column='val_Dice'):
        eval_df = pd.read_csv(os.path.join(model_dir, 'evaluation.csv'))
        eval_df = eval_df.sort_values(by=[dice_column], ascending=False)
        for index, row in eval_df.iterrows():
            selected_epoch = int(row['epoch'])
            if selected_epoch >= min_epoch:
                break

        print('best network validation dice is: ' + str(eval_df.loc[index, dice_column]))
        model_name = 'epoch_{epoch}_model.hdf5'.format(epoch="{:03d}".format(selected_epoch+1))
        model_name = 'epoch_049_model.hdf5'
        return os.path.join(model_dir, model_name)


class Postprocessing:
    @staticmethod
    def postprocess_binary(prediction_one_hot):
        """
        Transform one-hot-encoded network output to multiclass volume
        :param prediction_one_hot:
        :return:
        """
        prediction_multiclass = tf.math.argmax(prediction_one_hot, -1).numpy()
        prediction_multiclass = np.squeeze(prediction_multiclass, 0)
        final_prediction = np.zeros_like(prediction_multiclass)

        final_prediction[prediction_multiclass == 0] = 0
        final_prediction[prediction_multiclass == 1] = 1

        return final_prediction

    @staticmethod
    def binary_soft_prediction(prediction_one_hot):
        """
        This method assumes that the prediction is binary with one-hot-encoding
        The output will be probabilities of the binary class
        """
        if prediction_one_hot.shape[2]>1:#if 3D
            prediction_binary = prediction_one_hot[:,:,:,1]
        else:
            prediction_binary = prediction_one_hot[:, :, 1]
        return prediction_binary

    @staticmethod
    def extract_one_connected_component(prediction):
        """
        Extract a single connected component
        Args:
            prediction:

        Returns:

        """
        labeled_array, num_features = label(prediction)
        i = np.argmax([np.sum(labeled_array == _) for _ in range(1, num_features + 1)]) + 1
        return labeled_array == i

    @staticmethod
    def extract_one_connected_component_multiclass(prediction):
        """
        Extract a single connected component
        Args:
            prediction: multiclass prediction

        Returns:
            prediction after extraction of a single connected component for all classes (background is 0)
        """
        prediction_single = np.zeros_like(prediction)
        prediction_single[prediction != 0] = 1
        prediction_single = Postprocessing.extract_one_connected_component(prediction_single)
        prediction[prediction_single == 0] = 0

        return prediction


def map_values(labels, mapping):
    """
    Map labels according to mapping dictionary. Return labels with mapped values.
    :param labels: labels
    :param mapping: mapping distionary
    :return:
    """
    for mapping_value in mapping.keys():
        labels[labels == mapping_value] = mapping[mapping_value]
    return labels


class ValidationEvalUpdate(Callback):
    """
    Calculates Dice on validation set and adds it to Keras logs.
    """

    def __init__(self, validation_data, validation_truth, labels_out, in_shape, unet_model):
        super(ValidationEvalUpdate, self).__init__()

        self.valid_data = validation_data
        self.valid_labels = validation_truth
        self.in_shape = in_shape
        self.unet_model = unet_model
        self.labels_out = labels_out
        self.classes = np.unique(list(labels_out.values()))

        # Pre-process validation data once during init to save time
        for i in range(len(self.valid_data)):
            self.valid_data[i] = InferenceUtils.conform(self.valid_data[i])
            # Ensure labels are mapped correctly
            #self.valid_labels[i] = map_values(self.valid_labels[i], labels_out)

    def on_epoch_end(self, epoch, logs=None):
        """
        Run inference, calculate Dice, and update logs.
        """
        if logs is None:
            logs = {}

        validation_dice = self.calc_validation_dice()

        # KEY FIX: Add the value to the logs dictionary.
        # CSVLogger will see this and automatically write it to the CSV.
        logs['val_Dice'] = validation_dice

        # Optional: If you strictly need a 'val_loss' column filled:
        # logs['val_loss'] = 1 - validation_dice

        print(f"\nValidation Dice (Epoch {epoch + 1}): {validation_dice:.4f}")

    def calc_validation_dice(self):
        dice_scores = []
        for i in range(len(self.valid_data)):
            # Predict
            prediction = self.unet_model.predict(self.valid_data[i], verbose=0)
            prediction = Postprocessing.postprocess_binary(prediction)

            raw_truth = self.valid_labels[i]
            if len(raw_truth.shape) > len(self.in_shape):
                raw_truth = np.squeeze(raw_truth, -1)

            # --- NEW: Safely map GMM labels (e.g., 13-16) to binary (0 and 1) ---
            # We create a blank array so values don't accidentally overwrite each other
            truth = np.zeros_like(raw_truth)
            for gmm_label, binary_class in self.labels_out.items():
                truth[raw_truth == gmm_label] = binary_class

            # Calculate multiclass dice
            pred_scores_single_class = []
            for class_id_str in self.classes:
                class_id = int(class_id_str)
                if class_id == 0:
                    continue # We skip background (0) to strictly evaluate the Liver

                truth_single_class = (truth == class_id)
                pred_single_class = (prediction == class_id)

                score = Metrics.dice(truth_single_class, pred_single_class)
                pred_scores_single_class.append(score)

            if pred_scores_single_class:
                dice_scores.append(np.average(pred_scores_single_class))
            else:
                dice_scores.append(0.0)

        return np.average(dice_scores)

# class ValidationEvalUpdate(Callback):
#     """
#     A callback to perform validation by the end of each epoch and write to metrics file
#     """
#
#     def __init__(self, validation_data, validation_truth, metrics_path, labels_out, in_shape, model, unet_model):
#         super(Callback, self).__init__()
#
#         self.origin_shape = validation_data[0].shape
#         self.in_shape = in_shape
#         for i in range(len(validation_data)):
#             case_data = validation_data[i]
#             validation_data[i] = InferenceUtils.conform(case_data)
#
#         self.valid_data = validation_data
#         self.metrics_path = metrics_path
#         # model, _, unet_model = LabelsToImageUnet.get_model(origin_shape, in_shape, labels_in, labels_out,
#         #                                                    labels_to_image_params, unet_params, unet_type)
#         self.model = model
#         self.unet_model = unet_model
#         nb_classes = np.unique(list(labels_out.values()))
#         self.classes = nb_classes
#
#         # update validation ground truth
#         self.valid_labels = []
#         if len(self.in_shape) == 2:  # perform squeeze only in case of 2D data
#             validation_truth = np.squeeze(validation_truth, -1)
#         #for i in range(len(validation_truth)):
#         #    validation_truth[i] = map_values(validation_truth[i], labels_out)
#         self.valid_labels = validation_truth
#
#     def on_epoch_end(self, epoch, logs=None):
#         """
#         Apply inference on validation data and save to metrics file
#         Returns:
#         """
#         validation_dice = self.calc_validation_dice(epoch)
#         metrics_df = pd.read_csv(self.metrics_path)
#         row_index = metrics_df.loc[metrics_df['epoch'] == epoch].index[0]
#         metrics_df.loc[row_index, 'val_Dice'] = validation_dice
#
#         metrics_df.to_csv(self.metrics_path, index=False)
#
#
#     def calc_validation_dice(self, epoch):
#         """
#         Run inference on validation data with current weights and save average validation Dice score
#         in evaluation.csv file for the current epoch
#         Returns:
#
#         """
#         self.model.set_weights(self.model.get_weights())
#
#         dice_scores = []
#         for i in range(0, len(self.valid_data)):
#             prediction = self.unet_model.predict(self.valid_data[i])
#             prediction = Postprocessing.postprocess_binary(prediction)
#
#             truth = self.valid_labels[i]
#             if len(truth.shape) > len(self.in_shape):
#                 np.squeeze(truth, -1)
#
#             pred_scores_single_class = []
#             for class_id_str in self.classes:
#                 class_id = int(class_id_str)
#                 if class_id == 0:
#                     continue
#                 truth_single_class = np.zeros_like(truth)
#                 truth_single_class[truth == class_id] = 1
#                 pred_single_class = np.zeros_like(prediction)
#                 pred_single_class[prediction == class_id] = 1
#                 pred_scores_single_class.append(Metrics.dice(truth_single_class, pred_single_class))
#
#             dice_score = np.average(pred_scores_single_class)
#
#             dice_scores.append(dice_score)
#
#         return np.average(dice_scores)


# class Loss:
#
#     def soft_dice(self, a, b):
#         """
#         Soft Dice loss implementation
#         """
#         dim = len(a.shape) - 2
#         space = list(range(1, dim + 1))
#         top = 2 * tf.reduce_sum(a * b, axis=space)
#         bot = tf.reduce_sum(a ** 2, axis=space) + tf.reduce_sum(b ** 2, axis=space)
#
#         out = tf.divide(top, bot + 1e-6)
#         return -tf.reduce_mean(out)
#
#     def soft_dice_monai(self, a, b):
#         """
#         Soft Dice loss implementation
#         Added epsilon in denominator and 1-loss instead of -loss
#         """
#
#         dim = len(a.shape) - 2
#         space = list(range(1, dim + 1))
#         top = 2 * tf.reduce_sum(a * b, axis=space)
#         bot = tf.reduce_sum(a ** 2, axis=space) + tf.reduce_sum(b ** 2, axis=space)
#
#         out = tf.divide(top + tf.keras.backend.epsilon(), bot + tf.keras.backend.epsilon())
#         return 1 - tf.reduce_mean(out)
#
#     def dice_CE_loss(self, y_true, y_pred):
#         """
#         Dice and cross entropy loss
#         Args:
#             gt:ground truth mask
#             pred: network prediction
#
#         Returns: unified Dice and cross entropy loss
#
#         """
#         return self.soft_dice_monai(y_true, y_pred) + tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#
#
# def add_model_loss(model, loss_name, loss_params):
#     """
#     Add loss to the model according to loss name
#     """
#     if loss_name == 'soft_dice':
#         model.add_loss(Loss().soft_dice(*loss_params))
#     elif loss_name == 'soft_dice_monai':
#         model.add_loss(Loss().soft_dice_monai(*loss_params))
#     elif loss_name == 'dice_CE':
#         model.add_loss(Loss().dice_CE_loss(*loss_params))

class Metrics:

    @staticmethod
    def dice(y_true, y_pred, smooth=0.001):
        y_true_f = y_true.flatten() > 0
        y_pred_f = y_pred.flatten() > 0
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


    @staticmethod
    def IoU(y_true, y_pred, smooth=0.001):
        """
        Jaccard index or Intersection over Union (IoU) calculation
        :param y_true: truth mask
        :param y_pred: prediction mask
        :param smooth: smoothing parameter
        :return:
        """
        y_true_f = y_true.flatten() > 0
        y_pred_f = y_pred.flatten() > 0
        intersection = np.sum(y_true_f * y_pred_f)
        return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)

    @staticmethod
    def multiclass_dice(truth, pred, classes):
        """
        calculate multiclass dice by averaging single classes performance
        Expected to have the last channel representing the classes
        """
        class_dices = []
        for class_id in classes:
            class_id = int(class_id)
            if class_id == 0:
                continue
            truth_single_class = np.zeros_like(truth)
            truth_single_class[truth == class_id] = 1
            pred_single_class = np.zeros_like(pred)
            pred_single_class[pred == class_id] = 1
            dice_class = Metrics.dice(truth_single_class, pred_single_class)
            class_dices.append(dice_class)
        dice_multiclass = np.average(class_dices)
        return dice_multiclass

class Generators:
    @staticmethod
    def synth_unet_gen(label_maps, batch_size=1):
        """
        Generator for segmentation.

        Parameters:
            labels_maps: List of pre-loaded ND label maps, each as a NumPy array.
            batch_size: Batch size. Default is 1.
            same_subj: Whether the same label map is returned as the source and target for further
                augmentation. Default is False.
            flip: Whether axes are flipped randomly. Default is True.
        """
        # in_shape = label_maps[0].shape
        # num_dim = len(in_shape)
        #
        #
        # rand = np.random.default_rng()
        # prop = dict(replace=False, shuffle=False)
        # while True:
        #     ind = rand.integers(len(label_maps), size=batch_size)
        #     x = [label_maps[i] for i in ind]
        #
        #     sliced_x = []
        #     for vol in x:
        #         # vol.shape is currently (256, 256, 1, Z_depth)
        #         z_depth = vol.shape[-1]
        #
        #         # Pick a random slice index between 0 and the max depth
        #         random_z = np.random.randint(0, z_depth)
        #
        #         # Extract that single 2D slice.
        #         # This turns (256, 256, 1, Z) into a flat (256, 256) image!
        #         slice_2d = vol[:, :, 0, random_z]
        #
        #         sliced_x.append(slice_2d)
        #
        #     # Now stack the flat 2D slices together and add the channel dimension
        #     x = np.stack(sliced_x)[..., None]
        #     #x = np.stack(x)[..., None]
        #
        #     src = x[:batch_size, ...]
        #
        #     yield [src]

        rand = np.random.default_rng()

        while True:
            # 1. Pick random indices for the batch
            ind = rand.integers(len(label_maps), size=batch_size)

            # 2. Fetch the 2D images directly (they are already slices!)
            x = [label_maps[i] for i in ind]

            # 3. Stack the 2D slices together into a batch and add the channel dimension
            # Turns a list of (256, 256) into a numpy array of (batch_size, 256, 256, 1)
            x = np.stack(x)[..., None]

            src = x[:batch_size, ...]

            # Yield the batch to the Keras model
            yield [src]


def get_callbacks(save_dir, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, verbosity=1, early_stopping_patience=None, save_best_only=True,
                  reduce_plateau_with_restarts=False, num_cycle_epochs=60, gamma_dyn=None,
                  validation_callback_data=None):
    callbacks = list()

    # 1. Add Validation Callback FIRST
    # This allows it to calculate metrics and update 'logs' before logging happens
    if validation_callback_data is not None:
        callbacks.append(ValidationEvalUpdate(
            validation_data=validation_callback_data['validation_data'],
            validation_truth=validation_callback_data['validation_truth'],
            # Note: Removed metrics_path since we don't write to file manually anymore
            labels_out=validation_callback_data['labels_out'],
            in_shape=validation_callback_data['in_shape'],
            unet_model=validation_callback_data['unet_model']
        ))

    # 2. Add CSVLogger SECOND
    # It will now find 'val_Dice' in the logs and write it correctly
    callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'evaluation.csv'), append=True))

    # 3. Add ModelCheckpoint
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, 'epoch_{epoch:03d}_model.hdf5'),
        save_best_only=save_best_only,
        verbose=verbosity
    ))

    if gamma_dyn is not None:
        callbacks.append(gamma_dyn)

    if early_stopping_patience:
        callbacks.append(tf.keras.callbacks.EarlyStopping(verbose=verbosity, patience=early_stopping_patience))

    return callbacks
# def get_callbacks(save_dir, learning_rate_drop=0.5, learning_rate_epochs=None,
#                   learning_rate_patience=50, verbosity=1, early_stopping_patience=None, save_best_only=True,
#                   reduce_plateau_with_restarts=False, num_cycle_epochs=60, gamma_dyn=None,
#                   validation_callback_data=None):
#
#     callbacks = list()
#
#     callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, 'epoch_{epoch:03d}_model.hdf5'),
#                                      save_best_only=save_best_only, verbose=verbosity))
#     callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'evaluation.csv'), append=True))
#
#     if validation_callback_data is not None:
#         callbacks.append(ValidationEvalUpdate(validation_callback_data['validation_data'],
#                                               validation_callback_data['validation_truth'],
#                                               validation_callback_data['metrics_path'],
#                                               validation_callback_data['labels_out'],
#                                               validation_callback_data['in_shape'],
#                                               validation_callback_data['model'],
#                                               validation_callback_data['unet_model']))
#
#     if gamma_dyn is not None:
#         callbacks.append(gamma_dyn)
#
#     if early_stopping_patience:
#         callbacks.append(tf.keras.callbacks.EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
#
#     return callbacks

class ParamsReadWrite:
    @staticmethod
    def list_dump(lst, out_file):
        np.savetxt(out_file, lst, fmt='%s')

    @staticmethod
    def list_load(in_file):
        return list(np.loadtxt(in_file, dtype=str, ndmin=1))

    @staticmethod
    def save_split_data(model_dir, train_lst, valid_lst, test_lst):
        """
        Save training, validation andtest data
        """
        split_path = os.path.join(model_dir, 'data_split')
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        ParamsReadWrite.list_dump(train_lst, os.path.join(split_path, 'training_ids.txt'))
        ParamsReadWrite.list_dump(valid_lst, os.path.join(split_path, 'validation_ids.txt'))
        ParamsReadWrite.list_dump(test_lst, os.path.join(split_path, 'test_ids.txt'))
    @staticmethod
    def write_config(out_path, data_path, truth_filename, input_filename, in_shape, mapping_path, batch_size, steps_per_epoch, loss_name,
                     num_epochs, labels_to_image_params, unet_params):
        config = {
            "data_path": data_path,
            "truth_filename": truth_filename,
            "input_filename": input_filename,
            "in_shape": in_shape,
            "mapping_path": mapping_path,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "loss_name": loss_name,
            "num_epochs": num_epochs,
            "labels_to_image_params": labels_to_image_params,
            "unet_params": unet_params
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

def get_create_model_dir(experiments_dir):
    """
    List directories in experiments_dir and find the latest index
    Create a new index for the experiment with id=latest_id+1
    """
    dir_ids = os.listdir(experiments_dir)
    max_id = 0
    for dir_id in dir_ids:
        try:
            val = int(dir_id)  # handles '001', '42', '-3', etc.
        except ValueError:
            continue
        if val > max_id:
            max_id = val

    dir_path= os.path.join(experiments_dir, str(max_id + 1))
    if os.path.exists(dir_path) is False:
        print('creating experiment directory: ' + dir_path)
        os.mkdir(dir_path)

    return dir_path

#******************Inference utils******************************
def load_conform_test_data(data_path, split_dir, block_size, test_ids_filename='test_ids.txt',
                           labels_filename='labels.nii.gz'):
    """
    Load test data based on test ids
    :param data_path: path to data
    :param split_dir: path to data split ids
    :return:
    """
    test_images = []
    origin_test_images = []
    test_label_maps = []
    affines = []
    headers = []
    test_lst = ParamsReadWrite.list_load(os.path.join(split_dir, test_ids_filename))

    if '%' in data_path:
        data_paths = data_path.split('%')
    else:
        data_paths = [data_path]

    dirs = []
    for data_path in data_paths:
        substr_dirs = glob.glob(os.path.join(data_path, '*'))
        dirs = dirs + substr_dirs
    data_path_mapping = {}
    for data_dir in dirs:
        case_id = os.path.basename(data_dir)
        data_path_mapping[case_id] = data_dir

    for test_id in test_lst:
        input_path = os.path.join(data_path_mapping[test_id], labels_filename)
        truth, affine, header = NiftiReadWrite.read_nifti_img_meta(input_path)
        test_label_maps.append(np.int16(truth))
        images_path = os.path.join(data_path_mapping[test_id], 'image.nii.gz')
        image, affine, header = NiftiReadWrite.read_nifti_img_meta(images_path)
        origin_test_images.append(image)

        if len(block_size) == 3 and block_size != image.shape:
            print('applying image rescaling')
            image = ndimage.zoom(image, [block_size[0] / image.shape[0],
                                         block_size[1] / image.shape[1],
                                         block_size[2] / image.shape[2]], order=3)
            print('image shape after rescaling: ' + str(image.shape))
        conformed_img = InferenceUtils.conform(image)
        test_images.append(conformed_img)
        affines.append(affine)
        headers.append(header)

    return test_images, test_label_maps, affines, headers, test_lst, origin_test_images


def load_conform_test_masks(data_path, split_dir, block_size, test_ids_filename='test_ids.txt',
                            mask_filename='mask.nii.gz'):
    """
    Load test data and corresponding binary masks (0/1) based on test ids.
    :param data_path: path to data
    :param split_dir: path to data split ids
    :param block_size: desired output image size (rescaling)
    :param test_ids_filename: txt file listing test case IDs
    :param mask_filename: name of the mask file (binary 0/1)
    :return:
        test_images: list of conformed test images
        test_masks: list of masks (0/1)
        affines: list of affine matrices
        headers: list of NIfTI headers
        test_lst: list of test case IDs
        origin_test_images: list of original images (before conformation)
    """
    test_images = []
    origin_test_images = []
    test_masks = []
    affines = []
    headers = []

    # Load test IDs
    test_lst = ParamsReadWrite.list_load(os.path.join(split_dir, test_ids_filename))

    # Handle multiple data paths separated by '%'
    data_paths = data_path.split('%') if '%' in data_path else [data_path]

    # Collect all case directories
    dirs = []
    for dp in data_paths:
        dirs += glob.glob(os.path.join(dp, '*'))

    # Map case IDs to directories
    data_path_mapping = {os.path.basename(d): d for d in dirs}

    # Loop over test cases
    for test_id in test_lst:
        # Load mask
        mask_path = os.path.join(data_path_mapping[test_id], mask_filename)
        mask_data, affine, header = NiftiReadWrite.read_nifti_img_meta(mask_path)
        mask_data = np.int16(mask_data)  # ensure integer type 0/1

        # Load original image
        image_path = os.path.join(data_path_mapping[test_id], 'image.nii.gz')
        image, _, _ = NiftiReadWrite.read_nifti_img_meta(image_path)
        origin_test_images.append(image)

        # Rescale image and mask if needed
        if len(block_size) == 3 and block_size != image.shape:
            print('Applying rescaling to image and mask')
            factors = [block_size[i] / image.shape[i] for i in range(3)]
            image = ndimage.zoom(image, factors, order=3)  # image: high-order interpolation
            mask_data = ndimage.zoom(mask_data, factors, order=0)  # mask: nearest-neighbor
            print(f'Image shape after rescaling: {image.shape}, Mask shape: {mask_data.shape}')

        # Conform the image (normalize, pad/crop)
        # conformed_img = InferenceUtils.conform(image)
        conformed_img = InferenceUtils.conform_hanle_different_image_size(
            image)  # I changed this to the new conform function
        test_images.append(conformed_img)
        test_masks.append(mask_data)
        affines.append(affine)
        headers.append(header)

    return test_images, test_masks, affines, headers, test_lst, origin_test_images


def plot_multiclass_results(test_image, prediction_multiclass, truth_labels):
    """
    Plot multiclass results
    :param test_image: original image
    :param prediction_multiclass: multiclass prediction
    :param truth_labels: multiclass ground truth
    :return:
    """
    ne.plot.slices(
        slices_in=(test_image, prediction_multiclass, truth_labels),
        titles=('image', 'prediction', 'truth'),
        do_colorbars=True,
    );
    ne.plot.slices(
        slices_in=(prediction_multiclass, truth_labels),
        titles=('prediction', 'truth'),
        do_colorbars=True,
        cmaps=['tab20c']
    );

class InferenceWriter:
    def __init__(self, output_path, out_dirname):
        self.output_path = output_path
        self.out_dirname =  out_dirname
        if os.path.exists(output_path) is False:
            os.mkdir(os.path.join(output_path))
        if os.path.exists(os.path.join(output_path,  out_dirname)) is False:
            os.mkdir(os.path.join(output_path, out_dirname))

    def save_result(self, case_id, test_image, prediction, truth_labels, affine, header):
        """
        Write output of case case_id
        :param case_id: id of the case, folder to write
        :param test_image: test input
        :param prediction: prediction
        :param truth_labels: ground truth
        :param affine: affine
        :param header: header
        :return:
        """
        case_path = os.path.join(self.output_path,  self.out_dirname, case_id)
        if os.path.exists(case_path) is False:
            os.mkdir(case_path)
        prediction = np.expand_dims(prediction, -1)
        truth_labels = np.expand_dims(truth_labels, -1)
        NiftiReadWrite.save_nifti_with_metadata(test_image, affine, header, os.path.join(case_path, 'data.nii.gz'))
        NiftiReadWrite.save_nifti_with_metadata(prediction, affine, header,
                                                os.path.join(case_path, 'prediction.nii.gz'))
        NiftiReadWrite.save_nifti_with_metadata(truth_labels, affine, header,
                                                os.path.join(case_path, 'truth.nii.gz'))

    def save_inference_parameters(self, data_path, split_dir, mapping_path, in_shape, extract_single_component,
                                  model_path, inference_shape):
        """
        Write inference parameters for trackability
        :param data_path:
        :param split_dir:
        :param mapping_path:
        :return:
        """
        output_folder = os.path.join(self.output_path, self.out_dirname)
        params = {}
        params['output_path'] = output_folder
        params['data_path'] = data_path
        params['split_dir'] = split_dir
        params['mapping_path'] = mapping_path
        params['input_shape'] = str(in_shape)
        params['extract_single_component'] = extract_single_component
        params['model_path'] = model_path
        params['inference_shape'] = inference_shape

        with open(os.path.join(output_folder, 'inference_args.json'), 'wt') as f:
            json.dump(params, f, indent=4)

class InferenceUtils:
    @staticmethod
    def conform(x):
        '''Resize and normalize image.'''
        x = np.float32(x)
        x = np.squeeze(x)
        x = ne.utils.minmax_norm(x)
        #   x = ne.utils.zoom(x, zoom_factor=[o / i for o, i in zip(in_shape, x.shape)])
        return np.expand_dims(x, axis=(0, -1))

    @staticmethod
    def get_model(model_dir, origin_shape, in_shape, labels_in, labels_out, nb_classes, config):
        """
        Get model for inference
        Load model declaration and pick weights based on best performance on validation set if possible
        Args:
            model_dir: models directory
            origin_shape: shape of the volume
            in_shape: shape of the unet network. ideally should be the same as origin_shape
            labels_in:
            labels_out:
            config: training configuration file including networks parameters

        Returns:

        """
        model_path = ModelReadWrite.get_best_model_path(model_dir)
        if model_path is None or os.path.exists(model_path) is False:
            model_path = ModelReadWrite.get_last_model_path(model_dir)
        print('model path is: ' + model_path)

        model, _, unet_model = LabelsToImageUnet.get_model(origin_shape, in_shape, labels_in, labels_out, nb_classes,
                                                           config['labels_to_image_params'],
                                                           config['unet_params'])
        model.load_weights(model_path)

        return unet_model, model_path


    def conform_hanle_different_image_size(x, target_shape=(256, 256)):
        """Always resize to target_shape (ignores original aspect ratio)."""
        # make sure it's a proper numpy array with float dtype
        x = np.asarray(x).astype(np.float32)
        x = np.squeeze(x)  # -> (h, w)

        # resize to target
        th, tw = target_shape
        x = resize(x, (th, tw), order=1, preserve_range=True, anti_aliasing=True)

        # normalize
        x = ne.utils.minmax_norm(x)

        # add batch + channel
        return np.expand_dims(x, axis=(0, -1))  # -> (1, 256, 256, 1)
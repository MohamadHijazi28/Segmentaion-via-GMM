import string
from multiprocess.dummy import Pool
from tqdm.notebook import tqdm
from utils.lits_surface import Surface
import os
import pandas as pd
import nibabel as nib
import numpy as np

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
    def assd(y_true, y_pred, scaling):
        if (len(y_true.shape) == 3):
            evalsurf = Surface(y_pred, y_true, physical_voxel_spacing=scaling)
        else:  # 2D measure
            evalsurf = Surface(y_pred, y_true, physical_voxel_spacing=scaling, is_3D=False)
        assd = evalsurf.get_average_symmetric_surface_distance()
        return assd

    @staticmethod
    def hausdorff(y_true, y_pred, scaling):
        if (len(y_true.shape) == 3):
            evalsurf = Surface(y_pred, y_true, physical_voxel_spacing=scaling)
        else:  # 2D measure
            evalsurf = Surface(y_pred, y_true, physical_voxel_spacing=scaling, is_3D=False)
        Hausdorff = evalsurf.get_maximum_symmetric_surface_distance()
        return Hausdorff

    @staticmethod
    def hausdorff_robust(y_true, y_pred, scaling):
        if (len(y_true.shape) == 3):
            evalsurf = Surface(y_pred, y_true, physical_voxel_spacing=scaling)
        else:  # 2D measure
            evalsurf = Surface(y_pred, y_true, physical_voxel_spacing=scaling, is_3D=False)
        Hausdorff = evalsurf.get_percentile_surface_distance(95)
        return Hausdorff

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

def evaluate_all(path_list, metrics_without_rescaling, metrics_with_rescaling, truth_filename, result_filename, classes):
    """
    Perform evaluation in parallel for the requested cases
    """
    print('-----------------------------------------')
    print('-----------------------------------------')
    metrics = metrics_without_rescaling + metrics_with_rescaling

    pred_scores_single_class = {skey.__name__ : {} for skey in metrics}
    pred_scores_multiclass = {skey.__name__: {} for skey in metrics}

    def process_sub(subject_folder):
        subject_id = os.path.basename(os.path.dirname(subject_folder))
        print('subject folder is: ' + subject_folder)
        print('loading case: ' + subject_id)

        truth = np.int16(nib.load(os.path.join(subject_folder, truth_filename)).get_fdata())
        pred = np.int16(nib.load(os.path.join(subject_folder, result_filename)).get_fdata()).squeeze(-1)
        if truth.shape[-1] == 1:
            truth = truth.squeeze(-1)
        if pred.shape[-1] == 1:
            truth = pred.squeeze(-1)
        resolution = [1, 1]

        if(truth.shape != pred.shape):
            print("in case + " + subject_folder + " there is a dimensions mismatch")

        for score_method in metrics:
            score_key = score_method.__name__
            pred_scores_single_class[score_key][subject_id] = {}

        try:
            for class_id in classes:
                print('comparing class ' + str(class_id) + ' for input ' + subject_folder)
                class_id = int(class_id)
                if class_id == 0:
                    continue
                truth_single_class = np.zeros_like(truth)
                truth_single_class[truth==class_id] = 1
                pred_single_class = np.zeros_like(pred)
                pred_single_class[pred == class_id] = 1
                #calculate measures
                for score_method in metrics_without_rescaling:
                    pred_scores_single_class[score_method.__name__][subject_id][class_id] = (
                        score_method(truth_single_class, pred_single_class))
                for score_method in metrics_with_rescaling:
                    if len(np.nonzero(pred_single_class)[0]) != 0 and len(np.nonzero(truth_single_class)[0]) != 0:
                        pred_scores_single_class[score_method.__name__][subject_id][class_id] = (
                            score_method(truth_single_class, pred_single_class, resolution))

        except Exception as e:
            print('exception occurred when calculating evaluation in case: ' + subject_folder + '!')
            print(e)
            return None

        #calculate multiclass evaluation
        for score_method in metrics:
            pred_scores_multiclass[score_method.__name__][subject_id] = (
                np.average(list(pred_scores_single_class[score_method.__name__][subject_id].values())))

        del pred
        del truth

        #can give to Pool() less workers as a parameter
    with Pool() as pool:
        list(tqdm(pool.imap_unordered(process_sub, path_list), total=len(path_list)))

    return pred_scores_single_class, pred_scores_multiclass


def write_formula_row(df, row_ind, worksheet, formula_name):
    num_columns = df.shape[1]
    num_samples = df.shape[0]
    column_names = list(string.ascii_uppercase)
    worksheet.write('A' + str(row_ind), formula_name)
    for i in range(1, num_columns + 1):
        formula_cell = column_names[i] + str(row_ind)
        start_range = column_names[i] + '2'
        end_range = column_names[i] + str(num_samples + 1)
        formula = '=' + formula_name + '(' + start_range + ':' + end_range + ')'
        worksheet.write_formula(formula_cell, formula)

def write_to_excel(pred_scores_multiclass, pred_scores_single , excel_path, classes, mapping):
    """
    Write multiclass and single class results to excel file
    :param pred_scores_multiclass: multiclass evaluation
    :param pred_scores_single: singleclass evaluation
    :param excel_path: path to excel file with the results
    :return:
    """
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    df = pd.DataFrame.from_dict(pred_scores_multiclass)
    df = df.round(3)
    df.to_excel(writer, sheet_name='multiclass')

    worksheet = writer.sheets['multiclass']
    write_formula_row(df, df.shape[0] + 2, worksheet, 'AVERAGE')
    write_formula_row(df, df.shape[0] + 3, worksheet, 'MIN')
    write_formula_row(df, df.shape[0] + 4, worksheet, 'MAX')
    write_formula_row(df, df.shape[0] + 5, worksheet, 'STDEV.P')

    classes_results = {}
    for class_id_str in classes:
        class_id = int(class_id_str)
        if class_id == 0:
            continue
        classes_results[class_id] = {}
        for metric in pred_scores_single.keys():
            classes_results[class_id][metric] = {}
            for case_id in pred_scores_single[metric]:
                if class_id in pred_scores_single[metric][case_id]:
                    classes_results[class_id][metric][case_id] = pred_scores_single[metric][case_id][class_id]

    for class_id_str in classes:
        class_id = int(class_id_str)
        if class_id == 0:
            continue
        class_df = pd.DataFrame.from_dict(classes_results[class_id])
        try:
            class_name = mapping[class_id]
        except:
            print('class mapping for class ' + str(class_id) + ' was not found!')
        class_df.to_excel(writer, sheet_name=class_name)
        worksheet = writer.sheets[class_name]
        write_formula_row(class_df, class_df.shape[0] + 2, worksheet, 'AVERAGE')
        write_formula_row(class_df, class_df.shape[0] + 3, worksheet, 'MIN')
        write_formula_row(class_df, class_df.shape[0] + 4, worksheet, 'MAX')
        write_formula_row(class_df, class_df.shape[0] + 5, worksheet, 'STDEV.P')

    writer.close()
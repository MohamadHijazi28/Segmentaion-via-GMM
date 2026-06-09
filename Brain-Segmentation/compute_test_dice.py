import os
import glob
import numpy as np
import nibabel as nib
import json

def calculate_dice(y_true, y_pred, smooth=0.001):
    y_true_f = y_true.flatten() > 0
    y_pred_f = y_pred.flatten() > 0
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

print("Starting evaluation...")

experiments = glob.glob("experiments*/") + glob.glob("experiments*/*/*/")
results = {}

for exp in experiments:
    # Find all test subdirectories
    pred_files = glob.glob(os.path.join(exp, "Output", "*", "*", "prediction.nii.gz"))
    if not pred_files:
        pred_files = glob.glob(os.path.join(exp, "Output", "*", "prediction.nii.gz"))
        
    if not pred_files:
        continue
        
    dice_scores = []
    for pred_path in pred_files:
        truth_path = os.path.join(os.path.dirname(pred_path), "truth.nii.gz")
        if not os.path.exists(truth_path):
            continue
            
        try:
            pred = nib.load(pred_path).get_fdata()
            truth = nib.load(truth_path).get_fdata()
            
            dice = calculate_dice(truth, pred)
            dice_scores.append(dice)
        except Exception as e:
            print(f"Error evaluating {pred_path}: {e}")
            
    if dice_scores:
        results[exp] = np.mean(dice_scores)

print("\n--- Final Quantitative Evaluation (Test Set Dice) ---")
for exp, score in sorted(results.items()):
    print(f"{exp}: {score:.4f}")

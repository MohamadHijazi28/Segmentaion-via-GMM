import os
import glob
import numpy as np
import pandas as pd
from utils.utils import NiftiReadWrite

# --- Configuration ---
data_path = 'DataSet/CT'  # We must use the CT folder because it has the training masks
cluster_list = [2, 3, 4, 5]
output_dir = 'config'

os.makedirs(output_dir, exist_ok=True)


def generate_optimized_mappings():
    # Find all patient directories in the CT dataset
    patient_dirs = [d for d in glob.glob(os.path.join(data_path, '*')) if os.path.isdir(d)]
    print(f"Found {len(patient_dirs)} patients for mapping analysis.")

    for n_clusters in cluster_list:
        print(f"\n--- Analyzing {n_clusters}-class GMM ---")

        # Dictionary to track {cluster_id: {'total_pixels': X, 'overlap_pixels': Y}}
        cluster_stats = {}

        for patient_dir in patient_dirs:
            gmm_path = os.path.join(patient_dir, f'labels_gmm_{n_clusters}c.nii.gz')
            mask_path = os.path.join(patient_dir, 'mask.nii.gz')

            # Skip if files don't exist
            if not os.path.exists(gmm_path) or not os.path.exists(mask_path):
                continue

            # Load the volumes
            gmm_vol, _, _ = NiftiReadWrite.read_nifti_img_meta(gmm_path)
            mask_vol, _, _ = NiftiReadWrite.read_nifti_img_meta(mask_path)

            # Ensure the ground truth mask is strictly binary (1 for liver, 0 for background)
            mask_vol = (mask_vol > 0).astype(int)

            # Analyze each unique cluster in the GMM volume
            unique_clusters = np.unique(gmm_vol)
            for cluster_id in unique_clusters:
                cluster_id = int(cluster_id)
                if cluster_id not in cluster_stats:
                    cluster_stats[cluster_id] = {'total': 0, 'overlap': 0}

                # Create a binary mask of just THIS cluster
                cluster_mask = (gmm_vol == cluster_id)

                # Count total pixels belonging to this cluster
                cluster_stats[cluster_id]['total'] += np.sum(cluster_mask)

                # Count how many of those pixels overlap with the ground truth liver mask
                cluster_stats[cluster_id]['overlap'] += np.sum(cluster_mask & (mask_vol == 1))

        # Determine the final mapping based on the overlap ratio
        mapping_data = []
        for cluster_id, stats in sorted(cluster_stats.items()):
            if stats['total'] == 0:
                continue

            overlap_ratio = stats['overlap'] / stats['total']

            # THE RULE: If more than 50% of this cluster is inside the true liver, map to 1.
            # Otherwise, map to 0.
            is_liver = 1 if overlap_ratio > 0.5 else 0

            print(f"Cluster {cluster_id}: {overlap_ratio * 100:6.2f}% inside liver -> Mapped to {is_liver}")
            mapping_data.append({'label': cluster_id, 'mapping': is_liver})

        # Save the optimized mappings to a CSV file exactly how your U-Net expects it
        df = pd.DataFrame(mapping_data)
        csv_path = os.path.join(output_dir, f'mapping{n_clusters}c.csv')
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved {csv_path}")


if __name__ == "__main__":
    generate_optimized_mappings()
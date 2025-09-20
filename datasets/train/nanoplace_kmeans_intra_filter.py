import os
import argparse
import json
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import numpy as np
import pandas as pd 
from sklearn.cluster import AgglomerativeClustering
from analyse.dataset import * 
from analyse import DatasetAnalyzer
from .utils import read_image_paths, read_utm_coordinates, save_config
import torch 
from tqdm import tqdm
import torch.nn.functional as F


def filter_class_outliers(descriptors: np.ndarray, mad_threshold: float=2.5): 
    x = torch.as_tensor(descriptors)
    median = torch.median(x, dim=0).values
    x_normalized = F.normalize(x, p=2, dim=1)
    median_normalized = F.normalize(median.unsqueeze(0), p=2, dim=1)
    sim = (F.cosine_similarity(x_normalized, median_normalized, dim=1) + 1) / 2
    distances = 1 - sim 
    mad = torch.median(torch.abs(distances - torch.median(distances)))
    threshold = mad_threshold * mad
    is_outlier = distances > threshold
    outlier_indices = torch.nonzero(is_outlier, as_tuple=False).squeeze().cpu().numpy()
    if outlier_indices.ndim == 0:  # scalar case
        num_outliers = 0
        outlier_indices = np.array([])  # Convert to empty array for consistency
    else:  # array case
        num_outliers = len(outlier_indices)
    return outlier_indices

def filter_outlier(config: pd.DataFrame, descriptors: np.ndarray, mad_threshold: float=2.5, verbose: bool=True): 
    print(" ")
    outlier_mask = np.zeros(len(config), dtype=bool)
    # Statistics tracking
    total_samples = len(config)
    classes_with_outliers = 0
    total_outliers = 0
    class_stats = {}
    
    class_ids = config["class_id"].unique() 
    total_classes = len(class_ids)
    
    for class_id in tqdm(class_ids, desc="Filtering outliers"): 
        class_mask = config["class_id"] == class_id
        class_indices = np.where(class_mask)[0]
        class_descriptors = descriptors[class_mask]
        class_size = len(class_descriptors)
        
        outlier_indices = filter_class_outliers(class_descriptors, mad_threshold)
        
        if len(outlier_indices) > 0:
            classes_with_outliers += 1
            total_outliers += len(outlier_indices)
            
            # Map class outlier indices back to original dataframe indices
            original_outlier_indices = class_indices[outlier_indices]
            outlier_mask[original_outlier_indices] = True
            
            # Store per-class stats
            class_stats[class_id] = {
                'total': class_size,
                'outliers': len(outlier_indices),
                'percentage': (len(outlier_indices) / class_size) * 100
            }
    
    # Filter both config and descriptors
    filtered_config = config[~outlier_mask]
    filtered_descriptors = descriptors[~outlier_mask]
    
    if verbose:
        print(f"Outlier Removal Summary:")
        print(f"  Total samples: {total_samples}")
        print(f"  Outliers removed: {total_outliers} ({total_outliers/total_samples*100:.2f}%)")
        print(f"  Classes with outliers: {classes_with_outliers}/{total_classes} ({classes_with_outliers/total_classes*100:.1f}%)")
        print(f"  Remaining samples: {len(filtered_config)}")
        
        if class_stats and len(class_stats) <= 10:  # Only show details for reasonable number of classes
            print(f"\nPer-class outlier removal:")
            for class_id, stats in class_stats.items():
                print(f"  Class {class_id}: {stats['outliers']}/{stats['total']} ({stats['percentage']:.1f}%)")
        elif class_stats:
            print(f"\n{len(class_stats)} classes had outliers removed (use verbose=False to suppress details)")
    
    return filtered_config, filtered_descriptors, {
        'total_samples': total_samples,
        'total_outliers': total_outliers,
        'outlier_percentage': total_outliers/total_samples*100,
        'classes_with_outliers': classes_with_outliers,
        'total_classes': total_classes,
        'class_stats': class_stats
    }

def compute_classes(eastings, northings, class_max_distance=15.0):
    """
    Cluster points using complete linkage ensuring all points within 
    a cluster are within max_distance of each other.
    
    Parameters
    ----------
    eastings : np.ndarray
        Array of easting coordinates
    northings : np.ndarray  
        Array of northing coordinates
    max_distance : float
        Maximum distance allowed between any two points in the same cluster
        
    Returns
    -------
    cluster_labels : np.ndarray
        Array of cluster labels for each point
    """
    coords = np.stack([eastings, northings], axis=1)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=class_max_distance,
        linkage='complete',
        metric='euclidean'
    )
    
    class_ids = clustering.fit_predict(coords)
    return class_ids


def compute_defaultconfig(dataset_folder, class_max_distance=15.0) -> pd.DataFrame:
    image_paths = read_image_paths(dataset_folder)
    easting, northing = read_utm_coordinates(image_paths)
    class_ids = compute_classes(easting, northing, class_max_distance=class_max_distance)
    # Create DataFrame
    df = pd.DataFrame({
        "image_path": image_paths,
        "easting": easting,
        "northing": northing,
        "class_id": class_ids
    })
    
    # Store metadata in DataFrame attrs
    df.attrs = {
        "dataset_folder": dataset_folder,
    }

    return df


def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Compute default configuration for a dataset with spatial matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-folder",
        default="/home/oliver/datasets_drive/vpr_datasets/sf_xl/small/train",
        type=str,
        help="Path to the dataset folder containing images"
    )
    
    parser.add_argument(
        "--class-max-distance",
        type=float,
        default=15.0,
        help="Maximum distance allowed between any two points in the same class"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_path = Path(__file__).parent / "configs" / "kmeans_intra_filter" / "config.parquet"
    os.makedirs(config_path.parent, exist_ok=True)
    config = compute_defaultconfig(args.dataset_folder, class_max_distance=args.class_max_distance)
    save_config(config, config_path)
    dataset_analyzer = DatasetAnalyzer(config_path)
    descriptors = dataset_analyzer.compute_descriptors()
    config, descriptors, stats = filter_outlier(config, descriptors, mad_threshold=10.0)
    save_config(config, config_path)

    # Analyse the dataset 
    analyse_dataset = DatasetAnalyzer(config_path, max_num_classes=10)
    analyse_dataset.descriptors = descriptors 
    analyse_dataset.compute_descriptors()
    analyse_dataset.get_intra_class_sim()
    analyse_dataset.get_intra_class_sim_dist()

    analyse_dataset.plot_class_samples()
    analyse_dataset.plot_class_sim_heatmap()
    analyse_dataset.plot_intra_class_sim_dist()
    analyse_dataset.plot_intra_class_embedding()
    analyse_dataset.plot_inter_class_embedding()
    analyse_dataset.plot_inter_class_sim_dist()

    


    


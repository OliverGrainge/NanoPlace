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
from typing import List


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

def filter_classes(config: pd.DataFrame, 
                   inter_class_sim: np.ndarray, 
                   class_indices: List[int], 
                   threshold: float = 0.8,
                   removal_preference: str = 'random',
                   verbose: bool = False) -> pd.DataFrame:
    """
    Iteratively remove classes until max similarity between any pair is below threshold.
    
    Args:
        config: Configuration DataFrame with class metadata (optional, can be empty)
        inter_class_sim: Similarity matrix (n_classes x n_classes)
        class_indices: List of class indices
        threshold: Maximum allowed similarity between any two classes
        removal_preference: How to choose which class to remove from most similar pair:
            - 'random': Random choice
            - 'first': Always remove first class in pair
            - 'smaller_count': Remove class with fewer samples (requires 'count' in config)
            - 'lower_priority': Remove class with lower priority (requires 'priority' in config)
        verbose: Print removal details
        
    Returns:
        Filtered DataFrame containing only remaining classes
    """
    
    remaining_classes = set(class_indices)
    iteration = 0
    
    if verbose:
        print("=" * 70)
        print("CLASS FILTERING PROCESS")
        print("=" * 70)
        print(f"Initial classes: {len(remaining_classes)}")
        print(f"Similarity threshold: {threshold}")
        print(f"Removal preference: {removal_preference}")
        print(f"Similarity matrix shape: {inter_class_sim.shape}")
        
        # Show initial statistics
        initial_max_sim = _get_max_similarity_with_pair(list(remaining_classes), inter_class_sim, class_indices)
        print(f"Initial max similarity: {initial_max_sim['max_sim']:.4f} between classes {initial_max_sim['pair']}")
        print("\n" + "-" * 70)
    
    while len(remaining_classes) > 1:
        # Find the pair with maximum similarity among remaining classes
        max_similarity = -1
        best_pair = None
        
        remaining_list = list(remaining_classes)
        class_to_idx = {class_indices[i]: i for i in range(len(class_indices))}
        
        # Track all similarities for verbose output
        similarities = []
        
        for i, class_i in enumerate(remaining_list):
            for j in range(i + 1, len(remaining_list)):
                class_j = remaining_list[j]
                
                # Get similarity from matrix
                idx_i = class_to_idx[class_i]
                idx_j = class_to_idx[class_j]
                similarity = inter_class_sim[idx_i, idx_j]
                
                similarities.append((similarity, class_i, class_j))
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pair = (class_i, class_j)
        
        # If max similarity is below threshold, we're done
        if max_similarity <= threshold:
            if verbose:
                print(f"\n✓ FILTERING COMPLETE!")
                print(f"  Max similarity ({max_similarity:.4f}) is now below threshold ({threshold})")
                print(f"  Classes remaining: {len(remaining_classes)}")
            break
        
        iteration += 1
        
        if verbose:
            print(f"\nITERATION {iteration}")
            print(f"Classes remaining: {len(remaining_classes)}")
            
            # Show top 5 most similar pairs
            similarities.sort(reverse=True)
            print(f"Top similar pairs:")
            for i, (sim, c1, c2) in enumerate(similarities[:5]):
                marker = ">>> " if (c1, c2) == best_pair or (c2, c1) == best_pair else "    "
                print(f"{marker}Classes {c1}-{c2}: {sim:.4f}")
            
            if len(similarities) > 5:
                print(f"    ... and {len(similarities)-5} more pairs")
        
        # Decide which class to remove from the most similar pair
        class_i, class_j = best_pair
        to_remove, removal_reason = _choose_class_to_remove_verbose(class_i, class_j, config, removal_preference)
        
        # Remove the chosen class
        remaining_classes.remove(to_remove)
        
        if verbose:
            keep_class = class_j if to_remove == class_i else class_i
            print(f"  → Removing class {to_remove} (keeping {keep_class})")
            print(f"    Reason: {removal_reason}")
            
            # Show class details if available in config
            if not config.empty and 'class_id' in config.columns:
                if to_remove in config['class_id'].values:
                    removed_info = config[config['class_id'] == to_remove].iloc[0]
                    info_str = ""
                    if 'count' in config.columns:
                        info_str += f"count={removed_info.get('count', 'N/A')}, "
                    if 'priority' in config.columns:
                        info_str += f"priority={removed_info.get('priority', 'N/A')}, "
                    if info_str:
                        print(f"    Removed class info: {info_str.rstrip(', ')}")
                
                if keep_class in config['class_id'].values:
                    kept_info = config[config['class_id'] == keep_class].iloc[0]
                    info_str = ""
                    if 'count' in config.columns:
                        info_str += f"count={kept_info.get('count', 'N/A')}, "
                    if 'priority' in config.columns:
                        info_str += f"priority={kept_info.get('priority', 'N/A')}, "
                    if info_str:
                        print(f"    Kept class info: {info_str.rstrip(', ')}")
    
    result = list(remaining_classes)
    
    if verbose:
        print("\n" + "=" * 70)
        print("FILTERING SUMMARY")
        print("=" * 70)
        print(f"Classes removed: {len(class_indices) - len(result)}")
        print(f"Classes remaining: {len(result)}")
        print(f"Removal rate: {(len(class_indices) - len(result)) / len(class_indices) * 100:.1f}%")
        
        if len(result) > 1:
            # Calculate final statistics
            final_stats = _get_max_similarity_with_pair(result, inter_class_sim, class_indices)
            print(f"Final max similarity: {final_stats['max_sim']:.4f} between classes {final_stats['pair']}")
            
            # Show similarity distribution
            all_final_sims = []
            class_to_idx = {class_indices[i]: i for i in range(len(class_indices))}
            for i, class_i in enumerate(result):
                for j in range(i + 1, len(result)):
                    class_j = result[j]
                    idx_i = class_to_idx[class_i]
                    idx_j = class_to_idx[class_j]
                    all_final_sims.append(inter_class_sim[idx_i, idx_j])
            
            if all_final_sims:
                print(f"Final similarity stats:")
                print(f"  Mean: {np.mean(all_final_sims):.4f}")
                print(f"  Median: {np.median(all_final_sims):.4f}")
                print(f"  Min: {np.min(all_final_sims):.4f}")
                print(f"  Max: {np.max(all_final_sims):.4f}")
        
        print("=" * 70)

    # Return filtered config
    if config.empty or 'class_id' not in config.columns:
        return pd.DataFrame({'class_id': result})
    else:
        filtered_config = config[config["class_id"].isin(result)].copy()
        return filtered_config


def _choose_class_to_remove_verbose(class_i: int, class_j: int, config: pd.DataFrame, 
                                   preference: str) -> tuple[int, str]:
    """Choose which class to remove from a pair based on preference and return reason."""
    
    if preference == 'random':
        choice = np.random.choice([class_i, class_j])
        return choice, f"random selection from pair ({class_i}, {class_j})"
    
    elif preference == 'first':
        return class_i, f"always remove first class in pair ({class_i}, {class_j})"
    
    elif preference == 'smaller_count':
        if 'count' not in config.columns or config.empty:
            return class_i, f"smaller_count preference but no count data available, defaulting to first ({class_i})"
        
        count_i = config.loc[config['class_id'] == class_i, 'count'].iloc[0] if class_i in config['class_id'].values else 0
        count_j = config.loc[config['class_id'] == class_j, 'count'].iloc[0] if class_j in config['class_id'].values else 0
        
        if count_i <= count_j:
            return class_i, f"smaller count ({count_i} vs {count_j})"
        else:
            return class_j, f"smaller count ({count_j} vs {count_i})"
    
    elif preference == 'lower_priority':
        if 'priority' not in config.columns or config.empty:
            return class_i, f"lower_priority preference but no priority data available, defaulting to first ({class_i})"
        
        # Lower priority number = higher importance, so remove higher numbers
        priority_i = config.loc[config['class_id'] == class_i, 'priority'].iloc[0] if class_i in config['class_id'].values else float('inf')
        priority_j = config.loc[config['class_id'] == class_j, 'priority'].iloc[0] if class_j in config['class_id'].values else float('inf')
        
        if priority_i >= priority_j:
            return class_i, f"lower priority ({priority_i} vs {priority_j})"
        else:
            return class_j, f"lower priority ({priority_j} vs {priority_i})"
    
    else:
        return class_i, f"unknown preference '{preference}', defaulting to first ({class_i})"


def _get_max_similarity_with_pair(remaining_classes: List[int], 
                                 inter_class_sim: np.ndarray, 
                                 class_indices: List[int]) -> dict:
    """Get maximum similarity among remaining classes with pair information."""
    if len(remaining_classes) <= 1:
        return {'max_sim': 0.0, 'pair': None}
    
    class_to_idx = {class_indices[i]: i for i in range(len(class_indices))}
    max_sim = 0.0
    max_pair = None
    
    for i, class_i in enumerate(remaining_classes):
        for j in range(i + 1, len(remaining_classes)):
            class_j = remaining_classes[j]
            idx_i = class_to_idx[class_i]
            idx_j = class_to_idx[class_j]
            similarity = inter_class_sim[idx_i, idx_j]
            if similarity > max_sim:
                max_sim = similarity
                max_pair = (class_i, class_j)
    
    return {'max_sim': max_sim, 'pair': max_pair}

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
    df = df[df["class_id"].isin(np.arange(500))]
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
    config_path = Path(__file__).parent / "configs" / "kmeans_intra_inter_filter" / "config.parquet"
    os.makedirs(config_path.parent, exist_ok=True)
    config = compute_defaultconfig(args.dataset_folder, class_max_distance=args.class_max_distance)
    save_config(config, config_path)
    dataset_analyzer = DatasetAnalyzer(config_path)
    descriptors = dataset_analyzer.compute_descriptors()
    config, descriptors, stats = filter_outlier(config, descriptors, mad_threshold=10.0)
    class_sim = dataset_analyzer.get_inter_class_sim()
    config = filter_classes(config, class_sim, config["class_id"].unique(), threshold=0.5, verbose=True)
    save_config(config, config_path)

    # Analyse the dataset 
    #analyse_dataset = DatasetAnalyzer(config_path, max_num_classes=10)
    #analyse_dataset.descriptors = descriptors 
    #analyse_dataset.compute_descriptors()
    #analyse_dataset.get_intra_class_sim()
    #analyse_dataset.get_intra_class_sim_dist()

    #analyse_dataset.plot_class_samples()
    #analyse_dataset.plot_class_sim_heatmap()
    #analyse_dataset.plot_intra_class_sim_dist()
    #analyse_dataset.plot_intra_class_embedding()
   # analyse_dataset.plot_inter_class_embedding()
    #analyse_dataset.plot_inter_class_sim_dist()

    


    


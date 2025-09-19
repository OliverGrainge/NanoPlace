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




def compute_config(dataset_folder, class_max_distance=15.0) -> pd.DataFrame:
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
    #config = compute_config(args.dataset_folder, class_max_distance=args.class_max_distance)
    config_path = Path(__file__).parent / "configs" / "kmeans" / "config.parquet"
    #os.makedirs(config_path.parent, exist_ok=True)
    #save_config(config, config_path)


    # Analyse the dataset 
    analyse_dataset = DatasetAnalyzer(config_path)
    analyse_dataset.compute_descriptors() 
    analyse_dataset.get_intra_class_sim()
    analyse_dataset.get_intra_class_sim_dist()

    analyse_dataset.plot_class_samples()
    analyse_dataset.plot_class_sim_heatmap()
    analyse_dataset.plot_intra_class_sim_dist()
    analyse_dataset.plot_intra_class_embedding()
    analyse_dataset.plot_inter_class_embedding()
    analyse_dataset.plot_inter_class_sim_dist()

    


    


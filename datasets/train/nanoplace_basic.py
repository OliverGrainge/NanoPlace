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
from analyse import analyze_dataset

def read_image_paths(dataset_folder):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over very large folders might be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing images

    Returns
    -------
    images_paths : list[str], relative paths of images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        print(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = sorted(file.read().splitlines())
        # Sanity check that paths within the file exist
        abs_path = os.path.join(dataset_folder, images_paths[0])
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Image with path {abs_path} "
                f"does not exist within {dataset_folder}. It is likely "
                f"that the content of {file_with_paths} is wrong."
            )
    else:
        images_paths = sorted(glob(f"{dataset_folder}/**/*", recursive=True))
        images_paths = [
            os.path.relpath(p, dataset_folder)
            for p in images_paths
            if os.path.isfile(p)
            and os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        if len(images_paths) == 0:
            raise FileNotFoundError(
                f"Directory {dataset_folder} does not contain any images"
            )
    return images_paths


def read_utm_coordinates(image_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts easting and northing UTM coordinates from image paths.

    Parameters
    ----------
    image_paths : np.ndarray or list of str
        List or array of image paths containing UTM coordinates in the format ...@easting@northing...

    Returns
    -------
    easting : np.ndarray
        Array of easting coordinates.
    northing : np.ndarray
        Array of northing coordinates.
    """
    easting = np.array([float(path.split("@")[1]) for path in image_paths])
    northing = np.array([float(path.split("@")[2]) for path in image_paths])
    return easting, northing



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


def compute_defaultconfig(dataset_folder, class_max_distance=15.0, command_args=None) -> pd.DataFrame:
    image_paths = read_image_paths(dataset_folder)
    easting, northing = read_utm_coordinates(image_paths)
    class_ids = compute_classes(easting, northing, class_max_distance=class_max_distance)
    
    # Create DataFrame
    df = pd.DataFrame({
        "image_path": image_paths,
        "easting": easting,
        "northing": northing,
        "class_ids": class_ids
    })
    
    # Store metadata in DataFrame attrs
    df.attrs = {
        "dataset_folder": dataset_folder,
        "class_max_distance": class_max_distance,
        "num_images": len(image_paths),
    }
    
    # Save to parquet file in configs folder
    current_file = Path(__file__).stem
    configs_dir = Path(__file__).parent / "configs" / current_file
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = Path(dataset_folder).name
    parquet_filename = f"config.parquet"
    parquet_path = configs_dir / parquet_filename

    # Save to parquet (attrs will be preserved)
    df.to_parquet(parquet_path, index=False)
    
    analyze_dataset(parquet_path)
    print(f"Default config saved to: {parquet_path}")
    print(f"Metadata stored in DataFrame attrs")
    
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
    
    # Convert args to dictionary for metadata
    command_args = {
        "dataset_folder": args.dataset_folder,
        "class_max_distance": args.class_max_distance
    }
    
    compute_defaultconfig(args.dataset_folder, class_max_distance=args.class_max_distance, command_args=command_args)
    


    


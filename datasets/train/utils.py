import os
from glob import glob
import numpy as np
import pandas as pd 

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




def save_config(df: pd.DataFrame, config_path: str): 
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    df.to_parquet(config_path, index=False)
    return config_path


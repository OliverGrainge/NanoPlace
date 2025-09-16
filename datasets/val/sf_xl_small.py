
import yaml
from .base import ValDataset
from sklearn.neighbors import NearestNeighbors
import os 
import numpy as np 
from typing import Optional
from torchvision import transforms

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


def compute_ground_truth(query_paths, database_paths, radius=25.0): 
    """
    For each query, find all database images within the given radius (in meters).
    Returns a list of arrays, where each array contains the indices of database images within the radius for that query.
    """
    query_easting, query_northing = read_utm_coordinates(query_paths)
    database_easting, database_northing = read_utm_coordinates(database_paths)
    query_utm = np.stack([query_easting, query_northing], axis=1)
    database_utm = np.stack([database_easting, database_northing], axis=1)
    nbrs = NearestNeighbors(n_jobs=-1, radius=radius).fit(database_utm)
    indices = nbrs.radius_neighbors(query_utm, return_distance=False)
    ground_truth = indices
    return ground_truth


class SFXLSmall(ValDataset): 
    def __init__(self, transform: Optional[transforms.Compose] = None): 
        config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
        queries_dir = os.path.join(config["datasets"]["sf_xl_small"], "test/queries_v1")
        database_dir = os.path.join(config["datasets"]["sf_xl_small"], "test/database")
        queries_paths = [os.path.join(queries_dir, fname) for fname in os.listdir(queries_dir)]
        database_paths = [os.path.join(database_dir, fname) for fname in os.listdir(database_dir)]
        ground_truth = compute_ground_truth(queries_paths, database_paths) 
        super().__init__(queries_paths, database_paths, ground_truth, transform=transform)


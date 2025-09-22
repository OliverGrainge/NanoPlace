import dataclasses

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
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


class ValDataset(Dataset):
    def __init__(
        self,
        queries_paths: np.ndarray,
        database_paths: np.ndarray,
        ground_truth: np.ndarray,
        transform=None,
    ):
        self.queries_paths = queries_paths
        self.database_paths = database_paths
        self.image_paths = np.concatenate([self.queries_paths, self.database_paths])
        self.queries_num = len(self.queries_paths)
        self.database_num = len(self.database_paths)
        self.images_num = self.queries_num + self.database_num

        self.ground_truth = ground_truth
        self.transform = (
            transform if transform is not None else self._default_transform()
        )

    def _default_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform

    def __len__(self):
        return self.images_num

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, idx

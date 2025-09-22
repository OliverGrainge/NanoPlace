import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from datasets.train.analyzer.base import AnalyzerStep


class InterUTMDist(AnalyzerStep):
    def __init__(self):
        super().__init__()

    def __call__(self, dataconfig: pd.DataFrame, descriptors: np.ndarray):
        # Compute class centroids in UTM space
        easting = dataconfig["easting"].values
        northing = dataconfig["northing"].values
        utms = np.column_stack([easting, northing])
        class_ids = dataconfig["class_id"].values
        unique_class_ids = np.unique(class_ids)
        class_centroids = []
        for class_id in tqdm(unique_class_ids, desc="Calculating Inter UTM Centroids"):
            indices = np.where(class_ids == class_id)[0]
            utms_class = utms[indices]
            centroid = np.mean(utms_class, axis=0)
            class_centroids.append(centroid)
        class_centroids = np.array(class_centroids)

        dist_matrix = squareform(pdist(class_centroids))
        dataconfig.attrs["stats"]["inter_utmdist_matrix"] = dist_matrix
        return dataconfig, descriptors

    def name(self) -> str:
        return "InterUTMDist()"

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from datasets.train.analyzer.base import AnalyzerStep


class IntraUTMDist(AnalyzerStep):
    def __init__(self):
        super().__init__()

    def __call__(self, dataconfig: pd.DataFrame, descriptors: np.ndarray):
        class_mean_dists = []
        class_std_dists = []
        easting = dataconfig["easting"].values
        northing = dataconfig["northing"].values
        utms = np.column_stack([easting, northing])
        class_ids = dataconfig["class_id"].values
        for class_id in tqdm(
            np.unique(class_ids), desc="Calculating Intra UTM Distribution"
        ):
            indices = np.where(class_ids == class_id)[0]
            utms_class = utms[indices]
            dist_matrix = squareform(pdist(utms_class))
            # Exclude self-distances (diagonal) from calculations
            triu_indices = np.triu_indices_from(dist_matrix, k=1)
            upper_vals = dist_matrix[triu_indices]
            if upper_vals.size > 0:
                class_mean_dist = np.mean(upper_vals)
                class_std_dist = np.std(upper_vals)
            else:
                class_mean_dist = np.nan
                class_std_dist = np.nan
            class_mean_dists.append(class_mean_dist)
            class_std_dists.append(class_std_dist)
        dataconfig.attrs["stats"]["intra_utmdist_mean"] = class_mean_dists
        dataconfig.attrs["stats"]["intra_utmdist_std"] = class_std_dists
        return dataconfig, descriptors

    def name(self) -> str:
        return "IntraUTMDist()"

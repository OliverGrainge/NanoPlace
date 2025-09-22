from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from datasets.train.pipeline.base import CurationStep


class MinNumbPerClass(CurationStep):
    def __init__(self, min_num_per_class: int = 4):
        super().__init__()  # Call parent constructor

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        utms = np.column_stack(
            [dataconfig["easting"].values, dataconfig["northing"].values]
        )
        labels = self.clustering.fit_predict(utms)
        dataconfig = dataconfig.copy()  # Avoid modifying original
        dataconfig["class_id"] = labels
        return dataconfig, descriptors

    def name(self) -> str:
        dist_thresh_str = str(self.distance_threshold)
        return "AggroClust(distance_threshold=" + dist_thresh_str + ")"

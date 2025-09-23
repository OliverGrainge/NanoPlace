from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from typing import Union
from datasets.train.pipeline.base import CurationStep


class IntraClassMADFilter(CurationStep):
    def __init__(self, mad_threshold: float = 2.5):
        super().__init__()
        self.mad_threshold = mad_threshold

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[Union[np.ndarray, np.memmap]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        class_ids = dataconfig["class_id"].unique()
        keep_mask = np.zeros(len(dataconfig), dtype=bool)

        for class_id in class_ids:
            # Get indices for this class
            class_mask = dataconfig["class_id"] == class_id
            indices = np.where(class_mask)[0]
            class_descriptors = descriptors[indices]

            # Skip classes with only one sample
            if len(class_descriptors) <= 1:
                keep_mask[indices] = True
                continue

            # Compute cosine distances from each sample to class centroid
            class_centroid = class_descriptors.mean(axis=0).reshape(1, -1)
            cosine_dists = cosine_distances(class_descriptors, class_centroid).flatten()

            # Apply MAD filtering on cosine distances
            median_distance = np.median(cosine_dists)
            mad = np.median(np.abs(cosine_dists - median_distance))

            # Handle case where mad is 0 (all distances are the same)
            if mad == 0:
                keep_mask[indices] = True
                continue

            # Keep samples within mad_threshold * mad of the median distance
            distance_filter = (
                np.abs(cosine_dists - median_distance) < self.mad_threshold * mad
            )

            # Update keep_mask for this class
            keep_mask[indices] = distance_filter

        # Apply the mask
        filtered_dataconfig = dataconfig[keep_mask].reset_index(drop=True)
        filtered_descriptors = descriptors[keep_mask]

        if type(filtered_descriptors) == np.memmap:
            filtered_descriptors.flush()

        return filtered_dataconfig, filtered_descriptors

    def name(self) -> str:
        return f"IntraClassMADFilter(mad_threshold={self.mad_threshold})"

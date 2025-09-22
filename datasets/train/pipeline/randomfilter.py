from typing import Optional, Tuple

import numpy as np
import pandas as pd

from datasets.train.pipeline.base import CurationStep


class RandomClassFilter(CurationStep):
    def __init__(self, ratio: float = 0.05):
        super().__init__()
        self.ratio = ratio

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        class_ids = dataconfig["class_id"].unique()
        num_to_keep = int(len(class_ids) * self.ratio)
        if num_to_keep < 1:
            num_to_keep = 1
        keep_class_ids = np.random.choice(class_ids, size=num_to_keep, replace=False)
        mask = dataconfig["class_id"].isin(keep_class_ids)
        dataconfig_filtered = dataconfig[mask].reset_index(drop=True)
        if descriptors is not None:
            descriptors_filtered = descriptors[mask.values]
        else:
            descriptors_filtered = None
        return dataconfig_filtered, descriptors_filtered

    def name(self) -> str:
        return f"RandomClassFilter(ratio={self.ratio})"

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from datasets.train.pipeline.base import CurationStep


class MinNumPerClass(CurationStep):
    def __init__(self, min_num_per_class: int = 4):
        super().__init__()  # Call parent constructor
        self.min_num_per_class = min_num_per_class

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        class_counts = dataconfig["class_id"].value_counts()
        valid_classes = class_counts[
            class_counts >= self.min_num_per_class
        ].index.tolist()
        dataconfig = dataconfig[dataconfig["class_id"].isin(valid_classes)].copy()
        return dataconfig, descriptors

    def name(self) -> str:
        min_num_per_class_str = str(self.min_num_per_class)
        return "MinNumbPerClass(min_num_per_class=" + min_num_per_class_str + ")"

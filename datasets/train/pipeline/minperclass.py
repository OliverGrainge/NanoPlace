from typing import Optional, Tuple

import numpy as np
import pandas as pd

from datasets.train.pipeline.base import CurationStep
from typing import Union

class MinPerClass(CurationStep):
    def __init__(self, n_instances: int = 4):
        super().__init__()  # Call parent constructor
        self.n_instances = n_instances

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[Union[np.ndarray, np.memmap]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        class_counts = dataconfig["class_id"].value_counts()
        
        # Find classes that have at least n_instances samples
        valid_classes = class_counts[class_counts >= self.n_instances].index
        
        # Filter dataconfig to keep only classes with sufficient instances
        filtered_dataconfig = dataconfig[dataconfig["class_id"].isin(valid_classes)].reset_index(drop=True)
        
        return filtered_dataconfig, descriptors

    def name(self) -> str:
        return f"MinPerClass(n_instances={self.n_instances})"
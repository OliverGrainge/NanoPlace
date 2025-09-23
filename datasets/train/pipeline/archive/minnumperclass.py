from typing import Optional, Tuple

import numpy as np
import pandas as pd

from datasets.train.pipeline.base import CurationStep
from typing import Union

class MinNumPerClass(CurationStep):
    def __init__(self, min_num_per_class: int = 4):
        super().__init__()  # Call parent constructor
        self.min_num_per_class = min_num_per_class

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[Union[np.ndarray, np.memmap]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        class_counts = dataconfig["class_id"].value_counts()
        valid_classes = class_counts[
            class_counts >= self.min_num_per_class
        ].index.tolist()
        
        # Create mask for valid classes
        valid_mask = dataconfig["class_id"].isin(valid_classes)
        
        # Filter dataconfig and reset index
        filtered_dataconfig = dataconfig[valid_mask].copy().reset_index(drop=True)
        
        # Filter descriptors to match the filtered dataconfig
        if descriptors is not None:
            if type(descriptors) == np.memmap:
                descriptors.flush()
            # Get the indices of the kept rows
            kept_indices = np.where(valid_mask)[0]
            filtered_descriptors = descriptors[kept_indices]
        else:
            filtered_descriptors = None
        
        return filtered_dataconfig, filtered_descriptors

    def name(self) -> str:
        min_num_per_class_str = str(self.min_num_per_class)
        return "MinNumbPerClass(min_num_per_class=" + min_num_per_class_str + ")"
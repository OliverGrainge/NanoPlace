from typing import Optional, Tuple

import numpy as np
import pandas as pd
from typing import Union

from datasets.train.pipeline.base import CurationStep


class NClassSelection(CurationStep):
    def __init__(self, n: int = 100):
        super().__init__()  # Call parent constructor
        self.n = n

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[Union[np.ndarray, np.memmap]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        if self.n >= len(dataconfig["class_id"].unique()):
            return dataconfig, descriptors
        else:
            # Randomly select n classes
            class_ids = dataconfig["class_id"].unique()
            selected_class_ids = np.random.choice(class_ids, size=self.n, replace=False)
            
            # Filter dataconfig to keep only selected classes
            keep_mask = dataconfig["class_id"].isin(selected_class_ids)
            filtered_dataconfig = dataconfig[keep_mask].reset_index(drop=True)
            
            # Filter descriptors to match the filtered dataconfig
            if descriptors is not None:
                keep_positions = np.where(keep_mask)[0]
                filtered_descriptors = descriptors[keep_positions]
            else:
                filtered_descriptors = None
            
            if type(filtered_descriptors) == np.memmap:
                filtered_descriptors.flush()

            return filtered_dataconfig, filtered_descriptors

    def name(self) -> str:
        return "NClassSelection (n=" + str(self.n) + ")"
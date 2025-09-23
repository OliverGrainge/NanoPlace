from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets.train.pipeline.base import CurationStep


class RandomClassSelection(CurationStep):
    def __init__(
        self,
        n_classes: int = 500,
        seed: Optional[int] = None
    ):
        super().__init__()  # Call parent constructor
        self.n_classes = n_classes 
        self.seed = seed
        # Create a random number generator with the seed for reproducible results
        self.rng = np.random.default_rng(seed)

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        class_ids = dataconfig["class_id"].unique()
        if len(class_ids) <= self.n_classes:
            return dataconfig, descriptors
        else:
            selected_class_ids = self.rng.choice(class_ids, size=self.n_classes, replace=False)
            keep_mask = dataconfig["class_id"].isin(selected_class_ids)
            filtered_dataconfig = dataconfig[keep_mask].reset_index(drop=True)
            return filtered_dataconfig, descriptors

    def name(self) -> str:
        return f"RandomClassSelection(n_classes={self.n_classes}, seed={self.seed})"
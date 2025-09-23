from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets.train.pipeline.base import CurationStep


class RandomInstanceSelection(CurationStep):
    def __init__(
        self,
        n_instances: int = 500, 
        seed: Optional[int] = None
    ):
        super().__init__()  # Call parent constructor
        self.n_instances = n_instances
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
    
        class_ids = dataconfig["class_id"].unique()
        all_selected_idxs = []
        
        for class_id in class_ids: 
            class_idxs = dataconfig[dataconfig["class_id"] == class_id].index
            
            # Handle case where class has fewer instances than requested
            n_to_select = min(len(class_idxs), self.n_instances)
            selected_idxs = np.random.choice(class_idxs, size=n_to_select, replace=False)
            all_selected_idxs.extend(selected_idxs)
        
        # Filter dataconfig to keep only selected indices
        filtered_dataconfig = dataconfig.loc[all_selected_idxs].reset_index(drop=True)
        
        return filtered_dataconfig, descriptors

    def name(self) -> str:
        return f"RandomInstanceSelection(n_instances={self.n_instances}, seed={self.seed})"
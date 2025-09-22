from typing import Optional, Tuple

import numpy as np
import pandas as pd

from datasets.train.pipeline.base import CurationStep


class RandomFilter(CurationStep):
    def __init__(self, ratio: float = 0.05, min_images_per_class: int = 4):
        super().__init__()
        self.ratio = ratio
        self.min_images_per_class = min_images_per_class

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        original_size = len(dataconfig)

        # First filter classes by square root
        class_filter_num = int(self.ratio**0.6 * len(dataconfig["class_id"].unique()))
        keep_classes = np.random.choice(
            dataconfig["class_id"].unique(), class_filter_num, replace=False
        )
        dataconfig = dataconfig[dataconfig["class_id"].isin(keep_classes)]

        # Calculate target number of samples based on original dataset size
        target_samples = int(self.ratio * original_size)

        # If we already have fewer samples than target, return as is
        if len(dataconfig) <= target_samples:
            if descriptors is not None:
                descriptors = descriptors[dataconfig.index]
            return dataconfig, descriptors

        # Identify classes with <= min_images_per_class (protected classes)
        class_counts = dataconfig["class_id"].value_counts()
        protected_mask = dataconfig["class_id"].isin(
            class_counts[class_counts <= self.min_images_per_class].index
        )

        # Separate protected and removable samples
        protected_indices = dataconfig[protected_mask].index
        removable_indices = dataconfig[~protected_mask].index

        # Calculate how many more samples we need from removable classes
        samples_needed = target_samples - len(protected_indices)

        if samples_needed > 0 and len(removable_indices) > 0:
            # Randomly sample from removable indices
            samples_to_keep = min(samples_needed, len(removable_indices))
            kept_removable_indices = np.random.choice(
                removable_indices, samples_to_keep, replace=False
            )
            final_indices = np.concatenate([protected_indices, kept_removable_indices])
        else:
            final_indices = protected_indices

        # Filter dataconfig and descriptors
        final_dataconfig = dataconfig.loc[final_indices]

        if descriptors is not None:
            final_descriptors = descriptors[final_indices]
        else:
            final_descriptors = descriptors

        return final_dataconfig, final_descriptors

    def name(self) -> str:
        return f"RandomFilter(ratio={self.ratio}, min_images_per_class={self.min_images_per_class})"

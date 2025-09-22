import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from datasets.train.analyzer.base import AnalyzerStep


class ClassSamples(AnalyzerStep):
    def __init__(self):
        super().__init__()

    def __call__(self, dataconfig: pd.DataFrame, descriptors: np.ndarray):
        """
        Sample 5 random classes and then sample 5 image paths from each class.
        Stores the results in dataconfig.attrs["stats"]["class_samples"].
        """
        # Get unique class IDs
        unique_classes = dataconfig["class_id"].unique()

        # Sample 5 random classes (with replacement if fewer than 5 classes exist)
        sampled_class_ids = np.random.choice(
            unique_classes,
            size=min(6, len(unique_classes)),
            replace=len(unique_classes) < 6,
        )

        # Initialize the stats structure if it doesn't exist
        if "stats" not in dataconfig.attrs:
            dataconfig.attrs["stats"] = {}
        dataconfig.attrs["stats"]["class_samples"] = {}

        # For each sampled class, get  image paths
        for class_id in sampled_class_ids:
            class_data = dataconfig[dataconfig["class_id"] == class_id]

            # Sample image paths (with replacement if fewer than 6 samples exist)
            sampled_images = class_data.sample(
                n=min(6, len(class_data)), replace=len(class_data) < 6
            )
            image_paths = sampled_images["image_path"].tolist()

            dataconfig.attrs["stats"]["class_samples"][class_id] = image_paths
        return dataconfig, descriptors

    def name(self) -> str:
        return "ClassSamples()"

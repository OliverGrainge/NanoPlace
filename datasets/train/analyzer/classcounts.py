import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from datasets.train.analyzer.base import AnalyzerStep


class ClassCounts(AnalyzerStep):
    def __init__(self):
        super().__init__()

    def __call__(self, dataconfig: pd.DataFrame, descriptors: np.ndarray):
        dataconfig.attrs["stats"]
        class_counts = dataconfig["class_id"].value_counts()
        dataconfig.attrs["stats"]["class_counts"] = class_counts.to_dict()
        return dataconfig, descriptors

    def name(self) -> str:
        return "NumClasses()"

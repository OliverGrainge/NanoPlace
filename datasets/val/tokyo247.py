import os
from typing import Optional

import numpy as np
import yaml
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms

from .base import ValDataset, compute_ground_truth


class Tokyo247(ValDataset):
    def __init__(self, transform: Optional[transforms.Compose] = None):
        config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
        queries_dir = os.path.join(
            config["datasets"]["tokyo247"], "images/test/queries"
        )
        database_dir = os.path.join(
            config["datasets"]["tokyo247"], "images/test/database"
        )
        queries_paths = [
            os.path.join(queries_dir, fname) for fname in os.listdir(queries_dir)
        ]
        database_paths = [
            os.path.join(database_dir, fname) for fname in os.listdir(database_dir)
        ]
        ground_truth = compute_ground_truth(queries_paths, database_paths)
        super().__init__(
            queries_paths, database_paths, ground_truth, transform=transform
        )

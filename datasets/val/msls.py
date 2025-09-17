
import yaml
from .base import ValDataset
from sklearn.neighbors import NearestNeighbors
import os 
import numpy as np 
from typing import Optional
from torchvision import transforms
from .base import compute_ground_truth


class MSLS(ValDataset): 
    def __init__(self, transform: Optional[transforms.Compose] = None): 
        config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
        queries_dir = os.path.join(config["datasets"]["msls"], "val/queries")
        database_dir = os.path.join(config["datasets"]["msls"], "val/database")
        queries_paths = [os.path.join(queries_dir, fname) for fname in os.listdir(queries_dir)]
        database_paths = [os.path.join(database_dir, fname) for fname in os.listdir(database_dir)]
        ground_truth = compute_ground_truth(queries_paths, database_paths) 
        super().__init__(queries_paths, database_paths, ground_truth, transform=transform)

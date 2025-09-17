
import yaml
from .base import ValDataset
from sklearn.neighbors import NearestNeighbors
import os 
import numpy as np 
from typing import Optional
from torchvision import transforms
from .base import compute_ground_truth


def _get_queryset(query_set: Optional[str] = None): 
    if query_set is None: 
        return "queries"
    else: 
        return f"queries_{query_set}"

class SVOX(ValDataset): 
    def __init__(self, transform: Optional[transforms.Compose] = None, query_set: str = ""): 
        config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
        queries_dir = os.path.join(config["datasets"]["svox"], f"images/test/{_get_queryset(query_set)}")
        database_dir = os.path.join(config["datasets"]["svox"], "images/test/gallery")
        queries_paths = [os.path.join(queries_dir, fname) for fname in os.listdir(queries_dir)]
        database_paths = [os.path.join(database_dir, fname) for fname in os.listdir(database_dir)]
        ground_truth = compute_ground_truth(queries_paths, database_paths) 
        super().__init__(queries_paths, database_paths, ground_truth, transform=transform)


class SVOXNight(SVOX): 
    def __init__(self, transform: Optional[transforms.Compose] = None): 
        super().__init__(transform, query_set="night")


class SVOXOvercast(SVOX): 
    def __init__(self, transform: Optional[transforms.Compose] = None): 
        super().__init__(transform, query_set="overcast")


class SVOXRain(SVOX): 
    def __init__(self, transform: Optional[transforms.Compose] = None): 
        super().__init__(transform, query_set="rain")


class SVOXSnow(SVOX): 
    def __init__(self, transform: Optional[transforms.Compose] = None): 
        super().__init__(transform, query_set="snow")

class SVOXSun(SVOX): 
    def __init__(self, transform: Optional[transforms.Compose] = None): 
        super().__init__(transform, query_set="sun")
        

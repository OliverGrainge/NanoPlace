import torch 
from eval import compute_descriptors 
from models import get_model 
import pandas as pd
import numpy as np
from typing import List, Optional 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image 


class ImageDataset(Dataset): 
    def __init__(self, image_paths: List[str], transform: Optional[transforms.Compose]):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self): 
        return len(self.image_paths) 

    def __getitem__(self, idx): 
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, idx 
    

def compute_dataconfig_descriptors(dataconfig: pd.DataFrame, model_name: str, desc_dtype: np.dtype=np.float16, batch_size: int=32, num_workers: int=4, pbar: bool=True) -> np.ndarray:
    """
    Compute descriptors for a dataconfig.
    """
    model = get_model(model_name)
    model.eval()
    dataset = ImageDataset(dataconfig["image_path"].tolist(), model.transform)
    descriptors = compute_descriptors(model=model, dataset=dataset, desc_dtype=desc_dtype, batch_size=batch_size, num_workers=num_workers, pbar=pbar)
    return descriptors


def read_images(directory: str) -> List[str]:
    """
    Recursively find all image files in the given directory and return their paths as a list.
    """
    import os

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths


def read_gmberton_utm(image_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    easting = np.array([float(path.split("@")[1]) for path in image_paths])
    northing = np.array([float(path.split("@")[2]) for path in image_paths])
    utms = np.stack([easting, northing], axis=1)
    return utms


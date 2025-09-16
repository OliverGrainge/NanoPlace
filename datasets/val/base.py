import torch 
from torch.utils.data import Dataset
import dataclasses
import numpy as np
from torchvision import transforms
from PIL import Image


class ValDataset(Dataset):
    def __init__(self, queries_paths: np.ndarray, database_paths: np.ndarray, ground_truth: np.ndarray, transform=None):
        self.queries_paths = queries_paths
        self.database_paths = database_paths
        self.image_paths = np.concatenate([self.queries_paths, self.database_paths])
        self.queries_num = len(self.queries_paths)
        self.database_num = len(self.database_paths)
        self.images_num = self.queries_num + self.database_num
        
        self.ground_truth = ground_truth
        self.transform = transform if transform is not None else self._default_transform()

    def _default_transform(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def __len__(self):
        return self.images_num
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, idx
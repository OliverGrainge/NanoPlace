import pandas as pd 
from collections import defaultdict
import numpy as np 
from typing import Dict
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional
from torch.utils.data import DataLoader

class ImageDataset(Dataset): 
    def __init__(self, image_paths: np.ndarray, transform: Optional[transforms.Compose] = None): 
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self): 
        return len(self.image_paths)
    
    def __getitem__(self, idx): 
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, idx

        
def get_intra_class_sim(dataconfig: pd.DataFrame, descriptors: np.ndarray): 
    intra_class_sim = defaultdict(list) 
    class_ids = dataconfig['class_id'].to_numpy()
    for class_id in np.unique(class_ids):
        class_descriptors = descriptors[class_ids == class_id] 
        sim_matrix = np.dot(class_descriptors, class_descriptors.T) 
        intra_class_sim[class_id] = sim_matrix 
    return intra_class_sim


def get_inter_class_sim(dataconfig: pd.DataFrame, descriptors: np.ndarray): 
    unique_classes = dataconfig['class_id'].unique()
    place_desc = np.zeros((len(unique_classes), descriptors.shape[1]))
    
    # Create mapping from class_id to sequential index
    class_id_to_idx = {class_id: idx for idx, class_id in enumerate(unique_classes)}
    
    for class_id in unique_classes:
        class_idxs = dataconfig[dataconfig['class_id'] == class_id].index
        class_descriptors = descriptors[class_idxs]
        place_desc[class_id_to_idx[class_id]] = class_descriptors.mean(axis=0)
    
    inter_class_sim = np.dot(place_desc, place_desc.T)
    return inter_class_sim


def get_intra_class_sim_dist(intra_class_sim: Dict[int, np.ndarray]):
    means, stds = [], []
    for class_id, sim_matrix in intra_class_sim.items():
        # Extract upper-triangular (excluding diagonal) similarity values for this class
        s = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
        means.append(s.mean())
        stds.append(s.std())
    return means, stds


def get_inter_class_sim_dist(inter_class_sim: np.ndarray): 
    s = inter_class_sim[np.triu_indices(inter_class_sim.shape[0], k=1)]
    return s.mean(), s.std()
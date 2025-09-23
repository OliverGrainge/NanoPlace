from .base import CurationStep 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
from typing import Union
from k_means_constrained import KMeansConstrained

class ConfusableClassSelection(CurationStep):
    def __init__(self, n_classes: int = 1000):
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, dataconfig: pd.DataFrame, descriptors: np.memmap):
        unique_class_ids = dataconfig["class_id"].unique()
        
        # If we already have fewer classes than requested, return all
        if len(unique_class_ids) <= self.n_classes:
            return dataconfig, descriptors
        
        # Create mapping: class_id -> array index
        class_id_to_idx = {class_id: idx for idx, class_id in enumerate(unique_class_ids)}
        
        all_class_descriptors = np.zeros((len(unique_class_ids), len(descriptors[0])))
        
        # Compute class centroids (mean descriptor per class)
        for class_id in tqdm(unique_class_ids, desc="Computing class descriptors"): 
            class_mask = dataconfig["class_id"] == class_id
            class_positions = np.where(class_mask)[0]
            class_rows = dataconfig.iloc[class_positions]
            class_image_ids = class_rows['image_id'].values
            class_descriptors = descriptors[class_image_ids]
            class_descriptor = class_descriptors.mean(axis=0)
            
            # Use mapping to get correct array index
            array_idx = class_id_to_idx[class_id]
            all_class_descriptors[array_idx] = class_descriptor

        # Compute similarity matrix between class centroids
        sim_mat = np.dot(all_class_descriptors, all_class_descriptors.T)
        
        # Find all pairwise similarities (avoiding diagonal and duplicates)
        class_similarities = []
        n_classes_total = len(unique_class_ids)
        
        for i in range(n_classes_total):
            for j in range(i+1, n_classes_total):
                similarity = sim_mat[i, j]  # Higher similarity = more confusable
                class_id_a = unique_class_ids[i]
                class_id_b = unique_class_ids[j]
                class_similarities.append((similarity, class_id_a, class_id_b))
        
        # Sort by highest similarity (most confusable pairs first)
        class_similarities.sort(reverse=True)
        
        # Greedily select classes from most confusable pairs
        selected_classes = set()
        for similarity, class_a, class_b in tqdm(class_similarities, desc="Selecting confusable classes"):
            if len(selected_classes) >= self.n_classes:
                break
            
            # Add class_a if not already selected
            if class_a not in selected_classes:
                selected_classes.add(class_a)
            
            # Add class_b if not already selected and we still need more classes
            if class_b not in selected_classes and len(selected_classes) < self.n_classes:
                selected_classes.add(class_b)
        
        # Filter dataconfig to only include selected classes
        filtered_dataconfig = dataconfig[dataconfig["class_id"].isin(selected_classes)].reset_index(drop=True)
        
        print(f"Selected {len(selected_classes)} most confusable classes from {len(unique_class_ids)} total classes")
        
        return filtered_dataconfig, descriptors 
    
    def name(self) -> str: 
        return f"ConfusableClassSelection(n_classes={str(self.n_classes)})"
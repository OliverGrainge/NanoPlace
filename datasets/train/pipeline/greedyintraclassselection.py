from .base import CurationStep 
import pandas as pd 
import numpy as np 
import torch
from tqdm import tqdm 
from typing import Union

class GreedyIntraClassSelection(CurationStep):
    def __init__(self, n_instances: int = 4):
        super().__init__()
        self.n_instances = n_instances
        
        # Set up device for GPU acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available()

    def __call__(self, dataconfig: pd.DataFrame, descriptors: np.memmap):
        selected_indices = []  # These will be positional indices in dataconfig
        
        for class_id in tqdm(dataconfig["class_id"].unique(), desc="Selecting diverse class samples"): 
            class_mask = dataconfig["class_id"] == class_id
            class_positions = np.where(class_mask)[0]  # Positional indices in dataconfig
            class_rows = dataconfig.iloc[class_positions]  # The actual dataconfig rows for this class
            class_image_ids = class_rows['image_id'].values  # The image_ids for this class
            class_descriptors = descriptors[class_image_ids]  # Get descriptors using image_ids
            
            if len(class_descriptors) <= self.n_instances:
                # If class has fewer or equal samples than needed, take all
                selected_indices.extend(class_positions.tolist())
            else:
                # Use GPU-accelerated computation if available
                if self.use_gpu:
                    # Convert to torch tensor and move to GPU
                    descriptors_tensor = torch.from_numpy(class_descriptors).float().to(self.device)
                    
                    # Compute similarity matrix on GPU
                    sim_matrix = torch.mm(descriptors_tensor, descriptors_tensor.T)
                    # Convert similarity to distance
                    distance_matrix = 1 - sim_matrix
                    
                    # Move back to CPU for numpy operations
                    distance_matrix = distance_matrix.cpu().numpy()
                else:
                    # Fallback to numpy computation
                    sim_matrix = np.dot(class_descriptors, class_descriptors.T)
                    distance_matrix = 1 - sim_matrix
                
                # Select diverse samples using greedy algorithm
                diverse_local_indices = self._greedy_diverse_selection(
                    distance_matrix, self.n_instances
                )
                
                # Convert local indices back to position-based indices in dataconfig
                diverse_positions = class_positions[diverse_local_indices]
                selected_indices.extend(diverse_positions.tolist())
        
        # Filter dataconfig to keep only selected samples
        selected_indices = np.array(selected_indices)
        filtered_dataconfig = dataconfig.iloc[selected_indices].reset_index(drop=True)
        
        return filtered_dataconfig, descriptors
    
    def _greedy_diverse_selection(self, distance_matrix, n_select):
        """
        Greedy algorithm to select n_select most diverse samples.
        Selects samples that are furthest apart from each other.
        
        Args:
            distance_matrix: Square matrix of pairwise distances
            n_select: Number of samples to select
            
        Returns:
            List of selected indices (local to the distance matrix)
        """
        n_samples = len(distance_matrix)
        if n_samples <= n_select:
            return list(range(n_samples))
        
        selected = []
        remaining = list(range(n_samples))
        
        # Use GPU acceleration for mean calculation if available
        if self.use_gpu:
            distance_tensor = torch.from_numpy(distance_matrix).float().to(self.device)
            avg_distances = torch.mean(distance_tensor, dim=1).cpu().numpy()
        else:
            avg_distances = np.mean(distance_matrix, axis=1)
        
        # Start with the sample that has maximum average distance to all others
        first_idx = np.argmax(avg_distances)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Greedily select remaining samples
        for _ in range(n_select - 1):
            if not remaining:
                break
                
            best_idx = None
            best_min_dist = -1
            
            # For each remaining sample, find its minimum distance to any selected sample
            for idx in remaining:
                min_dist_to_selected = min(distance_matrix[idx][s] for s in selected)
                
                # Choose the sample with maximum minimum distance 
                # (i.e., furthest from its nearest selected neighbor)
                if min_dist_to_selected > best_min_dist:
                    best_min_dist = min_dist_to_selected
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return selected

    def name(self) -> str:
        device_info = f" (GPU: {self.use_gpu})"
        return "GreedyDiverseClassSelection(n_instances=" + str(self.n_instances) + ")" + device_info
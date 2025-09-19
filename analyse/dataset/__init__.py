import os
import numpy as np
import pandas as pd

from eval import compute_descriptors
from models import get_model
from analyse.dataset.plots import plot_class_samples, plot_class_sim_heatmap, plot_intra_class_sim_dist, plot_intra_class_embedding, plot_inter_class_embedding, plot_inter_class_sim_dist
from analyse.dataset.utils import get_intra_class_sim, get_intra_class_sim_dist, ImageDataset, get_inter_class_sim, get_inter_class_sim_dist
from typing import Optional

class DatasetAnalyzer: 
    """Analyzes dataset characteristics using visual embeddings."""
    
    def __init__(self, dataconfig_path: str, model_name: str = "cosplace", 
                 batch_size: int = 32, num_workers: int = 8, pbar: bool = True, max_num_classes: Optional[int] = None): 
        self.dataconfig = pd.read_parquet(dataconfig_path)
        self.dataconfig_path = dataconfig_path
        self.dataset_folder = self.dataconfig.attrs["dataset_folder"]
        self.eval_model = get_model(model_name, pretrained=True)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pbar = pbar
        self.max_plot_images = 8  # Default value for plotting
        self.max_num_classes = max_num_classes
        # Computed attributes
        self.intra_class_sim = None
        self.intra_class_sim_dist = None
        self.inter_class_sim = None
        self.inter_class_sim_dist = None
        self.descriptors = None

        self.sample_dataset()

    def sample_dataset(self): 
        if self.max_num_classes is not None:
            # Sample a subset of class_ids
            unique_classes = self.dataconfig['class_id'].unique()
            sampled_classes = np.random.choice(unique_classes, size=min(self.max_num_classes, len(unique_classes)), replace=False)
            self.dataconfig = self.dataconfig[self.dataconfig['class_id'].isin(sampled_classes)].reset_index(drop=True)



    def compute_descriptors(self): 
        """Compute visual descriptors for all images in the dataset."""
        paths = [os.path.join(self.dataset_folder, path) for path in self.dataconfig["image_path"].to_numpy()]
        dataset = ImageDataset(np.array(paths), transform=self.eval_model.transform)
        self.descriptors = compute_descriptors(
            self.eval_model, dataset, desc_dtype=np.float16, 
            batch_size=self.batch_size, num_workers=self.num_workers, pbar=self.pbar
        )
        return self.descriptors
        
    def get_intra_class_sim(self): 
        """Compute intra-class similarity statistics."""
        self.intra_class_sim = get_intra_class_sim(self.dataconfig, self.descriptors)
        return self.intra_class_sim

    def get_intra_class_sim_dist(self): 
        """Compute intra-class similarity distribution statistics."""
        if self.intra_class_sim is None:
            self.get_intra_class_similarity()
        mean, std = get_intra_class_sim_dist(self.intra_class_sim)
        self.intra_class_sim_dist = {"mean": mean, "std": std}
        return self.intra_class_sim_dist

    def get_inter_class_sim(self): 
        if self.descriptors is None:
            self.compute_descriptors()
        self.inter_class_sim = get_inter_class_sim(self.dataconfig, self.descriptors)
        return self.inter_class_sim

    def get_inter_class_sim_dist(self): 
        if self.inter_class_sim is None:
            self.get_inter_class_sim()
        mean, std = get_inter_class_sim_dist(self.inter_class_sim)
        self.inter_class_sim_dist = {"mean": mean, "std": std}

    def plot_class_samples(self): 
        """Plot sample images from different classes."""
        plot_class_samples(self.dataconfig, self.dataconfig_path.parent, n_classes=5, max_images=self.max_plot_images)

    def plot_class_sim_heatmap(self, by_std_dist: bool = True): 
        """Plot class similarity heatmap for selected classes."""
        if self.descriptors is None:
            self.compute_descriptors()

        if self.intra_class_sim is None:
            self.get_intra_class_similarity()

        plot_path = self.dataconfig_path.parent / "intra_class" / "class_sim_heatmap.png"
        os.makedirs(plot_path.parent, exist_ok=True)

        if self.intra_class_sim_dist is None:
            self.get_intra_class_sim_dist()

        # Get the actual class IDs from the similarity data
        class_ids = list(self.intra_class_sim.keys())
        sort_vals = np.array(self.intra_class_sim_dist["std"] if by_std_dist else self.intra_class_sim_dist["mean"])
        sorted_indices = np.argsort(sort_vals)
        index_idx = np.linspace(0, len(sorted_indices) - 1, 6, dtype=int)
        selected_indices = sorted_indices[index_idx]
        classes = [class_ids[i] for i in selected_indices]
        plot_class_sim_heatmap(self.dataconfig, self.intra_class_sim, plot_path=plot_path, classes=classes)


    def plot_intra_class_sim_dist(self): 
        """Plot intra-class similarity distribution."""
        if self.descriptors is None:
            self.compute_descriptors()

        if self.intra_class_sim is None:
            self.get_intra_class_similarity()
        plot_path = self.dataconfig_path.parent / "intra_class" / "intra_class_sim_dist.png"
        plot_intra_class_sim_dist(self.intra_class_sim, self.intra_class_sim_dist["mean"], self.intra_class_sim_dist["std"], plot_path=plot_path)

    def plot_inter_class_sim_dist(self): 
        if self.descriptors is None:
            self.compute_descriptors()
        if self.inter_class_sim is None:
            self.get_inter_class_sim()
        plot_path = self.dataconfig_path.parent / "inter_class" / "inter_class_sim_dist.png"
        os.makedirs(plot_path.parent, exist_ok=True)
        
        plot_inter_class_sim_dist(self.inter_class_sim, plot_path=plot_path)


    def plot_intra_class_embedding(self, by_std_dist: bool = True): 
        """Plot class embedding projection for selected classes."""
        if self.intra_class_sim is None:
            self.get_intra_class_similarity()

        if self.descriptors is None:
            self.compute_descriptors()

        if self.intra_class_sim_dist is None:
            self.get_intra_class_sim_dist()

        plot_path = self.dataconfig_path.parent / "intra_class" / "intra_class_embedding.png"
        os.makedirs(plot_path.parent, exist_ok=True)

        # Get the actual class IDs from the similarity data
        class_ids = list(self.intra_class_sim.keys())
        sort_vals = np.array(self.intra_class_sim_dist["std"] if by_std_dist else self.intra_class_sim_dist["mean"])
        sorted_indices = np.argsort(sort_vals)
        index_idx = np.linspace(0, len(sorted_indices) - 1, 6, dtype=int)
        selected_indices = sorted_indices[index_idx]
        classes = [class_ids[i] for i in selected_indices]
        
        # Create dictionaries mapping class_id to mean/std values
        means_dict = {class_ids[i]: self.intra_class_sim_dist["mean"][i] for i in range(len(class_ids))}
        stds_dict = {class_ids[i]: self.intra_class_sim_dist["std"][i] for i in range(len(class_ids))}
        
        plot_intra_class_embedding(
            self.dataconfig, self.descriptors, plot_path=plot_path, 
            means=means_dict, stds=stds_dict, 
            classes=classes
        )

    def plot_inter_class_embedding(self): 
        if self.descriptors is None: 
            self.compute_descriptors() 

        plot_path = self.dataconfig_path.parent / "inter_class" / "inter_class_embedding.png"
        os.makedirs(plot_path.parent, exist_ok=True)
        plot_inter_class_embedding(self.dataconfig, self.descriptors, plot_path=plot_path)

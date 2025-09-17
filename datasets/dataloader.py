import pytorch_lightning as pl 
from datasets.train import check_config
import os 
import pandas as pd 
from torch.utils.data import Dataset
from typing import Optional 
from torchvision import transforms
import numpy as np 
from PIL import Image 

class NanoPlaceDataset(Dataset): 
    def __init__(self, config: pd.DataFrame, min_images_per_place: int=4, transform: Optional[transforms.Compose] = None): 
        self.config = config.copy()  # Make a copy to avoid modifying original
        self.transform = transform if transform is not None else self._default_transform()
        self.min_images_per_place = min_images_per_place
        self._filter_classes_with_min_images()
        self._create_class_to_indices_mapping()
        
    def _filter_classes_with_min_images(self): 
        """Remove classes that don't have enough images and update config"""
        original_num_classes = self.config.attrs["num_classes"]
        
        # Count images per class
        class_counts = self.config["class_id"].value_counts().sort_index()
        
        # Find classes with sufficient images
        valid_classes = class_counts[class_counts >= self.min_images_per_place].index.tolist()
        
        # Filter config to only include valid classes
        self.config = self.config[self.config["class_id"].isin(valid_classes)].copy()
        
        # Remap class_id to be consecutive (0, 1, 2, ...)
        if len(valid_classes) > 0:
            old_to_new_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(valid_classes))}
            self.config["class_id"] = self.config["class_id"].map(old_to_new_mapping)
        
        # Update attrs
        new_num_classes = len(valid_classes)
        removed_classes = original_num_classes - new_num_classes
        
        self.config.attrs["num_classes"] = new_num_classes
        self.config.attrs["num_images"] = len(self.config)
        
        # Print warning/info
        if removed_classes > 0:
            print(f"⚠️  Warning: Removed {removed_classes} classes with fewer than {self.min_images_per_place} images")
            print(f"   Classes remaining: {new_num_classes}")
            print(f"   Images remaining: {len(self.config)}")
        else:
            print(f"✓ All {original_num_classes} classes have sufficient images (≥{self.min_images_per_place})")
        
        # Reset index after filtering
        self.config.reset_index(drop=True, inplace=True)
    
    def _create_class_to_indices_mapping(self):
        """Create a mapping from class_id to list of row indices for efficient sampling"""
        self.class_to_indices = {}
        for idx, row in self.config.iterrows():
            class_id = row["class_id"]
            if class_id not in self.class_to_indices:
                self.class_to_indices[class_id] = []
            self.class_to_indices[class_id].append(idx)
    
    def _default_transform(self): 
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def __len__(self): 
        return self.config.attrs["num_classes"]

    def __getitem__(self, class_idx): 
        """
        Sample min_images_per_place images from class class_idx
        
        Args:
            class_idx: The class ID to sample from (0 to num_classes-1)
            
        Returns:
            tuple: (images, class_id) where images is a tensor of shape 
                   [min_images_per_place, C, H, W] and class_id is the class ID
        """
        import torch
        import random
        
        # Get all indices for this class
        available_indices = self.class_to_indices[class_idx]
        
        # Sample min_images_per_place indices from this class
        sampled_indices = random.sample(available_indices, self.min_images_per_place)
        
        # Load and transform images
        images = []
        for idx in sampled_indices:
            row = self.config.iloc[idx]
            
            # Construct full image path
            image_path = row["image_path"]
            if not image_path.startswith('/'):  # If relative path
                image_path = f"{self.config.attrs['dataset_folder']}/{image_path}"
                
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            images.append(image)
        
        # Stack images into a tensor [min_images_per_place, C, H, W]
        images_tensor = torch.stack(images)
        
        return images_tensor, torch.tensor([class_idx] * self.min_images_per_place)



class NanoPlaceLoader(pl.LightningDataModule): 
    def __init__(self, config_path: str, min_images_per_place: int=4, transform: Optional[transforms.Compose] = None): 
        assert os.path.exists(config_path), f"Config file {config_path} does not exist"
        self.config_path = config_path
        self.config = pd.read_parquet(config_path)
        self.min_images_per_place = min_images_per_place
        check_config(self.config)
        self.dataset_folder = self.config.attrs["dataset_folder"]
        self.num_classes = self.config.attrs["num_classes"]
        self.num_images = self.config.attrs["num_images"]
        self.transform = transform

    def setup(self, stage: str): 
        self.dataset = ImageDataset(self.config, min_images_per_place=self.min_images_per_place, transform=self.transform)

    def train_dataloader(self): 
        pass 


if __name__ == "__main__": 
    parquet_path = "datasets/train/configs/nanoplace_basic/config.parquet"
    config = pd.read_parquet(parquet_path)
    ds = NanoPlaceDataset(config, min_images_per_place=4)
    imgs, idx = ds[0]
    print(imgs.shape)
    print(idx)

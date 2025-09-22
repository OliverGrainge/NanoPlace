import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NanoPlaceDataset(Dataset):
    """
    Dataset that samples images_per_place images from each class.

    Args:
        config: DataFrame containing image paths and class IDs
        images_per_place: Number of images to sample from each class
        transform: Optional image transformations
        seed: Random seed for reproducible sampling
    """

    def __init__(
        self,
        config: pd.DataFrame,
        images_per_place: int = 4,
        transform: Optional[transforms.Compose] = None,
        seed: Optional[int] = None,
    ):
        self.config = config.copy()
        self.images_per_place = images_per_place
        self.seed = seed

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Validate inputs
        self._validate_config()

        # Set up transforms
        self.transform = (
            transform if transform is not None else self._default_transform()
        )

        # Process dataset
        self._filter_classes_with_min_images()
        self._create_class_to_indices_mapping()

    def _validate_config(self):
        """Validate the input configuration"""
        required_columns = ["image_path", "class_id"]
        missing_cols = [
            col for col in required_columns if col not in self.config.columns
        ]
        if missing_cols:
            raise ValueError(f"Config missing required columns: {missing_cols}")

        if self.images_per_place < 1:
            raise ValueError("images_per_place must be >= 1")

        if len(self.config) == 0:
            raise ValueError("Config DataFrame is empty")

    def _filter_classes_with_min_images(self):
        """Remove classes that don't have enough images and remap class IDs"""
        original_num_classes = self.config.attrs.get(
            "num_classes", self.config["class_id"].nunique()
        )

        # Count images per class
        class_counts = self.config["class_id"].value_counts()
        valid_classes = class_counts[
            class_counts >= self.images_per_place
        ].index.tolist()

        if len(valid_classes) == 0:
            raise ValueError(f"No classes have at least {self.images_per_place} images")

        # Filter config to only include valid classes
        self.config = self.config[self.config["class_id"].isin(valid_classes)].copy()

        # Remap class_id to be consecutive (0, 1, 2, ...)
        sorted_classes = sorted(valid_classes)
        old_to_new_mapping = {
            old_id: new_id for new_id, old_id in enumerate(sorted_classes)
        }
        self.config["class_id"] = self.config["class_id"].map(old_to_new_mapping)

        # Store mapping for potential reverse lookup
        self.class_id_mapping = old_to_new_mapping
        self.reverse_class_mapping = {v: k for k, v in old_to_new_mapping.items()}

        # Update attributes
        self.num_classes = len(valid_classes)
        self.num_images = len(self.config)

        # Update DataFrame attrs if they exist
        if hasattr(self.config, "attrs"):
            self.config.attrs["num_classes"] = self.num_classes
            self.config.attrs["num_images"] = self.num_images

        # Log filtering results
        removed_classes = original_num_classes - self.num_classes
        if removed_classes > 0:
            print(
                f"Removed {removed_classes} classes with <{self.images_per_place} images"
            )

        # Reset index after filtering
        self.config.reset_index(drop=True, inplace=True)

    def _create_class_to_indices_mapping(self):
        """Create efficient mapping from class_id to list of row indices"""
        self.class_to_indices = defaultdict(list)

        for idx, class_id in enumerate(self.config["class_id"]):
            self.class_to_indices[class_id].append(idx)

        # Convert to regular dict and validate
        self.class_to_indices = dict(self.class_to_indices)

        # Validate that all classes have enough images
        for class_id, indices in self.class_to_indices.items():
            if len(indices) < self.images_per_place:
                raise ValueError(
                    f"Class {class_id} has only {len(indices)} images, need {self.images_per_place}"
                )

    def _default_transform(self) -> transforms.Compose:
        """Default image transformations following ImageNet standards"""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and transform a single image"""
        # Handle relative paths
        if not os.path.isabs(image_path):
            dataset_folder = getattr(self.config, "attrs", {}).get("dataset_folder", "")
            if dataset_folder:
                image_path = os.path.join(dataset_folder, image_path)

        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def __len__(self) -> int:
        """Return number of classes (each sample represents one class)"""
        return self.num_classes

    def __getitem__(self, class_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample images_per_place images from the specified class.

        Args:
            class_idx: The class index (0 to num_classes-1)

        Returns:
            tuple: (images, class_ids) where:
                - images: tensor of shape [images_per_place, C, H, W]
                - class_ids: tensor of shape [images_per_place] with repeated class_idx
        """
        if not (0 <= class_idx < self.num_classes):
            raise IndexError(
                f"Class index {class_idx} out of range [0, {self.num_classes})"
            )

        # Get all available indices for this class
        available_indices = self.class_to_indices[class_idx]

        # Sample images_per_place indices from this class
        if len(available_indices) == self.images_per_place:
            # If exactly the right number, use all
            sampled_indices = available_indices
        elif len(available_indices) > self.images_per_place:
            # If more than needed, sample without replacement
            sampled_indices = random.sample(available_indices, self.images_per_place)
        else:
            # This shouldn't happen due to filtering, but handle gracefully
            sampled_indices = random.choices(available_indices, k=self.images_per_place)

        # Load and transform images
        images = []
        for idx in sampled_indices:
            row = self.config.iloc[idx]
            image = self._load_image(row["image_path"])
            images.append(image)

        # Stack images into tensor [images_per_place, C, H, W]
        images_tensor = torch.stack(images)

        # Create class ID tensor
        class_ids = torch.full((self.images_per_place,), class_idx, dtype=torch.long)

        return images_tensor, class_ids

    def get_original_class_id(self, new_class_id: int) -> int:
        """Get the original class ID from the remapped ID"""
        return self.reverse_class_mapping.get(new_class_id, new_class_id)


class NanoPlaceLoader(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for NanoPlace dataset.
    """

    def __init__(
        self,
        config_path: str,
        batch_size: int = 12,
        images_per_place: int = 4,
        transform: Optional[transforms.Compose] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # Validate inputs
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} does not exist")

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        # Store parameters
        self.config_path = config_path
        self.batch_size = batch_size
        self.images_per_place = images_per_place
        self.transform = transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.seed = seed

        # Load and validate config
        self.config = pd.read_pickle(config_path)

        # Extract metadata
        self.dataset_folder = getattr(self.config, "attrs", {}).get(
            "dataset_folder", ""
        )

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset"""
        self.dataset = NanoPlaceDataset(
            self.config,
            images_per_place=self.images_per_place,
            transform=self.transform,
            seed=self.seed,
        )

        self.num_classes = self.dataset.num_classes
        self.num_images = self.dataset.num_images

    def collate_fn(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collate function to handle multiple images per sample.

        Args:
            batch: List of (images, class_ids) tuples

        Returns:
            tuple: (all_images, all_class_ids) where:
                - all_images: [batch_size * images_per_place, C, H, W]
                - all_class_ids: [batch_size * images_per_place]
        """
        images, class_ids = zip(*batch)

        # Stack all images and class IDs
        all_images = torch.vstack(images)  # [batch_size * images_per_place, C, H, W]
        all_class_ids = torch.hstack(class_ids)  # [batch_size * images_per_place]

        return all_images, all_class_ids

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        if not hasattr(self, "dataset"):
            raise RuntimeError("Must call setup() before creating dataloaders")

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def get_dataloader_info(self):
        return f"Config: {self.config_path}, Classes: {self.num_classes}, Images: {self.num_images}"


# Example usage and testing
if __name__ == "__main__":
    parquet_path = "datasets/train/configs/nanoplace_basic/config.parquet"

    # Test the dataloader
    loader = NanoPlaceLoader(
        parquet_path,
        batch_size=8,
        images_per_place=4,
        num_workers=0,  # Set to 0 for debugging
        seed=42,
    )

    loader.setup("fit")

    # Test dataloader
    dataloader = loader.train_dataloader()
    batch = next(iter(dataloader))

    print(f"\nBatch shapes:")
    print(
        f"Images: {batch[0].shape}"
    )  # Should be [batch_size * images_per_place, C, H, W]
    print(f"Labels: {batch[1].shape}")  # Should be [batch_size * images_per_place]

    print(loader.get_dataloader_info())
    print("\n")

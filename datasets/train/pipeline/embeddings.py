from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.train.pipeline.base import CurationStep
from eval.utils import compute_descriptors
from models import get_model


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


class Embeddings(CurationStep):
    def __init__(
        self,
        model_name: str = "cosplace",
        batch_size: int = 32,
        num_workers: int = 4,
        pbar: bool = True,
        mmap_path: str = None 
    ):
        super().__init__()  # Call parent constructor

        self.model_name = model_name
        self.model = get_model(model_name)
        self.model.eval()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pbar = pbar
        self.mmap_path = mmap_path

    def __call__(
        self, dataconfig: pd.DataFrame, descriptors: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        dataconfig = dataconfig.reset_index(drop=True)
        dataconfig["image_id"] = np.arange(len(dataconfig))
        self.dataset = ImageDataset(
            dataconfig["image_path"].tolist(), self.model.transform
        )
        dataset = ImageDataset(
            dataconfig["image_path"].tolist(), self.model.transform
        )
        descriptors = compute_descriptors(
            self.model,
            dataset,
            desc_dtype=np.float16,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pbar=self.pbar,
            mmap=True,
            feature_path=self.mmap_path
        )
        descriptors.flush()
        return dataconfig, descriptors

    def name(self) -> str:
        return "Embeddings(model_name={self.distance_threshold})"

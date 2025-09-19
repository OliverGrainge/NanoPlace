import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List, Optional
from torchvision import transforms
from datasets.val import get_val_dataset


class ValLoader(pl.LightningDataModule): 
    def __init__(self, val_set_names: List[str], batch_size: int = 4, num_workers: int = 4, transform: Optional[transforms.Compose] = None):
        super().__init__()
        self.val_set_names = val_set_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_sets = [get_val_dataset(val_set_name) for val_set_name in val_set_names]

    def val_dataloader(self): 
        return [DataLoader(vset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=self.num_workers > 0) for vset in self.val_sets]


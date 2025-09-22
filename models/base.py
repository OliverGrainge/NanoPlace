import faiss
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_metric_learning import losses, miners

from eval import compute_recall, match_cosine

from .archs import get_arch
from .losses import load_loss, load_miner


class NanoPlaceModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        descriptor_dim: int = 512,
        loss_name: str = "multisimilarity",
        miner_name: str = "multisimilarity",
        cache_dtype: str = "float16",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.arch = get_arch(model_name, descriptor_dim)
        self.descriptor_dim = descriptor_dim
        self.cache_dtype = self._cache_dtype(cache_dtype)
        self.loss = load_loss(loss_name)
        self.miner = load_miner(miner_name)

    def forward(self, x):
        return self.arch(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        features = self(images)
        if self.miner is not None:
            pairs = self.miner(features, labels)
        else:
            pairs = None
        loss = self.loss(features, labels, pairs)
        self.log("train_loss", loss)
        return loss

    def _cache_dtype(self, cache_dtype: str):
        if cache_dtype == "float16":
            return np.float16
        elif cache_dtype == "float32":
            return np.float32
        else:
            raise ValueError(f"Cache dtype {cache_dtype} not supported")

    def on_validation_start(self):
        self.cache = {}
        self.val_set_names = []
        for idx, val_loader in enumerate(self.trainer.val_dataloaders):
            self.val_set_names.append(val_loader.dataset.__class__.__name__)
            self.cache[idx] = np.zeros(
                (len(val_loader.dataset), self.descriptor_dim), dtype=self.cache_dtype
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, idxs = batch
        features = self(images)
        self.cache[dataloader_idx][idxs.cpu()] = (
            features.cpu().numpy().astype(self.cache_dtype)
        )

    def _compute_recall(self, dataset_idx: int, k=1):
        desc = self.cache[dataset_idx]
        ds_name = self.val_set_names[dataset_idx]
        num_queries = self.trainer.val_dataloaders[dataset_idx].dataset.queries_num
        query_desc = desc[:num_queries]
        database_desc = desc[num_queries:]
        matches, scores = match_cosine(query_desc, database_desc)
        recall = compute_recall(
            matches, self.trainer.val_dataloaders[dataset_idx].dataset.ground_truth, k
        )
        return ds_name, recall

    def on_validation_epoch_end(self):
        for idx in range(len(self.trainer.val_dataloaders)):
            for k in [1, 5, 10]:
                ds_name, recall = self._compute_recall(idx, k)
                self.log(f"val_recall@{k}_{ds_name}", recall, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.arch.parameters(), lr=1e-3)

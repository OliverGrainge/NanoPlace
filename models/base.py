from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F 
from .losses import load_miner, load_loss




class TripletMarginModel(LightningModule):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.model = load_model(model_name, pretrained)
        self.miner = load_miner(miner_name)
        self.loss = load_loss(loss_name)

    def forward(self, x):
        return F.normalize(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(batch) 
        pairs = self.miner(embeddings, labels) 
        loss = self.loss(embeddings, labels, pairs)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx): 
        images, labels = batch
        embeddings = self(batch) 
        loss = self.loss(embeddings, labels)
        self.log("val_loss", loss)
        return loss



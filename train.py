import sys
from omegaconf import OmegaConf, DictConfig
from datasets import NanoPlaceLoader, ValLoader
from models import NanoPlaceModel
import pytorch_lightning as pl
import torch 

torch.set_float32_matmul_precision("high")

def load_config_from_argv():
    """
    Loads a YAML config file specified as the first command-line argument using OmegaConf.

    Usage:
        python train.py path/to/config.yaml
    Returns:
        config (DictConfig): Parsed configuration.
    """
    if len(sys.argv) < 2:
        raise ValueError("Please provide a path to the config YAML file as the first argument.")
    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    return config

def load_trainmodule(config: DictConfig):
    return NanoPlaceLoader(
        config_path=config.config_path, 
        batch_size=config.batch_size, 
        images_per_place=config.images_per_place, 
        num_workers=config.num_workers, 
        seed=config.seed
    )

def load_valmodule(config: DictConfig):
    return ValLoader(
        val_set_names=config.val_set_names, 
        batch_size=config.batch_size, 
        num_workers=config.num_workers,
    )

def load_model(config: DictConfig):
    return NanoPlaceModel(config.model_name, config.descriptor_dim, config.loss_name, config.miner_name)

def load_trainer(config: DictConfig):
    logger_config = config.WandBLogger
    checkpoint_config = config.ModelCheckpoint
    Logger = pl.loggers.WandbLogger(project=logger_config.project, name=logger_config.name)
    Checkpoint = pl.callbacks.ModelCheckpoint(monitor=checkpoint_config.monitor, mode=checkpoint_config.mode, save_top_k=checkpoint_config.save_top_k)
    return pl.Trainer(
        max_steps=config.max_steps, 
        val_check_interval=config.val_check_interval,
        precision=config.precision, 
        callbacks=[Checkpoint],
        logger=[Logger])

def main(config: DictConfig):
    traindatamodule = load_trainmodule(config.NanoPlaceLoader) 
    valdatamodule = load_valmodule(config.ValLoader)
    module = load_model(config.NanoPlaceModel) 
    trainer = load_trainer(config.Trainer) 
    traindatamodule.setup("fit")
    valdatamodule.setup("validate")
    train_loader = traindatamodule.train_dataloader()
    val_loaders = valdatamodule.val_dataloader()
    trainer.fit(module, train_loader, val_loaders)

if __name__ == "__main__":
    config = load_config_from_argv()
    main(config)
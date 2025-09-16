import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

from data.datamodule import PathDataModule
from model.lightningmodule import PathPredictionModule


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Compute vocab_size dynamically
    if cfg.model.vocab_size is None:
        cfg.model.vocab_size = cfg.graph_generation.sphere_mesh.num_horizontal * cfg.graph_generation.sphere_mesh.num_vertical +
    
    # Set up data module
    datamodule = PathDataModule(
        train_file=cfg.data.train_file,
        test_file=cfg.data.test_file,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        max_path_length=cfg.data.max_path_length,
        vocab_size=cfg.model.vocab_size,
        data_dir=cfg.paths.data_dir
    )
    
    # Set up model
    model = PathPredictionModule(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        d_ff=cfg.model.d_ff,
        max_seq_length=cfg.model.max_seq_length,
        dropout=cfg.model.dropout,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps
    )
    
    # Set up logger
    logger = WandbLogger(
        project=cfg.logging.project_name,
        name=cfg.logging.experiment_name,
        log_model=True
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True
    )
    
    # Train the model
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
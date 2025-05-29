import json

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger

from dataset import MushroomDataModule
from model import MushroomClassifier
from utils import download_data, export_to_onnx, get_git_commit_id


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    download_data(cfg.data.data_dir, cfg.data.split_dir)

    mlflow_logger = MLFlowLogger(
        experiment_name="mushroom-classification",
        tracking_uri=cfg.mlflow.tracking_uri,
        tags={"git_commit": get_git_commit_id()},
    )

    mlflow_logger.log_hyperparams(
        {
            "batch_size": cfg.data.batch_size,
            "num_workers": cfg.data.num_workers,
            "image_size": cfg.data.image_size,
            "model": cfg.model.model_name,
            "optimizer": cfg.optimizer._target_,
            "learning_rate": cfg.optimizer.lr,
            "max_epochs": cfg.trainer.max_epochs,
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    data_module = MushroomDataModule(
        data_dir=cfg.data.split_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        transform_mean=cfg.data.mean,
        transform_std=cfg.data.std,
    )

    model = MushroomClassifier(cfg, class_to_category=data_module.get_class_to_category())

    tb_logger = TensorBoardLogger(save_dir="plots/", name="tb_logs")
    csv_logger = CSVLogger(save_dir="plots/", name="csv_logs")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=[tb_logger, csv_logger, mlflow_logger],
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    best_model = MushroomClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        cfg=cfg,
        class_to_category=data_module.get_class_to_category(),
    )

    export_to_onnx(
        model=best_model,
        output_path=cfg.export.onnx_path,
    )

    torch.save(
        best_model.state_dict(), cfg.checkpoint.dirpath + "/best_model_weights.pth"
    )
    with open(cfg.checkpoint.dirpath + "/class_to_name.json", "w") as f:
        json.dump(data_module.get_class_to_name(), f)
    with open(cfg.checkpoint.dirpath + "/class_to_category.json", "w") as f:
        json.dump(data_module.get_class_to_category(), f)

    mlflow.pytorch.log_model(best_model, "model")


if __name__ == "__main__":
    main()

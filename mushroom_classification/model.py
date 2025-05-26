import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score

from mushroom_classification.models.conv_net import ConvNet
from mushroom_classification.models.efficient_net import EfficientNetModel


class MushroomClassifier(pl.LightningModule):
    def __init__(self, cfg, class_to_category):
        super().__init__()

        self.cfg = cfg
        self.class_to_category = class_to_category

        self.num_species = len(class_to_category)
        self.register_buffer(
            "species_to_category",
            torch.tensor(
                [class_to_category[i] for i in range(self.num_species)], dtype=torch.long
            ),
        )

        self.register_buffer(
            "species_to_category_onehot",
            torch.zeros((self.num_species, 4)).scatter_(
                1, self.species_to_category.unsqueeze(1), 1.0
            ),
        )

        if cfg.model.model_name == "conv_net":
            self.model = self._create_baseline()
        elif cfg.model.model_name[:12] == "efficientnet":
            self.model = self._create_efficientnet()
        else:
            raise ValueError(f"Invalid model name: {cfg.model.model_name}")

        self.optimizer_config = cfg.optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.penalty_weight = cfg.model.penalty_weight

        self.train_acc = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)

        self.train_f1 = F1Score(
            task="multiclass", num_classes=cfg.model.num_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=cfg.model.num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=cfg.model.num_classes, average="macro"
        )

        self.train_group_acc = Accuracy(task="binary")
        self.val_group_acc = Accuracy(task="binary")
        self.test_group_acc = Accuracy(task="binary")

        self.train_group_f1 = F1Score(task="binary")
        self.val_group_f1 = F1Score(task="binary")
        self.test_group_f1 = F1Score(task="binary")

    def _create_baseline(self):
        return ConvNet(num_classes=self.cfg.model.num_classes)

    def _create_efficientnet(self):
        return EfficientNetModel(
            model_name=self.cfg.model.model_name, num_classes=self.cfg.model.num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_penalty(self, probs, true_cats):
        category_probs = torch.matmul(probs, self.species_to_category_onehot)
        dangerous_mask = (true_cats == 1) | (true_cats == 3)

        correct_group_probs = torch.where(
            dangerous_mask,
            category_probs[:, 1] + category_probs[:, 3],
            category_probs[:, 0] + category_probs[:, 2],
        )

        return (1 - correct_group_probs).mean() * self.penalty_weight

    def training_step(self, batch, batch_idx: int):
        x, species_labels, true_cats = batch
        logits = self(x)
        probs = logits.softmax(dim=-1)

        loss = self.loss_fn(logits, species_labels)

        penalty = self._compute_penalty(probs, true_cats)
        total_loss = loss + penalty

        self.train_acc(logits, species_labels)

        pred_species = logits.argmax(dim=1)
        pred_cats = self.species_to_category[pred_species]
        dangerous_pred = (pred_cats == 1) | (pred_cats == 3)
        dangerous_true = (true_cats == 1) | (true_cats == 3)

        self.train_group_acc(dangerous_pred, dangerous_true)
        self.train_group_f1(dangerous_pred, dangerous_true)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_group_acc",
            self.train_group_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_group_f1",
            self.train_group_f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx: int):
        x, species_labels, true_cats = batch
        logits = self(x)
        probs = logits.softmax(dim=-1)

        loss = self.loss_fn(logits, species_labels)
        penalty = self._compute_penalty(probs, true_cats)
        total_loss = loss + penalty

        self.val_acc(logits, species_labels)

        pred_species = logits.argmax(dim=1)
        pred_cats = self.species_to_category[pred_species]
        dangerous_pred = (pred_cats == 1) | (pred_cats == 3)
        dangerous_true = (true_cats == 1) | (true_cats == 3)

        self.val_group_acc(dangerous_pred, dangerous_true)
        self.val_group_f1(dangerous_pred, dangerous_true)

        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_group_acc", self.val_group_acc, on_epoch=True, prog_bar=True)
        self.log("val_group_f1", self.val_group_f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        x, species_labels, true_cats = batch
        logits = self(x)
        probs = logits.softmax(dim=-1)

        loss = self.loss_fn(logits, species_labels)
        penalty = self._compute_penalty(probs, true_cats)
        total_loss = loss + penalty

        self.test_acc(logits, species_labels)

        pred_species = logits.argmax(dim=1)
        pred_cats = self.species_to_category[pred_species]
        dangerous_pred = (pred_cats == 1) | (pred_cats == 3)
        dangerous_true = (true_cats == 1) | (true_cats == 3)

        self.test_group_acc(dangerous_pred, dangerous_true)
        self.test_group_f1(dangerous_pred, dangerous_true)

        self.log("test_loss", total_loss)
        self.log("test_acc", self.test_acc)
        self.log("test_group_acc", self.test_group_acc)
        self.log("test_group_f1", self.test_group_f1)

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.optimizer_config, params=self.parameters())

from __future__ import annotations


import torch
import torch.nn as nn
import tqdm
import wandb
from improvement_float import CheckImprovementFloat
from torch.utils.data import DataLoader

from ..utils import XTYPE, YTYPE, make_loader_from_xy


class SingleVariateMixin(CheckImprovementFloat, nn.Module):
    def train_model(
        self,
        train_loader: DataLoader | tuple[XTYPE, YTYPE],
        val_loader: DataLoader | tuple[XTYPE, YTYPE],
        epochs: int = 10,
    ):
        if not self._post_init_done:
            raise RuntimeError("post_init must be called before training the model.")

        device = self.device
        self.model_state["epochs_trained"] + epochs

        train_loader = (
            train_loader
            if isinstance(train_loader, DataLoader)
            else make_loader_from_xy(*train_loader, self.hparams["batchsize"])
        )
        val_loader = (
            val_loader
            if isinstance(val_loader, DataLoader)
            else make_loader_from_xy(*val_loader, self.hparams["batchsize"])
        )

        epochs_no_improve = 0

        for _ in tqdm.tqdm(range(epochs)):
            self.train()
            running_loss = 0.0
            total = 0

            for inputs, labels in train_loader:
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels.to(device)

                self.optimizer.zero_grad()
                outputs = self(inputs)

                loss = self.get_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                total += inputs.size(0)

            epoch_loss = running_loss / total

            val_loss = self.evaluate(val_loader)

            wandb.log(
                {
                    "Train Loss": epoch_loss,
                    "Validation Loss": val_loss,
                }
            )

            # Early stopping: check improvement
            improvements = self._improved(val_loss)
            epochs_no_improve = self._on_improvement(improvements, val_loss, epochs_no_improve)
            self.model_state["epochs_trained"] += 1
            if self._patience_reached(epochs_no_improve):
                break

    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model on a validation set.

        Returns:
            tuple: (validation loss, validation accuracy (subclass), validation accuracy (superclass))
        """
        # raise NotImplementedError("Subclasses must implement evaluate().")
        self.eval()
        running_loss = 0.0
        total = 0
        device = self.device

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels
                outputs = self(inputs)

                loss = self.get_loss(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

        val_loss = running_loss / total

        return val_loss

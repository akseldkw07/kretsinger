from __future__ import annotations

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from ..utils import XTYPE, YTYPE
from .abc_nn import ABCNN


class SingleVariateMixin(ABCNN, nn.Module):
    def train_model(
        self,
        train_loader: DataLoader | tuple[XTYPE, YTYPE],
        val_loader: DataLoader | tuple[XTYPE, YTYPE],
        epochs: int = 10,
        batch_size: int | None = 128,
    ):
        if not self._post_init_done:
            raise RuntimeError("post_init must be called before training the model.")

        device = self.device
        epochs_no_improve = 0
        batch_size = batch_size or self.hparams["batchsize"]

        train_loader = self._to_dataloader(train_loader, batch_size=batch_size)
        val_loader = self._to_dataloader(val_loader, batch_size=batch_size)

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

            eval = self.evaluate(val_loader)

            self._log_wandb({"train_loss": epoch_loss, **eval})

            # Early stopping: check improvement
            improvements = self._improved(eval)
            epochs_no_improve = self._on_improvement(improvements, eval, epochs_no_improve)

            self.model_state["epochs_trained"] += 1
            if self._patience_reached(epochs_no_improve):
                break

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
    ):
        if not self._post_init_done:
            raise RuntimeError("post_init must be called before training the model.")

        device = self.device
        epochs_no_improve = 0

        train_loader = self._to_dataloader(train_loader)
        val_loader = self._to_dataloader(val_loader)

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

    def evaluate(self, val_loader: DataLoader):
        """
        Evaluate the model on a validation set.

        Returns:
            (validation loss, R^2 score)
        """
        self.eval()
        running_loss = 0.0
        total = 0
        device = self.device

        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels.to(device)
                outputs: torch.Tensor = self(inputs)

                loss = self.get_loss(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

                all_preds.append(outputs.detach().view(-1))
                all_targets.append(labels.detach().view(-1))

        eval_loss = running_loss / total

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)

        # R^2 = 1 - SS_res / SS_tot
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
        eval_r2 = 1.0 - (ss_res / ss_tot)
        eval_r2 = float(eval_r2.item())

        return {"eval_loss": eval_loss, "eval_r2": eval_r2}

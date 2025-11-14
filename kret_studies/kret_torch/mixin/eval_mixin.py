from __future__ import annotations

import typing as t

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .abc_nn import ABCNN


class LinearEvalMixin(ABCNN, nn.Module):
    def predict(self, val_loader: DataLoader):
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
        return {"eval_loss": eval_loss, "y_pred": y_pred, "y_true": y_true}

    def evaluate(self, val_loader: DataLoader):
        """
        Evaluate the model on a validation set.

        Returns:
            (validation loss, R^2 score)
        """
        pred = self.predict(val_loader)

        # R^2 = 1 - SS_res / SS_tot
        ss_res = torch.sum((pred["y_true"] - pred["y_pred"]) ** 2)
        ss_tot = torch.sum((pred["y_true"] - pred["y_true"].mean()) ** 2)
        eval_r2 = 1.0 - (ss_res / ss_tot)
        eval_r2 = float(eval_r2.item())

        return {"eval_loss": pred["eval_loss"], "eval_r2": eval_r2}


class ClassificationEvalMixin(ABCNN, nn.Module):
    def predict(self, val_loader: DataLoader):
        self.eval()
        correct = 0
        total = 0
        device = self.device

        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels.to(device)
                outputs: torch.Tensor = self(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.append(predicted.detach())
                all_targets.append(labels.detach())

        eval_accuracy = correct / total

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)
        return {"eval_accuracy": eval_accuracy, "y_pred": y_pred, "y_true": y_true}

    def evaluate(self, val_loader: DataLoader):
        """
        Evaluate the model on a validation set.

        Returns:
            (validation accuracy, F1 score)
        """
        pred = self.predict(val_loader)

        # F1 Score calculation
        eval_f1 = t.cast(
            float, f1_score(pred["y_true"].cpu().numpy(), pred["y_pred"].cpu().numpy(), average="weighted")
        )

        return {"eval_accuracy": pred["eval_accuracy"], "eval_f1": eval_f1}

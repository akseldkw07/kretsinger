from torch import Tensor
from .abc_lightning import ABCLM
from .datamodule.data_module_custom import STAGE_LITERAL
import torch
import torch.nn as nn
import typing as t
from torchmetrics.regression import R2Score

PREDICT_TYPE = t.Literal["Regression", "Classification"]


class MetricMixin(ABCLM):
    @property
    def PredictType(self):
        if isinstance(self._criterion, nn.MSELoss):
            return "Regression"
        elif isinstance(self._criterion, (nn.BCEWithLogitsLoss, nn.CrossEntropyLoss)):
            return "Classification"

        raise ValueError(f"criterion {self._criterion} isn't registered as {t.get_args(PREDICT_TYPE)}")

    def compute_metrics(self, y_hat: Tensor, y: Tensor, stage: STAGE_LITERAL) -> dict[str, float]:
        assert stage in t.get_args(STAGE_LITERAL), f"Unknown stage: {stage!r}"
        metrics: dict[str, float] = {}

        match self.PredictType:
            case "Regression":
                mse = nn.MSELoss()(y_hat, y).item()
                mae = nn.L1Loss()(y_hat, y).item()
                r2 = R2Score()(y_hat, y).item()
                metrics.update({"MSE": mse, "MAE": mae, "R2": r2})
            case "Classification":
                preds = torch.argmax(y_hat, dim=1)
                accuracy = (preds == y).float().mean().item()
                metrics.update({"Accuracy": accuracy})
            case _:
                raise ValueError(f"Unknown PredictType: {self.PredictType!r}")

        return metrics

    def log_extra_metrics(self, y_hat: Tensor, y: Tensor, stage: STAGE_LITERAL) -> None:
        assert stage in t.get_args(STAGE_LITERAL), f"Unknown stage: {stage!r}"

        metrics = self.compute_metrics(y_hat, y, stage)
        self.log_dict({f"{stage}_{k}": v for k, v in metrics.items()}, prog_bar=True, on_epoch=True, on_step=False)

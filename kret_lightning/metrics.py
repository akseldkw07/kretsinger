import warnings

import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from ._core.constants_lightning import TASK_TYPE
from .abc_lightning import ABCLM


class MetricMixin(ABCLM):
    """
    Mixin that handles all metric computation and logging for Lightning models.

    Overrides training_step, validation_step, and test_step to:
    - Log loss values (train_loss, val_loss, test_loss)
    - Track additional metrics (accuracy, F1, R2, etc.) via torchmetrics
    - Compute and log epoch-level metrics

    Metrics are registered as module attributes, so they:
    - Automatically move to the correct device
    - Accumulate state across batches for epoch-level computation
    - Reset properly between epochs

    IMPORTANT: MetricMixin must come BEFORE BaseLightningNN in the inheritance list
    to ensure its step methods are called:

        class MyModel(MetricMixin, BaseLightningNN):  # Correct
            ...

        class MyModel(BaseLightningNN, MetricMixin):  # WRONG - won't track metrics
            ...

    Usage:
        class MyModel(MetricMixin, BaseLightningNN):
            def __init__(self, num_classes: int = 10, **kwargs):
                super().__init__(**kwargs)
                self._criterion = nn.CrossEntropyLoss()
                self._setup_metrics(task="multiclass", num_classes=num_classes)
    """

    _task: TASK_TYPE
    _num_classes: int | None = None
    _num_labels: int | None = None  # for multilabel

    # Metric collections for each stage
    _train_metrics: MetricCollection
    _val_metrics: MetricCollection
    _test_metrics: MetricCollection

    def setup_metrics(
        self, task: TASK_TYPE, num_classes: int | None = None, num_labels: int | None = None, threshold: float = 0.5
    ) -> None:
        """
        Initialize metrics based on task type. Call this in your __init__.

        Args:
            task: One of "regression", "binary", "multiclass", "multilabel"
            num_classes: Required for multiclass tasks
            num_labels: Required for multilabel tasks
            threshold: Decision threshold for binary/multilabel (default 0.5)
        """
        self._task = task
        self._num_classes = num_classes
        self._num_labels = num_labels

        metrics = self._create_metrics(task, num_classes, num_labels, threshold)

        # Create separate metric collections for each stage (they track state independently)
        self._train_metrics = metrics.clone(prefix="train_")
        self._val_metrics = metrics.clone(prefix="val_")
        self._test_metrics = metrics.clone(prefix="test_")

    def _create_metrics(
        self, task: TASK_TYPE, num_classes: int | None, num_labels: int | None, threshold: float
    ) -> MetricCollection:
        """Create a MetricCollection based on task type."""

        if task == "regression":
            return MetricCollection(
                {
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError(),
                    "r2": R2Score(),
                }
            )

        elif task == "binary":
            return MetricCollection(
                {
                    "acc": Accuracy(task="binary", threshold=threshold),
                    "precision": Precision(task="binary", threshold=threshold),
                    "recall": Recall(task="binary", threshold=threshold),
                    "f1": F1Score(task="binary", threshold=threshold),
                    "auroc": AUROC(task="binary"),
                }
            )

        elif task == "multiclass":
            if num_classes is None:
                raise ValueError("num_classes required for multiclass task")
            return MetricCollection(
                {
                    "acc": Accuracy(task="multiclass", num_classes=num_classes),
                    "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
                    "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
                    "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                }
            )

        elif task == "multilabel":
            if num_labels is None:
                raise ValueError("num_labels required for multilabel task")
            return MetricCollection(
                {
                    "acc": Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold),
                    "precision": Precision(
                        task="multilabel", num_labels=num_labels, threshold=threshold, average="macro"
                    ),
                    "recall": Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average="macro"),
                    "f1": F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average="macro"),
                }
            )

        else:
            raise ValueError(f"Unknown task: {task!r}")

    # region Step Overrides

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step with loss logging and metric tracking.
        """
        outputs, y, loss = self._compute_step(batch)

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Update metrics (accumulated, computed at epoch end)
        preds = self._prepare_preds(outputs)
        y_prepared = self._prepare_targets(y)
        self._train_metrics.update(preds, y_prepared)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step with loss logging and metric tracking.
        """
        outputs, y, val_loss = self._compute_step(batch)

        # Log loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Update metrics
        preds = self._prepare_preds(outputs)
        y_prepared = self._prepare_targets(y)
        self._val_metrics.update(preds, y_prepared)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Test step with loss logging and metric tracking.
        """
        outputs, y, test_loss = self._compute_step(batch)

        # Log loss
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Update metrics
        preds = self._prepare_preds(outputs)
        y_prepared = self._prepare_targets(y)
        self._test_metrics.update(preds, y_prepared)

    # endregion

    # region Epoch End Hooks

    def on_train_epoch_end(self) -> None:
        """Compute and log training metrics at epoch end."""
        self._log_and_reset_metrics(self._train_metrics)

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        self._log_and_reset_metrics(self._val_metrics)

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at epoch end."""
        self._log_and_reset_metrics(self._test_metrics)

    def _log_and_reset_metrics(self, metrics: MetricCollection) -> None:
        """Compute final metric values, log them, and reset for next epoch."""
        computed = metrics.compute()
        self.log_dict(computed, prog_bar=True, on_epoch=True)
        metrics.reset()

    # endregion

    # region Helpers

    def _prepare_preds(self, y_hat: Tensor) -> Tensor:
        """
        Prepare predictions for metric computation.

        - Regression: return as-is (squeezed)
        - Binary: squeeze if needed, apply sigmoid if logits
        - Multiclass: return as-is (metrics handle argmax internally)
        - Multilabel: apply sigmoid if logits
        """

        def _to_sigmoid_safe(tensor: Tensor) -> Tensor:
            if tensor.min() < 0 or tensor.max() > 1:
                warnings.warn("Applying sigmoid to logits for metric computation", UserWarning)
                tensor = torch.sigmoid(tensor)
            return tensor

        if self._task == "regression":
            return y_hat.squeeze()

        elif self._task == "binary":
            y_hat = y_hat.squeeze()
            return _to_sigmoid_safe(y_hat)

        elif self._task == "multiclass":
            # torchmetrics handles logits directly for multiclass
            return y_hat

        elif self._task == "multilabel":
            return _to_sigmoid_safe(y_hat)

        return y_hat

    def _prepare_targets(self, y: Tensor) -> Tensor:
        """
        Prepare targets for metric computation.

        - Regression: squeeze to match prediction shape
        - Classification: return as-is
        """
        if self._task == "regression":
            return y.squeeze()
        return y

    # endregion

from __future__ import annotations

from pathlib import Path
import typing as t

import torch
import torch.nn as nn
from lightning.fabric.utilities.data import AttributeDict

from kret_lightning.constants_lightning import LightningConstants
from kret_lightning.utils import LightningModuleAssert
from kret_torch_utils.priors import PriorLosses

from .abc_lightning import ABCLM


class HPDict(AttributeDict):  # type: ignore
    lr: float
    gamma: float
    stepsize: int
    l1_lambda: float
    l2_lambda: float


class HPasKwargs(t.TypedDict, total=False):
    lr: float
    gamma: float
    stepsize: int
    l1_lambda: float
    l2_lambda: float


class BaseLightningNN(ABCLM):
    """
    TODO

    1- finalize naming / filepath conventions
        how does load_from_checkpoint load hparams? how to save them?
    2- work on training loop defaults
    3- re-implement beijing nn with this base class
    """

    _criterion: nn.Module

    # region Init
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.5,
        stepsize: int = 12,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        **kwargs,
    ):
        """
        NOTE: don't call .to(device) here; Lightning handles device placement
        """
        LightningModuleAssert.assert_version_fmt(self.version)
        super().__init__()
        self.save_hyperparameters()

    @property
    def criterion(self):
        return self._criterion.to(self.device)

    def configure_optimizers(self, *args, **kwargs):
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        """
        hp = t.cast(HPDict, self.hparams_initial)

        optimizer = torch.optim.Adam(self.parameters(), lr=hp.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.stepsize, gamma=hp.gamma)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # endregion

    # region Naming

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}__{self.hparams_str}__{self.version}"

    @property
    def root_dir(self) -> Path:
        val = self._load_dir_override if self._load_dir_override is not None else LightningConstants.LIGHTNING_LOG_DIR
        return Path(val)

    @property
    def hparams_str(self) -> str:
        hp = t.cast(HPDict, self.hparams_initial)
        parts = [
            f"lr{hp.lr:g}",
            f"g{hp.gamma:g}",
            f"step{hp.stepsize}",
        ]
        if hp.l1_lambda > 0.0:
            parts.append(f"L1-{hp.l1_lambda:g}")
        if hp.l2_lambda > 0.0:
            parts.append(f"L2-{hp.l2_lambda:g}")
        return "-".join(parts)

    # endregion

    # region Loss

    def get_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss. If in training mode, includes prior regularization.
        """
        # Base loss (cross-entropy, MSE, etc.)
        base_loss = self.criterion(outputs, labels)

        # Prior regularization
        prior_loss = torch.tensor(0.0, device=self.device)
        if self.training:
            prior_loss = self.compute_prior_loss(outputs=outputs)

        return base_loss + prior_loss

    def compute_prior_loss(
        self, inputs: torch.Tensor | None = None, outputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        hp = t.cast(HPDict, self.hparams_initial)
        tot = torch.tensor(0.0, device=next(self.parameters()).device)
        if hp.l1_lambda > 0.0:
            tot += hp.l1_lambda * PriorLosses.compute_l1_loss(self)
        if hp.l2_lambda > 0.0:
            tot += hp.l2_lambda * PriorLosses.compute_l2_loss(self)
        return tot

    # endregion
    # region Training / Validation Steps
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """

        loss = ...
        return loss
        """
        x, y = batch
        outputs = self(x)
        loss = self.get_loss(outputs, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        val_loss = ...
        self.log('val_loss', val_loss)
        """
        x, y = batch
        outputs = self(x)
        val_loss = self.get_loss(outputs, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    # endregion

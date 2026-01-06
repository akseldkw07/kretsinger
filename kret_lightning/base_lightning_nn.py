from __future__ import annotations

import typing as t
from pathlib import Path
from typing import get_type_hints

import torch
import torch.nn as nn
from lightning.fabric.utilities.data import AttributeDict

from kret_lightning.constants_lightning import LightningConstants  # type: ignore
from kret_lightning.utils_lightning import LightningModuleAssert
from kret_torch_utils.priors import PriorLosses

from .abc_lightning import ABCLM, SaveLoadLoggingDict


class HPDict(AttributeDict):  # type: ignore
    lr: float
    gamma: float
    stepsize: int
    l1_penalty: float
    l2_penalty: float

    def as_str_safe(self, sanitize: bool = True) -> str:
        parts = [
            f"lr={self.lr:g}",
            f"gamma={self.gamma:g}",
            f"stepsize={self.stepsize}",
        ]
        if self.l1_penalty > 0.0:
            parts.append(f"L1={self.l1_penalty:g}")
        if self.l2_penalty > 0.0:
            parts.append(f"L2={self.l2_penalty:g}")

        for key in self.keys():
            if key not in get_type_hints(HPDict).keys():
                parts.append(f"{key}={self[key]}")

        ret = "--".join(parts)

        if sanitize:
            try:
                from pathvalidate import sanitize_filename  # lazy import in case others don't have
            except ImportError:
                raise ImportError("Please install 'pathvalidate' to use sanitize option, or set sanitize=False.")

            ret = ret.replace(" ", "")
            ret = sanitize_filename(ret, replacement_text="_")
        return ret


class HPasKwargs(t.TypedDict, total=False):
    lr: float
    gamma: float
    stepsize: int
    l1_penalty: float
    l2_penalty: float


class BaseLightningNN(ABCLM):
    """
    TODO

    1- finalize naming / filepath conventions
        how does load_from_checkpoint load hparams? how to save them?
    2- work on training loop defaults
    3- re-implement beijing nn with this base class
    4- add callbacks to assert correct behavior (tensor size? early stopping?)
    """

    _criterion: nn.Module

    # region Init
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.5,
        stepsize: int = 12,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        **kwargs,
    ):
        """
        NOTE: don't call .to(device) here; Lightning handles device placement
        """
        super().__init__()
        self.save_hyperparameters()
        LightningModuleAssert.initialization_check(self)

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
        return f"{self.__class__.__name__}__{self.hparams_str}"

    @property
    def root_dir(self) -> Path:
        val = self._load_dir_override if self._load_dir_override is not None else LightningConstants.LIGHTNING_LOG_DIR
        return Path(val)

    @property
    def hparams_str(self) -> str:
        hp = HPDict(self.hparams_initial)
        return hp.as_str_safe()

    @property
    def save_load_logging_dict(self) -> SaveLoadLoggingDict:
        ret: SaveLoadLoggingDict = {"save_dir": self.root_dir, "name": self.name, "version": self.version}
        return ret

    # endregion

    # region Loss

    def forward(self, *args, **kwargs) -> t.Any:
        """
        Note - don't call .to(device) here; bad for memory
        """
        raise NotImplementedError("Subclasses must implement forward method.")

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
        """
        Inputs and outputs are provided in case of more complex prior computations,
        such as entropic priors that depend on model outputs.
        """
        hp = t.cast(HPDict, self.hparams_initial)
        tot = torch.tensor(0.0, device=next(self.parameters()).device)

        # Check that penalty is non-zero before computing to save time
        if hp.l1_penalty > 0.0:
            tot += hp.l1_penalty * PriorLosses.compute_l1_loss(self)
        if hp.l2_penalty > 0.0:
            tot += hp.l2_penalty * PriorLosses.compute_l2_loss(self)
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

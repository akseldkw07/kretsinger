from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from pathlib import Path
from typing import get_type_hints

import lightning as L
import torch
import torch.nn as nn


class ABCLM(ABC, L.LightningModule):
    version: str = "v_000"  # eg. 'v_001'
    hparams: HPDict  # type: ignore
    hparams_initial: HPDict  # type: ignore

    __call__: t.Callable[..., torch.Tensor]
    _criterion: nn.Module
    _load_dir_override: str | Path | None = None

    @abstractmethod
    def configure_optimizers(self, *args, **kwargs) -> t.Any:
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def root_dir(self) -> Path: ...

    @property
    @abstractmethod
    def hparams_str(self) -> str: ...

    @property
    @abstractmethod
    def save_load_logging_dict(self) -> SaveLoadLoggingDict: ...

    @property
    @abstractmethod
    def ckpt_path(self) -> Path: ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> t.Any:
        """
        Note - don't call .to(device) here; bad for memory
        """
        ...

    @abstractmethod
    def get_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        loss = ...
        return loss
        """

    @abstractmethod
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """

        loss = ...
        return loss
        """

    @abstractmethod
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        val_loss = ...
        self.log('val_loss', val_loss)
        """

    """
    Other helpful methods:

    load_from_checkpoint()
    """


class SaveLoadLoggingDict(t.TypedDict):
    save_dir: str | Path
    name: str
    version: str


class HPDict(AttributeDict):  # type: ignore
    lr: float
    gamma: float
    stepsize: int
    l1_penalty: float
    l2_penalty: float
    patience: int

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
    patience: int

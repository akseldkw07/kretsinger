from __future__ import annotations

from pathlib import Path
import typing as t
from abc import ABC, abstractmethod

import lightning as L
import torch
import torch.nn as nn


class ABCLM(ABC, L.LightningModule):
    version: str = "v_000"  # eg. 'v_001'
    __call__: t.Callable[..., torch.Tensor]
    _criterion: nn.Module
    _load_dir_override: str | Path | None = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def root_dir(self) -> Path: ...

    @property
    @abstractmethod
    def hparams_str(self) -> str: ...

    # @abstractmethod
    # def forward(self, *args, **kwargs) -> t.Any:
    #     """
    #     Note - don't call .to(device) here; bad for memory
    #     """
    #     ...

    @abstractmethod
    def configure_optimizers(self, *args, **kwargs) -> t.Any:
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        """

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

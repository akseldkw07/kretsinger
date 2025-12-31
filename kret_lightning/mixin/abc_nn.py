from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import pytorch_lightning as L
import torch


class ABCNN(ABC, L.LightningModule):
    nickname: str = "v000"  # eg. 'v001', 'L2reg-v000', ' Dropout-0.2v000'

    @abstractmethod
    def forward(self, *args, **kwargs) -> t.Any:
        """
        Note - don't call .to(device) here; bad for memory
        """
        ...

    @abstractmethod
    def configure_optimizers(self, *args, **kwargs) -> t.Any:
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        """

    @abstractmethod
    def training_step(self, batch: t.Any, batch_idx: int) -> torch.Tensor:
        """
        loss = ...
        return loss
        """

    @abstractmethod
    def validation_step(self, batch: t.Any, batch_idx: int) -> None:
        """
        val_loss = ...
        self.log('val_loss', val_loss)
        """

    """
    Other helpful methods:

    load_from_checkpoint()
    """

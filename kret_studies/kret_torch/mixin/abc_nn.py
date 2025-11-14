from __future__ import annotations

import logging
import pathlib
import typing as t
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import XTYPE, YTYPE


class ModelPathDict(t.TypedDict):
    model_path: pathlib.Path
    state_path: pathlib.Path
    weight_path: pathlib.Path


class HyperParamDict(t.TypedDict, total=False):
    lr: float
    gamma: float
    stepsize: int
    batchsize: int
    # early stopping controls
    patience: int
    improvement_tol: float


class HyperParamTotalDict(t.TypedDict, total=True):
    lr: float
    gamma: float
    stepsize: int
    batchsize: int
    # early stopping controls
    patience: int
    improvement_tol: float


class ModelStateDict(t.TypedDict):
    epochs_trained: int
    best_eval_loss: float
    best_eval_r2: float
    best_accuracy: float
    best_f1: float


class FullStateDict(t.TypedDict):
    state: ModelStateDict
    hparams: HyperParamTotalDict


class ABCNN(ABC):
    version: str = "v000"
    model: nn.Module  # nn.Sequential or other nn.Module TODO define in subclass
    optimizer: torch.optim.Optimizer  # NOTE: NOT SET in __init__
    scheduler: torch.optim.lr_scheduler.LRScheduler  # NOTE: NOT SET in __init__
    device: t.Literal["cuda", "mps", "xpu", "cpu"]
    _criterion: nn.Module  # TODO define

    hparams: HyperParamTotalDict

    _load_weights_act: t.Literal["assert", "try", "fresh"]
    _post_init_done: bool = False
    model_state: ModelStateDict
    logger: logging.Logger
    _log: bool
    _wandb: bool

    @property
    @abstractmethod
    def root_dir(self) -> pathlib.Path: ...

    @property
    @abstractmethod
    def model_paths(self) -> ModelPathDict: ...

    @property
    @abstractmethod
    def FullStateDict(self) -> FullStateDict: ...

    @property
    @abstractmethod
    def FullStateDictDisplay(self) -> FullStateDict: ...

    @abstractmethod
    def save_weights(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def load_weights(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def name(self, *args, **kwargs) -> str: ...

    @abstractmethod
    def summary(self, *args, **kwargs) -> None: ...

    @property
    @abstractmethod
    def criterion(self) -> nn.Module: ...

    @abstractmethod
    def post_init(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> t.Any: ...

    @abstractmethod
    def get_loss(self, *args, **kwargs) -> torch.Tensor: ...

    # Helper training methods
    @abstractmethod
    def _patience_reached(self, *args, **kwargs) -> bool: ...

    @abstractmethod
    def _to_dataloader(self, data: tuple[XTYPE, YTYPE] | DataLoader) -> DataLoader: ...

    @abstractmethod
    def _log_wandb(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def set_model(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, float]: ...

    @abstractmethod
    def train_model(self, *args, **kwargs) -> None: ...

    # Helper methods for training with early stopping

    @abstractmethod
    def _improved(self, *args, **kwargs) -> dict[str, bool]: ...

    @abstractmethod
    def _on_improvement(self, *args, **kwargs) -> int: ...

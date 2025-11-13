from __future__ import annotations

import pathlib
import typing as t
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ModelPathDict(t.TypedDict):
    model_path: pathlib.Path
    state_path: pathlib.Path
    weight_path: pathlib.Path


class HyperParamDict(t.TypedDict, total=False):
    lr: float
    gamma: float
    stepsize: int
    # early stopping controls
    patience: int
    improvement_tol: float


class HyperParamTotalDict(t.TypedDict, total=True):
    lr: float
    gamma: float
    stepsize: int
    # early stopping controls
    patience: int
    improvement_tol: float


class ModelStateDict(t.TypedDict):
    best_loss: float
    epochs_trained: int


class FullStateDict(t.TypedDict):
    state: ModelStateDict
    hparams: HyperParamTotalDict


class ABCNN(ABC):
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

    @abstractmethod
    def set_model(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def train_model(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float | t.Any: ...

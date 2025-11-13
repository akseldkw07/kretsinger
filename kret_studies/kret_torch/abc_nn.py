from __future__ import annotations
import pathlib
import typing as t
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ABCNN(ABC):
    @property
    @abstractmethod
    def root_dir(self) -> pathlib.Path: ...

    @property
    @abstractmethod
    def model_paths(self) -> t.Dict[str, pathlib.Path]: ...

    @property
    @abstractmethod
    def FullStateDict(self) -> t.Dict[str, t.Any]: ...

    @property
    @abstractmethod
    def FullStateDictDisplay(self) -> t.Dict[str, t.Any]: ...

    @property
    @abstractmethod
    def criterion(self) -> nn.Module: ...

    @abstractmethod
    def set_model(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def post_init(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    @abstractmethod
    def get_loss(self, *args, **kwargs) -> torch.Tensor: ...

    @abstractmethod
    def train_model(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float | t.Any: ...

    @abstractmethod
    def save_weights(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def load_weights(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def name(self, *args, **kwargs) -> str: ...

    @abstractmethod
    def summary(self, *args, **kwargs) -> None: ...

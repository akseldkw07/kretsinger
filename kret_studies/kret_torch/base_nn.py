from __future__ import annotations

import logging
import pathlib
import typing as t

import torch
import torch.nn as nn

from kret_studies.kret_torch.abc_nn import ABCNN
from kret_studies.kret_torch.constants import DEVICE_TORCH_STR

LOAD_LTRL = t.Literal["assert", "try", "fresh"]


class HyperParamDict(t.TypedDict, total=False):
    lr: float
    # early stopping controls
    patience: int
    improvement_tol: float


class HyperParamTotalDict(t.TypedDict, total=True):
    lr: float
    # early stopping controls
    patience: int
    improvement_tol: float


class ModelStateDict(t.TypedDict):
    best_loss: float
    epochs_trained: int


class FullStateDict(t.TypedDict):
    state: ModelStateDict
    hparams: HyperParamTotalDict


class ModelPathDict(t.TypedDict):
    model_path: pathlib.Path
    state_path: pathlib.Path
    weight_path: pathlib.Path


# Default training state
DEFAULT_HYPER_PARAMS = HyperParamTotalDict(lr=1e-3, patience=25, improvement_tol=1e-4)
DEFAULT_MODEL_STATE = ModelStateDict(best_loss=float("inf"), epochs_trained=0)


class BaseNN(ABCNN, nn.Module):
    version: str = "v000"
    model: nn.Module  # nn.Sequential or other nn.Module TODO define in subclass
    optimizer: torch.optim.Optimizer  # NOTE: NOT SET in __init__
    scheduler: torch.optim.lr_scheduler.LRScheduler  # NOTE: NOT SET in __init__
    device: t.Literal["cuda", "mps", "xpu", "cpu"]  # DEVICE_TORCH_STR
    _criterion: nn.Module  # TODO define

    hparams: HyperParamTotalDict

    _load_weights_act: t.Literal["assert", "try", "fresh"]
    _post_init_done: bool = False
    model_state: ModelStateDict
    _log: bool

    def __init__(self, log: bool = True, **hparams: t.Unpack[HyperParamDict]):
        super().__init__()

        # Initialize logging and model components
        self.logger = logging.getLogger(self.name())
        self.device = DEVICE_TORCH_STR

        # Initialize default Hyperparameters (may be overridden by load)
        hp = DEFAULT_HYPER_PARAMS.copy()
        hp.update(hparams)
        self.hparams = hp

        # Initialize default training state (may be overridden by load)
        self.model_state = DEFAULT_MODEL_STATE.copy()
        self._log = log

        self.to(self.device)

    @property
    def criterion(self):
        return self._criterion.to(self.device)

    def set_model(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement set_model().")

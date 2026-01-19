from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import Literal, TypedDict

import optuna
from lightning import Callback, LightningDataModule
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.loggers import CSVLogger, Logger
from lightning.pytorch.plugins import LayerSync, Precision
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy
from optuna.integration import PyTorchLightningPruningCallback

from kret_lightning.abc_lightning import ABCLM
from kret_lightning.custom_callbacks import CallbackConfig


class TrainerStaticDefaults:
    MIN_EPOCHS: int = 5
    MAX_EPOCHS_TEST: int = 15

    CHECK_VAL_EVERY_N_EPOCH: int = 1
    LOG_EVERY_N_STEPS: int = 50

    ENABLE_CHECKPOINTING: bool = True
    ENABLE_PROGRESS_BAR: bool = True
    ENABLE_MODEL_SUMMARY: bool = True

    VAL_CHECK_INTERVAL: int | float = 1

    BAREBONES: bool = False

    # for fast dev run testing
    TRAINER_QUICK_ITER: Trainer___init___TypedDict = {
        "min_epochs": 5,
        "max_epochs": 5,
        "check_val_every_n_epoch": 1,
        "log_every_n_steps": 10,
        # "limit_train_batches": 0.1,
        "limit_val_batches": 0.1,
        "limit_test_batches": 0.1,
    }

    # for first full run testing
    TRAINER_FIRST_FULL_RUN: Trainer___init___TypedDict = {}

    # for overnight full run testing
    TRAINER_OVERNIGHT_FULL_RUN: Trainer___init___TypedDict = {}

    # for debugging
    TRAINER_DEBUG: Trainer___init___TypedDict = {
        "detect_anomaly": True,
        "fast_dev_run": True,
    }

    TRAINER_FIT: Trainer_Fit_TypedDict = {"ckpt_path": "best", "weights_only": False}


class TrainerDynamicDefaults:
    @classmethod
    def trainer_dynamic_defaults(
        cls, nn: ABCLM, datamodule: LightningDataModule, trial: optuna.trial.Trial | None = None
    ):

        logger = CSVLogger(**nn.save_load_logging_dict)
        checkpoints = CallbackConfig.trainer_dynamic_defaults(nn, datamodule)

        if trial is not None:
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
            checkpoints.append(pruning_callback)

        ret: Trainer___init___TypedDict = {"logger": logger, "default_root_dir": nn.ckpt_path, "callbacks": checkpoints}
        return ret


# region TYPES
"""
TYPES
"""


class Trainer___init___TypedDict(TypedDict, total=False):
    accelerator: str | Accelerator  # = 'auto'
    strategy: str | Strategy  # = 'auto'
    devices: list[int] | str | int  # = 'auto'
    num_nodes: int  # = 1
    precision: (
        Literal[64, 32, 16]
        | Literal[
            "transformer-engine",
            "transformer-engine-float16",
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
        ]
        | Literal["64", "32", "16", "bf16"]
        | None
    )  # = None
    logger: Logger | Iterable[Logger] | bool | None  # = None
    callbacks: list[Callback] | Callback | None  # = None
    fast_dev_run: int | bool  # = False
    max_epochs: int | None  # = None
    min_epochs: int | None  # = None
    max_steps: int  # = -1
    min_steps: int | None  # = None
    max_time: str | timedelta | dict[str, int] | None  # = None
    limit_train_batches: int | float | None  # = None
    limit_val_batches: int | float | None  # = None
    limit_test_batches: int | float | None  # = None
    limit_predict_batches: int | float | None  # = None
    overfit_batches: int | float  # = 0.0
    val_check_interval: int | float | str | timedelta | dict[str, int] | None  # = None
    check_val_every_n_epoch: int | None  # = 1
    num_sanity_val_steps: int | None  # = None
    log_every_n_steps: int | None  # = None
    enable_checkpointing: bool | None  # = None
    enable_progress_bar: bool | None  # = None
    enable_model_summary: bool | None  # = None
    accumulate_grad_batches: int  # = 1
    gradient_clip_val: int | float | None  # = None
    gradient_clip_algorithm: str | None  # = None
    deterministic: bool | Literal["warn"] | None  # = None
    benchmark: bool | None  # = None
    inference_mode: bool  # = True
    use_distributed_sampler: bool  # = True
    profiler: Profiler | str | None  # = None
    detect_anomaly: bool  # = False
    barebones: bool  # = False
    plugins: (
        Precision
        | ClusterEnvironment
        | CheckpointIO
        | LayerSync
        | list[Precision | ClusterEnvironment | CheckpointIO | LayerSync]
        | None
    )  # = None
    sync_batchnorm: bool  # = False
    reload_dataloaders_every_n_epochs: int  # = 0
    default_root_dir: str | Path | None  # = None
    enable_autolog_hparams: bool  # = True
    model_registry: str | None  # = None


class Trainer_Fit_TypedDict(TypedDict, total=False):
    # model: LightningModule
    # train_dataloaders: Any | LightningDataModule | None  # = None
    # val_dataloaders: Any | None  # = None
    # datamodule: LightningDataModule | None  # = None
    ckpt_path: str | Path | None  # = None
    weights_only: bool | None  # = None


# endregion

from __future__ import annotations
import collections.abc
import datetime
import pathlib
import typing as t

import lightning
import lightning as L
import lightning.fabric.plugins.environments.cluster_environment
import lightning.fabric.plugins.io.checkpoint_io
import lightning.pytorch
import lightning.pytorch.accelerators.accelerator
import lightning.pytorch.callbacks.callback
import lightning.pytorch.core.datamodule
import lightning.pytorch.loggers.logger
import lightning.pytorch.plugins.layer_sync
import lightning.pytorch.plugins.precision.precision
import lightning.pytorch.profilers.profiler
import lightning.pytorch.strategies.strategy


class TrainingDefaults:
    MIN_EPOCHS: int = 5

    CHECK_VAL_EVERY_N_EPOCHS: int = 1
    LOG_EVERY_N_STEPS: int = 50

    ENABLE_CHECKPOINTING: bool = True
    ENABLE_PROGRESS_BAR: bool = True
    ENABLE_MODEL_SUMMARY: bool = True

    VAL_CHECK_INTERVAL: int | float = 1

    BAREBONES: bool = False


# region TYPES
"""
TYPES
"""


class Lightning__Trainer__init___TypedDict(t.TypedDict, total=False):
    self: t.Any
    accelerator: str | lightning.pytorch.accelerators.accelerator.Accelerator
    strategy: str | lightning.pytorch.strategies.strategy.Strategy
    devices: list[int] | str | int
    num_nodes: int
    precision: (
        t.Literal[64, 32, 16]
        | t.Literal[
            "transformer-engine",
            "transformer-engine-float16",
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
        ]
        | t.Literal["64", "32", "16", "bf16"]
        | None
    )
    logger: (
        lightning.pytorch.loggers.logger.Logger
        | collections.abc.Iterable[lightning.pytorch.loggers.logger.Logger]
        | bool
        | None
    )
    callbacks: (
        list[lightning.pytorch.callbacks.callback.Callback] | lightning.pytorch.callbacks.callback.Callback | None
    )
    fast_dev_run: int | bool
    max_epochs: int | None
    min_epochs: int | None
    max_steps: int
    min_steps: int | None
    max_time: str | datetime.timedelta | dict[str, int] | None
    limit_train_batches: float | int | None
    limit_val_batches: float | int | None
    limit_test_batches: float | int | None
    limit_predict_batches: float | int | None
    overfit_batches: int | float
    val_check_interval: int | float | str | datetime.timedelta | dict[str, int] | None
    check_val_every_n_epoch: int | None
    num_sanity_val_steps: int | None
    log_every_n_steps: int | None
    enable_checkpointing: bool | None
    enable_progress_bar: bool | None
    enable_model_summary: bool | None
    accumulate_grad_batches: int
    gradient_clip_val: float | int | None
    gradient_clip_algorithm: str | None
    deterministic: bool | t.Literal["warn"] | None
    benchmark: bool | None
    inference_mode: bool
    use_distributed_sampler: bool
    profiler: lightning.pytorch.profilers.profiler.Profiler | str | None
    detect_anomaly: bool
    barebones: bool
    plugins: (
        lightning.pytorch.plugins.precision.precision.Precision
        | lightning.fabric.plugins.environments.cluster_environment.ClusterEnvironment
        | lightning.fabric.plugins.io.checkpoint_io.CheckpointIO
        | lightning.pytorch.plugins.layer_sync.LayerSync
        | list[
            (
                lightning.pytorch.plugins.precision.precision.Precision
                | lightning.fabric.plugins.environments.cluster_environment.ClusterEnvironment
                | lightning.fabric.plugins.io.checkpoint_io.CheckpointIO
                | lightning.pytorch.plugins.layer_sync.LayerSync
            )
        ]
        | None
    )
    sync_batchnorm: bool
    reload_dataloaders_every_n_epochs: int
    default_root_dir: str | pathlib.Path | None
    enable_autolog_hparams: bool
    model_registry: str | None


class Fit_TypedDict(t.TypedDict, total=False):
    self: t.Any
    model: L.LightningModule
    train_dataloaders: t.Any | lightning.pytorch.core.datamodule.LightningDataModule | None
    val_dataloaders: t.Any | None
    datamodule: lightning.pytorch.core.datamodule.LightningDataModule | None
    ckpt_path: str | pathlib.Path | None
    weights_only: bool | None


# endregion

import lightning as L
import torch
import typing as t
import torch.nn as nn
from lightning.fabric.utilities.data import AttributeDict


class TrainingDefaults:
    MIN_EPOCHS: int = 5

    CHECK_VAL_EVERY_N_EPOCHS: int = 1
    LOG_EVERY_N_STEPS: int = 50

    ENABLE_CHECKPOINTING: bool = True
    ENABLE_PROGRESS_BAR: bool = True
    ENABLE_MODEL_SUMMARY: bool = True

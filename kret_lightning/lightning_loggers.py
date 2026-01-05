from __future__ import annotations
from lightning.pytorch.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from pathlib import Path
import typing as t
from abc import ABC, abstractmethod

import lightning as L
import torch
import torch.nn as nn

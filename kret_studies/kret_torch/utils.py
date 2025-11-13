from functools import cache
from math import ceil, log

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from kret_studies.helpers.numpy_utils import SingleReturnArray


@cache
def _exp_decay(required_len: int, initial_epsilon: float = 0.95, half_life: float = 1000, min_value: float = 0.01):
    t = np.arange(required_len)
    decay = np.exp(-np.log(2.0) * t / half_life)  # exp with half-life semantics
    arr: SingleReturnArray[np.float32] = np.maximum(min_value, initial_epsilon * decay).astype(np.float32)
    return arr


def exp_decay(episode: int, initial_epsilon: float = 0.95, half_life: float = 1000.0, min_value: float = 0.01):
    eff_episode = 2 ** (ceil(log(episode + 1, 2)))
    arr = _exp_decay(eff_episode, initial_epsilon, half_life, min_value)

    return arr[episode]


def freeze_model_weights(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model_weights(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True


XTYPE = pd.DataFrame | np.ndarray | torch.Tensor
YTYPE = pd.Series | np.ndarray | torch.Tensor


def make_loader_from_xy(X: XTYPE, y: YTYPE, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
    """
    Convert (X, y) in various common formats into a PyTorch DataLoader.
    Supported:
        - X: pd.DataFrame | np.ndarray | torch.Tensor
        - y: pd.Series    | np.ndarray | torch.Tensor
    """
    # X -> numpy
    if isinstance(X, pd.DataFrame):
        X = X.astype(np.float32).to_numpy()
    if isinstance(X, np.ndarray):
        X_tensor = torch.from_numpy(X).float()
    elif isinstance(X, torch.Tensor):
        X_tensor = X.float()
    else:
        raise TypeError(f"Unsupported X type: {type(X)}")

    # y -> numpy
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
    if isinstance(y, np.ndarray):
        y_tensor = torch.from_numpy(y).float()  # [N]
    elif isinstance(y, torch.Tensor):
        y_tensor = y.float()
    else:
        raise TypeError(f"Unsupported y type: {type(y)}")

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

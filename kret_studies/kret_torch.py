from collections.abc import Callable
from functools import cache
from math import ceil, log

import numpy as np
import torch

LossSpec = str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
from kret_studies.helpers.numpy_utils import SingleReturnArray
from kret_studies.helpers.torch_helper import train_regression  # autoflake: ignore


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

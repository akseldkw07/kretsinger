import typing as t
from functools import cache

import numpy as np

if t.TYPE_CHECKING:
    from kret_studies.helpers.numpy_utils import SingleReturnArray


def get_gamma_from_half_life(half_life: float) -> float:
    """
    Utility to convert half-life to gamma, half-life is easier to reason about.
    """
    return 2.0 ** (-1.0 / half_life)


@cache
def _exp_decay_half_life_cache(length: int, half_life: float, gamma: float):
    gamma = gamma or get_gamma_from_half_life(half_life)
    t_ = np.arange(length, dtype=np.float32)
    arr = t.cast("SingleReturnArray[np.float32]", gamma**t_)
    return arr


def exp_decay_half_life(length: int, half_life: float | None = None, gamma: float | None = None):
    """
    Exponential decay array with specified half-life.
    """
    assert (half_life is not None) or (gamma is not None), "Either half_life or gamma must be provided."
    return _exp_decay_half_life_cache(length=length, half_life=half_life, gamma=gamma)

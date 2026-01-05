import typing as t
from functools import cache
from math import ceil, log

import numpy as np

if t.TYPE_CHECKING:
    from kret_studies.helpers.numpy_utils import SingleReturnArray


@cache
def _exp_decay(required_len: int, initial_epsilon: float = 0.95, half_life: float = 1000, min_value: float = 0.01):
    t_ = np.arange(required_len)
    decay = np.exp(-np.log(2.0) * t_ / half_life)  # exp with half-life semantics
    arr = np.maximum(min_value, initial_epsilon * decay).astype(np.float32)
    return t.cast("SingleReturnArray[np.float32]", arr)


def exp_decay(episode: int, initial_epsilon: float = 0.95, half_life: float = 1000.0, min_value: float = 0.01):
    eff_episode = 2 ** (ceil(log(episode + 1, 2)))
    arr = _exp_decay(eff_episode, initial_epsilon, half_life, min_value)

    return arr[episode]

import typing as t
from functools import cache

import numpy as np

if t.TYPE_CHECKING:
    from kret_np_pd.single_ret_ndarray import SingleReturnArray


class NP_ExpDecay_Utils:
    @classmethod
    def get_gamma_from_half_life(cls, half_life: float) -> float:
        """
        Utility to convert half-life to gamma, half-life is easier to reason about.
        """
        return 2.0 ** (-1.0 / half_life)

    @classmethod
    @cache
    def _exp_decay_half_life_cache(cls, length: int, half_life: float, gamma: float):
        gamma = gamma or cls.get_gamma_from_half_life(half_life)
        t_ = np.arange(length, dtype=np.float32)
        arr = t.cast("SingleReturnArray[np.float32]", gamma**t_)
        return arr

    @classmethod
    def exp_decay_half_life(cls, length: int, half_life: float | None = None, gamma: float | None = None):
        """
        Exponential decay array with specified half-life.
        """
        assert (half_life is not None) or (gamma is not None), "Either half_life or gamma must be provided."
        return cls._exp_decay_half_life_cache(length=length, half_life=half_life, gamma=gamma)

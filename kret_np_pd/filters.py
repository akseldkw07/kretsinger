import typing as t
from functools import cache

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_bool_dtype

from kret_np_pd.single_ret_ndarray import SingleReturnArray
from kret_rosetta.UTILS_rosetta import UTILS_rosetta

FILT_TYPE = np.ndarray | pd.Series | torch.Tensor | pd.DataFrame


class FilterSampleUtils:
    @classmethod
    def process_filter(cls, filter: FILT_TYPE | None, shape: tuple[int, int] | tuple[int] | int | None = None):
        if filter is None:
            assert shape is not None, "Shape must be provided when filter is None"
            ret = np.full((shape[0] if isinstance(shape, tuple) else shape), True)
        else:
            ret = UTILS_rosetta.coerce_to_ndarray(filter, assert_1dim=True, attempt_flatten_1d=True)
        cls.assert_bool_dtype(ret)

        return t.cast(SingleReturnArray[bool], ret)

    @classmethod
    @cache
    def gen_sample_filter(cls, hot: int, total_size: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        indices = rng.choice(total_size, size=hot, replace=False)
        ret = np.full(total_size, False)
        ret[indices] = True

        cls.assert_bool_dtype(ret)

        return t.cast(SingleReturnArray[bool], ret)

    @classmethod
    def downsample_bool(cls, ret: np.ndarray, k: int, seed: int | None = None):
        """
        Randomly flip True values to False so that sum(arr) == k.
        If arr.sum() <= k, returns arr unchanged.
        """
        cls.assert_bool_dtype(ret)
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        idx = np.flatnonzero(ret.copy())

        if len(idx) <= k:
            return t.cast(SingleReturnArray[bool], ret)

        keep = rng.choice(idx, size=k, replace=False)
        ret[:] = False
        ret[keep] = True

        return t.cast(SingleReturnArray[bool], ret)

    @classmethod
    def assert_bool_dtype(cls, arr: np.ndarray):
        assert is_bool_dtype(arr), f"Expected boolean filter type, got {arr.dtype}"

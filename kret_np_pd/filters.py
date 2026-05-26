import typing as t
from functools import cache

import numpy as np
import pandas as pd
import polars as pl
import torch
from pandas.api.types import is_bool_dtype

from kret_np_pd.single_ret_ndarray import SingleReturnArray
from kret_rosetta.UTILS_rosetta import UTILS_rosetta

FILT_NOT_NONE = np.ndarray | pd.Series | torch.Tensor | pd.DataFrame | pl.Series | pl.DataFrame
FILT_TYPE = FILT_NOT_NONE | None
SHAPE_TYPE = tuple[int, ...] | tuple[int] | int


class FilterSampleUtils:
    # Overloads: (None, None) → None ; (None, shape) → ndarray ; (filter, ...) → ndarray.
    # Plain np.ndarray return (not SingleReturnArray) so fancy-indexing keys like pd.Index work.
    @t.overload
    @classmethod
    def process_filter(cls, filter: None, shape: None = ...) -> None: ...  # type: ignore[overload-overlap]
    @t.overload
    @classmethod
    def process_filter(cls, filter: None, shape: SHAPE_TYPE) -> np.ndarray: ...
    @t.overload
    @classmethod
    def process_filter(cls, filter: FILT_NOT_NONE, shape: SHAPE_TYPE | None = ...) -> np.ndarray: ...

    @classmethod
    def process_filter(cls, filter: FILT_TYPE, shape: SHAPE_TYPE | None = None) -> np.ndarray | None:
        if filter is None:
            if shape is None:
                return None
            ret = np.full((shape[0] if isinstance(shape, tuple) else shape), True)
        else:
            ret = UTILS_rosetta.coerce_to_ndarray(filter, assert_1dim=True, attempt_flatten_1d=True)
        cls.assert_bool_dtype(ret)
        return ret

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

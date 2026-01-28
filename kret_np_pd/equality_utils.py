import numpy as np
import pandas as pd
import torch

from kret_rosetta.UTILS_rosetta import UTILS_rosetta
from kret_utils.assert_type import TypeAssert

from .filters import FilterSampleUtils

ARG_TYPE = pd.DataFrame | pd.Series | np.ndarray | pd.Categorical | torch.Tensor
TABULAR_TYPE = pd.DataFrame | torch.Tensor  # technically np.ndarray too, but we'll treat as array for now
ARR_TYPE = np.ndarray | pd.Series | pd.Categorical


class EqualityUtils:
    @classmethod
    def _should_coerce_np(cls, arg: ARG_TYPE, try_coerce_np: bool | None = None):
        if isinstance(try_coerce_np, bool):
            return try_coerce_np
        if isinstance(arg, pd.Categorical):
            return True
        if isinstance(arg, (pd.DataFrame, torch.Tensor)):
            if arg.ndim > 1:
                return False
            else:
                return True
        return False

    @classmethod
    def is_close(
        cls,
        a: ARG_TYPE,
        b: ARG_TYPE,
        filt: np.ndarray | None = None,
        nan_true: bool = True,
        rtol: float = 1e-05,
        try_coerce_np: bool | None = None,
    ):
        """
        Compare two arrays/dataframes element-wise for closeness within a tolerance.

        1. If `filt` is provided, only compare elements where `filt` is True.
        2. If `try_coerce_np` is True, attempt to convert inputs to ndarray. If None, decide based on input properties.
        3. Assert that shapes match after any coercion.
        4. Return a boolean array/dataframe indicating closeness.
        """
        TypeAssert.assert_type(a, ARG_TYPE)
        TypeAssert.assert_type(b, ARG_TYPE)
        filt = FilterSampleUtils.process_filter(filt, a.shape)

        def np_isclose(arr1: ARR_TYPE, arr2: ARR_TYPE) -> np.ndarray:
            ret = np.where(filt, np.isclose(arr1, arr2, rtol=rtol, equal_nan=nan_true), -1)
            return ret

        try_coerce_np_a = cls._should_coerce_np(a, try_coerce_np)
        try_coerce_np_b = cls._should_coerce_np(b, try_coerce_np)
        a = UTILS_rosetta.coerce_to_ndarray(a, assert_1dim=True, attempt_flatten_1d=True) if try_coerce_np_a else a
        b = UTILS_rosetta.coerce_to_ndarray(b, assert_1dim=True, attempt_flatten_1d=True) if try_coerce_np_b else b

        assert (
            a.shape == b.shape
        ), f"Shapes must match for is_close comparison, got {a.shape} and {b.shape}. Types are {type(a)} and {type(b)}. Try setting try_coerce_np=True."

        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            assert set(a.columns) == set(b.columns), "DataFrames must have the same columns for is_close comparison."
            ret = pd.DataFrame()
            for col in a.columns:
                ret[col] = np_isclose(a[col], b[col])
            return ret
        elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            a_np = UTILS_rosetta.coerce_to_ndarray(a, assert_1dim=False, attempt_flatten_1d=True)
            b_np = UTILS_rosetta.coerce_to_ndarray(b, assert_1dim=False, attempt_flatten_1d=True)
            return np_isclose(a_np, b_np)
        elif isinstance(a, ARR_TYPE) and isinstance(b, ARR_TYPE):
            return np_isclose(a, b)
        else:
            raise TypeError(
                f"Both inputs must be of the same type, either np.ndarray or pd.DataFrame. Got {type(a)} and {type(b)}. Try setting try_coerce_np=True."
            )

import numpy as np
import pandas as pd

from kret_rosetta.UTILS_rosetta import UTILS_rosetta

from .filters import FilterSampleUtils


class EqualityUtils:
    @classmethod
    def is_close(
        cls,
        a: np.ndarray | pd.DataFrame,
        b: np.ndarray | pd.DataFrame,
        filt: np.ndarray | None = None,
        nan_true: bool = True,
        rtol: float = 1e-05,
        try_coerce_1d: bool = False,
    ):
        filt = FilterSampleUtils.process_filter(filt, a.shape)

        def np_isclose(arr1: np.ndarray | pd.Series, arr2: np.ndarray | pd.Series) -> np.ndarray:
            ret = np.where(filt, np.isclose(arr1, arr2, rtol=rtol, equal_nan=nan_true), -1)
            return ret

        if try_coerce_1d:
            a = UTILS_rosetta.coerce_to_ndarray(a, assert_1dim=True, attempt_flatten_1d=True)
            b = UTILS_rosetta.coerce_to_ndarray(b, assert_1dim=True, attempt_flatten_1d=True)

        assert (
            a.shape == b.shape
        ), f"Shapes must match for is_close comparison, got {a.shape} and {b.shape}. Types are {type(a)} and {type(b)}."

        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            ret = pd.DataFrame()
            for col in a.columns:
                ret[col] = np_isclose(a[col], b[col])
            return ret
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np_isclose(a, b)
        else:
            raise TypeError(
                f"Both inputs must be of the same type, either np.ndarray or pd.DataFrame. Got {type(a)} and {type(b)}."
            )

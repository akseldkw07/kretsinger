import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_bool_dtype

from kret_np_pd.filters import FilterUtils
from kret_rosetta.UTILS_rosetta import UTILS_rosetta


class FilterUtils:
    @classmethod
    def process_filter(
        cls, filter: np.ndarray | pd.Series | torch.Tensor | pd.DataFrame | None, shape: tuple[int] | None = None
    ):
        if filter is None:
            assert shape is not None, "Shape must be provided when filter is None"
            return np.full(shape[0], True)

        ret = UTILS_rosetta.coerce_to_ndarray(filter, assert_1dim=True, attempt_flatten_1d=True)
        assert is_bool_dtype(ret), f"Expected boolean filter type, got {ret.dtype}"

        return ret

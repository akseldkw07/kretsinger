"""
Utility class to convert to pandas and numpy
"""

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype
from torch.utils.data import TensorDataset

from .conversion_protocols import PandasConvertibleWithColumns

TO_NP_TYPE = pd.DataFrame | pd.Series | np.ndarray | torch.Tensor | list | tuple
TO_PD_TYPE = pd.DataFrame | pd.Series | np.ndarray | list | tuple | object | torch.Tensor | TensorDataset


class To_NP_PD:
    @classmethod
    def coerce_to_df(cls, obj: TO_PD_TYPE, cols: list[str] | None = None):
        if isinstance(obj, PandasConvertibleWithColumns):
            # This covers TensorDatasetCustom and any other custom types implementing the protocol
            ret = obj.to_pandas()
        elif isinstance(obj, pd.DataFrame):
            ret = obj
        elif isinstance(obj, pd.Series):
            ret = obj.to_frame()
            ret.columns = cols if cols is not None else [obj.name]
        elif isinstance(obj, np.ndarray):
            ret = pd.DataFrame({i: obj[:, i] for i in range(obj.shape[1])}) if obj.ndim > 1 else pd.DataFrame({0: obj})
        elif isinstance(obj, (list, tuple)):
            ret = pd.DataFrame(obj)
        elif isinstance(obj, torch.Tensor):
            ret = pd.DataFrame(obj.numpy(force=True))
        elif isinstance(obj, torch.utils.data.TensorDataset):
            data_list = [obj[i] for i in range(len(obj))]
            ret = pd.DataFrame(data_list)

        else:
            ret = pd.DataFrame([obj])

        ret.columns = cols if cols is not None else ret.columns
        return ret

    @classmethod
    def df_to_np_safe(cls, df: pd.DataFrame):
        for col in df.columns:
            arr = df[col]
            if is_datetime64_any_dtype(arr):
                df[col] = df[col].to_numpy(dtype="datetime64[ns]").view("int64") / 1e9  # int64
            elif is_timedelta64_dtype(arr):
                df[col] = df[col].to_numpy(dtype="timedelta64[ns]").view("int64") / 1e9  # int64

        ret = df.to_numpy()
        return ret

    @classmethod
    def coerce_to_ndarray(cls, arr: TO_NP_TYPE, assert_1dim: bool = False, attempt_flatten_1d: bool = True):
        """
        Convert to np.ndarray, with optional dimensionality check
        """
        from kret_np_pd.categoricals import CategoricalUtils

        if isinstance(arr, torch.Tensor):
            ret = arr.numpy(force=True)
        elif isinstance(arr, pd.DataFrame):
            ret = cls.df_to_np_safe(arr)
        elif isinstance(arr, pd.Series):
            ret = CategoricalUtils.to_numpy_cat(arr)
        elif isinstance(arr, (list, tuple)):
            ret = np.array(arr)
        else:
            raise ValueError(f"Type {type(arr)} not accepted")

        if attempt_flatten_1d and len(ret.shape) > 1 and ret.shape[1] == 1:
            ret = ret.flatten()

        assert not assert_1dim or ret.ndim == 1, f"Expected 1-dim ndarray output, got{ret.shape}"
        return ret

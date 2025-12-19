import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype


class PD_NP_Torch_Translation:
    @classmethod
    def coerce_to_df(cls, obj: pd.DataFrame | pd.Series | np.ndarray | list | tuple | object | torch.Tensor):
        if isinstance(obj, pd.DataFrame):
            ret = obj
        elif isinstance(obj, pd.Series):
            ret = obj.to_frame()
        elif isinstance(obj, np.ndarray):
            ret = pd.DataFrame({i: obj[:, i] for i in range(obj.shape[1])}) if obj.ndim > 1 else pd.DataFrame({0: obj})
        elif isinstance(obj, (list, tuple)):
            ret = pd.DataFrame(obj)
        elif isinstance(obj, torch.Tensor):
            ret = pd.DataFrame(obj.detach().cpu().numpy())
        else:
            ret = pd.DataFrame([obj])

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
    def coerce_to_ndarray(cls, obj: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor | list | tuple): ...

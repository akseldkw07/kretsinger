import warnings

import numpy as np
import pandas as pd
import torch

from .to_pd_np import To_NP_PD


class To_Tensor:
    @classmethod
    def coerce_to_tensor(
        cls, obj: pd.DataFrame | np.ndarray | torch.Tensor | pd.Series, dtype=torch.float32
    ) -> torch.Tensor:
        """
        Coerce various data types to a PyTorch tensor.
        """
        if isinstance(obj, pd.DataFrame):
            warnings.warn(
                "Converting DataFrame to tensor. Ensure all columns are numeric to avoid unexpected results.",
                UserWarning,
            )
        if isinstance(obj, torch.Tensor):
            return obj

        df = To_NP_PD.coerce_to_df(obj)
        tensor = torch.tensor(df.values, dtype=dtype)
        return tensor

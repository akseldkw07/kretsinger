import numpy as np
import pandas as pd
import torch

from .to_pd_np import To_NP_PD


class To_Tensor:
    @classmethod
    def coerce_to_tensor_ds(cls, obj: pd.DataFrame | np.ndarray | torch.Tensor | pd.Series, dtype=torch.float32):
        if isinstance(obj, torch.utils.data.TensorDataset):
            return obj
        elif isinstance(obj, torch.Tensor):
            tensor = obj
        elif isinstance(obj, pd.DataFrame):
            tensor = torch.tensor(To_NP_PD.df_to_np_safe(obj), dtype=dtype)
        elif isinstance(obj, pd.Series):
            tensor = torch.tensor(obj.to_numpy(), dtype=dtype)
        elif isinstance(obj, np.ndarray):
            tensor = torch.tensor(obj, dtype=dtype)
        else:
            raise ValueError(f"Type {type(obj)} not accepted")

        dataset = torch.utils.data.TensorDataset(tensor)
        return dataset

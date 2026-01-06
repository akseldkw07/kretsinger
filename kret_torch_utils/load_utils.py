import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

XTYPE = pd.DataFrame | np.ndarray | torch.Tensor
YTYPE = pd.Series | np.ndarray | torch.Tensor


class TorchLoadUtils:
    """
    NOTE probably should use L.LightningDataModule instead of this class
    """

    @classmethod
    def make_loader_from_xy(cls, X: XTYPE, y: YTYPE, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
        """
        Convert (X, y) in various common formats into a PyTorch DataLoader.
        Supported:
            - X: pd.DataFrame | np.ndarray | torch.Tensor
            - y: pd.Series    | np.ndarray | torch.Tensor
        """
        # X -> tensor
        if isinstance(X, pd.DataFrame):
            X = X.copy()

            # Turn categorical / object columns into integer codes
            cat_cols = X.select_dtypes(include=["category", "object"]).columns
            for col in cat_cols:
                X[col] = X[col].astype("category").cat.codes

            X = X.astype(np.float32).to_numpy()

        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

        # y -> tensor
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.copy()
            # If y is categorical/string, encode to integer class indices
            if str(y.dtype) in ("category", "object"):
                y = y.astype("category").cat.codes
            y = y.to_numpy()

        if isinstance(y, np.ndarray):
            # keep a writable copy to avoid the non-writable warning
            y = np.array(y, copy=True)

            if np.issubdtype(y.dtype, np.integer):
                # flatten to 1D for CrossEntropyLoss
                y_tensor = torch.from_numpy(y).long().view(-1)
            else:
                y_tensor = torch.from_numpy(y).float()
        elif isinstance(y, torch.Tensor):
            if y.dtype in (torch.int32, torch.int64):
                y_tensor = y.long().view(-1)
            else:
                y_tensor = y.float()
        else:
            raise TypeError(f"Unsupported y type: {type(y)}")

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

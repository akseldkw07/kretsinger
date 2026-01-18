"""
PandasTensorDataset: Seamless conversion between pandas DataFrames and PyTorch TensorDataset
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from kret_rosetta.UTILS_rosetta import UTILS_rosetta


class TensorDatasetCustom(TensorDataset):
    """
    A TensorDataset subclass that preserves column names and enables
    seamless conversion between pandas DataFrames and PyTorch tensors.

    Stores column names internally and provides properties/methods for
    easy conversion back to pandas.
    """

    _columns: tuple[list[str], ...]

    def __init__(self, *tensors, columns: tuple[list[str], ...] | None = None):
        super().__init__(*tensors)

        cols_manual = tuple([[f"col_{i}" for i in range(tensor.shape[1])] for tensor in tensors])
        self._columns = columns if columns is not None else cols_manual
        assert len(self._columns) == len(self.tensors), "Number of column sets must match number of tensors."

    @property
    def columns(self) -> tuple[list[str], ...]:
        """Get column names."""
        return self._columns

    @property
    def shape(self) -> tuple:
        """Get shape as (num_rows, num_columns)."""
        return (len(self), sum(len(cols) for cols in self.columns))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TensorDatasetCustom(shape={self.shape}, "
            f"columns={self.columns}, "
            f"dtype={[t.dtype for t in self.tensors]})"
        )

    @staticmethod
    def from_pd_xy(
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame | np.ndarray | None = None,
        dtype=torch.float32,
        label_dtype=torch.float32,
    ) -> "TensorDatasetCustom":
        columns = (list(X.columns),)
        tensors = [torch.tensor(X.values, dtype=dtype)]

        if y is not None:
            y_df = UTILS_rosetta.coerce_to_df(y)
            y_cols = y_df.columns.tolist()
            y_cols = [f"label_{col}" for col in y_cols] if isinstance(y, np.ndarray) else y_cols
            y_tensors = [torch.tensor(y_df.values, dtype=label_dtype)]

            tensors.extend(y_tensors)
            columns += (y_cols,)
        return TensorDatasetCustom(*tensors, columns=columns)

    @staticmethod
    def from_pd(
        df: pd.DataFrame,
        label_col: str | None = None,
        feature_cols: list[str] | None = None,
        dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.float32,
    ) -> "TensorDatasetCustom":
        # Determine feature columns
        if feature_cols is None:
            if label_col is not None:
                feature_cols = [col for col in df.columns if col != label_col]
            else:
                feature_cols = list(df.columns)

        # Create tensors
        tensors = []
        columns = (feature_cols,)

        # Add feature tensor
        feature_tensor = torch.tensor(df[feature_cols].values, dtype=dtype)
        tensors.append(feature_tensor)

        # Add label tensor if specified
        if label_col is not None:
            label_tensor = torch.tensor(df[label_col].values, dtype=label_dtype)
            tensors.append(label_tensor)
            columns += ([label_col],)

        return TensorDatasetCustom(*tensors, columns=columns)

    def to_pandas(self, copy: bool = False) -> pd.DataFrame:
        """
        Convert back to pandas DataFrame.

        Returns:
            pandas DataFrame with original column names

        Example:
            >>> dataset = TensorDatasetCustom.from_pandas(df)
            >>> df_recovered = dataset.to_pandas()
            >>> pd.testing.assert_frame_equal(df, df_recovered)
        """
        # Convert all tensors to numpy
        data = []
        for col, tensor in zip(self.columns, self.tensors):
            arr = tensor.numpy(force=True)
            df_part = pd.DataFrame(arr, columns=col, copy=copy)
            data.append(df_part)

        return pd.concat(data, axis=1)

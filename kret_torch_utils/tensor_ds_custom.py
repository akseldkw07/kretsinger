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

    _columns: list[str]

    def __init__(self, *tensors, columns: list[str] | None = None):
        super().__init__(*tensors)

        columns = [f"tensor_{i}" for i in range(len(tensors))] if columns is None else columns

        self._columns = columns

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return self._columns

    @property
    def shape(self) -> tuple:
        """Get shape as (num_rows, num_columns)."""
        return (len(self), len(self.tensors))

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

        y = UTILS_rosetta.coerce_to_df(y) if y is not None else None
        columns = list(X.columns) + (list(y.columns) if y is not None else [])
        tensors = [torch.tensor(X.values, dtype=dtype)]
        if y is not None:
            tensors.append(torch.tensor(y.values, dtype=label_dtype))
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
        columns = []

        # Add feature tensor
        feature_tensor = torch.tensor(df[feature_cols].values, dtype=dtype)
        tensors.append(feature_tensor)
        columns.extend(feature_cols)

        # Add label tensor if specified
        if label_col is not None:
            label_tensor = torch.tensor(df[label_col].values, dtype=label_dtype)
            tensors.append(label_tensor)
            columns.append(label_col)

        return TensorDatasetCustom(*tensors, columns=columns)

    def to_pandas(self) -> pd.DataFrame:
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
        data = {}
        for col, tensor in zip(self.columns, self.tensors):
            data[col] = tensor.numpy(force=True)

        return pd.DataFrame(data)

    def get_feature_tensor(self, col_name: str) -> torch.Tensor:
        """
        Get a specific column as a tensor.

        Args:
            col_name: Column name

        Returns:
            Tensor for that column

        Example:
            >>> ages = dataset.get_feature_tensor('age')
        """
        try:
            idx = self.columns.index(col_name)
            return self.tensors[idx]
        except ValueError:
            raise KeyError(f"Column '{col_name}' not found. Available: {self.columns}")

    def get_column_dtype(self, col_name: str) -> torch.dtype:
        """Get dtype of a specific column."""
        tensor = self.get_feature_tensor(col_name)
        return tensor.dtype

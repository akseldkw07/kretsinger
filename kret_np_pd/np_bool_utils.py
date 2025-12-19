import numpy as np
import pandas as pd
import torch
import typing as t

AnyAll = t.Literal["any", "all"]
IndexLabel = t.Hashable | t.Sequence[t.Hashable]


DF_SERIES_NP_TENSOR = pd.DataFrame | pd.Series | np.ndarray | torch.Tensor


class NP_Boolean_Utils:
    @classmethod
    def mask_and(cls, *masks: np.ndarray | pd.DataFrame):
        """
        Pass in a list of boolean masks (numpy arrays or pandas DataFrames) and return single mask that is the logical AND across all masks.
        """
        arr = [m.astype(bool).all(axis=1).T if isinstance(m, pd.DataFrame) else m.astype(bool) for m in masks]
        mask_and = np.logical_and.reduce(arr)
        assert isinstance(mask_and, np.ndarray)
        return mask_and

    @classmethod
    def mask_or(cls, *masks: np.ndarray | pd.DataFrame):
        """
        Pass in a list of boolean masks (numpy arrays or pandas DataFrames) and return single mask that is the logical OR across all masks.
        """
        arr = [m.astype(bool).any(axis=1).T if isinstance(m, pd.DataFrame) else m.astype(bool) for m in masks]
        mask_or = np.logical_or.reduce(arr)
        assert isinstance(mask_or, np.ndarray)
        return mask_or

    @classmethod
    def nan_filter(cls, *args: DF_SERIES_NP_TENSOR | None, how: AnyAll = "any"):
        """
        Create a boolean mask that is True where rows have NaN values according to the specified `how` across all inputs.

        If `how` is "any", the mask is True if any input has NaN in that row.
        If `how` is "all", the mask is True only if all inputs are NaN in that row.
        """
        masks: list[np.ndarray | pd.DataFrame] = []
        for arr in args:
            if arr is None:
                continue
            if isinstance(arr, torch.Tensor):
                arr = arr.numpy(force=True)
            assert isinstance(arr, (np.ndarray | pd.Series | pd.DataFrame))

            if isinstance(arr, np.ndarray):
                nan_mask = np.isnan(arr)
            elif (isinstance(arr, pd.DataFrame)) or isinstance(arr, pd.Series):
                nan_mask = arr.isna() if isinstance(arr, pd.DataFrame) else arr.isna().to_frame()
            else:
                raise TypeError(f"Unsupported type for nan_filter: {type(arr)}")

            masks.append(nan_mask)

        combined_mask = cls.mask_or(*masks) if how == "any" else cls.mask_and(*masks)
        return combined_mask

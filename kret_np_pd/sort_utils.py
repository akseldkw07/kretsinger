import numpy as np
import pandas as pd


class SortUtils:
    @classmethod
    def is_sorted(cls, arr: np.ndarray | pd.Series):

        is_sorted = arr.is_monotonic_increasing if isinstance(arr, pd.Series) else np.all(np.diff(arr) >= 0)
        return is_sorted

import numpy as np
import pandas as pd


class SortUtils:
    @classmethod
    def is_sorted(cls, arr: np.ndarray | pd.Series):
        """
        NOTE this is horrible inefficient, can be parallelized etc
        """
        is_sorted = np.array_equal(arr, np.sort(arr))
        return is_sorted

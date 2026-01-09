import numpy as np
import pandas as pd


class CategoricalUtils:
    @classmethod
    def is_categorical(cls, ser: pd.Series | np.ndarray) -> bool:
        """
        Check if a pandas Series is of categorical dtype.
        """
        return isinstance(ser.dtype, pd.CategoricalDtype)

    @classmethod
    def to_categorical(
        cls,
        ser: pd.Series | np.ndarray | pd.Categorical,
        categories: list | None = None,
        ordered: bool = False,
    ) -> pd.Categorical:
        """
        Convert a pandas Series or numpy ndarray to a pandas Categorical.
        If categories are provided, they are used to define the categorical levels.
        """
        if isinstance(ser, pd.Categorical):
            return ser

        cat = pd.Categorical(ser)

        if categories is not None:
            cat = cat.set_categories(categories, ordered=ordered)

        return cat

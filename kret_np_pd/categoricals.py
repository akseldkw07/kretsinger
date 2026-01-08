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
    def to_numpy_cat(cls, ser: pd.Series) -> np.ndarray:
        """
        Convert a pandas Series of categorical dtype to a numpy ndarray of integer codes.
        If the Series is not categorical, it is returned as a numpy array as is.
        """
        if cls.is_categorical(ser):
            return ser.cat.codes.to_numpy()
        else:
            return ser.to_numpy()

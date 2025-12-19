import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

STR_TO_BOOL = {"yes": True, "no": False, "y": True, "n": False, "0": False, "1": True}


class NP_Dtype_Utils:
    @classmethod
    def is_int_dtype(cls, ser: pd.Series | np.ndarray) -> bool:
        if isinstance(ser, pd.Series):
            return pd.api.types.is_integer_dtype(ser)
        return np.issubdtype(ser.dtype, np.integer)

    @classmethod
    def is_str_dtype(cls, ser: pd.Series | np.ndarray, convert_na: bool = True) -> bool:
        """
        Return True if the series/array is string-like.

        If `convert_na` is True, NaN/None values are treated as empty strings before testing
        (this helps when pandas reports non-string dtype due to NaNs).
        """
        has_na = ser.isna().any() if isinstance(ser, pd.Series) else np.isnan(ser).any()
        is_cat = isinstance(ser.dtype, CategoricalDtype) if isinstance(ser, pd.Series) else ser.dtype.name == "category"
        if convert_na and has_na and not is_cat:
            ser = ser.fillna("") if isinstance(ser, pd.Series) else np.where(np.isnan(ser), "", ser)
        if isinstance(ser, pd.Series):
            return pd.api.types.is_string_dtype(ser)
        return np.issubdtype(ser.dtype, np.object_)

    @classmethod
    def _is_int_bool_candidate(cls, ser: pd.Series | np.ndarray):
        is_int = cls.is_int_dtype(ser)
        if not is_int:
            return False

        is_0_1 = ser.dropna().isin([0, 1]).all() if isinstance(ser, pd.Series) else np.all(np.isin(ser, [0, 1]))

        return is_int and bool(is_0_1)

    @classmethod
    def _is_str_bool_candidate(cls, ser: pd.Series | np.ndarray):
        is_str = cls.is_str_dtype(ser, False)
        if not is_str:
            return False

        uniqes = ser.dropna().str.lower().unique() if isinstance(ser, pd.Series) else np.unique(ser)
        is_yes_no = np.all(np.isin(uniqes, list(STR_TO_BOOL.keys())))

        return is_str and bool(is_yes_no)

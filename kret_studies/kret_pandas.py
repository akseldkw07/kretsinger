import pandas as pd
import numpy as np

STR_TO_BOOL = {"yes": True, "no": False, "y": True, "n": False, "0": False, "1": True}

# region DTYPE


def is_int_dtype(ser: pd.Series | np.ndarray) -> bool:
    if isinstance(ser, pd.Series):
        return pd.api.types.is_integer_dtype(ser)
    return np.issubdtype(ser.dtype, np.integer)


def is_str_dtype(ser: pd.Series | np.ndarray) -> bool:
    if isinstance(ser, pd.Series):
        return pd.api.types.is_string_dtype(ser)
    return np.issubdtype(ser.dtype, np.object_)


def _is_int_bool_candidate(ser: pd.Series | np.ndarray):
    is_int = is_int_dtype(ser)
    if not is_int:
        return False

    is_0_1 = ser.dropna().isin([0, 1]).all() if isinstance(ser, pd.Series) else np.all(np.isin(ser, [0, 1]))

    return is_int and bool(is_0_1)


def _is_str_bool_candidate(ser: pd.Series | np.ndarray):
    is_str = is_str_dtype(ser)
    if not is_str:
        return False

    uniqes = ser.dropna().str.lower().unique() if isinstance(ser, pd.Series) else np.unique(ser)
    is_yes_no = np.all(np.isin(uniqes, list(STR_TO_BOOL.keys())))

    return is_str and bool(is_yes_no)


# endregion


# region CLEANUP


def cols_to_bool(df: pd.DataFrame):
    """
    Convert columns in the DataFrame to boolean in-place if:
    - The column contains only 'yes'/'no' (case-insensitive, ignoring NaN)
    - The column is integer type and contains only 0/1 (ignoring NaN)
    Modifies the DataFrame in-place. Returns None.
    """
    for col in df.columns:
        ser = df[col]
        # Check for 'yes'/'no' columns (string/object)
        if _is_str_bool_candidate(ser):
            df[col] = ser.str.lower().map(STR_TO_BOOL)
            continue

        # Check for 0/1 columns (integer)
        if _is_int_bool_candidate(ser):
            df[col] = ser.astype(bool)
            continue


# endregion

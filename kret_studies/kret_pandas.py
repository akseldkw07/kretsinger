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


# region CLEANING


def data_cleanup(df: pd.DataFrame):
    _cols_to_bool(df)
    _cols_to_categorical(df)
    _cols_to_datetime(df)


def _cols_to_bool(df: pd.DataFrame):
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


# region CATEGORICAL


def _cols_to_categorical(df: pd.DataFrame, k: int = 10) -> None:
    """
    Convert string columns in the DataFrame to categorical in-place if the column has under k unique values (excluding NaN).
    Modifies the DataFrame in-place. Returns None.
    """
    for col in df.columns:
        if "id" in col.lower()[2:]:
            df[col] = df[col].astype("category")
            continue

        ser = df[col]
        if is_str_dtype(ser):
            nunique = ser.nunique(dropna=True)
            if nunique <= k:
                df[col] = ser.astype("category")


# endregion
# region DATETIME


def _cols_to_datetime(df: pd.DataFrame, thresh: float = 0.95) -> None:
    """
    Attempt to convert string columns to datetime in-place if they appear to be datetime candidates.
    A column is considered a candidate if at least `thresh` fraction of non-null values can be parsed as datetime.
    Modifies the DataFrame in-place. Returns None.
    """
    for col in df.columns:
        if "datetime" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            continue

        ser = df[col]
        if is_str_dtype(ser):
            non_null = ser.dropna()
            if non_null.empty:
                continue
            parsed = pd.to_datetime(non_null, errors="coerce")
            success_frac = parsed.notna().mean()
            if success_frac >= thresh:
                df[col] = pd.to_datetime(ser, errors="coerce")


# endregion
# endregion


# region COLUMN ORDER
def move_columns(df: pd.DataFrame, start: list[str] | None = None, end: list[str] | None = None):
    """
    Return a DataFrame with the specified columns moved to the start and/or end.
    Args:
        df: The DataFrame.
        start: List of column names to move to the beginning.
        end: List of column names to move to the end.
    Returns:
        A new DataFrame with columns reordered.
    """
    start = [col for col in (start or []) if col in df.columns]
    end = [col for col in (end or []) if col in df.columns and col not in start]

    middle = [col for col in df.columns if col not in start and col not in end]
    new_order = start + middle + end
    return df[new_order]


# endregion


def split_x_y(df: pd.DataFrame, y_col: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if y_col is not None and y_col in df.columns:
        X = df.drop(columns=[y_col])
        y = df[y_col]
    else:
        X = df
        y = df[-1]

    return X, y

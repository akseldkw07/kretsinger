"""
NOTE most of this code could probably be replaced with sklearn etc, it's for data cleanup
"""

import typing as t
import warnings

import pandas as pd

from kret_np_pd.np_dtype_utils import STR_TO_BOOL, NP_Dtype_Utils


class PD_Cleanup:
    @t.overload
    @classmethod
    def data_cleanup(cls, df: pd.DataFrame, ret: t.Literal[True]) -> pd.DataFrame: ...

    @t.overload
    @classmethod
    def data_cleanup(cls, df: pd.DataFrame, ret: t.Literal[False] = ...) -> None: ...

    @classmethod
    def data_cleanup(cls, df: pd.DataFrame, ret: bool = False):
        """
        Clean the DataFrame by converting columns to appropriate dtypes.
        Convert boolean candidates to bool
        Convert string candidates to category (if < 10 unique values). Casts nan to str for is_str check
        Convert datetime candidates to datetime

        INPLACE
        """
        cls._cols_to_bool(df)
        cls._cols_to_categorical(df)
        cls._cols_to_datetime(df)
        if ret:
            return df

    @classmethod
    def _cols_to_bool(cls, df: pd.DataFrame):
        """
        Convert columns in the DataFrame to boolean in-place if:
        - The column contains only 'yes'/'no' (case-insensitive, ignoring NaN)
        - The column is integer type and contains only 0/1 (ignoring NaN)
        Modifies the DataFrame in-place. Returns None.
        """
        for col in df.columns:
            ser = df[col]
            # Check for 'yes'/'no' columns (string/object)
            if NP_Dtype_Utils._is_str_bool_candidate(ser):
                df[col] = ser.str.lower().map(STR_TO_BOOL)
                continue

            # Check for 0/1 columns (integer)
            if NP_Dtype_Utils._is_int_bool_candidate(ser):
                df[col] = ser.astype(bool)
                continue

    # region CATEGORICAL

    @classmethod
    def _cols_to_categorical(cls, df: pd.DataFrame, k: int = 10) -> None:
        """
        Convert string columns in the DataFrame to categorical in-place if the column has under k unique values (excluding NaN).
        Modifies the DataFrame in-place. Returns None.
        """
        for col in df.columns:
            name = col.lower()
            if name.startswith("id") or name.endswith("id") or "category" in name or "cat" in name:
                df[col] = df[col].astype("category")
                continue

            ser = df[col]
            if NP_Dtype_Utils.is_str_dtype(ser, True):
                nunique = ser.nunique(dropna=True)
                if nunique <= k:
                    df[col] = ser.astype("category")

    # endregion
    # region DATETIME

    @classmethod
    def _cols_to_datetime(cls, df: pd.DataFrame, thresh: float = 0.95) -> None:
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
            if NP_Dtype_Utils.is_str_dtype(ser, False):
                non_null = ser.dropna()
                if non_null.empty:
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`.",
                        category=UserWarning,
                    )
                    parsed = pd.to_datetime(non_null, errors="coerce")
                success_frac = parsed.notna().mean()
                if success_frac >= thresh:
                    df[col] = pd.to_datetime(ser, errors="coerce")

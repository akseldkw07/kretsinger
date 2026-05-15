import typing as t

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from kret_rosetta.UTILS_rosetta import UTILS_rosetta


class PD_Convenience_utils_Col_filter_TypedDict(t.TypedDict, total=False):
    df: pd.DataFrame
    include: list[str]  # = []
    exclude: list[str]  # = []


class PD_Convenience_utils:
    @classmethod
    def float_cols(cls, df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=["float"]).columns.tolist()

    @classmethod
    def int_cols(cls, df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=["integer"]).columns.tolist()

    @classmethod
    def numeric_cols(cls, df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=["number"]).columns.tolist()

    @classmethod
    def cat_cols(cls, df: pd.DataFrame) -> list[str]:
        # This catches pandas 'category', strings/objects, and booleans.
        # (If you treat bool as numeric, remove "bool".)
        return df.select_dtypes(include=["category", "object", "string", "bool"]).columns.tolist()

    @classmethod
    def move_columns(cls, df: pd.DataFrame, start: list[str] | None = None, end: list[str] | None = None):
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

    @t.overload
    @classmethod
    def col_filter(  # type: ignore
        cls, df: pd.DataFrame, include: t.Sequence[str] = ..., exclude: t.Sequence[str] = ...
    ) -> pd.DataFrame: ...
    @t.overload
    @classmethod
    def col_filter(cls, df: Styler, include: t.Sequence[str] = ..., exclude: t.Sequence[str] = ...) -> Styler: ...

    @classmethod
    def col_filter(cls, df: Styler | pd.DataFrame, include: t.Sequence[str] = [], exclude: t.Sequence[str] = []):
        """
        Return a DataFrame/Styler with only the specified columns included and/or excluded.

        NOTE: `include` and `exclude` are sequences of substrings, not exact column names.

        For a Styler, columns are *hidden* via `Styler.hide(subset=..., axis="columns")`
        rather than dropped — this preserves any styling rules bound to column positions.

        Args:
            df: The DataFrame or Styler.
            include: Sequence of column substrings to include (empty = include all).
            exclude: Sequence of column substrings to exclude (empty = exclude none).
        Returns:
            A new DataFrame (or Styler with the matching columns hidden).
        """
        data: pd.DataFrame = getattr(df, "data") if isinstance(df, Styler) else df
        keep = (
            [col for col in data.columns if any(substr in col for substr in include)]
            if len(include)
            else data.columns.tolist()
        )
        drop = [col for col in data.columns if any(substr in col for substr in exclude)]
        keep_final = [c for c in keep if c not in drop]
        cols_gone: list[t.Hashable] = [col for col in data.columns if col not in keep_final]
        print(f"Returning df without {len(cols_gone)} columns: {cols_gone}")

        if isinstance(df, Styler):
            return df.hide(subset=cols_gone, axis="columns") if cols_gone else df
        return data[keep_final]

    @t.overload
    @classmethod
    def pop_label_and_drop(  # type: ignore
        cls,
        df: pd.DataFrame,
        label_col: list[str] | str,
        drop_cols: list[str] | str | None = ...,
        keep_labels: bool = ...,
        label_ret_type: t.Literal["np"] = "np",
    ) -> tuple[pd.DataFrame, np.ndarray]: ...

    @t.overload
    @classmethod
    def pop_label_and_drop(
        cls,
        df: pd.DataFrame,
        label_col: list[str] | str,
        drop_cols: list[str] | str | None = ...,
        keep_labels: bool = ...,
        label_ret_type: t.Literal["df"] = "df",
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @classmethod
    def pop_label_and_drop(
        cls,
        df: pd.DataFrame,
        label_col: list[str] | str,
        drop_cols: list[str] | str | None = None,
        keep_labels: bool = False,
        label_ret_type: t.Literal["np", "df"] = "np",
    ):
        """
        Pop the label column(s) from the DataFrame and optionally drop other columns.

        """
        if isinstance(label_col, str):
            label_col = [label_col]
        if isinstance(drop_cols, str):
            drop_cols = [drop_cols]
        elif drop_cols is None:
            drop_cols = []

        labels = df[label_col]
        cols_to_drop = drop_cols + ([] if keep_labels else label_col)
        df.drop(columns=cols_to_drop, inplace=True, axis=1)

        if label_ret_type == "np":
            labels = UTILS_rosetta.coerce_to_ndarray(labels, assert_1dim=True, attempt_flatten_1d=True)

        return df, labels

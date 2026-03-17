import typing as t

import numpy as np
import pandas as pd

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

    @classmethod
    def col_filter(cls, df: pd.DataFrame, include: list[str] = [], exclude: list[str] = []) -> pd.DataFrame:
        """
        Return a DataFrame with only the specified columns included and/or excluded.
        Args:
            df: The DataFrame.
            include: List of column substrings to include (if None, include all).
            exclude: List of column substrings to exclude (if None, exclude none).
        Returns:
            A new DataFrame with columns filtered.
        """
        include = (
            [col for col in df.columns if any(substr in col for substr in include)]
            if len(include)
            else df.columns.tolist()
        )

        exclude = [col for col in df.columns if any(substr in col for substr in exclude)]

        ret = df[include].drop(columns=exclude, errors="ignore")
        print(
            f"Returning df without {len(cols_gone:=[col for col in df.columns if col not in ret.keys()])} columns: {cols_gone}"
        )
        return ret

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

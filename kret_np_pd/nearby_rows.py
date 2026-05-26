import datetime as dt
import typing as t

import numpy as np
import pandas as pd
import polars as pl
import torch

from kret_rosetta.UTILS_rosetta import UTILS_rosetta
from kret_utils.assert_type import TypeAssert

from .filters import FilterSampleUtils

FILT_NOT_NONE = np.ndarray | pd.Series | torch.Tensor | pd.DataFrame | pl.Series | pl.DataFrame
FILT_TYPE = FILT_NOT_NONE | None
SHAPE_TYPE = tuple[int, ...] | tuple[int] | int


class NearbyRows:
    NUMERIC_DTYPE = int | float | np.number | dt.datetime | dt.timedelta | pd.Timestamp | pd.Timedelta
    DIRECTION = t.Literal["both", "forward", "backward"]

    @classmethod
    def get_nearby_rows(
        cls,
        df: pd.DataFrame | pl.DataFrame,
        filt: FILT_TYPE,
        hard_match: t.Sequence[str] | str = [],
        hard_match_apply: t.Literal["and", "or"] = "and",
        *,
        soft_match_default_direction: DIRECTION = "both",
        soft_match: dict[str, NUMERIC_DTYPE | tuple[NUMERIC_DTYPE, DIRECTION]] = {},
        soft_match_apply: t.Literal["and", "or"] = "and",
        join_hard_soft: t.Literal["and", "or"] = "and",
    ):
        if isinstance(df, pd.DataFrame):
            return cls.get_nearby_rows_df(
                df=df,
                filt=filt,
                hard_match=hard_match,
                hard_match_apply=hard_match_apply,
                soft_match_default_direction=soft_match_default_direction,
                soft_match=soft_match,
                soft_match_apply=soft_match_apply,
                join_hard_soft=join_hard_soft,
            )
        raise ValueError(f"Not yet implemented for input df type {type(df)}")

    @classmethod
    def get_nearby_rows_df(
        cls,
        df: pd.DataFrame,
        filt: FILT_TYPE,
        hard_match: t.Sequence[str] | str = [],
        hard_match_apply: t.Literal["and", "or"] = "and",
        *,
        soft_match_default_direction: DIRECTION = "both",
        soft_match: dict[str, NUMERIC_DTYPE | tuple[NUMERIC_DTYPE, DIRECTION]] = {},
        soft_match_apply: t.Literal["and", "or"] = "and",
        join_hard_soft: t.Literal["and", "or"] = "and",
    ) -> np.ndarray:
        """
        Return a boolean filter selecting rows "nearby" the True rows in `filt`.

        Nearness has two flavors, combinable via `join_hard_soft`:
        - `hard_match`: listed columns must exactly equal a seed row's values.
        - `soft_match`: listed columns must be within a numeric distance of a seed row's
          value. Pass `{col: dist}` to use `soft_match_default_direction`, or
          `{col: (dist, direction)}` to override per column. `direction` is one of
          "both" (|d| <= dist), "forward" (0 <= d <= dist), "backward" (-dist <= d <= 0).

        `hard_match_apply` / `soft_match_apply` ('and'/'or') combine conditions within each
        side; `join_hard_soft` ('and'/'or') combines hard and soft. Returns a length-`len(df)`
        boolean ndarray.

        Full worked examples are appended below at runtime — see
        `kret_np_pd/_core/get_nearby_rows_examples.py` (or
        `help(UKS_NP_PD.get_nearby_rows)` / `UKS_NP_PD.get_nearby_rows?` in Jupyter).
        """
        filt = FilterSampleUtils.process_filter(filt, shape=len(df))
        n = len(df)
        if not filt.any():
            return np.zeros(n, dtype=bool)
        if not hard_match and not soft_match:
            return np.asarray(filt, dtype=bool)

        # Normalize to pandas: polars subclasses (e.g. Enriched_DF_PL) override
        # to_numpy() to expect their full column_order, which breaks on subsets.
        df = UTILS_rosetta.coerce_to_df(df)

        per_seed_masks: list[np.ndarray] = []

        # HARD: one DF lookup → (n, H), then broadcast-equal against seeds → (k, n, H)
        if hard_match:
            hard_match = [hard_match] if isinstance(hard_match, str) else hard_match
            hard_arr = df[list(hard_match)].to_numpy()
            seeds_hard = hard_arr[filt]
            eq = seeds_hard[:, None, :] == hard_arr[None, :, :]
            hard_ps = TypeAssert.assert_type(eq.all(-1) if hard_match_apply == "and" else eq.any(-1), np.ndarray)
            per_seed_masks.append(hard_ps)

        # SOFT: one DF lookup → (n, S), broadcast-diff against seeds → (k, n, S)
        if soft_match:
            soft_cols = list(soft_match)
            dists = np.array([v[0] if isinstance(v, tuple) else v for v in soft_match.values()])
            dirs = np.array(
                [v[1] if isinstance(v, tuple) else soft_match_default_direction for v in soft_match.values()]
            )

            soft_arr = df[soft_cols].to_numpy()
            seeds_soft = soft_arr[filt]
            diffs = soft_arr[None, :, :] - seeds_soft[:, None, :]

            # dtype-preserving zero so datetime/timedelta cols compare cleanly
            zero = dists * 0
            lower = np.where(dirs == "forward", zero, -dists)
            upper = np.where(dirs == "backward", zero, dists)
            in_win = (diffs >= lower) & (diffs <= upper)

            soft_ps = t.cast(np.ndarray, in_win.all(-1) if soft_match_apply == "and" else in_win.any(-1))
            per_seed_masks.append(soft_ps)

        if len(per_seed_masks) == 1:
            combined = per_seed_masks[0]
        else:
            h, s = per_seed_masks
            combined = (h & s) if join_hard_soft == "and" else (h | s)

        return t.cast(np.ndarray, combined.any(axis=0))


# Append the verbose examples onto get_nearby_rows.__doc__. cleandoc strips the source
# indent of the short docstring so it concatenates flush against the (unindented) examples.
import inspect as _inspect

from kret_np_pd._core.get_nearby_rows_examples import EXAMPLES_DOC as _EXAMPLES_DOC

NearbyRows.get_nearby_rows.__func__.__doc__ = (
    _inspect.cleandoc(NearbyRows.get_nearby_rows.__doc__ or "") + "\n\n" + _EXAMPLES_DOC
)

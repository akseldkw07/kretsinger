import typing as t
from functools import cache

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_bool_dtype

from kret_np_pd.single_ret_ndarray import SingleReturnArray
from kret_rosetta.UTILS_rosetta import UTILS_rosetta
import polars as pl
import datetime as dt

FILT_TYPE = np.ndarray | pd.Series | torch.Tensor | pd.DataFrame | pl.Series | pl.DataFrame | None


class FilterSampleUtils:
    @classmethod
    def process_filter(cls, filter: FILT_TYPE, shape: tuple[int, ...] | tuple[int] | int | None = None):
        if filter is None:
            assert shape is not None, "Shape must be provided when filter is None"
            ret = np.full((shape[0] if isinstance(shape, tuple) else shape), True)
        else:
            ret = UTILS_rosetta.coerce_to_ndarray(filter, assert_1dim=True, attempt_flatten_1d=True)
        cls.assert_bool_dtype(ret)

        return t.cast(SingleReturnArray[bool], ret)

    @classmethod
    @cache
    def gen_sample_filter(cls, hot: int, total_size: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        indices = rng.choice(total_size, size=hot, replace=False)
        ret = np.full(total_size, False)
        ret[indices] = True

        cls.assert_bool_dtype(ret)

        return t.cast(SingleReturnArray[bool], ret)

    @classmethod
    def downsample_bool(cls, ret: np.ndarray, k: int, seed: int | None = None):
        """
        Randomly flip True values to False so that sum(arr) == k.
        If arr.sum() <= k, returns arr unchanged.
        """
        cls.assert_bool_dtype(ret)
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        idx = np.flatnonzero(ret.copy())

        if len(idx) <= k:
            return t.cast(SingleReturnArray[bool], ret)

        keep = rng.choice(idx, size=k, replace=False)
        ret[:] = False
        ret[keep] = True

        return t.cast(SingleReturnArray[bool], ret)

    @classmethod
    def assert_bool_dtype(cls, arr: np.ndarray):
        assert is_bool_dtype(arr), f"Expected boolean filter type, got {arr.dtype}"

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
        filt = cls.process_filter(filt, shape=len(df))
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
            hard_ps = eq.all(-1) if hard_match_apply == "and" else eq.any(-1)
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

FilterSampleUtils.get_nearby_rows.__func__.__doc__ = (
    _inspect.cleandoc(FilterSampleUtils.get_nearby_rows.__doc__ or "") + "\n\n" + _EXAMPLES_DOC
)

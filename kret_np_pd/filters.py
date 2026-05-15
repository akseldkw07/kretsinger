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
        Accepts a DataFrame and a boolean filter, and returns a new filter that includes
        rows that are "nearby" the True rows in the original filter.

        There are two types of "nearness" that can be applied:
        1. Hard match: The columns specified in `hard_match` must match exactly for a
        row to be considered nearby.

        2. Soft match: The columns specified in `soft_match` must be within a certain distance
        of the values in the True rows for a row to be considered nearby. The value in
        `soft_match` is the maximum distance that is allowed, and the direction can be
        "both", "forward", or "backward". Direction is customizeable per column by passing both
        (distance, direction).

        The `hard_match_apply` and `soft_match_apply` parameters determine how multiple
        conditions are combined. If "and", a row must satisfy all conditions to be considered nearby.
        If "or", a row must satisfy at least one condition to be considered nearby.

        The `join_hard_soft` parameter determines how the hard and soft matches are combined.
        If "and", a row must satisfy BOTH the hard and soft conditions to be considered nearby.
        If "or", a row must satisfy EITHER the hard or soft conditions to be considered nearby.

        Examples:

        >>> df = pd.DataFrame({
        ...     "gameId": [1, 1, 2, 2],
        ...     "teamId": ["A", "B", "A", "B"],
        ...     "event":  ["play", "timeout", "play", "play"]
        ... })
        >>> # Seed: The timeout in Game 1 for Team B
        >>> filt = df["event"] == "timeout"
        >>> # Hard match on both: only find rows that are (Game 1, Team B)
        >>> get_nearby_rows(df, filt, hard_match=["gameId", "teamId"])

            [False, True, False, False]

        -----------Example 2-----------

        >>> df = pd.DataFrame({
        ...     "time": [10, 14, 15, 20, 25],
        ...     "whistle": [False, False, False, True, False]
        ... })
        >>> # Seed: The whistle at time 20
        >>> filt = df["whistle"]
        >>> # Soft match: 5 units 'backward' (looking at time 15 to 20)
        >>> get_nearby_rows(
        ...     df, filt,
        ...     soft_match={"time": (5, "backward")}
        ... )

            [False, False, True, True, False]

        -----------Example 3-----------

        >>> df = pd.DataFrame({
        ...     "time": [10, 11, 12],
        ...     "dist": [0, 5, 20],
        ...     "foul": [True, False, False]
        ... })
        >>> # Must be within 2 seconds AFTER (forward) AND within 10 distance units (both)
        >>> get_nearby_rows(
        ...     df, df["foul"],
        ...     soft_match={"time": (2, "forward"), "dist": 10},
        ...     soft_match_apply="and"
        ... )

        -----------Example 4-----------
        >>> df = pd.DataFrame({
        ...     "gameId": [1, 1, 2, 2, 3],
        ...     "period": [1, 2, 1, 2, 1],
        ...     "event":  ["goal", "play", "play", "goal", "play"]
        ... })
        >>> # Seed: Goals in (Game 1, Period 1) and (Game 2, Period 2)
        >>> filt = df["event"] == "goal"
        >>> # Hard match ensures we only get rows within those specific (Game, Period) pairs.
        >>> # It will NOT match Game 1, Period 2 or Game 2, Period 1.
        >>> get_nearby_rows(df, filt, hard_match=["gameId", "period"])
        0     True  # Match (1, 1)
        1    False  # No match for (1, 2)
        2    False  # No match for (2, 1)
        3     True  # Match (2, 2)
        4    False  # No match for (3, 1)

        -----------Example 5-----------
        >>> df = pd.DataFrame({
        ...     "playId": [10, 10, 10, 10, 20, 20],
        ...     "time":   [1, 4, 5, 6, 5, 6],
        ...     "dist":   [2, 2, 0, 15, 0, 2],
        ...     "is_hit": [False, False, True, False, True, False]
        ... })
        >>> # Seed: Two hits (one in Play 10 at T=5, one in Play 20 at T=5)
        >>> filt = df["is_hit"]
        >>> # 1. hard_match: Stay within the same playId.
        >>> # 2. soft_match 'time': Overwrite default to look 3s 'backward'.
        >>> # 3. soft_match 'dist': Use global default 'both' for a 5-unit radius.
        >>> get_nearby_rows(
        ...     df, filt,
        ...     hard_match=["playId"],
        ...     soft_match_default_direction="both",
        ...     soft_match={"time": (3, "backward"), "dist": 5},
        ...     soft_match_apply="and"
        ... )
        0    False  # Play 10, T=1 (Out of time window 2-5)
        1     True  # Play 10, T=4 (In window, In dist)
        2     True  # Play 10, T=5 (The Seed)
        3    False  # Play 10, T=6 (Wrong direction: forward)
        4     True  # Play 20, T=5 (The Seed)
        5    False  # Play 20, T=6 (Wrong direction: forward)

        -----------Example 6-----------
        >>> df = pd.DataFrame({
        ...     "gameId": [1, 1, 2, 2],
        ...     "time":   [59, 60, 1, 2],  # Game 1 ends at 60, Game 2 starts at 1
        ...     "event":  [None, "buzzer", None, None]
        ... })
        >>> # Seed: The buzzer at the end of Game 1
        >>> filt = df["event"] == "buzzer"
        >>> # We want rows within 2 seconds of the buzzer, but ONLY in the same game.
        >>> get_nearby_rows(
        ...     df, filt,
        ...     hard_match=["gameId"],
        ...     soft_match={"time": 2},
        ...     join_hard_soft="and"
        ... )
        0     True  # Within 2s AND same gameId
        1     True  # The Seed
        2    False  # Within 2s (time=1 is close to 60) BUT different gameId
        3    False  # Different gameId
        dtype: bool

        >>> # If we changed join_hard_soft to "or", Index 2 would become True
        >>> # because it satisfies the "nearby time" condition.
        """
        filt = cls.process_filter(filt, shape=len(df))
        n = len(df)
        if not filt.any():
            return np.zeros(n, dtype=bool)
        if not hard_match and not soft_match:
            return np.asarray(filt, dtype=bool)

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

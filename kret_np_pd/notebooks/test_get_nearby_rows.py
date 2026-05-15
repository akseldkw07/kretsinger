"""Unit tests for FilterSampleUtils.get_nearby_rows.

Each test is a function returning a NearbyTestCase. The notebook runs them
via run_all(), which calls get_nearby_rows on each and compares against the
expected mask. Individual cases are also importable for inspection.
"""

from __future__ import annotations

import dataclasses as dc
import typing as t

import numpy as np
import pandas as pd

from kret_np_pd.filters import FilterSampleUtils


@dc.dataclass
class NearbyTestCase:
    name: str
    df: pd.DataFrame
    filt: pd.Series | np.ndarray
    expected: np.ndarray
    kwargs: dict[str, t.Any] = dc.field(default_factory=dict)


# ============================================================================
# Docstring examples (Examples 1-6)
# ============================================================================
def case_ex1_hard_match_two_cols():
    df = pd.DataFrame(
        {
            "gameId": [1, 1, 2, 2],
            "teamId": ["A", "B", "A", "B"],
            "event": ["play", "timeout", "play", "play"],
        }
    )
    filt = df["event"] == "timeout"
    return NearbyTestCase(
        name="ex1: hard match (gameId, teamId)",
        df=df,
        filt=filt,
        kwargs={"hard_match": ["gameId", "teamId"]},
        expected=np.array([False, True, False, False]),
    )


def case_ex2_soft_match_backward():
    df = pd.DataFrame(
        {
            "time": [10, 14, 15, 20, 25],
            "whistle": [False, False, False, True, False],
        }
    )
    return NearbyTestCase(
        name="ex2: soft match {time: (5, backward)}",
        df=df,
        filt=df["whistle"],
        kwargs={"soft_match": {"time": (5, "backward")}},
        expected=np.array([False, False, True, True, False]),
    )


def case_ex3_soft_forward_and_both():
    # Seed: idx 0 (time=10, dist=0). time forward 2 → candidate time in [10,12].
    # dist both 10 → candidate dist in [-10,10]. AND-joined.
    df = pd.DataFrame(
        {
            "time": [10, 11, 12],
            "dist": [0, 5, 20],
            "foul": [True, False, False],
        }
    )
    return NearbyTestCase(
        name="ex3: soft {time:(2,forward), dist:10 (default both)}, and",
        df=df,
        filt=df["foul"],
        kwargs={
            "soft_match": {"time": (2, "forward"), "dist": 10},
            "soft_match_apply": "and",
        },
        expected=np.array([True, True, False]),
    )


def case_ex4_hard_match_game_period():
    df = pd.DataFrame(
        {
            "gameId": [1, 1, 2, 2, 3],
            "period": [1, 2, 1, 2, 1],
            "event": ["goal", "play", "play", "goal", "play"],
        }
    )
    return NearbyTestCase(
        name="ex4: hard match (gameId, period)",
        df=df,
        filt=df["event"] == "goal",
        kwargs={"hard_match": ["gameId", "period"]},
        expected=np.array([True, False, False, True, False]),
    )


def case_ex5_hard_play_soft_time_dist():
    df = pd.DataFrame(
        {
            "playId": [10, 10, 10, 10, 20, 20],
            "time": [1, 4, 5, 6, 5, 6],
            "dist": [2, 2, 0, 15, 0, 2],
            "is_hit": [False, False, True, False, True, False],
        }
    )
    return NearbyTestCase(
        name="ex5: hard playId AND soft {time:(3,backward), dist:5 (default both)}",
        df=df,
        filt=df["is_hit"],
        kwargs={
            "hard_match": ["playId"],
            "soft_match_default_direction": "both",
            "soft_match": {"time": (3, "backward"), "dist": 5},
            "soft_match_apply": "and",
        },
        expected=np.array([False, True, True, False, True, False]),
    )


def case_ex6_buzzer_same_game():
    df = pd.DataFrame(
        {
            "gameId": [1, 1, 2, 2],
            "time": [59, 60, 1, 2],
            "event": [None, "buzzer", None, None],
        }
    )
    return NearbyTestCase(
        name="ex6: hard gameId AND soft {time:2 (both)}",
        df=df,
        filt=df["event"] == "buzzer",
        kwargs={
            "hard_match": ["gameId"],
            "soft_match": {"time": 2},
            "join_hard_soft": "and",
        },
        expected=np.array([True, True, False, False]),
    )


# ============================================================================
# Edge cases & combinator coverage
# ============================================================================
def case_empty_filt():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    return NearbyTestCase(
        name="empty filt → all False",
        df=df,
        filt=pd.Series([False, False, False]),
        kwargs={"hard_match": ["a"]},
        expected=np.array([False, False, False]),
    )


def case_no_match_criteria():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    return NearbyTestCase(
        name="no hard, no soft → returns filt unchanged",
        df=df,
        filt=pd.Series([True, False, True]),
        kwargs={},
        expected=np.array([True, False, True]),
    )


def case_hard_apply_or():
    # Seed: row 0 (a=1, b=X). Match if a==1 OR b=="X".
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["X", "Y", "X", "Z"],
            "seed": [True, False, False, False],
        }
    )
    return NearbyTestCase(
        name="hard_match_apply=or",
        df=df,
        filt=df["seed"],
        kwargs={"hard_match": ["a", "b"], "hard_match_apply": "or"},
        expected=np.array([True, False, True, False]),
    )


def case_soft_apply_or():
    # Seed: row 0 (x=0, y=0). Match if |dx|<=1 OR |dy|<=1.
    df = pd.DataFrame(
        {
            "x": [0, 0, 5, 5],
            "y": [0, 5, 0, 5],
            "seed": [True, False, False, False],
        }
    )
    return NearbyTestCase(
        name="soft_match_apply=or",
        df=df,
        filt=df["seed"],
        kwargs={"soft_match": {"x": 1, "y": 1}, "soft_match_apply": "or"},
        expected=np.array([True, True, True, False]),
    )


def case_join_or_includes_extra():
    # Seed: idx 1 (gameId=1, time=11).
    # hard (gameId==1): rows 0,1 ✓ ; rows 2,3 ✗
    # soft (|time-11|<=2): row 0 (10) ✓, row 1 (11) ✓, row 2 (10) ✓, row 3 (99) ✗
    # OR distinguishes row 2 (no hard, but yes soft) — AND would drop it.
    df = pd.DataFrame(
        {
            "gameId": [1, 1, 2, 2],
            "time": [10, 11, 10, 99],
            "seed": [False, True, False, False],
        }
    )
    return NearbyTestCase(
        name="join_hard_soft=or includes row passing only soft",
        df=df,
        filt=df["seed"],
        kwargs={
            "hard_match": ["gameId"],
            "soft_match": {"time": 2},
            "join_hard_soft": "or",
        },
        expected=np.array([True, True, True, False]),
    )


def case_join_and_excludes_extra():
    # Same DF as case_join_or_includes_extra; AND drops row 2.
    df = pd.DataFrame(
        {
            "gameId": [1, 1, 2, 2],
            "time": [10, 11, 10, 99],
            "seed": [False, True, False, False],
        }
    )
    return NearbyTestCase(
        name="join_hard_soft=and excludes row passing only soft",
        df=df,
        filt=df["seed"],
        kwargs={
            "hard_match": ["gameId"],
            "soft_match": {"time": 2},
            "join_hard_soft": "and",
        },
        expected=np.array([True, True, False, False]),
    )


def case_mixed_directions():
    # Seed: idx 3 (t1=3, t2=3). t1 forward 1 → [3,4]; t2 default both 1 → [2,4].
    df = pd.DataFrame(
        {
            "t1": [0, 1, 2, 3, 4, 5],
            "t2": [0, 1, 2, 3, 4, 5],
            "seed": [False, False, False, True, False, False],
        }
    )
    return NearbyTestCase(
        name="mixed directions: t1 forward 1, t2 default both 1",
        df=df,
        filt=df["seed"],
        kwargs={
            "soft_match_default_direction": "both",
            "soft_match": {"t1": (1, "forward"), "t2": 1},
            "soft_match_apply": "and",
        },
        expected=np.array([False, False, False, True, True, False]),
    )


def case_per_col_overrides_default():
    # default=forward, but column explicitly set to backward.
    # Seed: idx 2 (t=2). Backward 1 → candidate t in [1,2].
    df = pd.DataFrame(
        {
            "t": [0, 1, 2, 3, 4],
            "seed": [False, False, True, False, False],
        }
    )
    return NearbyTestCase(
        name="per-col direction overrides default (default=forward, col=backward)",
        df=df,
        filt=df["seed"],
        kwargs={
            "soft_match_default_direction": "forward",
            "soft_match": {"t": (1, "backward")},
        },
        expected=np.array([False, True, True, False, False]),
    )


def case_datetime_soft_match():
    base = pd.Timestamp("2024-01-01 12:00:00")
    df = pd.DataFrame(
        {
            "ts": [
                base,
                base + pd.Timedelta(seconds=30),
                base + pd.Timedelta(seconds=120),
                base + pd.Timedelta(seconds=200),
            ],
            "seed": [False, False, True, False],
        }
    )
    return NearbyTestCase(
        name="datetime soft match: within 60s of seed timestamp",
        df=df,
        filt=df["seed"],
        kwargs={"soft_match": {"ts": pd.Timedelta(seconds=60)}},
        expected=np.array([False, False, True, False]),
    )


def case_multiple_seeds_union():
    # Two seeds in different games; expect union of their nearby neighborhoods.
    df = pd.DataFrame(
        {
            "gameId": [1, 1, 1, 2, 2, 2],
            "time": [0, 5, 100, 0, 5, 100],
            "seed": [True, False, False, False, True, False],
        }
    )
    return NearbyTestCase(
        name="multiple seeds: union of per-seed neighborhoods",
        df=df,
        filt=df["seed"],
        kwargs={
            "hard_match": ["gameId"],
            "soft_match": {"time": 10},
            "join_hard_soft": "and",
        },
        # Seed A: (1, 0); Seed B: (2, 5).
        # Per seed A: gameId==1 AND |time-0|<=10 → rows 0,1.
        # Per seed B: gameId==2 AND |time-5|<=10 → rows 3,4.
        # Union: rows 0,1,3,4.
        expected=np.array([True, True, False, True, True, False]),
    )


# ============================================================================
# Runner
# ============================================================================
ALL_TEST_CASES: list[t.Callable[[], NearbyTestCase]] = [
    case_ex1_hard_match_two_cols,
    case_ex2_soft_match_backward,
    case_ex3_soft_forward_and_both,
    case_ex4_hard_match_game_period,
    case_ex5_hard_play_soft_time_dist,
    case_ex6_buzzer_same_game,
    case_empty_filt,
    case_no_match_criteria,
    case_hard_apply_or,
    case_soft_apply_or,
    case_join_or_includes_extra,
    case_join_and_excludes_extra,
    case_mixed_directions,
    case_per_col_overrides_default,
    case_datetime_soft_match,
    case_multiple_seeds_union,
]


def run_one(case: NearbyTestCase) -> tuple[np.ndarray | None, bool, str | None]:
    """Returns (actual, passed, error_msg)."""
    try:
        actual = FilterSampleUtils.get_nearby_rows(case.df, case.filt, **case.kwargs)
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}"
    actual_np = np.asarray(actual)
    if actual_np.shape != case.expected.shape:
        return actual_np, False, f"shape mismatch: got {actual_np.shape}, expected {case.expected.shape}"
    passed = bool(np.array_equal(actual_np, case.expected))
    return actual_np, passed, None


def run_all(verbose: bool = True) -> list[tuple[NearbyTestCase, np.ndarray | None, bool, str | None]]:
    results = []
    for fn in ALL_TEST_CASES:
        case = fn()
        actual, passed, err = run_one(case)
        results.append((case, actual, passed, err))
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {case.name}")
            if not passed:
                if err is not None:
                    print(f"  error:    {err}")
                else:
                    print(f"  expected: {case.expected.astype(int).tolist()}")
                    print(f"  actual:   {actual.astype(int).tolist() if actual is not None else None}")
    n_pass = sum(1 for _, _, p, _ in results if p)
    if verbose:
        print(f"\n{n_pass}/{len(results)} passed")
    return results

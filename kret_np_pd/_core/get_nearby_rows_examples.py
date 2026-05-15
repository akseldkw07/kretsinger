"""Examples appended at import time to FilterSampleUtils.get_nearby_rows.__doc__.

Kept out of filters.py to keep the function's source-level docstring scannable.
The text is concatenated onto the function's runtime __doc__ in filters.py, so
help(get_nearby_rows) and Jupyter's `?` show it in full. Static IDE hover only
sees the short docstring at the function definition.
"""

EXAMPLES_DOC = """\
Examples
--------

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

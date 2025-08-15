from __future__ import annotations

import typing as t
from collections.abc import Sequence
from typing import Any, Literal
from requests import Session
import matplotlib.colors as mcolors
import numpy as np

INTERVAL_LITERAL = t.Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
PERIOD_LITERAL = t.Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]


class Subplots_TypedDict(t.TypedDict, total=False):
    # nrows: int
    # ncols: int
    sharex: bool | Literal["none", "all", "row", "col"]
    sharey: bool | Literal["none", "all", "row", "col"]
    squeeze: bool
    width_ratios: Sequence[float] | None
    height_ratios: Sequence[float] | None
    subplot_kw: dict[str, Any] | None
    gridspec_kw: dict[str, Any] | None


class Sns_Heatmap_TypedDict(t.TypedDict, total=False):
    vmin: float
    vmax: float
    cmap: str | mcolors.LinearSegmentedColormap
    center: t.Any
    robust: t.Any
    annot: t.Any
    fmt: str
    annot_kws: t.Any
    linewidths: t.Any
    linecolor: t.Any
    cbar: t.Any
    cbar_kws: t.Any
    cbar_ax: t.Any
    square: t.Any
    xticklabels: t.Any
    yticklabels: t.Any
    mask: t.Any
    ax: t.Any


class Heatmap_Params_TD(t.TypedDict, total=False):
    vmin: float
    vmax: float
    cmap: str | mcolors.LinearSegmentedColormap
    fmt: str


class Download_TypedDictLite(t.TypedDict, total=False):
    actions: bool
    threads: bool | int
    ignore_tz: bool | None
    group_by: t.Literal["column", "ticker"]
    auto_adjust: bool | None
    back_adjust: t.Any
    repair: bool
    keepna: bool
    progress: t.Any
    interval: INTERVAL_LITERAL
    period: PERIOD_LITERAL
    prepost: bool
    proxy: t.Any
    rounding: bool
    timeout: None | float
    session: None | Session
    multi_level_index: bool


class Download_TypedDict(Download_TypedDictLite):
    tickers: str | list[str]
    start: str | None
    end: str | None


class LBGMRegressor__init___TypedDict(t.TypedDict, total=False):
    self: t.Any
    boosting_type: str
    num_leaves: int
    max_depth: int
    learning_rate: float
    n_estimators: int
    subsample_for_bin: int
    objective: (
        str
        | t.Callable[[np.ndarray | None, np.ndarray], t.Tuple[np.ndarray, np.ndarray]]
        | t.Callable[[np.ndarray | None, np.ndarray, np.ndarray | None], t.Tuple[np.ndarray, np.ndarray]]
        | t.Callable[
            [np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None],
            t.Tuple[np.ndarray, np.ndarray],
        ]
        | None
    )
    class_weight: t.Dict | str | None
    min_split_gain: float
    min_child_weight: float
    min_child_samples: int
    subsample: float
    subsample_freq: int
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    random_state: int | np.random.mtrand.RandomState | np.random._generator.Generator | None
    n_jobs: int | None | None
    importance_type: str
    verbose: int


class Pairplot_TypedDict(t.TypedDict, total=False):
    # data: pd.DataFrame
    hue: str | None
    hue_order: t.Iterable[str] | None
    palette: t.Any | None
    vars: t.Iterable[str] | None
    x_vars: t.Iterable[str] | str | None
    y_vars: t.Iterable[str] | str | None
    kind: t.Literal["scatter", "kde", "hist", "reg"]
    diag_kind: t.Literal["auto", "hist", "kde"] | None
    markers: t.Any | None
    height: float
    aspect: float
    corner: bool
    dropna: bool
    plot_kws: dict[str, t.Any] | None
    diag_kws: dict[str, t.Any] | None
    grid_kws: dict[str, t.Any] | None
    size: float | None

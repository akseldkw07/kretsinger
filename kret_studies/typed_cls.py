from __future__ import annotations

import typing as t
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.colors as mcolors


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


class Download_TypedDict(t.TypedDict, total=False):
    tickers: str | list[str]
    start: str | None
    end: str | None
    actions: bool
    threads: bool | int
    ignore_tz: bool | None
    group_by: str
    auto_adjust: bool | None
    back_adjust: t.Any
    repair: bool
    keepna: bool
    progress: t.Any
    period: t.Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    interval: t.Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    prepost: bool
    proxy: t.Any
    rounding: bool
    timeout: None | float
    session: None | Session
    multi_level_index: bool

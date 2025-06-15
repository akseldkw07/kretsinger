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
    vmin: t.Any
    vmax: t.Any
    cmap: t.Any
    center: t.Any
    robust: t.Any
    annot: t.Any
    fmt: t.Any
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

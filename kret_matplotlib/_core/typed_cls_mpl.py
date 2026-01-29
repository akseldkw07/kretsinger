import typing as t
from collections.abc import Sequence
from typing import Any, Literal

INTERVAL_LITERAL = t.Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
PERIOD_LITERAL = t.Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
if t.TYPE_CHECKING:
    import matplotlib.colors as mcolors
    import numpy as np
    from matplotlib.axes import Axes
    from pandas._typing import Axis
    from pandas.io.formats.style_render import Subset


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


class Background_gradient_TypedDict(t.TypedDict, total=False):
    cmap: "t.Literal['RedWhiteGreen', 'WhiteGreen', 'WhiteRed'] | mcolors.Colormap"
    low: float
    high: float
    axis: "Axis | None"
    subset: "Subset | None"
    text_color_threshold: float
    vmin: float | None
    vmax: float | None
    gmap: Sequence | None


class Format_TypedDict(t.TypedDict, total=False):
    formatter: str
    subset: "Subset | None"
    na_rep: str | None
    precision: int | None
    decimal: str
    thousands: str | None
    escape: str | None
    hyperlinks: t.Literal["html", "latex"] | None


class Pandas_Styler_TypedDict(Background_gradient_TypedDict, Format_TypedDict, total=False):
    pass


class Sns_Heatmap_TypedDict(t.TypedDict, total=False):
    vmin: float
    vmax: float
    cmap: "t.Literal['RedWhiteGreen', 'WhiteGreen', 'WhiteRed'] | mcolors.LinearSegmentedColormap"
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
    mask: "np.ndarray"
    ax: "Axes"


class Heatmap_Params_TD(t.TypedDict, total=False):
    vmin: float
    vmax: float
    cmap: "t.Literal['RedWhiteGreen', 'WhiteGreen', 'WhiteRed'] | mcolors.LinearSegmentedColormap"
    fmt: str


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

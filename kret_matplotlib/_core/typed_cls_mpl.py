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


class PxHistogram_TypedDict(t.TypedDict, total=False):
    # plotly.express.histogram kwargs
    # https://plotly.com/python/histograms/

    # data_frame: pd.DataFrame | None NOTE passed explicitly
    # x: str | t.Sequence[Any] | None NOTE passed explicitly
    y: str | t.Sequence[Any] | None
    # color: str | None NOTE passed explicitly
    pattern_shape: str | None
    facet_row: str | None
    facet_col: str | None
    facet_col_wrap: int
    facet_row_spacing: float | None
    facet_col_spacing: float | None

    histfunc: Literal["count", "sum", "avg", "min", "max"] | None
    histnorm: Literal["percent", "probability", "density", "probability density"] | None
    nbins: int | None
    barmode: Literal["relative", "group", "overlay"]
    barnorm: Literal["fraction", "percent"] | None

    cumulative: bool | None
    marginal: Literal["rug", "box", "violin", "histogram"] | None

    opacity: float | None
    orientation: Literal["v", "h"] | None

    category_orders: dict[str, list[Any]] | None
    labels: dict[str, str] | None
    color_discrete_sequence: t.Sequence[Any] | None
    color_discrete_map: dict[Any, Any] | None
    # color_continuous_scale: t.Sequence[Any] | None
    # range_color: tuple[float, float] | None

    hover_name: str | None
    hover_data: dict[str, bool | str | list[Any]] | list[str] | None

    animation_frame: str | None
    animation_group: str | None

    template: str | Any | None
    title: str | None
    width: int | None
    height: int | None


# class GoHistogram_TypedDict(t.TypedDict, total=False):
#     # plotly.graph_objects.Histogram kwargs

#     # -----------------
#     # Data
#     # -----------------
#     # x: t.Sequence[Any] | np.ndarray | None
#     y: t.Sequence[Any] | np.ndarray | None
#     orientation: Literal["v", "h"] | None
#     name: str | None
#     legendgroup: str | None
#     showlegend: bool | None
#     opacity: float | None
#     visible: bool | Literal["legendonly"] | None

#     # -----------------
#     # Histogram behavior
#     # -----------------
#     histfunc: Literal["count", "sum", "avg", "min", "max"] | None
#     histnorm: Literal["", "percent", "probability", "density", "probability density"] | None
#     cumulative: dict[str, Any] | None
#     autobinx: bool | None
#     autobiny: bool | None
#     nbinsx: int | None
#     nbinsy: int | None
#     xbins: dict[str, float | int] | None
#     ybins: dict[str, float | int] | None

#     # -----------------
#     # Styling
#     # -----------------
#     marker: dict[str, Any] | None
#     marker_color: Any | None
#     marker_line: dict[str, Any] | None

#     # -----------------
#     # Hover / text
#     # -----------------
#     hoverinfo: str | None
#     hovertemplate: str | None
#     text: t.Sequence[str] | None
#     texttemplate: str | None

#     # -----------------
#     # Axes / subplot binding
#     # -----------------
#     xaxis: str | None
#     yaxis: str | None
#     offsetgroup: str | None
#     alignmentgroup: str | None

#     # -----------------
#     # Animation / transforms
#     # -----------------
#     uid: str | None
#     meta: Any | None
#     customdata: Any | None
#     transforms: list[dict[str, Any]] | None

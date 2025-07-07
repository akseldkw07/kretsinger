from __future__ import annotations
import matplotlib.colors as mcolors
import warnings
import typing as t

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .typed_cls import *
from .numpy_utils import SingleReturnArray
import pandas as pd
import seaborn as sns
from kret_studies.float_utils import get_precision


@t.overload
def subplots(  # type: ignore
    ncols: t.Literal[1] = ...,
    nrows: t.Literal[1] = ...,
    width_per: float = ...,
    height_per: float = ...,
    **subplot_args: t.Unpack[Subplots_TypedDict],
) -> t.Tuple[Figure, Axes]: ...


@t.overload
def subplots(  # type: ignore
    ncols: int = ...,
    nrows: t.Literal[1] = ...,
    width_per: float = ...,
    height_per: float = ...,
    **subplot_args: t.Unpack[Subplots_TypedDict],
) -> t.Tuple[Figure, SingleReturnArray[Axes]]: ...


@t.overload
def subplots(  # type: ignore
    ncols: t.Literal[1] = ...,
    nrows: int = ...,
    width_per: float = ...,
    height_per: float = ...,
    **subplot_args: t.Unpack[Subplots_TypedDict],
) -> t.Tuple[Figure, SingleReturnArray[Axes]]: ...


@t.overload
def subplots(
    ncols: int = ...,
    nrows: int = ...,
    width_per: float = ...,
    height_per: float = ...,
    **subplot_args: t.Unpack[Subplots_TypedDict],
) -> t.Tuple[Figure, SingleReturnArray[SingleReturnArray[Axes]]]: ...


def subplots(
    ncols: int = 1,
    nrows: int = 1,
    width_per: float = 8,
    height_per: float = 6,
    **subplot_args: t.Unpack[Subplots_TypedDict],
):
    fig_width = ncols * width_per
    fig_height = nrows * height_per
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), **subplot_args)
    style_axes(fig, ax)

    plt.close(fig)
    return fig, ax


def style_axes(fig: Figure, axes: Axes | t.Iterable[Axes]):
    """
    Add grid
    idk what else
    """
    if isinstance(axes, Axes):
        axes = [axes]

    for axes in axes:
        axes.grid(True, which="both", axis="both")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            axes.legend()
    fig.tight_layout()


rwg = ["red", "white", "green"]
wg = ["white", "green"]
wr = ["red", "white"]

red_green_centered = mcolors.LinearSegmentedColormap.from_list("RedWhiteGreen", rwg)
white_green = mcolors.LinearSegmentedColormap.from_list("WhiteGreen", wg)
white_red = mcolors.LinearSegmentedColormap.from_list("WhiteRed", wr)

try:
    plt.colormaps.register(cmap=red_green_centered, name="RedWhiteGreen")
    plt.colormaps.register(cmap=white_green, name="WhiteGreen")
    plt.colormaps.register(cmap=white_red, name="WhiteRed")
except ValueError:
    # Re-registering raises ex
    pass


def set_title_label(ax: Axes, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    if title is not None:
        ax.set_title(title, fontsize=16)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=16)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=16)


def format_residual_plot(ax: Axes):
    style_color = zip([0, 2, 3], ["green", "purple", "r"])
    linestyle = "-"
    linewidth = 0.6

    for y, color in style_color:
        ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=linewidth)
        ax.axhline(y=y * -1, color=color, linestyle=linestyle, linewidth=linewidth)

    ax.set_title("Studentized Residuals vs. Fitted Values")
    ax.set_xlabel(r"Fitted Values ($\hat{Y}$)")
    ax.set_ylabel("Studentized Residuals")


def _generate_heatmap_colors(df: pd.DataFrame) -> Heatmap_Params_TD:
    df_min = float(df.min(axis=None))  # type: ignore
    df_max = float(df.max(axis=None))  # type: ignore
    abs_max = max(abs(df_min), abs(df_max))

    if df_min >= 0 and df_max >= 0:
        return {"vmin": 0, "vmax": abs_max, "cmap": white_green}
    if df_min <= 0 and df_max <= 0:
        return {"vmin": -abs_max, "vmax": 0, "cmap": white_red}
    if df_min < 0 and df_max >= 0:
        return {"vmin": -abs_max, "vmax": abs_max, "cmap": red_green_centered}
    else:
        raise ValueError(f"{df_min=} {df_max=}, {abs_max=}")


def _generate_heatmap_params(df: pd.DataFrame):
    colors = _generate_heatmap_colors(df)

    fmt = get_precision(df.values.flatten())

    ret = colors | {"fmt": fmt}
    return ret


def heatmap_df(df: pd.DataFrame, **kwargs: t.Unpack[Sns_Heatmap_TypedDict]):
    computed_params = _generate_heatmap_params(df)

    kwargs_default: Sns_Heatmap_TypedDict = {"annot": True, "cmap": red_green_centered, "linewidths": 0.1, "cbar": True}
    kwargs_compute = kwargs_default | computed_params
    kwargs = {**kwargs_compute, **kwargs}
    print(kwargs)
    sns.heatmap(df, **kwargs)

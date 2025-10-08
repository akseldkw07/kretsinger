from __future__ import annotations
import warnings
import typing as t
from kret_studies.helpers.matplot_helper import _generate_heatmap_params, red_green_centered
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .low_prio.typed_cls import *
from .numpy_utils import SingleReturnArray
import pandas as pd
import seaborn as sns
from kret_studies.helpers.float_utils import smart_round
import numpy as np


# region EDA
def plot_pairwise_sns(df: pd.DataFrame, **kwargs: t.Unpack[Pairplot_TypedDict]):
    """
    Plot pairwise relationships in the dataset.

    >>> uks_mpl.plot_pairwise_sns(df, vars=["area", "bedrooms", "stories", "price"], hue="bedrooms")

    """
    kwargs_default: Pairplot_TypedDict = {"kind": "reg", "diag_kind": "kde", "palette": "coolwarm"}
    kwargs_effective = kwargs_default | kwargs
    pair = sns.pairplot(df, **kwargs_effective)
    return pair


def plot_pairwise_sns_cat_bool(df: pd.DataFrame, **kwargs: t.Unpack[Pairplot_TypedDict]):
    """
    Automatically select categorical and boolean variables for pairwise plotting.
    """
    cat_vars = df.select_dtypes(include=["category"]).columns.tolist()
    bool_vars = df.select_dtypes(include=["bool"]).columns.tolist()
    kwargs_default: Pairplot_TypedDict = {"vars": cat_vars + bool_vars}
    kwargs_effective = kwargs_default | kwargs
    return plot_pairwise_sns(df, **kwargs_effective)


# endregion
# region PLOT UTILS
def plot_scatter_and_ols(
    ax: Axes,
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    cat: pd.Series | None = None,
    is_y_pred: bool = False,
):
    """
    If `cat` is provided, plot scatter and OLS for each category (hue).
    """
    if cat is not None:
        categories = cat.unique()
        for category in categories:
            mask = cat == category
            sns.scatterplot(x=x[mask], y=y[mask], ax=ax, alpha=0.5, label=str(category))
            sns.regplot(x=x[mask], y=y[mask], ax=ax, scatter=False, label=f"OLS {category}")
    else:
        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.5)
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color="red", line_kws={"linestyle": "--", "linewidth": 1.5})
    if is_y_pred:
        # identity line (y = x)
        xy_min = float(min(x.min(), y.min()))
        xy_max = float(max(x.max(), y.max()))
        ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", linewidth=1, alpha=0.7, label="y = x")


def plot_residual_hist(ax: Axes, y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray):
    residuals = np.array(y_true - y_pred)
    sns.histplot(residuals, ax=ax, kde=True, color="blue", bins=30)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="x = 0")


def df_in_ax(ax: Axes, df: pd.DataFrame, round_: int | None = None, fontsize=10, scale=(1.0, 1.2)):
    """Draw df as a table inside `ax` and return the Matplotlib Table."""
    ax.set_axis_off()
    round_val = smart_round(df) if round_ is None else round_
    data = df.round(round_val)
    # Make each column width proportional so that total fills the axes width
    n_cols = len(data.columns)
    col_widths = [1.0 / n_cols] * n_cols
    tbl = pd.plotting.table(ax, data, loc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(*scale)
    return tbl


# endregion
# region STYLE


def style_axes(
    fig: Figure, axes: Axes | list[Axes] | SingleReturnArray[Axes] | SingleReturnArray[SingleReturnArray[Axes]]
):
    """
    Add grid
    idk what else
    """
    if isinstance(axes, Axes):
        axes = [axes]

    if isinstance(axes, list):
        axes = t.cast(SingleReturnArray, np.array(axes))

    for ax in axes.ravel():
        ax.grid(True, which="both", axis="both")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax.legend()
    fig.tight_layout()


def set_title_label(ax: Axes, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    if title is not None:
        ax.set_title(title, fontsize=16)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=16)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=16)


# endregion
# region HEATMAP


def heatmap_df(df: pd.DataFrame, **kwargs: t.Unpack[Sns_Heatmap_TypedDict]):
    computed_params = _generate_heatmap_params(df)

    kwargs_default: Sns_Heatmap_TypedDict = {"annot": True, "cmap": red_green_centered, "linewidths": 0.1, "cbar": True}
    kwargs_compute = kwargs_default | computed_params
    kwargs = {**kwargs_compute, **kwargs}
    # print(kwargs)
    sns.heatmap(df, **kwargs)


# endregion
# region SUBPLOTS
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
    height_per: float = 7,
    **subplot_args: t.Unpack[Subplots_TypedDict],
):
    fig_width = ncols * width_per
    fig_height = nrows * height_per
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), **subplot_args)

    plt.close(fig)
    style_axes(fig, ax)

    if nrows > 1:
        fig.subplots_adjust(hspace=0.15)

    return fig, ax


# endregion

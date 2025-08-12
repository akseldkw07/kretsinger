from __future__ import annotations
import matplotlib.colors as mcolors
import warnings
import typing as t
import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .low_prio.typed_cls import *
from .numpy_utils import SingleReturnArray
import pandas as pd
import seaborn as sns
from kret_studies.float_utils import get_precision, smart_round
import numpy as np
import kret_studies.kret_sklearn as uks_sklearn


# region PLOT UTILS
def plot_scatter_and_ols(ax: Axes, x: pd.Series | np.ndarray, y: pd.Series | np.ndarray):
    sns.scatterplot(x=x, y=y, ax=ax, alpha=0.5)
    sns.regplot(x=x, y=y, ax=ax, scatter=False, color="red", line_kws={"linestyle": "--", "linewidth": 1.5})


def plot_residual_hist(ax: Axes, y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray):
    residuals = np.array(y_true - y_pred)
    sns.histplot(residuals, ax=ax, kde=True, color="blue", bins=30)
    ax.axhline(0, color="red", linestyle="--")


def plot_diagnostics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, X: pd.DataFrame | np.ndarray):
    # Residuals
    ret = uks_sklearn.model_diagnostics(y_true, y_pred, X)
    metrics_df, residuals, vif, slope, intercept, resid_std = (
        ret["metrics_df"],
        ret["residuals"],
        ret["vif"],
        ret["slope"],
        ret["intercept"],
        ret["resid_std"],
    )

    # True vs Predicted with identity line and OLS slope/intercept annotation
    fig, axes = subplots(3, 3)
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[0, 0])
    axes[0, 0].set_title("True vs Predicted")
    axes[0, 0].set_xlabel("True Values")
    axes[0, 0].set_ylabel("Predicted Values")

    # identity line (y = x)
    xy_min = float(min(y_true.min(), y_pred.min()))
    xy_max = float(max(y_true.max(), y_pred.max()))
    axes[0, 0].plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", linewidth=1, alpha=0.7)

    # draw the fitted line explicitly
    x_line = np.linspace(xy_min, xy_max, 100)
    y_line = intercept + slope * x_line
    axes[0, 0].plot(x_line, y_line, linewidth=1.5)

    # annotate slope & intercept in figure coordinates
    axes[0, 0].text(
        0.02,
        0.98,
        f"slope = {slope:.3f}\nintercept = {intercept:.3g}",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        fontsize=9,
    )

    sns.residplot(x=y_pred, y=residuals, lowess=True, ax=axes[0, 1])
    axes[0, 1].set_title("Residuals vs Fitted")
    axes[0, 1].set_xlabel("Fitted Values")
    axes[0, 1].set_ylabel("Residuals")

    sns.histplot(residuals, kde=True, ax=axes[0, 2])
    axes[0, 2].set_title("Residual Distribution")

    # QQ plot using standardized residuals to avoid scale/outlier dominance
    sm.qqplot(resid_std, line="45", ax=axes[1, 0])
    axes[1, 0].set_title("QQ Plot (standardized residuals)")

    # ACF only if we have enough residuals
    try:
        sm.graphics.tsa.plot_acf(residuals, ax=axes[1, 1])
        axes[1, 1].set_title("ACF of Residuals")
    except Exception:
        axes[1, 1].text(0.5, 0.5, "ACF unavailable", ha="center", va="center")
        axes[1, 1].set_axis_off()

    if not vif.empty:
        sns.barplot(data=vif, x="feature", y="VIF", ax=axes[1, 2])
        axes[1, 2].set_title("VIF per Feature")
        axes[1, 2].tick_params(axis="x", rotation=45)
    else:
        axes[1, 2].text(0.5, 0.5, "VIF unavailable", ha="center", va="center")
        axes[1, 2].set_axis_off()

    plt.tight_layout()
    return metrics_df, fig


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


# endregion
# region HEATMAP
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
    height_per: float = 6,
    **subplot_args: t.Unpack[Subplots_TypedDict],
):
    fig_width = ncols * width_per
    fig_height = nrows * height_per
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), **subplot_args)
    style_axes(fig, ax)

    plt.close(fig)
    return fig, ax


# endregion

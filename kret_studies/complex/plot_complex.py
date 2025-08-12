from __future__ import annotations
import statsmodels.api as sm

from matplotlib.axes import Axes

import pandas as pd
import seaborn as sns
import numpy as np
from .model_diagnostics import model_diagnostics
from kret_studies.kret_mpl import subplots, df_in_ax


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


def plot_diagnostics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, X: pd.DataFrame | np.ndarray):
    # Residuals
    ret = model_diagnostics(y_true, y_pred, X)
    metrics, residuals, vif, slope, intercept, resid_std = (
        ret[k] for k in ["metrics", "residuals", "vif", "slope", "intercept", "resid_std"]
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
    df_in_ax(axes[2, 0], metrics, round_=3, fontsize=10, scale=(1.0, 1.2))
    return fig

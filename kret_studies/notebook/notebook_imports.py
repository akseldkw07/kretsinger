from __future__ import annotations

import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


def model_diagnostics(y_true, y_pred, residuals):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # True vs Predicted with identity line and OLS slope/intercept annotation
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[0, 0])
    axes[0, 0].set_title("True vs Predicted")
    axes[0, 0].set_xlabel("True Values")
    axes[0, 0].set_ylabel("Predicted Values")

    # identity line (y = x)
    xy_min = float(min(y_true.min(), y_pred.min()))
    xy_max = float(max(y_true.max(), y_pred.max()))
    axes[0, 0].plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", linewidth=1, alpha=0.7)

    # OLS fit of y_pred on y_true to expose bias
    X_line = sm.add_constant(y_true)
    line_model = sm.OLS(y_pred, X_line).fit()
    intercept, slope = line_model.params[0], line_model.params[1]

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

    # Residuals vs Fitted
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0, 1])
    axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Residuals vs Fitted")
    axes[0, 1].set_xlabel("Fitted values")
    axes[0, 1].set_ylabel("Residuals")

    # QQ plot using standardized residuals to avoid scale/outlier dominance
    resid_std = (residuals - residuals.mean()) / (residuals.std(ddof=1) if residuals.std(ddof=1) != 0 else 1.0)
    sm.qqplot(resid_std, line="45", ax=axes[1, 0])
    axes[1, 0].set_title("QQ Plot (standardized residuals)")

    # Residuals histogram
    sns.histplot(residuals, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Residuals Histogram")

    plt.tight_layout()
    plt.show()

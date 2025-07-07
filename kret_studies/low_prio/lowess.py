"""
Some implementations of LOWESS (Locally Weighted Scatterplot Smoothing) in Python.
# LOWESS is a non-parametric regression method that combines multiple regression models in a k-nearest-neighbor-based meta-model.
# It is particularly useful for smoothing scatterplots and can be used to fit a smooth curve to data points.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


def demo_lowess_smoothing():
    # Sample data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + np.random.normal(0, 0.4, 100)

    # Apply LOWESS
    smoothed = sm.nonparametric.lowess(y, x, frac=0.2)
    smoothed_x = smoothed[:, 0]
    smoothed_y = smoothed[:, 1]

    print("First 5 smoothed points (x, y):")
    print(np.round(smoothed[:5], 3))

    # Using the same x and y from the example above
    df = pd.DataFrame({"x_values": x, "y_values": y})

    # Create a scatter plot with a LOWESS trendline
    sns.lmplot(
        x="x_values",
        y="y_values",
        data=df,
        lowess=True,
        line_kws={"color": "red"},
        scatter_kws={"alpha": 0.6},
    )
    plt.title("Scatter Plot with LOWESS Trendline")
    plt.show()

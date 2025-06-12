from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .typed_cls import Subplots_TypedDict


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    width_per: float = 6,
    height_per: float = 6,
    **kwargs: t.Unpack[Subplots_TypedDict],
):
    fig_width = ncols * width_per
    fig_height = nrows * height_per
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    plt.close(fig)
    return fig, ax


def style_axes(axes: Axes):
    """
    Add grid
    idk what else
    """
    axes.grid(True, which="both", axis="both")

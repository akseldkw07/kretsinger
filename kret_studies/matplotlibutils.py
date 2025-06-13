from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .typed_cls import Subplots_TypedDict
from .numpy_utils import SingleReturnArray


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
    plt.close(fig)
    return fig, ax


def style_axes(axes: Axes):
    """
    Add grid
    idk what else
    """
    axes.grid(True, which="both", axis="both")

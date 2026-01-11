import typing as t
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kret_np_pd.single_ret_ndarray import SingleReturnArray

from .typed_cls_mpl import Subplots_TypedDict


class SubplotHelper:
    # region AXES STYLING

    @classmethod
    def set_title_label(cls, ax: Axes, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
        if title is not None:
            ax.set_title(title, fontsize=16)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=16)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=16)

    @classmethod
    def style_axes(
        cls, fig: Figure, axes: Axes | list[Axes] | SingleReturnArray[Axes] | SingleReturnArray[SingleReturnArray[Axes]]
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

    # endregion
    @classmethod
    def subplots_smart_dims(cls, nplots: int, max_cols: int = 5) -> tuple[int, int]:
        rows = int(np.ceil(np.sqrt(nplots)))
        cols = int(np.ceil(nplots / rows))

        if cols > max_cols:
            cols = max_cols
            rows = int(np.ceil(nplots / cols))
        return rows, cols

    # region SUBPLOTS
    @t.overload
    @classmethod
    def subplots(  # type: ignore[override]
        cls,
        ncols: t.Literal[1] = ...,
        nrows: t.Literal[1] = ...,
        width_per: float = ...,
        height_per: float = ...,
        **subplot_args: t.Unpack[Subplots_TypedDict],
    ) -> t.Tuple[Figure, Axes]: ...

    @t.overload
    @classmethod
    def subplots(
        cls,
        ncols: int = ...,
        nrows: t.Literal[1] = ...,
        width_per: float = ...,
        height_per: float = ...,
        **subplot_args: t.Unpack[Subplots_TypedDict],
    ) -> t.Tuple[Figure, SingleReturnArray[Axes]]: ...

    @t.overload
    @classmethod
    def subplots(
        cls,
        ncols: t.Literal[1] = ...,
        nrows: int = ...,
        width_per: float = ...,
        height_per: float = ...,
        **subplot_args: t.Unpack[Subplots_TypedDict],
    ) -> t.Tuple[Figure, SingleReturnArray[Axes]]: ...

    @t.overload
    @classmethod
    def subplots(
        cls,
        ncols: int = ...,
        nrows: int = ...,
        width_per: float = ...,
        height_per: float = ...,
        **subplot_args: t.Unpack[Subplots_TypedDict],
    ) -> t.Tuple[Figure, SingleReturnArray[SingleReturnArray[Axes]]]: ...

    @classmethod
    def subplots(
        cls,
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
        cls.style_axes(fig, ax)

        if nrows > 1:
            fig.subplots_adjust(hspace=0.15)

        return fig, ax

    # endregion

from __future__ import annotations

import typing as t
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.io.formats.style import Styler

from kret_matplotlib.matplot_helper import KretMatplotHelper
from kret_np_pd.single_ret_ndarray import SingleReturnArray
from kret_type_hints.typed_cls import Pairplot_TypedDict, Sns_Heatmap_TypedDict, Subplots_TypedDict


class Plotting_Utils(KretMatplotHelper):
    """
    Utility class for common plotting functions using Matplotlib and Seaborn.
    """

    # region SNS
    @classmethod
    def plot_pairwise_sns(cls, df: pd.DataFrame, **kwargs: t.Unpack[Pairplot_TypedDict]):
        """
        Plot pairwise relationships in the dataset.

        >>> uks_mpl.plot_pairwise_sns(df, vars=["area", "bedrooms", "stories", "price"], hue="bedrooms")

        """
        kwargs_default: Pairplot_TypedDict = {"kind": "reg", "diag_kind": "kde", "palette": "coolwarm"}
        kwargs_effective = kwargs_default | kwargs
        pair = sns.pairplot(df, **kwargs_effective)
        return pair

    @classmethod
    def plot_pairwise_sns_cat_bool(cls, df: pd.DataFrame, **kwargs: t.Unpack[Pairplot_TypedDict]):
        """
        Automatically select categorical and boolean variables for pairwise plotting.
        """
        cat_vars = df.select_dtypes(include=["category"]).columns.tolist()
        bool_vars = df.select_dtypes(include=["bool"]).columns.tolist()
        kwargs_default: Pairplot_TypedDict = {"vars": cat_vars + bool_vars}
        kwargs_effective = kwargs_default | kwargs
        return cls.plot_pairwise_sns(df, **kwargs_effective)

    # endregion
    # region HEATMAP

    @classmethod
    def heatmap_df(cls, df: pd.DataFrame | Styler, **kwargs: t.Unpack[Sns_Heatmap_TypedDict]):
        # Accept a pandas Styler (presentation wrapper) and unwrap to the underlying DataFrame
        # If a Styler is passed, extract the underlying DataFrame; otherwise leave as-is
        df_data = df if isinstance(df, pd.DataFrame) else df.data  # type: ignore

        computed_params = cls._generate_heatmap_params(df_data)

        kwargs_default: Sns_Heatmap_TypedDict = {
            "annot": True,
            "cmap": cls.red_green_centered,
            "linewidths": 0.1,
            "cbar": True,
        }
        kwargs_compute = kwargs_default | computed_params
        kwargs = {**kwargs_compute, **kwargs}
        # print(kwargs)
        sns.heatmap(df_data, **kwargs)

    # endregion
    # region AXES STYLING

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

    @classmethod
    def set_title_label(cls, ax: Axes, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
        if title is not None:
            ax.set_title(title, fontsize=16)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=16)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=16)

    # endregion
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

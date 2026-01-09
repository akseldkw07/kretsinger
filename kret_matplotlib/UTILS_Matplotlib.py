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
from kret_type_hints.typed_cls import (
    Background_gradient_TypedDict,
    Format_TypedDict,
    Pairplot_TypedDict,
    Pandas_Styler_TypedDict,
    Sns_Heatmap_TypedDict,
    Subplots_TypedDict,
)
from kret_type_hints.typed_dict_utils import TypedDictUtils


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

        kwargs_compute = cls.sns_heatmap_defaults | computed_params
        kwargs = {**kwargs_compute, **kwargs}
        # print(kwargs)
        sns.heatmap(df_data, **kwargs)

    @classmethod
    def heatmap_styler(
        cls, df: pd.DataFrame | Styler, show_legend: bool = True, **kwargs: t.Unpack[Pandas_Styler_TypedDict]
    ) -> Styler:
        """
        Generate a heatmap-styled DataFrame as a pandas Styler object (returns HTML without plotting).

        This follows the same logic as heatmap_df but returns a Styler object suitable for
        displaying in Jupyter notebooks and for integration with dtt().

        Args:
            df: DataFrame or Styler to create heatmap from
            **kwargs: Additional seaborn heatmap parameters

        Returns:
            pd.io.formats.style.Styler: Styled DataFrame with heatmap coloring
        """
        # Extract DataFrame if a Styler was passed
        df_data = df if isinstance(df, pd.DataFrame) else df.data  # type: ignore

        computed_params = cls._generate_heatmap_params(df_data)
        fmt = computed_params.pop("fmt")
        formatter = lambda v: format(v, fmt)

        heatmap_params = cls.background_grad_defaults | computed_params | kwargs  # type: ignore
        heatmap_params = TypedDictUtils.filter_dict_by_typeddict(heatmap_params, Background_gradient_TypedDict)
        fmt_params = cls.format_defaults | {"formatter": formatter} | kwargs
        fmt_params = TypedDictUtils.filter_dict_by_typeddict(fmt_params, Format_TypedDict)

        # Create a Styler with background gradient
        styled: Styler = df_data.style.background_gradient(**heatmap_params)
        styled = styled.format(**fmt_params)

        styled = styled.set_properties(**{"text-align": "center"})  # type: ignore[arg-type]

        if show_legend:
            legend_html = cls._legend_html(**computed_params)
            styled = styled.set_caption(legend_html)
        return styled

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

    @classmethod
    def subplots_smart_dims(cls, nplots: int, max_cols: int = 5) -> tuple[int, int]:
        rows = int(np.ceil(np.sqrt(nplots)))
        cols = int(np.ceil(nplots / rows))

        if cols > max_cols:
            cols = max_cols
            rows = int(np.ceil(nplots / cols))
        return rows, cols

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

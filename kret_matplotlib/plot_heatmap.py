import typing as t

import pandas as pd
import seaborn as sns
from pandas.io.formats.style import Styler

from kret_type_hints.typed_dict_utils import TypedDictUtils

from ._core.constants_mpl import MPLDefaults as C
from ._core.typed_cls_mpl import (
    Background_gradient_TypedDict,
    Format_TypedDict,
    Pandas_Styler_TypedDict,
    Sns_Heatmap_TypedDict,
)
from .matplot_helper import KretMatplotHelper


class HeatmapUtils(KretMatplotHelper):
    # region HEATMAP

    @classmethod
    def heatmap_df(cls, df: pd.DataFrame | Styler, **kwargs: t.Unpack[Sns_Heatmap_TypedDict]):
        # Accept a pandas Styler (presentation wrapper) and unwrap to the underlying DataFrame
        # If a Styler is passed, extract the underlying DataFrame; otherwise leave as-is
        df_data = df if isinstance(df, pd.DataFrame) else df.data  # type: ignore

        computed_params = cls._generate_heatmap_params(df_data)

        kwargs_compute = C.sns_heatmap_defaults | computed_params
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

        heatmap_params = C.background_grad_defaults | computed_params | kwargs  # type: ignore
        heatmap_params = TypedDictUtils.filter_dict_by_typeddict(heatmap_params, Background_gradient_TypedDict)
        fmt_params = C.format_defaults | {"formatter": formatter} | kwargs
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

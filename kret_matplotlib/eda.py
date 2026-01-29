import typing as t

import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from ._core.typed_cls_mpl import *
import plotly.express as px
from .subplot_utils import SubplotHelper
from plotly.subplots import make_subplots
from ._core.constants_mpl import MPLDefaults


class EDA_Utils:
    # region SNS
    @classmethod
    def plot_pairwise_sns(cls, df: pd.DataFrame, **kwargs: t.Unpack[Pairplot_TypedDict]):
        """
        Plot pairwise relationships in the dataset.

        >>> EDA_Utils.plot_pairwise_sns(df, vars=["area", "bedrooms", "stories", "price"], hue="bedrooms")

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

    @classmethod
    def histogram_on_target(
        cls, df: pd.DataFrame, target: str, x: list[str] | str | None, **kwargs_: t.Unpack[PxHistogram_TypedDict]
    ):
        """
        Takes in a dataframe and a target column name, and plots histograms of the specified columns

        """
        x = x if isinstance(x, list) else [x] if isinstance(x, str) else [col for col in df.columns if col != target]
        rows, col = SubplotHelper.subplots_smart_dims(len(x))
        fig = make_subplots(rows=rows, cols=col, subplot_titles=x)

        kwargs = MPLDefaults.PX_HIST_DEFAULTS | kwargs_

        for i, col_name in enumerate(x):
            r = i // col + 1
            c = i % col + 1
            hist = go.Histogram(x=df[col_name], color=target, **kwargs)
            fig.add_trace(hist, row=r, col=c)
        # match px-like stacking/overlay behavior
        # fig.update_layout(barmode=barmode)

        # nice legend title (px does this)
        # if legend:
        # fig.update_layout(legend_title_text=group_label)

        return fig
        # fig = px.histogram(df, x=x, color=target, **kwargs)
        # fig.update_layout(height=400, width=500, showlegend=True)
        # fig.update_traces(marker_line_width=1, marker_line_color="black")

    @classmethod
    def histogram_on_target_basic(
        cls, df: pd.DataFrame, target: str, x: list[str] | str | None, **kwargs_: t.Unpack[PxHistogram_TypedDict]
    ):
        x = x if isinstance(x, list) else [x] if isinstance(x, str) else [col for col in df.columns if col != target]
        figs: list[go.Figure] = []
        for col_name in x:
            kwargs = MPLDefaults.PX_HIST_DEFAULTS | kwargs_
            fig = px.histogram(df, x=col_name, color=target, **kwargs)
            fig.update_layout(height=400, width=500, showlegend=True)
            fig.update_traces(marker_line_width=1, marker_line_color="black")
            figs.append(fig)
        return figs

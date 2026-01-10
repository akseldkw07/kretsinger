import typing as t

import pandas as pd
import seaborn as sns

from .typed_cls_mpl import Pairplot_TypedDict


class SeabornUtils:
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

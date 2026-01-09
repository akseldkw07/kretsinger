from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.linear_model import HuberRegressor, LinearRegression

from kret_matplotlib.UTILS_Matplotlib import Plotting_Utils
from kret_np_pd.filters import FILT_TYPE, FilterSampleUtils
from kret_np_pd.UTILS_np_pd import NP_PD_Utils
from kret_rosetta.to_pd_np import TO_NP_TYPE
from kret_rosetta.UTILS_rosetta import UTILS_rosetta

REG_FUNC = t.Literal["OLS", "Huber"]

if t.TYPE_CHECKING:
    pass


# ==============================================================================================
# Helper classes
# ==============================================================================================


class StateDF(pd.DataFrame):
    y_true: pd.Series
    y_pred: pd.Series
    category: pd.Series
    filt: pd.Series


class RichDF(StateDF):
    y_true_centroid: pd.Series
    y_pred_centroid: pd.Series
    centroid_bin: pd.Categorical


class AttrBase:
    _df_input: StateDF
    _df_sorted: StateDF
    _df_enriched: RichDF
    downsample: int
    seed: int

    n_centroids: int
    regression_func: REG_FUNC


class FilterCalcMixin(AttrBase):
    def calc_filter_mask(self):

        filt = NP_PD_Utils.mask_and(self._df_input.filt.to_numpy(), ~self._df_input.isna())
        if self.downsample is not None:
            filt = FilterSampleUtils.downsample_bool(filt, k=self.downsample, seed=self.seed)
        return filt

    def reset(self):
        for attr in ("_df_enriched", "_df_sorted"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.seed += 1
        self.calc_filter_mask()


class DataFrameMixin(FilterCalcMixin):

    @property
    def DfBaseSorted(self) -> StateDF:
        """
        Returns the base DataFrame sorted by y_true after applying the effective filter mask.
        """
        try:
            self._df_sorted
        except AttributeError:
            self.df_setup()
        return self._df_sorted

    def df_setup(self):
        eff_filt = self.calc_filter_mask()
        df = self._df_input[eff_filt].copy()
        df = df.sort_values(by="y_true").reset_index(drop=True)

        self._df_sorted = df

    @property
    def DfFull(self) -> RichDF:
        """
        Returns the enriched DataFrame with centroid calculations.
        """
        try:
            self._df_enriched
        except AttributeError:
            self.df_enrich()
        return self._df_enriched

    def df_enrich(self):
        df = self.DfBaseSorted.copy()
        centroid_bin = df.y_pred.groupby(df.category, observed=False).transform(
            lambda s: pd.qcut(s, q=self.n_centroids, labels=False, duplicates="drop")
        )
        df["centroid_bin"] = pd.Categorical(centroid_bin)
        df["y_true_centroid"] = df.groupby(["category", "centroid_bin"], observed=False)["y_true"].transform("mean")
        df["y_pred_centroid"] = df.groupby(["category", "centroid_bin"], observed=False)["y_pred"].transform("mean")

        self._df_enriched = RichDF(df)


# ==============================================================================================
# Main Class
# ==============================================================================================
class GroupScatter(DataFrameMixin):
    """
    Plot a group scatter with regression line. Helpful for visualizing model performance, especially when there are thousands or millions of points.

    Take in y and y_hat, optional categorical column, optional filter, # centroids=25, downsample=False, and regression funcion (OLS, Huber, etc)=OLS

    TODO - add option for 10,25,75,90 percentiles as error bars around centroids
    """

    model_dict: dict[tuple[t.Any, int], HuberRegressor | LinearRegression]
    centroid_scatter_kwargs: dict[t.Any, t.Any] = dict(
        s=60, alpha=0.6, marker="D", edgecolor="black", linewidth=0.5, zorder=3
    )
    raw_scatter_kwargs: dict[t.Any, t.Any] = dict(s=10, alpha=0.2, marker="o", edgecolor="none", zorder=2)

    def __init__(
        self,
        y: TO_NP_TYPE,
        y_hat: TO_NP_TYPE,
        category: np.ndarray | pd.Series | pd.Categorical | None = None,
        filter: FILT_TYPE | None = None,
        n_centroids: int = 25,
        downsample: int | None = None,
        regression_func: REG_FUNC = "OLS",
    ):
        y_true = UTILS_rosetta.coerce_to_ndarray(y, assert_1dim=True, attempt_flatten_1d=True)
        y_pred = UTILS_rosetta.coerce_to_ndarray(y_hat, assert_1dim=True, attempt_flatten_1d=True)

        if category is None:
            category = pd.Categorical(np.zeros_like(y_true))
        elif isinstance(category, pd.Categorical):
            category = category
        else:
            arr = UTILS_rosetta.coerce_to_ndarray(category, assert_1dim=True, attempt_flatten_1d=True)
            category = pd.Categorical(arr)

        _filter_mask_input = FilterSampleUtils.process_filter(filter, shape=y_true.shape[0])
        self._df_input = StateDF({"y_true": y_true, "y_pred": y_pred, "category": category, "filt": _filter_mask_input})

        self.n_centroids = n_centroids
        self.downsample = downsample if downsample is not None else len(y)
        self.regression_func = regression_func

        self.seed = 0

    @staticmethod
    def fit_line(x: np.ndarray, y: np.ndarray, kind: REG_FUNC):
        x2 = x.reshape(-1, 1)
        if kind == "OLS":
            model = LinearRegression().fit(x2, y)
            return model
        if kind == "Huber":
            model = HuberRegressor().fit(x2, y)
            return model
        raise ValueError(kind)

    def plot(
        self,
        ax: Axes | t.Iterable[Axes] | t.Literal["shared", "separate"] = "shared",
        scatters: tuple[t.Literal["raw", "centroids"], ...] = ("centroids",),
        fit_on: t.Literal["raw", "centroids"] = "raw",
    ):
        """
        Plot the group scatter with regression line.
        """
        shared = ax == "shared" or isinstance(ax, Axes)
        centroids = self.DfFull.drop_duplicates(subset=["category", "centroid_bin"])
        categories = centroids.category.cat.categories

        if isinstance(ax, str):
            rows, cols = Plotting_Utils.subplots_smart_dims(len(categories))
            fig, ax_ = Plotting_Utils.subplots(1, 1) if shared else Plotting_Utils.subplots(rows, cols)
            ax = ax_ if isinstance(ax_, Axes) else ax_.ravel()
        else:
            fig = plt.gcf()

        centroids = self.DfFull.drop_duplicates(subset=["category", "centroid_bin"])
        model_dict: dict[tuple[t.Any, int], HuberRegressor | LinearRegression] = {}

        for i, cat in enumerate(categories):
            color = plt.get_cmap("tab10")(i % 10)
            ax_curr: Axes = ax if isinstance(ax, Axes) else ax[i]  # type: ignore[index]
            df: RichDF = self.DfFull[self.DfFull.category == cat]
            cent_cat: RichDF = centroids[centroids.category == cat]

            if "raw" in scatters:
                ax_curr.scatter(
                    df["y_true"].to_numpy(),
                    df["y_pred"].to_numpy(),
                    color=color,
                    label=f"{cat} Raw",
                    **self.raw_scatter_kwargs,
                )
            if "centroids" in scatters:
                ax_curr.scatter(
                    cent_cat["y_true_centroid"].to_numpy(),
                    cent_cat["y_pred_centroid"].to_numpy(),
                    color=color,
                    label=f"{cat} Centroids",
                    **self.centroid_scatter_kwargs,
                )

            # Fit regression line
            x = df["y_true"].to_numpy() if fit_on == "raw" else cent_cat["y_true_centroid"].to_numpy()
            y = df["y_pred"].to_numpy() if fit_on == "raw" else cent_cat["y_pred_centroid"].to_numpy()

            model = self.fit_line(x, y, self.regression_func)
            x_line: np.ndarray = np.linspace(x.min(), x.max())
            y_line = model.predict(x_line.reshape(-1, 1))
            ax_curr.plot(x_line, y_line, linewidth=2)

            model_dict[(cat, self.seed)] = model

            ax_curr.set_title(f"Group: {cat}")
            ax_curr.set_xlabel("True Values")
            ax_curr.set_ylabel("Predicted Values")
            ax_curr.legend()

        df = self.DfFull
        for ax_curr in ax if not isinstance(ax, Axes) else [ax]:
            lo = min(df["y_true"].min(), df["y_pred"].min())
            hi = max(df["y_true"].max(), df["y_pred"].max())
            ax_curr.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray", zorder=1)
        self.model_dict = model_dict

        if isinstance(ax, Axes):
            """
            If shared:

                ax.set_title("Group Scatter")
            """
            ax.set_title(f"Group Scatter. Groups: {', '.join(map(str, categories))}")
        return fig

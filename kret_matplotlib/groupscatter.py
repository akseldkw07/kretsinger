from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score

from kret_np_pd.filters import FILT_TYPE, FilterSampleUtils
from kret_np_pd.UTILS_np_pd import NP_PD_Utils
from kret_rosetta.to_pd_np import TO_NP_TYPE
from kret_rosetta.UTILS_rosetta import UTILS_rosetta
from .typed_cls_mpl import Subplots_TypedDict
from .UTILS_Matplotlib import UTILS_Plotting

REG_FUNC = t.Literal["OLS", "Huber"]


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
    n_samples: int
    seed: int

    n_centroids: int
    regression_func: REG_FUNC
    model_dict: dict[tuple[t.Any, int], HuberRegressor | LinearRegression]  # (catval, seed) | Model


class FilterCalcMixin(AttrBase):
    def calc_filter_mask(self):

        filt = NP_PD_Utils.mask_and(self._df_input.filt.to_numpy(), ~self._df_input.isna())
        if self.n_samples is not None:
            filt = FilterSampleUtils.downsample_bool(filt, k=self.n_samples, seed=self.seed)
        return filt

    def reset(self, n_samples: int | None = None, new_filter: np.ndarray | None = None, new_seed: int | None = None):
        for attr in ("_df_enriched", "_df_sorted"):
            if hasattr(self, attr):
                delattr(self, attr)
        if new_filter is not None:
            FilterSampleUtils.assert_bool_dtype(new_filter)
            self._df_input.filt = pd.Series(new_filter, dtype=bool)

        self.seed = (self.seed + 1) if new_seed is None else new_seed
        self.n_samples = n_samples if n_samples is not None else self.n_samples
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
        centroid_bin = df.y_true.groupby(df.category, observed=False).transform(
            lambda s: pd.qcut(s, q=self.n_centroids, labels=False, duplicates="drop")
        )
        df["centroid_bin"] = pd.Categorical(centroid_bin)
        df["y_true_centroid"] = df.groupby(["category", "centroid_bin"], observed=False)["y_true"].transform("mean")
        df["y_pred_centroid"] = df.groupby(["category", "centroid_bin"], observed=False)["y_pred"].transform("mean")

        self._df_enriched = RichDF(df)

    def add_percentiles(self, percentiles: tuple[int, ...], recompute: bool = True):
        """
        Add percentile columns to the enriched DataFrame.
        """
        assert not any(
            perc < 0 or perc > 100 for perc in percentiles
        ), f"Percentiles must be between 0 and 100, passed {percentiles}"

        percentile_invert = {100 - perc for perc in percentiles}
        perc_symmetric = set(percentiles).union(percentile_invert)
        df = self.DfFull

        for perc in perc_symmetric:
            col_name = f"y_pred_perc_{perc}"
            if not recompute and col_name in df.columns:
                continue
            df[col_name] = df.groupby(["category", "centroid_bin"], observed=False)["y_pred"].transform(
                lambda s: np.percentile(s, perc)
            )

        self._df_enriched = df
        return perc_symmetric


# ==============================================================================================
# Main Class
# ==============================================================================================


class GroupScatter(DataFrameMixin):
    """
    Plot a group scatter with regression line. Helpful for visualizing model performance, especially when there are thousands or millions of points.

    Take in y and y_hat, optional categorical column, optional filter, # centroids=25, downsample=False, and regression funcion (OLS, Huber, etc)=OLS

    TODO - don't recalculate models if already calculated
    """

    subplot_args: Subplots_TypedDict = {"sharex": True, "sharey": True}
    centroid_scatter_kwargs: dict[str, t.Any] = dict(
        s=60, alpha=0.6, marker="D", edgecolor="black", linewidth=0.5, zorder=1
    )
    raw_scatter_kwargs: dict[str, t.Any] = dict(s=3, alpha=0.2, marker="o", edgecolor="none", zorder=0)

    def __init__(
        self,
        y_true: TO_NP_TYPE,
        y_hat: TO_NP_TYPE,
        category: np.ndarray | pd.Series | pd.Categorical | None = None,
        filter: FILT_TYPE | None = None,
        n_centroids: int = 25,
        n_samples: int | None = None,
        regression_func: REG_FUNC = "OLS",
    ):
        y_true = UTILS_rosetta.coerce_to_ndarray(y_true, assert_1dim=True, attempt_flatten_1d=True)
        y_pred = UTILS_rosetta.coerce_to_ndarray(y_hat, assert_1dim=True, attempt_flatten_1d=True)

        if category is None:
            category = pd.Categorical(np.full(y_true.shape, "All"))
        elif isinstance(category, pd.Categorical):
            category = category
        else:
            arr = UTILS_rosetta.coerce_to_ndarray(category, assert_1dim=True, attempt_flatten_1d=True)
            category = pd.Categorical(arr)

        _filter_mask_input = FilterSampleUtils.process_filter(filter, shape=y_true.shape[0])
        self._df_input = StateDF({"y_true": y_true, "y_pred": y_pred, "category": category, "filt": _filter_mask_input})

        self.n_centroids = n_centroids
        self.n_samples = n_samples if n_samples is not None else len(y_true)
        self.regression_func = regression_func
        self.model_dict = {}

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

    def scatter_args_dynamic(self, which: t.Literal["raw", "centroids"], df: RichDF):
        """ """
        if which == "centroids":
            return self.centroid_scatter_kwargs
        elif which == "raw":
            alpha = np.clip((2_000 / len(df)), a_min=0.1, a_max=0.4)
            size = np.clip((5e3 / len(df)), a_min=2, a_max=20)

            return self.raw_scatter_kwargs | {"alpha": alpha, "s": size}

    def plot(
        self,
        ax: Axes | t.Iterable[Axes] | t.Literal["shared", "separate"] = "shared",
        scatters: tuple[t.Literal["raw", "centroids"], ...] | t.Literal["raw", "centroids"] = "centroids",
        percentiles: tuple[int, ...] = (),
        addtl_plots: tuple[t.Literal["identity", "y_0"], ...] | t.Literal["identity", "y_0"] = "identity",
        fit_on: t.Literal["raw", "centroids"] = "raw",
    ):
        """
        Plot the group scatter with regression line.
        """
        shared = ax == "shared" or isinstance(ax, Axes)
        centroids = self.DfFull.drop_duplicates(subset=["category", "centroid_bin"])
        categories = centroids.category.cat.categories
        if isinstance(scatters, str):
            scatters = (scatters,)
        if isinstance(addtl_plots, str):
            addtl_plots = (addtl_plots,)

        perc_symmetric = self.add_percentiles(percentiles)

        if isinstance(ax, str):
            rows, cols = UTILS_Plotting.subplots_smart_dims(len(categories)) if not shared else (1, 1)
            fig, ax_ = UTILS_Plotting.subplots(rows, cols, **self.subplot_args)
            ax = ax_ if isinstance(ax_, Axes) else ax_.ravel()
        else:
            fig = plt.gcf()

        centroids = self.DfFull.drop_duplicates(subset=["category", "centroid_bin"])
        model_dict = self.model_dict

        for i, cat in enumerate(categories):
            color = plt.get_cmap("tab10")(i % 10)
            ax_curr: Axes = ax if isinstance(ax, Axes) else ax[i]  # type: ignore[index]
            df: RichDF = self.DfFull[self.DfFull.category == cat]
            cent_cat: RichDF = centroids[centroids.category == cat]

            # scatters

            if "raw" in scatters:
                args = self.scatter_args_dynamic("raw", df)
                ax_curr.scatter(df["y_true"], df["y_pred"], color=color, label=f"{cat} Raw", **args)
            if "centroids" in scatters:
                args = self.scatter_args_dynamic("centroids", df)
                ax_curr.scatter(
                    cent_cat["y_true_centroid"],
                    cent_cat["y_pred_centroid"],
                    color=color,
                    label=f"{cat} Centroids",
                    **self.centroid_scatter_kwargs,
                )

            # regression line
            x = df["y_true"].to_numpy() if fit_on == "raw" else cent_cat["y_true_centroid"].to_numpy()
            y = df["y_pred"].to_numpy() if fit_on == "raw" else cent_cat["y_pred_centroid"].to_numpy()
            r2 = r2_score(x, y)

            model = self.fit_line(x, y, self.regression_func)
            x_line: np.ndarray = np.linspace(x.min(), x.max())
            y_line = model.predict(x_line.reshape(-1, 1))
            ax_curr.plot(
                x_line,
                y_line,
                linewidth=2,
                color=color,
                label=f"{self.regression_func} best fit. r2={r2:.3f}",
                zorder=10,
            )

            model_dict[(cat, self.seed)] = model

            # percentiles
            for perc in perc_symmetric:
                col = f"y_pred_perc_{perc}"
                ax_curr.plot(
                    cent_cat["y_true_centroid"], cent_cat[col], linestyle="--", linewidth=1, color=color, label=col
                )

            ax_curr.set_title(f"Group: {cat}")
            ax_curr.set_xlabel("True Values")
            ax_curr.set_ylabel("Predicted Values")
            ax_curr.legend()

        df = self.DfFull
        for ax_curr in ax if not isinstance(ax, Axes) else [ax]:
            if "identity" in addtl_plots:
                lo = min(df["y_true"].min(), df["y_pred"].min())
                hi = max(df["y_true"].max(), df["y_pred"].max())
                ax_curr.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray", zorder=2, label="Identity")
            if "y_0" in addtl_plots:
                ax_curr.axhline(0, linestyle="--", linewidth=1, color="gray", zorder=2, label="y=0")
        self.model_dict = model_dict

        if isinstance(ax, Axes):
            """
            Default single Axes case
            """
            ax.set_title(f"Group Scatter. Groups: {', '.join(map(str, categories))}")
        return fig

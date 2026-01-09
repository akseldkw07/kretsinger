from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd

from kret_np_pd.filters import FILT_TYPE, FilterSampleUtils
from kret_rosetta.UTILS_rosetta import UTILS_rosetta
from kret_rosetta.to_pd_np import TO_NP_TYPE

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


class AttrBase:
    _df_input: StateDF
    _df_sorted: pd.DataFrame
    _df_enriched: pd.DataFrame
    downsample: int
    seed: int

    # y_true: np.ndarray
    # y_pred: np.ndarray
    # category: np.ndarray
    n_centroids: int
    regression_func: REG_FUNC


class FilterCalcMixin(AttrBase):
    def calc_filter_mask(self):
        filt = self._df_input.filt.to_numpy()
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
    def DfBaseSorted(self) -> pd.DataFrame:
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
    def DfFull(self) -> pd.DataFrame:
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

        self._df_enriched = df


# ==============================================================================================
# Main Class
# ==============================================================================================
class GroupScatter(DataFrameMixin):
    """
    Plot a group scatter with regression line. Helpful for visualizing model performance, especially when there are thousands or millions of points.

    Take in y and y_hat, optional categorical column, optional filter, # centroids=25, downsample=False, and regression funcion (OLS, Huber, etc)=OLS
    """

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
            self.category = pd.Categorical(np.zeros_like(y_true))
        elif isinstance(category, pd.Categorical):
            self.category = category
        else:
            arr = UTILS_rosetta.coerce_to_ndarray(category, assert_1dim=True, attempt_flatten_1d=True)
            self.category = pd.Categorical(arr)

        _filter_mask_input = FilterSampleUtils.process_filter(filter, shape=y_true.shape[0])
        self._df_input = StateDF(
            {"y_true": y_true, "y_pred": y_pred, "category": self.category, "filt": _filter_mask_input}
        )

        self.n_centroids = n_centroids
        self.downsample = downsample if downsample is not None else len(y)
        self.regression_func = regression_func

        self.seed = 0

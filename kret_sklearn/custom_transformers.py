"""
TODO FunctionTransformer? Seems like a good experimental first step
    > clip outliers?
    > lagged features? (probably fillna)
    > Label vs OneHotEncoder?
    > how does featureunion work
"""

import typing as t

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, LogisticRegression

from kret_np_pd.np_bool_utils import AnyAll, IndexLabel
from kret_np_pd.UTILS_np_pd1 import NP_PD_Utils

from ._core.typed_cls_sklean import (
    ElasticNet_Params_TypedDict,
    HuberRegressor_Params_TypedDict,
    LinearRegression_Params_TypedDict,
    LogisticRegression_Params_TypedDict,
)


class PandasColumnOrderBase(BaseEstimator, TransformerMixin):
    """
    A base class for transformers that preserve the column order of pandas DataFrames.
    """

    feature_names_in_: np.ndarray
    new_columns: list[str]  # NOTE To be set by subclasses during fit/transform

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None) -> "PandasColumnOrderBase":
        # Store input feature names for later use
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self._fit(X, y)
        return self

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None):
        """Actual fitting logic to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the _fit method.")

    def transform(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        """
        Call subclass-specific transformation logic and reorder columns to match new_columns
        """
        err_msg = f"Input columns do not match those seen during fit. {X.columns=}, {self.feature_names_in_=}"
        assert set(X.columns) == set(self.feature_names_in_), err_msg

        X = X.copy()

        # Call the actual transformation logic
        X_transformed = self._transform(X, y)

        # Reorder columns to match original
        X_transformed = X_transformed[self.new_columns]

        return X_transformed

    def _transform(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        """Actual transformation logic to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the _transform method.")

    def get_feature_names_out(self, *args, **kwargs):
        # TODO return np.array(self.new_columns, dtype=object) if config set that way
        return self.new_columns

    def get_feature_names_out_list(self, *args, **kwargs):
        return self.new_columns


DEF_DATETIME_COLS = {"month": 12, "day": 31, "dayofweek": 7, "hour": 24, "minute": 60}


REG_FUNC = t.Literal["OLS", "Huber", "ElasticNet", "Logistic"]
INIT_PARAM_DICT = (
    LinearRegression_Params_TypedDict
    | HuberRegressor_Params_TypedDict
    | ElasticNet_Params_TypedDict
    | LogisticRegression_Params_TypedDict
)


class RegressionResidualAdder(PandasColumnOrderBase):
    """
    Add residuals from a linear regression model as new features to the DataFrame.
    Maintains the original column order - appends residual columns at the end.
    """

    model: HuberRegressor | LinearRegression | LogisticRegression | ElasticNet

    def __init__(self, regression: REG_FUNC, args: INIT_PARAM_DICT) -> None:
        """
        Args:
            target_col: Name of the target column in the DataFrame
            model: A fitted linear regression model with predict method
        """
        match regression:
            case "OLS":
                self.model = LinearRegression(**args)  # type: ignore[args]
            case "Huber":
                self.model = HuberRegressor(**args)  # type: ignore[args]
            case "ElasticNet":
                self.model = ElasticNet(**args)  # type: ignore[args]
            case "Logistic":
                self.model = LogisticRegression(**args)  # type: ignore[args]

    def _fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None
    ) -> "RegressionResidualAdder":
        assert y is not None, "Must pass y to perform fit"
        self.model.fit(X, y)
        return self

    def _transform(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        """
        Add residuals and predictions to the DataFrame as new columns.
        """

        predictions = self.model.predict(X)
        X["y_hat"] = predictions

        self.new_columns = list(X.columns)

        return X


class DateTimeSinCosNormalizer(PandasColumnOrderBase):
    """
    Normalize datetime features (e.g., month, day, hour) using sine and cosine transformations.
    Maintains the original column order - replaces each datetime column with its sin/cos versions in place.
    """

    datetime_cols: dict[str, int]

    def __init__(self, datetime_cols: dict[str, int] | None) -> None:
        """
        Args:
            datetime_cols: Dictionary mapping column names to their periodicity (e.g., {'month': 12, 'hour': 24})
        """
        self.datetime_cols = datetime_cols if datetime_cols is not None else DEF_DATETIME_COLS

    def _fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None
    ) -> "DateTimeSinCosNormalizer":

        return self

    def _transform(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        """Transform datetime columns using sine and cosine transformations."""
        error_msg = f"Input columns must contain all datetime columns specified during initialization. {X.columns=}, {self.datetime_cols.keys()=}"
        assert set(self.datetime_cols.keys()).issubset(set(X.columns)), error_msg

        new_columns = []
        for col in X.columns:
            if col in self.datetime_cols:
                period = self.datetime_cols[col]
                X[f"{col}_sin"] = np.sin(2 * np.pi * X[col] / period)
                X[f"{col}_cos"] = np.cos(2 * np.pi * X[col] / period)
                new_columns.extend([f"{col}_sin", f"{col}_cos"])
            else:
                new_columns.append(col)

        self.new_columns = new_columns

        return X


class MissingValueRemover(PandasColumnOrderBase):
    """
    Remove rows with missing values (NaNs) from the DataFrame based on the specified criteria.

    NOTE WARNING this transformer removes rows, which can lead to misalignment between X and y if not handled carefully.
    """

    how: AnyAll
    subset: IndexLabel | None

    def __init__(self, how: AnyAll = "any", subset: IndexLabel | None = None) -> None:
        super().__init__()
        self.how = how
        self.subset = subset

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None) -> "MissingValueRemover":
        return self

    def _transform(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        # Remove rows with any NaN values in X
        self._orig_columns = list(X.columns)

        nan_mask = NP_PD_Utils.nan_filter(X[self.subset] if self.subset is not None else X, y, how=self.how)
        ret = X.loc[~nan_mask]  # NOTE KEEP INDEX ON PURPOSE .reset_index(drop=True)
        if y is not None:
            y = y[~nan_mask] if isinstance(y, np.ndarray) else y.loc[~nan_mask].reset_index(drop=True)
            error_msg = f"Y and x must have same number of rows after dropping NaNs: {len(ret)} vs {len(y)}"
            assert len(ret) == len(y), error_msg

        error_msg = f"Column count mismatch after dropping NaNs: {len(ret.columns)} vs {len(X.columns)} vs {len(self._orig_columns)}"
        assert len(ret.columns) == len(X.columns) == len(self._orig_columns), error_msg
        self.new_columns = list(X.columns)
        print(f"Removed {nan_mask.sum()} rows, representing {nan_mask.mean():.2%} of the data")

        return ret

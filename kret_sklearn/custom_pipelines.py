"""
TODO FunctionTransformer? Seems like a good experimental first step
    > clip outliers?
    > lagged features? (probably fillna)
    > Label vs OneHotEncoder?
    > how does featureunion work
"""

from __future__ import annotations
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np
import typing as t


class PandasColumnOrderBase(BaseEstimator, TransformerMixin):
    """
    A base class for transformers that preserve the column order of pandas DataFrames.
    """

    feature_names_in_: np.ndarray
    new_columns: list[str]

    def fit(self, X: pd.DataFrame, y: t.Any = None) -> PandasColumnOrderBase:
        # Store input feature names for later use
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self._fit(X, y)
        return self

    def _fit(self, X: pd.DataFrame, y: t.Any = None):
        """Actual fitting logic to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the _fit method.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform while preserving column order"""
        err_msg = f"Input columns do not match those seen during fit. {X.columns=}, {self.feature_names_in_=}"
        assert set(X.columns) == set(self.feature_names_in_), err_msg

        X = X.copy()

        # Call the actual transformation logic
        X_transformed = self._transform(X)

        # Reorder columns to match original
        X_transformed = X_transformed[self.feature_names_in_]

        return X_transformed

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Actual transformation logic to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the _transform method.")

    def get_feature_names_out(self, **kwargs):
        return np.array(self.new_columns, dtype=object)


AnyAll = t.Literal["any", "all"]
IndexLabel = t.Hashable | t.Sequence[t.Hashable]


class MissingValueRemover(PandasColumnOrderBase):
    """Remove rows with missing values from X only."""

    how: AnyAll
    subset: IndexLabel | None

    def __init__(self, how: AnyAll = "any", subset: IndexLabel | None = None) -> None:
        super().__init__()
        self.how = how
        self.subset = subset

    def _fit(self, X: pd.DataFrame, y: t.Any = None) -> MissingValueRemover:
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Remove rows with any NaN values in X
        return X.dropna(how=self.how, subset=self.subset, ignore_index=True)  # pyright: ignore[reportArgumentType]

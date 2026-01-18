"""
Protocol classes for pandas conversion.

Protocols define a structural interface - any class that implements
the required methods is considered to satisfy the protocol, without
explicit inheritance.
"""

import typing as t
from typing import Protocol, runtime_checkable

import pandas as pd

# ============================================================================
# PROTOCOL APPROACH (Recommended for typing/duck typing)
# ============================================================================

T = t.TypeVar("T", bound="PandasConvertibleWithColumns")


@runtime_checkable
class PandasConvertibleWithColumns(Protocol):
    """Protocol for objects with column information and pandas conversion."""

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        ...

    def to_pandas(self, copy: bool) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        ...

    @staticmethod
    def from_pd(df: pd.DataFrame, **kwargs) -> "PandasConvertibleWithColumns":
        """Create from pandas DataFrame."""
        ...


@runtime_checkable
class ImplementsToPandas(Protocol):
    """Protocol for objects that can convert to pandas DataFrame."""

    def to_pandas(self, copy: bool) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        ...

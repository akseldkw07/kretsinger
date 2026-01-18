import typing as t

import numpy as np
import pandas as pd

from kret_rosetta.to_pd_np import To_NP_PD

from .typed_cls_np_pd import DataFrame___init___TypedDict


class InputTypedDict(t.TypedDict):
    """
    Base class for input param.

    NOTE `data` is required - will get passed onto `pd.DataFrame.__init__`
    """

    data: pd.DataFrame


# Type variables for generic MemoDataFrame and memo_array
T = t.TypeVar("T", bound="InputTypedDict")
MDF = t.TypeVar("MDF", bound="MemoDataFrame[t.Any]")


class MemoDataFrame(pd.DataFrame, t.Generic[T]):
    """
    Subclass of pd.DataFrame that implements memoized np.ndarrays

    @memo_array behaves similarly to @cached_property, but
    1. registers objects to self._memo_dict
    2. When MemoDataFrame is viewed, the calculated memo arrays (but NOT the uncalculated arrays) are displayed,
    as if they were normal columns
    """

    _metadata = ["_inputs", "_memo_dict"]
    _inputs: T
    _memo_dict: dict[str, np.ndarray]

    def __init__(self, input: T, /, **kwargs: t.Unpack[DataFrame___init___TypedDict]) -> None:
        object.__setattr__(self, "_inputs", input)
        object.__setattr__(self, "_memo_dict", {})
        super().__init__(input["data"], **kwargs)  # type: ignore[arg-type]

    @property
    def inputs(self) -> T:
        return self._inputs

    @property
    def data(self):
        return self.inputs["data"]

    @property
    def _constructor(self):
        return type(self)

    def clear(self):
        """Clears the memoization cache."""
        self._memo_dict.clear()

    def _repr_html_(self):
        """Custom HTML representation that includes memoized arrays as columns."""
        display_df = self.to_pandas(copy=False)

        return display_df._repr_html_()  # type: ignore

    def __repr__(self):
        """Custom string representation that includes memoized arrays as columns."""
        display_df = self.to_pandas(copy=False)

        return display_df.__repr__()

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        """Return self as a pandas DataFrame."""
        input = pd.DataFrame(self, copy=copy)
        memo = pd.DataFrame(self._memo_dict, copy=copy)
        return pd.concat([input, memo], axis=1)


class memo_array(t.Generic[MDF]):
    """Descriptor similar to cached_property, but caches np.ndarrays in MemoDataFrame.

    Works with subclasses of MemoDataFrame by being generic over the instance type.
    Accepts pd.Series or np.ndarray as input type, coerces to np.ndarray.
    """

    def __init__(self, func: t.Callable[[MDF], np.ndarray | pd.Series]) -> None:
        self.func = func
        self.name = func.__name__

    def __get__(self, instance: MDF | None, owner: type[MDF]) -> "np.ndarray | memo_array[MDF]":
        if instance is None:
            return self
        if self.name not in instance._memo_dict:
            result = self.func(instance)
            instance._memo_dict[self.name] = To_NP_PD.coerce_to_ndarray(
                result, assert_1dim=True, attempt_flatten_1d=True
            )
        return instance._memo_dict[self.name]

    def __set__(self, instance: MDF, value: np.ndarray) -> None:
        instance._memo_dict[self.name] = value

    def __delete__(self, instance: MDF) -> None:
        if self.name in instance._memo_dict:
            del instance._memo_dict[self.name]

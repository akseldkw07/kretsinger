import typing as t

import numpy as np
import pandas as pd

from kret_rosetta.to_pd_np import To_NP_PD

from .typed_cls_np_pd import DataFrame___init___TypedDict

# Type variables for generic MemoDataFrame and memo_array
T = t.TypeVar("T", bound="InputTypedDict")
MDF = t.TypeVar("MDF", bound="MemoDataFrame[t.Any]")


class InputTypedDict(t.TypedDict):
    """
    Base class for input param. General pattern is for the "primary" dataset to get stored in `"data"`,
    and supplmenetal datasets under different keys. NOTE they don't need to be the same len as `"data"`

    NOTE `data` is required - will get passed onto `pd.DataFrame.__init__`
    """

    data: pd.DataFrame


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
        return t.cast(t.Callable, display_df._repr_html_)()

    def __repr__(self):
        """Custom string representation that includes memoized arrays as columns."""
        display_df = self.to_pandas(copy=False)
        return display_df.__repr__()

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        """Return self as a pandas DataFrame."""
        input = pd.DataFrame(self, copy=copy)
        memo = pd.DataFrame(self._memo_dict, copy=copy)
        memo.insert(0, "SEP", "...")
        return pd.concat([input, memo], axis=1, copy=False)  # no need to copy again


class memo_array(t.Generic[MDF]):
    """Descriptor similar to cached_property, but caches np.ndarrays in MemoDataFrame.

    Works with subclasses of MemoDataFrame by being generic over the instance type.
    Accepts pd.Series or np.ndarray as input type, coerces to np.ndarray.
    TODO fix type-hinting to always return np.ndarray
    """

    def __init__(self, func: t.Callable[[MDF], np.ndarray | pd.Series]) -> None:
        self.func: t.Callable[[MDF], np.ndarray | pd.Series] = func
        self.name = func.__name__
        # Copy over function metadata for better type checking
        self.__doc__ = func.__doc__

    def __get__(self, instance: MDF | None, owner: type[MDF]) -> np.ndarray:
        if instance is None:
            return self  # type: ignore
        if self.name not in instance._memo_dict:
            print(f"Calculating {self.name}")
            result = self.func(instance)
            instance._memo_dict[self.name] = To_NP_PD.coerce_to_ndarray(
                result, assert_1dim=True, attempt_flatten_1d=True
            )
        ret = instance._memo_dict[self.name]
        return ret

    def __delete__(self, instance: MDF) -> None:
        if self.name in instance._memo_dict:
            del instance._memo_dict[self.name]


# ===============================================================
# Test class
# ===============================================================


class MyInputDict(InputTypedDict):
    aux: pd.DataFrame


class MyMemoDataFrame(MemoDataFrame[MyInputDict]):
    @memo_array
    def a_sq(self):
        print("Calculating a_sq")
        return self.data["a"] ** 2

    @memo_array
    def compute_sum(self):
        return self.data.sum(axis=1) + self.inputs["aux"].sum(axis=1)

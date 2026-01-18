import typing as t

import pandas as pd
import numpy as np
from .typed_cls_np_pd import DataFrame___init___TypedDict


class InputTypedDict(t.TypedDict):
    """
    Base class for input param.

    NOTE `data` is required - will get passed onto `pd.DataFrame.__init__`
    """

    data: pd.DataFrame


# Type variables for generic MemoDataFrame and memo_array
T = t.TypeVar("T", bound=InputTypedDict)
InputT = t.TypeVar("InputT", bound=InputTypedDict)
MDF = t.TypeVar("MDF", bound="MemoDataFrame[t.Any]")


class MemoDataFrame(pd.DataFrame, t.Generic[InputT]):
    """
    Subclass of pd.DataFrame that implements memoized np.ndarrays

    @memo_array behaves similarly to @cached_property, but
    1. registers objects to self._memo_dict
    2. When MemoDataFrame is viewed, the calculated memo arrays (but NOT the uncalculated arrays) are displayed,
    as if they were normal columns


    TODO    1) handle clearing out the cache
            2) validate that displaying looks good
    """

    _metadata = ["_inputs", "_memo_dict"]
    _inputs: InputT
    _memo_dict: dict[str, np.ndarray]

    def __init__(self, input: InputT, /, **kwargs: t.Unpack[DataFrame___init___TypedDict]) -> None:
        object.__setattr__(self, "_inputs", input)
        object.__setattr__(self, "_memo_dict", {})
        super().__init__(input["data"], **kwargs)  # type: ignore[arg-type]

    @property
    def inputs(self) -> InputT:
        return self._inputs

    @property
    def data(self):
        return self.inputs["data"]

    @property
    def _constructor(self):
        return type(self)


class memo_array(t.Generic[MDF]):
    """Descriptor similar to cached_property, but caches np.ndarrays in MemoDataFrame.

    Works with subclasses of MemoDataFrame by being generic over the instance type.
    """

    def __init__(self, func: t.Callable[[MDF], np.ndarray]) -> None:
        self.func = func
        self.name = func.__name__

    def __get__(self, instance: MDF | None, owner: type[MDF]) -> "np.ndarray | memo_array[MDF]":
        if instance is None:
            return self
        if self.name not in instance._memo_dict:
            instance._memo_dict[self.name] = self.func(instance)
        return instance._memo_dict[self.name]

    def __set__(self, instance: MDF, value: np.ndarray) -> None:
        instance._memo_dict[self.name] = value

    def __delete__(self, instance: MDF) -> None:
        if self.name in instance._memo_dict:
            del instance._memo_dict[self.name]

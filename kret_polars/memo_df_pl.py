import typing as t

import polars as pl

# Type variables for generic MemoDataFramePL and memo_series
T = t.TypeVar("T", bound="InputTypedDictPL")
MDF = t.TypeVar("MDF", bound="MemoDataFramePL[t.Any]")
TData = t.TypeVar("TData", bound=pl.DataFrame)


class InputTypedDictPL(t.TypedDict, t.Generic[TData]):
    """
    Base class for input param. General pattern is for the "primary" dataset to get stored in `"data"`,
    and supplemental datasets under different keys. NOTE they don't need to be the same len as `"data"`

    NOTE `data` is required - will get passed onto `pl.DataFrame.__init__`
    """

    data: TData


def _coerce_to_series(result: "pl.Series | pl.Expr | list | tuple", *, name: str) -> pl.Series:
    """Coerce a memo result to a pl.Series."""
    if isinstance(result, pl.Series):
        return result.alias(name)
    if isinstance(result, (list, tuple)):
        return pl.Series(name=name, values=result)
    raise TypeError(f"memo_series expected pl.Series | list | tuple, got {type(result).__name__}")


class MemoDataFramePL(pl.DataFrame, t.Generic[T]):
    """
    Subclass of pl.DataFrame that implements memoized pl.Series.

    @memo_series behaves similarly to @cached_property, but
    1. registers objects to self._memo_dict
    2. When MemoDataFramePL is displayed, the calculated memo series (but NOT the uncalculated ones) are shown,
    as if they were normal columns

    NOTE: Polars does not support _metadata propagation like pandas. Any Polars operation
    that returns a new DataFrame (filter, select, join, etc.) will return a plain pl.DataFrame,
    losing memo state. This is designed for "construct once, compute memos, inspect" workflows.
    """

    _inputs: T
    _memo_dict: dict[str, pl.Series]
    _df_dict: dict[str, pl.DataFrame]

    def __init__(self, input: T, /) -> None:
        object.__setattr__(self, "_inputs", input)
        object.__setattr__(self, "_memo_dict", {})
        object.__setattr__(self, "_df_dict", {})
        super().__init__(input["data"])

    @property
    def inputs(self) -> T:
        return self._inputs

    @property
    def data(self):
        return self.inputs["data"]

    def clear_memo(self):
        """Clears the memoization cache."""
        self._memo_dict.clear()
        self._df_dict.clear()

    def to_polars_display(self) -> pl.DataFrame:
        """Return self with memoized series concatenated as columns."""
        base = pl.DataFrame._from_pydf(self._df)
        if not self._memo_dict:
            return base
        sep = pl.Series("SEP", ["..."] * self.height)
        memo = pl.DataFrame([s.alias(k) for k, s in self._memo_dict.items()])
        return pl.concat([base, sep.to_frame(), memo], how="horizontal")

    def to_polars_memo(self) -> pl.DataFrame:
        """Return only the memoized series as a pl.DataFrame."""
        if not self._memo_dict:
            return pl.DataFrame()
        return pl.DataFrame([s.alias(k) for k, s in self._memo_dict.items()])


class memo_fn(t.Generic[MDF]):
    """
    Like ``memo_series`` but for methods that accept arguments.

    The result is cached in ``_memo_dict`` under a key derived from the method
    name and the call arguments, so each unique ``(name, args, kwargs)``
    combination is computed once and reused. No coercion is applied - the
    return value is stored as-is.

    Cleared by ``MemoDataFramePL.clear_memo()`` just like ``memo_series`` results.
    """

    def __init__(self, func: t.Callable) -> None:
        self.func = func
        self.name = func.__name__
        self.__doc__ = func.__doc__

    @t.overload
    def __get__(self, instance: None, owner: type[MDF]) -> "memo_fn[MDF]": ...
    @t.overload
    def __get__(self, instance: MDF, owner: type[MDF]) -> t.Callable[..., pl.Series]: ...

    def __get__(self, instance: MDF | None, owner: type[MDF]) -> "memo_fn[MDF] | t.Callable[..., t.Any]":
        if instance is None:
            return self

        def wrapper(*args, **kwargs):
            key = f"{self.name}__{args}" + (f"__{tuple(sorted(kwargs.items()))}" if kwargs else "")
            if key not in instance._memo_dict:
                print(f"Calculating {self.name}{args or ''}")
                instance._memo_dict[key] = self.func(instance, *args, **kwargs)
            return instance._memo_dict[key]

        return wrapper


class memo_series(t.Generic[MDF]):
    """Descriptor similar to cached_property, but caches pl.Series in MemoDataFramePL.

    Works with subclasses of MemoDataFramePL by being generic over the instance type.
    Accepts pl.Series, list, or tuple as return type, coerces to pl.Series.
    """

    def __init__(self, func: t.Callable[[MDF], pl.Series | list | tuple]) -> None:
        self.func: t.Callable[[MDF], pl.Series | list | tuple] = func
        self.name = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, instance: MDF | None, owner: type[MDF]) -> pl.Series:
        if instance is None:
            return self  # type: ignore
        if self.name not in instance._memo_dict:
            print(f"Calculating {self.name}")
            result = self.func(instance)
            instance._memo_dict[self.name] = _coerce_to_series(result, name=self.name)
        return instance._memo_dict[self.name]

    def __delete__(self, instance: MDF) -> None:
        if self.name in instance._memo_dict:
            del instance._memo_dict[self.name]

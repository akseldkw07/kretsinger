from __future__ import annotations

import typing as t

import numpy as np

NPT = t.TypeVar("NPT", bound=t.Any)
T = t.TypeVar("T")


class SingleReturnArray(np.ndarray, t.Generic[NPT]):
    """
    A subclass of np.ndarray that returns a single value when indexed with a single index.
    """

    @t.overload
    def __getitem__(self: SingleReturnArray[T], key: int) -> T:  # type: ignore
        return super().__getitem__(key)

    @t.overload
    def __getitem__(self: SingleReturnArray[SingleReturnArray[T]], key: tuple[int, int]) -> T:  # type: ignore
        return super().__getitem__(key)

    def __getitem__(self, key):  # type: ignore
        return super().__getitem__(key)

    @t.overload
    def __iter__(self: SingleReturnArray[T]) -> t.Iterator[T]:
        return super().__iter__()

    @t.overload
    def __iter__(self: SingleReturnArray[SingleReturnArray[T]]) -> t.Iterator[SingleReturnArray[T]]:  # type: ignore
        return super().__iter__()

    def __iter__(self):  # type: ignore
        return super().__iter__()

    @t.overload
    def ravel(self: SingleReturnArray[SingleReturnArray[SingleReturnArray[T]]]) -> SingleReturnArray[T]: ...
    @t.overload
    def ravel(self: SingleReturnArray[SingleReturnArray[T]]) -> SingleReturnArray[T]: ...
    @t.overload
    def ravel(self: SingleReturnArray[T]) -> SingleReturnArray[T]: ...

    def ravel(self) -> SingleReturnArray[T]:  # type: ignore
        return super().ravel()  # type: ignore

from __future__ import annotations

import typing as t
import weakref

import numpy as np
import pandas as pd

T = t.TypeVar("T", bound=pd.DataFrame)


class MemoNDArray(np.ndarray):
    """NumPy ndarray subclass that tracks its originating MemoDataFrame."""

    def __new__(cls, input_array: t.Any, owner: MemoDataFrame | None = None, **kwargs: t.Any) -> MemoNDArray:
        obj = np.asarray(input_array, **kwargs).view(cls)
        obj._owner_ref = weakref.ref(owner) if owner is not None else None
        return obj

    def __array_finalize__(self, obj: t.Optional[np.ndarray]) -> None:
        # NumPy calls this when creating views/copies; carry the owner reference forward.
        self._owner_ref = getattr(obj, "_owner_ref", None)

    @property
    def owner(self) -> MemoDataFrame | None:
        """Return the originating MemoDataFrame, if still alive."""
        if getattr(self, "_owner_ref", None) is None:
            return None
        return self._owner_ref()


class MemoDataFrame(pd.DataFrame):
    """DataFrame subclass that lazily exposes a paired MemoNDArray view."""

    _metadata = ["_memo_array_cache"]

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self._memo_array_cache: MemoNDArray | None = None

    @property
    def _constructor(self) -> t.Type[MemoDataFrame]:  # type: ignore[override]
        return MemoDataFrame

    @property
    def memo_array(self) -> MemoNDArray:
        """Lazily materialize and cache an ndarray view of the frame."""
        if self._memo_array_cache is None:
            # to_numpy(copy=False) keeps sharing data when possible, so the memo array stays in sync.
            base = self.to_numpy(copy=False)
            self._memo_array_cache = MemoNDArray(base, owner=self)
        return self._memo_array_cache

    def invalidate_memo_array(self) -> None:
        """Explicitly drop the cached MemoNDArray, forcing regeneration on next access."""
        self._memo_array_cache = None

    # -- pandas hooks -----------------------------------------------------
    def __finalize__(self, other: t.Any, method: t.Optional[str] = None, **kwargs: t.Any) -> MemoDataFrame:
        result = super().__finalize__(other, method=method, **kwargs)
        self._memo_array_cache = None
        return result

    def __setitem__(self, key: t.Any, value: t.Any) -> None:
        self.invalidate_memo_array()
        super().__setitem__(key, value)

    def assign(self, *args: t.Any, **kwargs: t.Any) -> MemoDataFrame:
        result = super().assign(*args, **kwargs)
        result.invalidate_memo_array()
        return result

    def drop(self, *args: t.Any, **kwargs: t.Any) -> MemoDataFrame | None:
        inplace = kwargs.get("inplace", False)
        result = super().drop(*args, **kwargs)
        if inplace:
            self.invalidate_memo_array()
            return None
        if hasattr(result, "invalidate_memo_array"):
            result.invalidate_memo_array()
        return t.cast(MemoDataFrame, result)

    def pop(self, item: t.Any) -> t.Any:
        self.invalidate_memo_array()
        return super().pop(item)

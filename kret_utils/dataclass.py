from __future__ import annotations

import typing as t
from dataclasses import fields, is_dataclass, replace


class ResettableDataclassMixin:
    """
    Mixin for @dataclass config-ish objects.

    Captures the initial field values once, so reset() can restore them later.
    Works for frozen dataclasses too (uses object.__setattr__).

    If a subclass defines __post_init__, it should call super().__post_init__().
    """

    _defaults_snapshot: dict[str, t.Any] | None = None

    def __post_init__(self) -> None:
        if self._defaults_snapshot is None:
            if not is_dataclass(self):
                raise TypeError("ResettableDataclass must be used with @dataclass classes")
            object.__setattr__(
                self,
                "_defaults_snapshot",
                {f.name: getattr(self, f.name) for f in fields(self)},
            )

    def reset(self) -> None:
        if self._defaults_snapshot is None:
            self.__post_init__()
        assert self._defaults_snapshot is not None
        for name, value in self._defaults_snapshot.items():
            object.__setattr__(self, name, value)

    def update(self, /, **kwargs: t.Any) -> None:
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

    def replaced(self, /, **kwargs: t.Any):
        assert is_dataclass(self), "ResettableDataclass.replaced() requires a dataclass instance"
        return replace(self, **kwargs)

    def as_dict(self) -> dict[str, t.Any]:
        if not is_dataclass(self):
            raise TypeError("ResettableDataclass.as_dict() requires a dataclass instance")
        return {f.name: getattr(self, f.name) for f in fields(self)}

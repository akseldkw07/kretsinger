from typing import TypeVar, Generic
from collections.abc import Callable

T = TypeVar("T")


class classproperty(Generic[T]):
    """Descriptor that acts like a combined @classmethod + @property.

    Works on Python 3.9-3.13+ (the built-in stacked ``@classmethod @property``
    was deprecated in 3.11 and removed in 3.13).

    Usage::

        class Foo:
            @classproperty
            def bar(cls) -> int:
                return 42
    """

    def __init__(self, func: Callable[..., T]) -> None:
        self.fget = func

    def __get__(self, obj: object, owner: type | None = None) -> T:
        if owner is None:
            owner = type(obj)
        return self.fget(owner)

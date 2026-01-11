from __future__ import annotations

import functools
import typing as t

from kret_decorators.protocols import DefinesPostInit

F = t.TypeVar("F", bound=t.Callable[..., t.Any])
T = t.TypeVar("T", bound=DefinesPostInit)


def post_init(cls: type[T], *, method_name: str = "__post_init__") -> type[T]:
    """Class decorator that calls a post-init hook after `__init__`.

    This is intentionally "dataclass-like": if the hook method exists, it is called
    *after* the original `__init__` completes.

    Parameters
    ----------
    cls:
        The class whose `__init__` should be wrapped.
    method_name:
        Name of the post-init hook to call (defaults to `__post_init__`).

    Notes
    -----
    - Avoids double-wrapping the same class by tagging the wrapper.
    - Preserves `__init__` metadata via `functools.wraps`.
    """

    # Nothing to do if there is no __init__ to wrap (very uncommon).
    original_init = getattr(cls, "__init__", None)
    if original_init is None:
        return cls

    # Avoid double-wrapping (e.g., if both a base and derived class are decorated,
    # or the decorator is applied twice).
    if getattr(original_init, "__kret_calls_post_init__", False):
        return cls

    @functools.wraps(original_init)
    def __init__(self: T, *args: t.Any, **kwargs: t.Any) -> None:
        original_init(self, *args, **kwargs)

        hook = getattr(self, method_name, None)
        if hook is None:
            raise AttributeError(f"{cls.__qualname__} instances have no '{method_name}' method to call after __init__.")

        if not callable(hook):
            raise TypeError(
                f"Expected '{method_name}' on {cls.__qualname__} instances to be callable; got {type(hook)!r}."
            )

        hook()

    # Tag the wrapper so we can detect already-wrapped __init__.
    setattr(__init__, "__kret_calls_post_init__", True)

    cls.__init__ = __init__
    return cls

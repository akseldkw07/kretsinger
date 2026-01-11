from __future__ import annotations

import functools
import typing as t

from kret_decorators.protocols import PostInitCompatible

F = t.TypeVar("F", bound=t.Callable[..., t.Any])
T = t.TypeVar("T", bound=PostInitCompatible)


def post_init(cls: type[T], *, method_name: str = "__post_init__") -> type[T]:
    """Class decorator that calls a post-init hook after *all* __init__ calls complete.

    Why this exists
    ---------------
    If you decorate a base class and call `hook()` at the end of the base `__init__`,
    the subclass may not have finished its own initialization yet.

    This implementation delays the hook until the *outermost* `__init__` in the
    inheritance chain completes, so `__post_init__` sees fully-initialized state.

    How it works
    ------------
    - Wraps `__init__` to maintain a per-instance init-depth counter.
    - Only the outermost `__init__` (depth 0 -> 1) triggers the hook.
    - Installs `__init_subclass__` so decorating a base class also wraps subclasses,
      allowing the depth tracking to work across `super()` chains.
    """

    INIT_DEPTH_ATTR = "__kret_init_depth__"
    RAN_ATTR = "__kret_post_init_ran__"
    WRAP_TAG = "__kret_calls_post_init__"

    def wrap_init(target_cls: type[t.Any]) -> None:
        original_init = getattr(target_cls, "__init__", None)
        if original_init is None:
            return

        # Avoid double-wrapping
        if getattr(original_init, WRAP_TAG, False):
            return

        @functools.wraps(original_init)
        def __init__(self: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
            depth_before = int(getattr(self, INIT_DEPTH_ATTR, 0))
            setattr(self, INIT_DEPTH_ATTR, depth_before + 1)
            try:
                original_init(self, *args, **kwargs)
            finally:
                # Always decrement depth, even if __init__ raises.
                depth_after = int(getattr(self, INIT_DEPTH_ATTR, 1)) - 1
                if depth_after <= 0:
                    # Clean up so we don't leave state on the instance.
                    try:
                        delattr(self, INIT_DEPTH_ATTR)
                    except AttributeError:
                        pass
                else:
                    setattr(self, INIT_DEPTH_ATTR, depth_after)

            # Only the *outermost* init call triggers post-init.
            if depth_before != 0:
                return

            # Run exactly once per instance.
            if getattr(self, RAN_ATTR, False):
                return
            setattr(self, RAN_ATTR, True)

            hook = getattr(self, method_name, None)
            if hook is None:
                raise AttributeError(
                    f"{target_cls.__qualname__} instances have no '{method_name}' method to call after __init__."
                )

            if not callable(hook):
                raise TypeError(
                    f"Expected '{method_name}' on {target_cls.__qualname__} instances to be callable; got {type(hook)!r}."
                )

            hook()

        # Tag the wrapper so we can detect already-wrapped __init__.
        setattr(__init__, WRAP_TAG, True)

        target_cls.__init__ = __init__  # type: ignore[method-assign]

    # Wrap the decorated class itself.
    wrap_init(cls)

    # Ensure subclasses are also wrapped so the outermost init is the one that
    # triggers post-init.
    prev_init_subclass = getattr(cls, "__init_subclass__", None)

    @classmethod
    def __init_subclass__(subcls: type[PostInitCompatible], **kwargs: t.Any) -> None:
        if prev_init_subclass is not None:
            prev_init_subclass(**kwargs)  # type: ignore[misc]
        else:
            super(cls, subcls).__init_subclass__(**kwargs)

        wrap_init(subcls)

    cls.__init_subclass__ = __init_subclass__  # type: ignore[assignment]

    return cls


def post_init_no_inheritance(cls: type[T], *, method_name: str = "__post_init__") -> type[T]:
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

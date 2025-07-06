import typing as t
from typing import TypeVar, Type, cast

T = TypeVar("T")


def assert_type(cls: type[T], obj: t.Any) -> T:
    """
    Assert that obj is an instance of cls, and return obj as type cls (for type checkers).
    Usage: my_obj = assert_type(ExpectedClass, my_obj)
    """
    assert isinstance(obj, cls), f"Object {obj!r} is not of type {cls}"
    return cast(T, obj)

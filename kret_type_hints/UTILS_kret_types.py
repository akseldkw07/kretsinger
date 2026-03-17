import typing as t
from typing import TypeVar, cast

from kret_type_hints.typed_dict_utils import TypedDictUtils

from .func_2_typed_dict import FuncToTypedDict

T = TypeVar("T")


class KretTypeHints(FuncToTypedDict, TypedDictUtils):
    @classmethod
    def assert_type(cls, obj: t.Any, clstype: type[T]) -> T:
        """
        Assert that obj is an instance of clstype, and return obj as type clstype (for type checkers).
        Usage: my_obj = assert_type(ExpectedClass, my_obj)
        """
        assert isinstance(obj, clstype), f"Object {obj!r} is not of type {clstype}"
        return cast(T, obj)

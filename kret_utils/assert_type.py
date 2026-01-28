import typing as t
from types import UnionType

T = t.TypeVar("T")


class TypeAssert:
    @classmethod
    def assert_type(cls, var: t.Any, expected_type: type[T] | UnionType) -> T:
        if not isinstance(var, expected_type):
            raise TypeError(f"Expected type {expected_type}, but got type {type(var)}")
        return var

    @classmethod
    def assert_not_none(cls, var: t.Any):
        """
        TODO this function might be redundant since assert_type already checks for None
        """
        if var is None:
            raise TypeError(f"Expected non-None value, but got None")
        # if not isinstance(var, expected_type):
        #     raise TypeError(f"Expected type {expected_type}, but got type {type(var)}")
        return var

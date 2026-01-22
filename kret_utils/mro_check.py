import typing as t


class MROUtils:
    @classmethod
    def assert_mro_order(cls, obj, type_a: type, type_b: type, on_missing: t.Literal["raise", "ignore"] = "raise"):
        """
        Asserts that type_a comes before type_b in the
        Method Resolution Order (MRO) of the given object.
        """
        mro = type(obj).mro()

        # Ensure both types actually exist in the inheritance chain
        if on_missing == "ignore" and (type_a not in mro or type_b not in mro):
            return True

        assert type_a in mro, f"{type_a.__name__} not found in MRO of {type(obj).__name__}"
        assert type_b in mro, f"{type_b.__name__} not found in MRO of {type(obj).__name__}"

        index_a = mro.index(type_a)
        index_b = mro.index(type_b)

        assert index_a < index_b, (
            f"MRO Order Violation: {type_a.__name__} (index {index_a}) "
            f"must come before {type_b.__name__} (index {index_b}) "
            f"in {type(obj).__name__}'s MRO."
        )

        return True

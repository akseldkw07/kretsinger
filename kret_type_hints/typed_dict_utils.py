import typing as t
from typing import TypeVar

T = TypeVar("T", bound=t.Any)


class TypedDictUtils:
    """
    TypedDict Filtering Utility

    Provides a function to filter dictionary objects to only include keys
    that are declared in a TypedDict specification (including inherited keys).
    """

    @classmethod
    def filter_dict_by_typeddict(cls, data: dict | t.Any, typed_dict_class: type[T], strict: bool = False) -> T:
        """
        Filter a dictionary to only include keys specified in a TypedDict.

        Removes any keys not declared in the TypedDict or its parent TypedDicts.
        Automatically handles inheritance - includes keys from parent TypedDicts.

        Args:
            data: Dictionary to filter
            typed_dict_class: TypedDict class defining allowed keys
            strict: If True, raise KeyError if required keys are missing (default: False)

        Returns:
            Filtered dictionary with only TypedDict-specified keys

        Raises:
            TypeError: If typed_dict_class is not a TypedDict
            KeyError: If strict=True and required (non-optional) keys are missing
        """
        # Validate input
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        # Check if it's actually a TypedDict
        if not cls._is_typeddict(typed_dict_class):
            raise TypeError(
                f"{typed_dict_class} is not a TypedDict. "
                "TypedDict classes have __annotations__ and __total__ attributes."
            )

        # Get all allowed keys from TypedDict (including parent keys)
        allowed_keys = cls._get_typeddict_keys(typed_dict_class)
        required_keys = cls._get_required_keys(typed_dict_class) if strict else set()

        # Check for missing required keys if strict mode
        if strict:
            missing = required_keys - set(data.keys())
            if missing:
                raise KeyError(f"Missing required keys for {typed_dict_class.__name__}: {missing}")

        # Filter dictionary to only include allowed keys
        filtered = {k: v for k, v in data.items() if k in allowed_keys}

        return filtered  # type: ignore

    @classmethod
    def _is_typeddict(cls, obj: t.Any) -> bool:
        """Check if an object is a TypedDict class."""
        return (
            isinstance(obj, type)
            and issubclass(obj, dict)
            and hasattr(obj, "__annotations__")
            and hasattr(obj, "__total__")
            and hasattr(obj, "__required_keys__")
        )

    @classmethod
    def _get_typeddict_keys(cls, typed_dict_class: t.Type) -> set[str]:
        """Extract all keys from a TypedDict, including inherited keys.

        NOTE: We intentionally do NOT use `typing.get_type_hints()` here.
        `get_type_hints()` evaluates forward references (e.g. "Axis") and can fail
        at runtime when those names only exist under `TYPE_CHECKING`.

        We only need field *names*, and those are available via `__annotations__`
        without evaluating the annotation values.
        """
        if not cls._is_typeddict(typed_dict_class):
            raise TypeError(f"{typed_dict_class} is not a TypedDict")

        keys: set[str] = set()
        for base in reversed(getattr(typed_dict_class, "__mro__", ())):
            if cls._is_typeddict(base):
                ann = getattr(base, "__annotations__", None)
                if ann:
                    keys.update(ann.keys())
        return keys

    @classmethod
    def _get_required_keys(cls, typed_dict_class: t.Type) -> set[str]:
        """
        Extract keys that are required (not optional) in a TypedDict.

        A key is required if:
        - The TypedDict or its containing parent has total=True, OR
        - The key is explicitly in __required_keys__
        """
        # Use the __required_keys__ attribute if available
        if hasattr(typed_dict_class, "__required_keys__"):
            return set(typed_dict_class.__required_keys__)

        # Fallback: if total=True, all keys are required
        if getattr(typed_dict_class, "__total__", True):
            return cls._get_typeddict_keys(typed_dict_class)

        return set()


# Convenience aliases
drop_unknown_keys = TypedDictUtils.filter_dict_by_typeddict
validate_dict_keys = TypedDictUtils.filter_dict_by_typeddict
validate_required_keys = TypedDictUtils.filter_dict_by_typeddict

import inspect
import typing as t
from collections import defaultdict
from types import UnionType
from typing import Union, get_args, get_origin

from .typed_func_helper import TypedFuncHelper


class FuncToTypedDict(TypedFuncHelper):

    @classmethod
    def func_to_typed_dict(cls, func: t.Callable, include_defaults: bool = True, include_ret: bool = False):
        """
        1. Print required imports
        2. Print TypedDict class definition based on function annotations

        Args:
            func: The callable to convert
            include_ret: Include return type annotation
            include_defaults: Include default values as comments

        TODO define special cases where we want to be extra specific with imports
            1. E.g. instead of from pandas import DataFrame, do import pandas as pd; pd.DataFrame
                > this helps avoid name conflicts with polars DataFrame (for example)
        """
        cls.print_imports(func)
        print()  # Blank line between imports and class definition
        cls.print_typed_dict_from_callable(func, include_ret=include_ret, include_defaults=include_defaults)

    @classmethod
    def print_imports(cls, func: t.Callable):
        """
        Print necessary imports based on function annotations.

        Rules:
        - Always use python builtin types when possible. (E.g., use 'list' instead of 'typing.List', type1 | type2 instead of 'typing.Union[type1, type2]')
        - Extract true import statement from Optional[], Callable[], Union[], etc. We want to print copy-pastable code for Python 3.10+ (PEP 604).
        - Import from as highest level possible (e.g. `from pandas import DataFrame` instead of `from pandas.core.frame import DataFrame`).
        """

        sig = inspect.signature(func)
        imports = defaultdict(set[str])  # {module: set(names)}

        # Extract imports from parameters
        for param in sig.parameters.values():
            cls.extract_imports_from_annotation(imports, param.annotation)

        # Extract imports from return annotation
        cls.extract_imports_from_annotation(imports, sig.return_annotation)

        # Always import TypedDict from typing
        if imports.get("typing"):
            imports["typing"].add("TypedDict")
        else:
            imports["typing"] = {"TypedDict"}

        # Print imports in sorted order
        for module in sorted(imports.keys()):
            names = sorted(imports[module])
            for name in names:
                print(f"from {module} import {name}")

    @classmethod
    def print_typed_dict_from_callable(
        cls, callable: t.Callable, include_defaults: bool = True, include_ret: bool = False
    ):
        """
        Print a TypedDict definition from a callable's annotations.

        Args:
            callable: The callable to convert
            include_defaults: Include default values as comments
            include_ret: Include return type annotation
        """
        annotations = getattr(callable, "__annotations__", {})

        if not annotations:
            print(f"No annotations found on {callable}")
            return

        dict_name = cls.resolve_dict_name(callable)
        sig = inspect.signature(callable) if include_defaults else None

        print(f"class {dict_name}(TypedDict, total=False):")
        for param_name, annotation in annotations.items():
            if not include_ret and param_name == "return":
                continue
            formatted = cls.format_annotation(annotation)

            # Add default value as comment if requested
            if include_defaults and sig and param_name in sig.parameters:
                param = sig.parameters[param_name]
                if param.default is not inspect.Parameter.empty:
                    default_repr = cls._format_default_value(param.default)
                    print(f"    {param_name}: {formatted}  # = {default_repr}")
                else:
                    print(f"    {param_name}: {formatted}")
            else:
                print(f"    {param_name}: {formatted}")

        return annotations

    @classmethod
    def _format_default_value(cls, value) -> str:
        """Format a default value for display in a comment."""
        if isinstance(value, str):
            return repr(value)
        elif isinstance(value, (list, dict, tuple)):
            # For mutable defaults, show a summary
            if isinstance(value, (list, dict)) and len(str(value)) > 50:
                type_name = type(value).__name__
                return f"{type_name}(...)"
            return repr(value)
        elif value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            # For complex objects, just show the type
            return f"{type(value).__name__}(...)"

    @classmethod
    def format_annotation(cls, annotation) -> str:
        """Convert a type annotation to Python 3.10+ syntax (PEP 604)."""

        # Handle None type
        if annotation is type(None):
            return "None"

        # NEW - CORRECT
        if isinstance(annotation, (str, int, float, bool)):
            return repr(annotation)  # Handles all literal types

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Union types -> use | syntax
        if origin is Union or origin is UnionType:
            return " | ".join(cls.format_annotation(arg) for arg in args)

        # Handle Optional types -> convert to | None
        if origin is Union and type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return f"{cls.format_annotation(non_none_args[0])} | None"

        # Handle generic types like List, Dict, etc.
        if origin is not None:
            origin_name = getattr(origin, "__name__", str(origin))
            if args:
                formatted_args = ", ".join(cls.format_annotation(arg) for arg in args)
                return f"{origin_name}[{formatted_args}]"
            return origin_name

        # Handle regular types
        if hasattr(annotation, "__name__"):
            return annotation.__name__

        return str(annotation)

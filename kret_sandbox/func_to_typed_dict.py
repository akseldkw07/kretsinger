from __future__ import annotations

from collections import defaultdict
import inspect
import typing as t
from types import UnionType
from kret_type_hints.typed_func_helper import TypedFuncHelper

from typing import get_args, get_origin, Union


# Types from typing module that we convert to PEP 604 syntax (don't import these)
PEP604_REPLACEMENTS = {"Union", "Optional", "List", "Dict", "Set", "Tuple", "UnionType"}


class FuncToTypedDict(TypedFuncHelper):

    @classmethod
    def func_to_typed_dict(cls, func: t.Callable, include_ret: bool = False):
        """
        1. Print required imports
        2. Print TypedDict class definition based on function annotations
        """
        cls.print_imports(func)
        print()  # Blank line between imports and class definition
        cls.print_typed_dict_from_callable(func, include_ret=include_ret)

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

        def extract_imports_from_annotation(annotation):
            """Recursively extract import statements from an annotation."""
            if annotation is inspect.Parameter.empty or annotation is type(None):
                return

            # Handle string annotations
            if isinstance(annotation, str):
                return

            origin = get_origin(annotation)
            args = get_args(annotation)

            # Process the origin type
            if origin is not None:
                type_to_process = origin
            else:
                type_to_process = annotation

            # Skip builtin types
            if isinstance(type_to_process, type) and type_to_process.__module__ == "builtins":
                pass
            elif hasattr(type_to_process, "__module__") and hasattr(type_to_process, "__name__"):
                module_path = type_to_process.__module__
                name = type_to_process.__name__

                # Skip types we convert to PEP 604 syntax
                skip_typing = module_path == "typing" and name in PEP604_REPLACEMENTS
                skip_uniontype = module_path == "types" and name == "UnionType"
                if skip_typing or skip_uniontype:
                    pass
                else:
                    # Resolve to highest-level import
                    module, resolved_name = cls._resolve_import_location(type_to_process, name)

                    imports[module].add(resolved_name)

            # Recursively process generic arguments (but skip Literal string values)
            for arg in args:
                # Don't try to extract imports from literal values (strings, ints, etc)
                if not isinstance(arg, (str, int, float, bool, type(None))):
                    extract_imports_from_annotation(arg)

        # Extract imports from parameters
        for param in sig.parameters.values():
            extract_imports_from_annotation(param.annotation)

        # Extract imports from return annotation
        extract_imports_from_annotation(sig.return_annotation)

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
        cls, callable: t.Callable, dict_name: str | None = None, include_ret: bool = False
    ):
        """
        Print a TypedDict definition from a callable's annotations.
        """
        annotations = getattr(callable, "__annotations__", {})

        if not annotations:
            print(f"No annotations found on {callable}")
            return

        dict_name = dict_name or cls.resolve_dict_name(callable)

        print(f"class {dict_name}(TypedDict):")
        for param_name, annotation in annotations.items():
            if not include_ret and param_name == "return":
                continue
            formatted = cls.format_annotation(annotation)
            print(f"    {param_name}: {formatted}")

        return annotations

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

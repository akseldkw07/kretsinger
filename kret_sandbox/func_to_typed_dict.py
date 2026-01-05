from __future__ import annotations

import inspect
import re
import typing as t

from kret_type_hints.typed_func_helper import TypedFuncHelper

from typing import get_args, get_origin, Union, Optional
import sys


class FuncToTypedDict(TypedFuncHelper):

    @classmethod
    def func_to_typed_dict(cls, func: t.Callable, include_ret: bool = False):
        """
        1. Print required imports
        2. Print TypedDict class definition based on function annotations
        """
        cls.print_imports(func)
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
        imports = set()

        for param in sig.parameters.values():
            arg_type = param.annotation
            str_argtype = str(arg_type)
            # imports.add(str_argtype)
            # if "<class" in str_argtype:
            #     str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
            # for to_replace, replacement in TypedFuncHelper.import_replace_dict.items():
            #     str_argtype = str_argtype.replace(to_replace, replacement)
            # TypedFuncHelper.collect_imports(str_argtype, imports)

        for imp in sorted(imports):
            print(imp)

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
    def resolve_dict_name(cls, func: t.Callable) -> str:
        qualname = getattr(func, "__qualname__", "")
        parts = qualname.split(".")
        if len(parts) > 1 and "<locals>" not in qualname:
            class_name = parts[-2]
            dict_name = f"{class_name}_{func.__name__.capitalize()}_TypedDict"
        else:
            dict_name = f"{func.__name__.capitalize()}_TypedDict"
        return dict_name

    @classmethod
    def format_annotation(cls, annotation) -> str:
        """Convert a type annotation to Python 3.10+ syntax (PEP 604)."""

        # Handle None type
        if annotation is type(None):
            return "None"

        # Handle string annotations
        if isinstance(annotation, str):
            return annotation

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Union types -> use | syntax
        if origin is Union:
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

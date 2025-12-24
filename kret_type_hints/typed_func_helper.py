from __future__ import annotations

import inspect
import re
import typing as t
from typing import TypeVar

T = TypeVar("T")


class TypedFuncHelper:

    @classmethod
    # --- Shared printer ---
    def print_typed_dict(cls, func: t.Callable, typed_dict: dict | None, include_ret: bool = False):
        name = func.__name__
        sig = inspect.signature(func)
        extra_imports = set()
        # If typed_dict is None, fallback to signature-based printing (like func_to_typed_dict)
        if typed_dict is None:
            # Reuse func_to_typed_dict logic for printing
            for param_name, param in sig.parameters.items():
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    continue
                arg_type = param.annotation
                if arg_type is inspect.Parameter.empty:
                    str_argtype = "t.Any"
                else:
                    str_argtype = str(arg_type)
                    if "<class" in str_argtype:
                        str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
                    for to_replace, replacement in cls.import_replace_dict.items():
                        str_argtype = str_argtype.replace(to_replace, replacement)
                if param.default is None and "None" not in str_argtype:
                    str_argtype = f"{str_argtype} | None"
                cls.collect_imports(str_argtype, extra_imports)
            # Return type
            if include_ret and sig.return_annotation is not inspect.Parameter.empty:
                return_type = sig.return_annotation
                str_return_type = str(return_type)
                if "<class" in str_return_type:
                    str_return_type = str_return_type.replace("<class '", "").replace(">", "").replace("'", "")
                for to_replace, replacement in cls.import_replace_dict.items():
                    str_return_type = str_return_type.replace(to_replace, replacement)
                cls.collect_imports(str_return_type, extra_imports)
            for imp in sorted(extra_imports):
                print(imp)
            print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")
            for param_name, param in sig.parameters.items():
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    continue
                arg_type = param.annotation
                if arg_type is inspect.Parameter.empty:
                    str_argtype = "t.Any"
                else:
                    str_argtype = str(arg_type)
                    if "<class" in str_argtype:
                        str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
                    for to_replace, replacement in cls.import_replace_dict.items():
                        str_argtype = str_argtype.replace(to_replace, replacement)
                if param.default is None and "None" not in str_argtype:
                    str_argtype = f"{str_argtype} | None"
                print(f"    {param_name}: {str_argtype}")
            if include_ret and sig.return_annotation is not inspect.Parameter.empty:
                return_type = sig.return_annotation
                str_return_type = str(return_type)
                if "<class" in str_return_type:
                    str_return_type = str_return_type.replace("<class '", "").replace(">", "").replace("'", "")
                for to_replace, replacement in cls.import_replace_dict.items():
                    str_return_type = str_return_type.replace(to_replace, replacement)
                print(f"    # return: {str_return_type}")
            return
        # If typed_dict is a dict (from LLM/JSON), print using its keys/values
        for v in typed_dict.values():
            cls.collect_imports(str(v), extra_imports)
        for imp in sorted(extra_imports):
            print(imp)
        print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")
        for k, v in typed_dict.items():
            print(f"    {k}: {v}")

    import_replace_dict = {
        "pandas.core.frame.DataFrame": "pd.DataFrame",
        "pandas.core.series.Series": "pd.Series",
        "numpy": "np",
        "pandas": "pd",
        "t.Union": "Union",
        "t.Optional": "Optional",
        "typing.Union": "Union",
        "typing.Optional": "Optional",
        "typing.": "t.",
        "numba": "nb",
    }

    # --- Shared import collector ---
    @classmethod
    def collect_imports(cls, type_str: str, extra_imports: set):
        known_type_map = {
            "Session": "from requests import Session",
        }
        for match in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", type_str):
            if match in known_type_map:
                extra_imports.add(known_type_map[match])
            elif match not in {
                "None",
                "Any",
                "Literal",
                "Optional",
                "Union",
                "Sequence",
                "Mapping",
                "Dict",
                "List",
                "Tuple",
                "Set",
                "Type",
                "Callable",
                "str",
                "int",
                "float",
                "bool",
                "object",
            }:
                extra_imports.add(f"from {match.lower()} import {match}")

from __future__ import annotations

import datetime as dt
import inspect
import typing as t
from types import GenericAlias

import numpy as np
import pandas as pd
import pytz
from IPython.display import display
from IPython.display import HTML
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure
from matplotlib.typing import ColorType
from plotly.basedatatypes import BaseTraceType

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


def func_to_typed_dict(func: t.Callable, include_ret: bool = False):
    """Convert function signature / annotations to a typed dict of accepted types."""

    import re

    sig = inspect.signature(func)
    name = func.__name__
    doc = func.__doc__ or ""

    # Parse docstring for :Parameters: section
    param_types_from_doc = {}
    param_descs = {}
    param_literals = {}
    param_section = False
    doc_lines = doc.splitlines()
    i = 0
    while i < len(doc_lines):
        line = doc_lines[i].strip()
        if line.lower().startswith(":parameters:"):
            param_section = True
            i += 1
            continue
        if param_section:
            if not line or line.startswith(":"):
                break
            # Try to parse lines like: name : type
            m = re.match(r"([\w*]+)\s*:\s*([^#]+)", line)
            if m:
                pname, ptype = m.group(1).strip(), m.group(2).strip()
                desc = line
                # Look ahead for valid values/intervals/periods in next lines
                lookahead = 1
                while i + lookahead < len(doc_lines):
                    next_line = doc_lines[i + lookahead].strip()
                    if not next_line or next_line.startswith(":") or re.match(r"^[\w*]+\s*:\s*", next_line):
                        break
                    valid_match = re.search(r"Valid (?:values|periods|intervals): ([^\n]+)", next_line, re.IGNORECASE)
                    if valid_match:
                        vals = [v.strip() for v in valid_match.group(1).replace(" ", "").split(",") if v.strip()]
                        param_literals[pname] = vals
                        desc += " " + next_line
                    lookahead += 1
                param_types_from_doc[pname] = ptype
                param_descs[pname] = desc
                i += lookahead - 1
            else:
                # Try to parse lines like: name : type, description
                m = re.match(r"([\w*]+)\s*:\s*([^,]+),?\s*(.*)", line)
                if m:
                    pname, ptype, desc = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                    # Look ahead for valid values/intervals/periods in next lines
                    lookahead = 1
                    while i + lookahead < len(doc_lines):
                        next_line = doc_lines[i + lookahead].strip()
                        if not next_line or next_line.startswith(":") or re.match(r"^[\w*]+\s*:\s*", next_line):
                            break
                        valid_match = re.search(
                            r"Valid (?:values|periods|intervals): ([^\n]+)",
                            next_line,
                            re.IGNORECASE,
                        )
                        if valid_match:
                            vals = [v.strip() for v in valid_match.group(1).replace(" ", "").split(",") if v.strip()]
                            param_literals[pname] = vals
                            desc += " " + next_line
                        lookahead += 1
                    param_types_from_doc[pname] = ptype
                    param_descs[pname] = desc
                    i += lookahead - 1
        i += 1

    def doc_type_to_hint(ptype: str, desc: str = "", literals: t.Optional[list[str]] = None) -> str:
        # Try to convert docstring type to python type hint
        ptype = ptype.strip()
        if literals:
            return f"t.Literal[{', '.join([repr(v) for v in literals])}]"
        # Handle common cases
        if ptype in {"str", "string"}:
            return "str"
        if ptype in {"int", "integer"}:
            return "int"
        if ptype in {"float"}:
            return "float"
        if ptype in {"bool", "boolean"}:
            return "bool"
        if ptype in {"list", "array", "sequence"}:
            return "list"
        if ptype in {"dict", "mapping"}:
            return "dict"
        if ptype == "str, list":
            return "str | list[str]"
        # Try to extract from ptype itself
        if any(x in ptype for x in [" or ", ",", "/"]):
            # e.g. "str, list" or "bool / int"
            parts = re.split(r",|/| or ", ptype)
            parts = [p.strip() for p in parts if p.strip()]
            return " | ".join(parts)
        return ptype

    print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")

    for param_name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue

        # Prefer docstring type if available
        if param_name in param_types_from_doc:
            doc_type = param_types_from_doc[param_name]
            desc = param_descs.get(param_name, "")
            literals = param_literals.get(param_name)
            str_argtype = doc_type_to_hint(doc_type, desc, literals)
        else:
            arg_type = param.annotation
            if arg_type is inspect.Parameter.empty:
                str_argtype = "t.Any"
            else:
                str_argtype = str(arg_type)
                if "<class" in str_argtype:
                    str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
                for to_replace, replacement in import_replace_dict.items():
                    str_argtype = str_argtype.replace(to_replace, replacement)

        # Incorporate None if default is None and not already present
        if param.default is None and "None" not in str_argtype:
            str_argtype = f"{str_argtype} | None"

        print(f"    {param_name}: {str_argtype}")

    # Handle return type if include_ret is True
    if include_ret and sig.return_annotation is not inspect.Parameter.empty:
        return_type = sig.return_annotation
        str_return_type = str(return_type)
        if "<class" in str_return_type:
            str_return_type = str_return_type.replace("<class '", "").replace(">", "").replace("'", "")
        for to_replace, replacement in import_replace_dict.items():
            str_return_type = str_return_type.replace(to_replace, replacement)
        print(f"    # return: {str_return_type}")

    return sig.parameters


def func_to_typed_dict_depr(func: t.Callable, include_ret: bool = False):
    """Convert function signature / annotations to a typed dict of accepted types."""
    annot = func.__annotations__
    if not include_ret:
        annot.pop("return", None)
    name = func.__name__

    print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")
    for arg, arg_type in annot.items():

        str_argtype = str(arg_type)
        if "<class" in str_argtype:
            str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
        for to_replace, replacement in import_replace_dict.items():
            str_argtype = str_argtype.replace(to_replace, replacement)

        print(f"    {arg}: {str(str_argtype)}")
    return annot

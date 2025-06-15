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


def func_to_typed_dict2(func: t.Callable, include_ret: bool = False):
    """Convert function signature / annotations to a typed dict of accepted types."""
    sig = inspect.signature(func)
    name = func.__name__

    print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")

    # Iterate through parameters from the signature
    for param_name, param in sig.parameters.items():
        # Exclude positional-only arguments if any (like '/' in some signatures)
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            continue

        # Exclude keyword arguments passed via **kwargs
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # You might want to add a comment or a placeholder for **kwargs
            # print(f"    # {param_name}: Any # Additional keyword arguments")
            continue

        # Exclude positional arguments passed via *args
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # print(f"    # {param_name}: Any # Positional arguments")
            continue

        arg_type = param.annotation
        # If no annotation is present, it's inspect.Parameter.empty
        if arg_type is inspect.Parameter.empty:
            str_argtype = "t.Any"  # Default to Any if type hint is missing
        else:
            str_argtype = str(arg_type)
            if "<class" in str_argtype:
                str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
            for to_replace, replacement in import_replace_dict.items():
                str_argtype = str_argtype.replace(to_replace, replacement)

        print(f"    {param_name}: {str_argtype}")

    # Handle return type if include_ret is True
    if include_ret and sig.return_annotation is not inspect.Parameter.empty:
        return_type = sig.return_annotation
        str_return_type = str(return_type)
        if "<class" in str_return_type:
            str_return_type = str_return_type.replace("<class '", "").replace(">", "").replace("'", "")
        for to_replace, replacement in import_replace_dict.items():
            str_return_type = str_return_type.replace(to_replace, replacement)
        print(f"    # return: {str_return_type}")  # Return type is not part of TypedDict fields

    return sig.parameters  # Return parameters for inspection if needed

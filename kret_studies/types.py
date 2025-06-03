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
    'pandas.core.frame.DataFrame': 'pd.DataFrame',
    'pandas.core.series.Series': 'pd.Series',
    'numpy': 'np',
    'pandas': 'pd',
    't.Union': 'Union',
    't.Optional': 'Optional',
    'typing.Union': 'Union',
    'typing.Optional': 'Optional',
    'typing.': 't.',
    'numba': 'nb',
}


def func_to_typed_dict(func: t.Callable, include_ret: bool = False):
    """Convert function signature / annotations to a typed dict of accepted types."""
    annot = func.__annotations__
    if not include_ret:
        annot.pop('return', None)
    name = func.__name__

    print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")
    for arg, arg_type in annot.items():

        str_argtype = str(arg_type)
        if '<class' in str_argtype:
            str_argtype = str_argtype.replace("<class '", '').replace('>', '').replace("'", '')
        for to_replace, replacement in import_replace_dict.items():
            str_argtype = str_argtype.replace(to_replace, replacement)

        print(f"    {arg}: {str(str_argtype)}")
    return annot

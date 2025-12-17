from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from IPython.display import display_html


def _coerce_to_df(obj: pd.DataFrame | pd.Series | np.ndarray | list | tuple | object | torch.Tensor) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, pd.Series):
        return obj.to_frame()
    elif isinstance(obj, np.ndarray):
        return pd.DataFrame(obj)
    elif isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)
    elif isinstance(obj, torch.Tensor):
        return pd.DataFrame(obj.detach().cpu().numpy())
    else:
        return pd.DataFrame([obj])


def display_side_by_side(
    dfs: Sequence[pd.DataFrame | pd.Series | np.ndarray | list | tuple | torch.Tensor],
    names: list[str] | None = None,
    spacing: int = 12,
    font_size: str = "0.9em",
) -> None:
    """
    Display multiple tables (DataFrames, Series, arrays, lists, or tuples) side by side in Jupyter/VS Code.
    """
    html_str = f'<div style="display:flex; gap:{spacing}px; align-items:flex-start; flex-wrap:nowrap;">'
    for i, obj in enumerate(dfs):
        df = _coerce_to_df(obj)
        title = f"<h3>{names[i]}</h3>" if names and i < len(names) else ""
        html_table = df.to_html()
        html_table = html_table.replace(
            "<table ",
            f'<table style="width:auto; table-layout:auto; white-space:nowrap; font-size:{font_size}; border-collapse:collapse;" ',
            1,
        )
        html_str += f'<div style="flex:0 0 auto; overflow:auto;">' f"{title}{html_table}</div>"
    html_str += "</div>"
    display_html(html_str, raw=True)


# def dataset_to_table(
#     *dfs: pd.DataFrame | np.ndarray | torch.Tensor, names: list[str] | None = None, spacing: int = 10
# ) -> None:
#     """
#     Display a list of pandas DataFrames side-by-side in Jupyter or VS Code.
#     """
#     display_side_by_side(list(dfs), names=names, spacing=spacing)

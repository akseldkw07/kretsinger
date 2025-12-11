import typing as t
from functools import cache
from itertools import chain, cycle

import numpy as np
import pandas as pd
import torch
from IPython.display import display_html


@cache
def gen_display_mask(n: int, hot: int, seed: int, display_method: t.Literal["sample", "head", "tail"]):
    rng = np.random.default_rng(seed)
    if display_method == "sample":
        mask = np.zeros(n, dtype=bool)
        hot_indices = rng.choice(n, size=hot, replace=False)
        mask[hot_indices] = True
    elif display_method == "head":
        mask = np.zeros(n, dtype=bool)
        mask[:hot] = True
    else:  # tail
        mask = np.zeros(n, dtype=bool)
        mask[-hot:] = True
    return mask


def process_filter(
    filter: np.ndarray | pd.Series | torch.Tensor | pd.DataFrame | None,
) -> np.ndarray | None:
    if isinstance(filter, torch.Tensor):
        assert filter.dim() == 1, "Filter tensor must be 1-dimensional."
        filter = filter.detach().cpu().numpy()
    elif isinstance(filter, pd.DataFrame):
        assert filter.shape[1] == 1, "Filter DataFrame must have a single column."
        filter = filter.iloc[:, 0].to_numpy()
    elif isinstance(filter, pd.Series):
        filter = filter.to_numpy()

    if isinstance(filter, np.ndarray):
        assert filter.ndim == 1, "Filter array must be 1-dimensional."
        assert filter.dtype == bool or np.issubdtype(
            filter.dtype, np.integer
        ), "Filter array must be of boolean or integer type."
        if np.issubdtype(filter.dtype, np.integer):
            filter = np.asarray(filter, dtype=bool)
    return filter


def coerce_to_df(obj: pd.DataFrame | pd.Series | np.ndarray | list | tuple | object | torch.Tensor) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, pd.Series):
        return obj.to_frame()
    elif isinstance(obj, np.ndarray):
        return pd.DataFrame({i: obj[:, i] for i in range(obj.shape[1])}) if obj.ndim > 1 else pd.DataFrame({0: obj})
    elif isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)
    elif isinstance(obj, torch.Tensor):
        return pd.DataFrame(obj.detach().cpu().numpy())
    else:
        return pd.DataFrame([obj])


def dtt(
    args_: t.Sequence[pd.DataFrame | pd.Series | np.ndarray | torch.Tensor],
    n: int = 5,
    filter: np.ndarray | pd.Series | torch.Tensor | pd.DataFrame | None = None,
    how: t.Literal["sample", "head", "tail"] = "sample",
    titles: list[str] | cycle = cycle([""]),
    seed: int | None = None,
    round: int | None = 3,
) -> None:
    seed = seed or np.random.randint(0, 1_000_000)
    filter = process_filter(filter)

    args: list[pd.DataFrame] = []
    for arg in args_:
        arg = arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg
        df = arg[filter] if filter is not None else arg
        df = coerce_to_df(df)
        if round is not None:
            df = df.round(round)
        args.append(df)

    # Add overflow-x: auto for horizontal scrolling
    html_str = '<div style="display: flex; gap: 20px; overflow-x: auto;">'

    for df, title in zip(args, chain(titles, cycle([""]))):
        # Add flex-shrink: 0 to prevent tables from shrinking
        html_str += '<div style="flex: 0 0 auto; min-width: 30px;">'

        if title:
            html_str += (
                f'<div style="text-align: left; font-weight: bold; font-size: 18px; margin-bottom: 8px;">{title}</div>'
            )

        mask = gen_display_mask(len(df), min(n, len(df)), seed, how)
        table_html = df[mask].to_html()
        html_str += table_html
        html_str += "</div>"
    html_str += "</div>"
    display_html(html_str, raw=True)

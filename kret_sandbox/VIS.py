from __future__ import annotations

import typing as t
from functools import cache
from itertools import chain, cycle

import numpy as np
import pandas as pd
import torch
from IPython.display import display_html

DEFAULT_DTT_PARAMS: DTTParams = {"seed": None, "round_float": 3, "max_col_width": 150, "cols_per_row": None}


def dtt(
    args_: t.Sequence[pd.DataFrame | pd.Series | np.ndarray | torch.Tensor],
    n: int = 5,
    how: t.Literal["sample", "head", "tail"] = "sample",
    filter: np.ndarray | pd.Series | torch.Tensor | pd.DataFrame | None = None,
    titles: list[str] | cycle = cycle([""]),
    **hparams: t.Unpack[DTTParams],
) -> None:
    hparams = {**DEFAULT_DTT_PARAMS, **hparams}
    seed = hparams.get("seed") or np.random.randint(0, 1_000_000)
    filter = process_filter(filter)

    args: list[pd.DataFrame] = []
    for arg in args_:
        arg = arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg
        df = arg[filter] if filter is not None else arg
        df = coerce_to_df(df)
        if (round_float := hparams.get("round_float")) is not None:
            df = df.round(round_float)
        args.append(df)

    # Add overflow-x: auto for horizontal scrolling
    html_str = '<div style="display: flex; gap: 20px; overflow-x: auto;">'
    html_str = fmt_css(hparams, html_str)

    for df, title in zip(args, chain(titles, cycle([""]))):
        # Add flex-shrink: 0 to prevent tables from shrinking
        html_str += '<div style="flex: 0 0 auto; min-width: 30px;">'

        if title:
            html_str += (
                f'<div style="text-align: left; font-weight: bold; font-size: 18px; margin-bottom: 8px;">{title}</div>'
            )

        mask = gen_display_mask(len(df), min(n, len(df)), seed, how)
        table_html = generate_table_with_dtypes(df[mask])
        html_str += table_html
        html_str += "</div>"
    html_str += "</div>"
    display_html(html_str, raw=True)


def generate_table_with_dtypes(df: pd.DataFrame) -> str:
    """Generate HTML table with datatypes displayed below column headers."""
    # Start table
    html = '<table border="1" class="dataframe">\n'

    # Header row with column names
    html += '  <thead>\n    <tr style="text-align: right;">\n'
    html += "      <th></th>\n"  # Index column
    for col in df.columns:
        html += f"      <th>{col}</th>\n"
    html += "    </tr>\n"

    # Datatype row
    html += '    <tr style="text-align: right; font-size: 0.85em; color: #666; font-style: italic;">\n'
    html += "      <th></th>\n"  # Index column
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        html += f"      <th>{dtype_str}</th>\n"
    html += "    </tr>\n  </thead>\n"

    # Body
    html += "  <tbody>\n"
    for idx, row in df.iterrows():
        html += "    <tr>\n"
        html += f"      <th>{idx}</th>\n"
        for val in row:
            html += f"      <td>{val}</td>\n"
        html += "    </tr>\n"
    html += "  </tbody>\n</table>"

    return html


def fmt_css(hparams: DTTParams, html_str: str):
    # Add CSS for column width limits if specified
    if (max_col_width := hparams.get("max_col_width")) is not None:
        html_str += f"""
        <style>
            table td, table th {{
                max-width: {max_col_width}px;
                overflow: hidden;
                white-space: nowrap;
                position: relative;
            }}
            table td {{
                text-overflow: ellipsis;
            }}
            table td.truncated::after, table th.truncated::after {{
                content: '';
                position: absolute;
                right: 0;
                top: 0;
                bottom: 0;
                width: 20px;
                background: linear-gradient(to right, transparent, rgba(255, 0, 0, 0.4));
                pointer-events: none;
            }}
        </style>
        <script>
            (function() {{
                // Wait for the DOM to be ready
                setTimeout(function() {{
                    document.querySelectorAll('table td, table th').forEach(function(cell) {{
                        if (cell.scrollWidth > cell.clientWidth) {{
                            cell.classList.add('truncated');
                        }}
                    }});
                }}, 100);
            }})();
        </script>
        """
    return html_str


class DTTParams(t.TypedDict, total=False):
    seed: int | None
    round_float: int | None
    max_col_width: int | None
    cols_per_row: int | None


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

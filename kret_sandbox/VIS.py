from __future__ import annotations

import typing as t
from functools import cache
from itertools import chain, cycle

import numpy as np
import pandas as pd
import torch
from IPython.display import display_html


if t.TYPE_CHECKING:
    pass

    from pandas._typing import ListLike, ColspaceArgType, FormattersType, FloatFormatType


DEFAULT_DTT_PARAMS: DTTParams = {"seed": None, "max_col_width": 150, "num_cols": None}
PD_TO_HTML_KWARGS: To_html_TypedDict = {
    "border": 1,
    # "index": False,
    # "justify": "left",
    # "classes": "dataframe dtt-table",
}


def dtt(
    args_: t.Sequence[pd.DataFrame | pd.Series | np.ndarray | torch.Tensor],
    n: int = 5,
    how: t.Literal["sample", "head", "tail"] = "sample",
    filter: np.ndarray | pd.Series | torch.Tensor | pd.DataFrame | None = None,
    titles: list[str] | cycle = cycle([""]),
    **hparams: t.Unpack[DTTKwargs],
):
    hparams = {**DEFAULT_DTT_PARAMS, **DEFAULT_DTT_PARAMS, **hparams}
    seed = hparams.get("seed") or np.random.randint(0, 1_000_000)
    filter = process_filter(filter)

    args: list[pd.DataFrame] = []
    for arg in args_:
        arg = arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg
        df = arg[filter] if filter is not None else arg
        df = coerce_to_df(df)
        args.append(df)

    display_df_list(args, titles, n, seed, how, hparams, num_cols=hparams.get("num_cols"))


TITLE_FMT = '<div style="text-align: left; font-weight: bold; font-size: 18px; margin-bottom: 8px;">{title}</div>'
OUTER_DIV = "<div style='display: flex; gap: 20px; overflow-x: auto;'>"
PER_ROW_DIV = "<div style='display: flex; flex-direction: column; gap: 20px; flex-shrink: 0; overflow: hidden;'>"
PER_TABLE_DIV = "<div style='flex: 0 0 auto; min-width: 30px;'>"


def display_df_list(
    args: list[pd.DataFrame],
    titles: t.Iterable[str],
    n: int,
    seed: int,
    how: str,
    hparams: DTTKwargs,
    num_cols: int | None = None,
):
    """Original behavior: display all items in a single row with horizontal scroll."""
    # Add overflow-x: auto for horizontal scrolling
    html_str = OUTER_DIV
    html_str = fmt_css(hparams, html_str)

    for idx, (df, title) in enumerate(zip(args, chain(titles, cycle([""])))):
        if num_cols is not None and idx % num_cols == 0:
            # Close previous row and start a new one
            if idx > 0:
                html_str += "</div>"  # Close previous div

            html_str += PER_ROW_DIV  # Start new div

        # Add flex-shrink: 0 to prevent tables from shrinking
        html_str += PER_TABLE_DIV

        if title:
            html_str += TITLE_FMT.format(title=title)

        mask = gen_display_mask(len(df), min(n, len(df)), seed, how)
        table_html = generate_table_with_dtypes(df[mask], **hparams)
        html_str += table_html
        html_str += "</div>"
    html_str += "</div>" + ("</div>" if num_cols is not None else "")
    display_html(html_str, raw=True)


class To_html_TypedDict(t.TypedDict, total=False):
    buf: None
    columns: ListLike | None
    col_space: ColspaceArgType | None
    header: bool
    index: bool
    na_rep: str
    formatters: FormattersType | None
    float_format: FloatFormatType | None
    sparsify: bool | None
    index_names: bool
    justify: str | None
    max_rows: int | None
    max_cols: int | None
    show_dimensions: bool
    decimal: str
    bold_rows: bool
    classes: str | list | tuple | None
    escape: bool
    notebook: bool
    border: int | bool | None
    table_id: str | None
    render_links: bool
    encoding: str | None


def generate_table_with_dtypes(df: pd.DataFrame, **hparams: t.Unpack[To_html_TypedDict]) -> str:
    """Generate HTML table with datatypes displayed below column headers."""
    # Use pandas' fast to_html() method
    import inspect

    html_params = {k: v for k, v in hparams.items() if k in inspect.signature(df.to_html).parameters}
    base_html = df.to_html(**html_params)  # type: ignore

    # Build the dtype row HTML
    dtype_row = '    <tr style="text-align: right; font-size: 0.85em; color: #666; font-style: italic;">\n'
    dtype_row += "      <th></th>\n"  # Index column
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        dtype_row += f"      <th>{dtype_str}</th>\n"
    dtype_row += "    </tr>\n"

    # Inject dtype row after the first </tr> in thead
    # Find the end of the first header row
    first_tr_end = base_html.find("</tr>", base_html.find("<thead>"))
    if first_tr_end != -1:
        # Insert dtype row after the first </tr>
        insert_pos = first_tr_end + len("</tr>\n")
        html = base_html[:insert_pos] + dtype_row + base_html[insert_pos:]
    else:
        # Fallback if structure is unexpected
        html = base_html

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
    max_col_width: int | None
    num_cols: int | None


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


class DTTKwargs(To_html_TypedDict, DTTParams, total=False):
    pass

import inspect
import typing as t
from functools import cache
from itertools import chain, cycle

import numpy as np
import pandas as pd
import torch
from IPython.display import display_html
from pandas.io.formats.style import Styler

from kret_np_pd.filters import FilterUtils
from kret_rosetta.UTILS_rosetta import UTILS_rosetta

if t.TYPE_CHECKING:
    from pandas._typing import ColspaceArgType, FloatFormatType, FormattersType, ListLike

    from kret_torch_utils.tensor_ds_custom import TensorDatasetCustom

    VectorMatrixType = (
        pd.DataFrame
        | Styler
        | pd.Series
        | np.ndarray
        | torch.Tensor
        | torch.utils.data.TensorDataset
        | TensorDatasetCustom
    )

TITLE_FMT = '<div style="text-align: left; font-weight: bold; margin-left: 5px; font-size: 18px; margin-bottom: 8px;">{title}</div>'
OUTER_STYLE_TABLE = "<div style='display: flex; flex-direction: column; gap: 20px; overflow-x: auto;'>"
OUTER_STYLE_ROW = "<div style='display: flex; gap: 20px; overflow-x: auto;'>"
PER_ROW_DIV = "<div style='display: flex; gap: 20px; flex-shrink: 0; overflow: hidden; width: fit-content;'>"
PER_TABLE_DIV = (
    "<div style='flex: 0 0 auto; min-width: 30px; border-left: 2px solid #ccc; padding-left: {addtl_width}px;'>"
)
TABLE_FMT = """
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


class DTTParams(t.TypedDict, total=False):
    seed: int | None
    max_col_width: int | None
    num_cols: int | None
    show_dimensions: bool  # TODO make this nicer, add to the bottom of the dataframe instead of applying it to title
    align_cols: bool  # NOTE not implemented


class To_html_TypedDict(t.TypedDict, total=False):
    buf: None
    columns: "ListLike | None"
    col_space: "ColspaceArgType | None"
    header: bool
    index: bool
    na_rep: str
    formatters: "FormattersType | None"
    float_format: "FloatFormatType | None"
    sparsify: bool | None
    index_names: bool
    justify: str | None
    max_rows: int | None
    max_cols: int | None
    decimal: str
    bold_rows: bool
    classes: str | list | tuple | None
    escape: bool
    notebook: bool
    border: int | bool | None
    table_id: str | None
    render_links: bool
    encoding: str | None


class DTTKwargs(To_html_TypedDict, DTTParams, total=False):
    pass


DEFAULT_DTT_PARAMS: DTTParams = {"seed": None, "max_col_width": 150, "num_cols": None, "show_dimensions": False}
PD_TO_HTML_KWARGS: To_html_TypedDict = {"border": 1, "float_format": "{:.3f}".format}

ViewHow = t.Literal["sample", "head", "tail"]


class PD_Display_Utils:

    @classmethod
    def dtt(
        cls,
        input: "list[VectorMatrixType] | VectorMatrixType",
        n: int = 5,
        how: ViewHow = "sample",
        filter: np.ndarray | pd.Series | torch.Tensor | pd.DataFrame | None = None,
        titles: list[str] | cycle = cycle([""]),
        **hparams: t.Unpack[DTTKwargs],
    ):
        """
        TODO add shape to the titles
        TODO ability to view slice of rows
        Display one or more DataFrames / arrays / tensors in a Jupyter notebook with datatypes shown below column headers.
        """

        input = input if isinstance(input, (list)) else [input]
        hparams = {**DEFAULT_DTT_PARAMS, **PD_TO_HTML_KWARGS, **hparams}
        hparams["seed"] = hparams.get("seed") or np.random.randint(0, 1_000_000)
        filter = FilterUtils.process_filter(filter) if filter is not None else None

        args: list[pd.DataFrame | Styler] = []
        for arg in input:
            arg = arg.numpy(force=True) if isinstance(arg, torch.Tensor) else arg
            if not isinstance(arg, Styler):
                df = arg[filter] if (filter is not None) else arg
                df = UTILS_rosetta.coerce_to_df(df)
            else:
                df = arg
            args.append(df)

        cls.display_df_list(args, titles, n, how, hparams)

    @classmethod
    def display_df_list(
        cls, args: list[pd.DataFrame | Styler], titles: t.Iterable[str], n: int, how: ViewHow, hparams: DTTKwargs
    ):
        """Original behavior: display all items in a single row with horizontal scroll."""
        # Add overflow-x: auto for horizontal scrolling
        num_cols = hparams.get("num_cols")
        html_str = OUTER_STYLE_TABLE if num_cols else OUTER_STYLE_ROW
        html_str = cls.fmt_css(hparams, html_str)

        for idx, (df, title) in enumerate(zip(args, chain(titles, cycle(["NO_TITLE"])))):
            if num_cols is not None and idx % num_cols == 0:
                # Close previous row and start a new one
                if idx > 0:
                    html_str += "</div>"  # Close previous div

                html_str += PER_ROW_DIV  # Start new div

            # Add flex-shrink: 0 to prevent tables from shrinking
            html_str += PER_TABLE_DIV.format(addtl_width=0)
            if isinstance(df, Styler):
                table_html = cls.generate_table_with_dtypes(df, **hparams)
                html_str += TITLE_FMT.format(title=title) if title else ""
                html_str += table_html
                html_str += "</div>"
                continue

            title += f"{df.shape[0]} rows x {df.shape[1] } columns" if hparams.get("show_dimensions", False) else ""
            html_str += TITLE_FMT.format(title=title) if title else ""
            assert "seed" in hparams, f"Seed must be set in hparams, got {hparams}"
            mask = gen_display_mask(len(df), min(n, len(df)), hparams["seed"], how)
            table_html = cls.generate_table_with_dtypes(df[mask], **hparams)
            html_str += table_html
            html_str += "</div>"
        html_str += "</div>" + ("</div>" if num_cols is not None else "")
        display_html(html_str, raw=True)

    @classmethod
    def generate_table_with_dtypes(cls, df: pd.DataFrame | Styler, **hparams: t.Unpack[To_html_TypedDict]) -> str:
        """Generate HTML table with datatypes displayed below column headers."""
        # Use pandas' fast to_html() method

        html_params = {k: v for k, v in hparams.items() if k in inspect.signature(df.to_html).parameters}
        base_html = df.to_html(**html_params)  # type: ignore
        if not isinstance(df, pd.DataFrame):
            return base_html

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

    @classmethod
    def fmt_css(cls, hparams: DTTParams, html_str: str):
        # Add CSS for column width limits if specified
        if (max_col_width := hparams.get("max_col_width")) is not None:
            html_str += TABLE_FMT.format(max_col_width=max_col_width)
        return html_str


@cache
def gen_display_mask(n: int, hot: int, seed: int, display_method: ViewHow):
    # print(n)
    if hot == -1:
        return np.full(n, True, dtype=bool)
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

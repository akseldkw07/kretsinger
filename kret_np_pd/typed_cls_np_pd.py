import typing as t

if t.TYPE_CHECKING:
    from pandas._typing import ColspaceArgType, FloatFormatType, FormattersType, ListLike


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

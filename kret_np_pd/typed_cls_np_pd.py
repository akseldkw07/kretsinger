# from pandas._libs import lib
import collections.abc as c_abc
import csv
import typing as t

if t.TYPE_CHECKING:
    from pandas._typing import (
        ColspaceArgType,
        CompressionOptions,
        CSVEngine,
        DtypeArg,
        DtypeBackend,
        FloatFormatType,
        FormattersType,
        IndexLabel,
        ListLike,
        StorageOptions,
        UsecolsArgType,
    )


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


class Read_csv_TypedDict(t.TypedDict, total=False):
    """
    NOTE this doesn't work with type checkers for some reason
    """

    # filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str]
    sep: str | None  # = _NoDefault(...)
    delimiter: str | None  # = None
    header: int | t.Sequence[int] | None | t.Literal["infer"]  # = infer
    names: c_abc.Sequence[t.Hashable] | None  # = _NoDefault(...)
    index_col: "IndexLabel | t.Literal[False] | None"  # = None
    usecols: "UsecolsArgType"  # = None
    dtype: "DtypeArg | None"  # = None
    engine: "CSVEngine | None"  # = None
    converters: t.Mapping[t.Hashable, t.Callable] | None  # = None
    true_values: list | None  # = None
    false_values: list | None  # = None
    skipinitialspace: bool  # = False
    skiprows: list[int] | int | t.Callable[[t.Hashable], bool] | None  # = None
    skipfooter: int  # = 0
    nrows: int | None  # = None
    na_values: t.Hashable | t.Iterable[t.Hashable] | t.Mapping[t.Hashable, t.Iterable[t.Hashable]] | None  # = None
    keep_default_na: bool  # = True
    na_filter: bool  # = True
    verbose: bool  # = _NoDefault(...)
    skip_blank_lines: bool  # = True
    parse_dates: bool | t.Sequence[t.Hashable] | None  # = None
    infer_datetime_format: bool  # = _NoDefault(...)
    keep_date_col: bool  # = _NoDefault(...)
    date_parser: t.Callable  # = _NoDefault(...)
    date_format: str | dict[t.Hashable, str] | None  # = None
    dayfirst: bool  # = False
    cache_dates: bool  # = True
    iterator: bool  # = False
    chunksize: int | None  # = None
    compression: "CompressionOptions"  # = infer
    thousands: str | None  # = None
    decimal: str  # = '.'
    lineterminator: str | None  # = None
    quotechar: str  # = '"'
    quoting: int  # = 0
    doublequote: bool  # = True
    escapechar: str | None  # = None
    comment: str | None  # = None
    encoding: str | None  # = None
    encoding_errors: str | None  # = strict
    dialect: str | csv.Dialect | None  # = None
    on_bad_lines: str  # = error
    delim_whitespace: bool  # = _NoDefault(...)
    low_memory: bool  # = True
    memory_map: bool  # = False
    float_precision: t.Literal["high", "legacy"] | None  # = None
    storage_options: "StorageOptions | None"  # = None
    dtype_backend: "DtypeBackend "  # = _NoDefault(...)

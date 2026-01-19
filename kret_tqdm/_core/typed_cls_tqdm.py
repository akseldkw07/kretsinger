import typing as t

if t.TYPE_CHECKING:
    from _typeshed import SupportsWrite


class TQDM__init__TypedDict(t.TypedDict, total=False):
    # iterable: t.Iterable[t.Any] | t.AsyncIterator[t.Any]
    desc: str | None  # = ...,
    # total: float | None  # = ...,
    leave: bool | None  # = ...,
    file: "SupportsWrite[str] | None "  # = ...,
    ncols: int | None  # = ...,
    mininterval: float  # = ...,
    maxinterval: float  # = ...,
    miniters: float | None  # = ...,
    ascii: bool | str | None  # = ...,
    disable: bool | None  # = ...,
    unit: str  # = ...,
    unit_scale: bool | float  # = ...,
    dynamic_ncols: bool  # = ...,
    smoothing: float  # = ...,
    bar_format: str | None  # = ...,
    initial: float  # = ...,
    position: int | None  # = ...,
    postfix: t.Mapping[str, object] | str | None  # = ...,
    unit_divisor: float  # = ...,
    write_bytes: bool | None  # = ...,
    lock_args: tuple[bool | None, float | None] | tuple[bool | None] | None  # = ...,
    nrows: int | None  # = ...,
    colour: str | None  # = ...,
    delay: float | None  # = ...,
    gui: bool  # = ...,


"""
TODO solve with chatgpt

def __init__(
    self: tqdm,
    iterable: Iterable[Unknown] | AsyncIterator[Unknown],
    desc: str | None # = ...,
    total: float | None # = ...,
    leave: bool | None # = ...,
    file: SupportsWrite[str] | None # = ...,
    ncols: int | None # = ...,
    mininterval: float # = ...,
    maxinterval: float # = ...,
    miniters: float | None # = ...,
    ascii: bool | str | None # = ...,
    disable: bool | None # = ...,
    unit: str # = ...,
    unit_scale: bool | float # = ...,
    dynamic_ncols: bool # = ...,
    smoothing: float # = ...,
    bar_format: str | None # = ...,
    initial: float # = ...,
    position: int | None # = ...,
    postfix: Mapping[str, object] | str | None # = ...,
    unit_divisor: float # = ...,
    write_bytes: bool | None # = ...,
    lock_args: tuple[bool | None, float | None] | tuple[bool | None] | None # = ...,
    nrows: int | None # = ...,
    colour: str | None # = ...,
    delay: float | None # = ...,
    gui: bool # = ...,
    **kwargs: Unknown
    ) -> None:
"""

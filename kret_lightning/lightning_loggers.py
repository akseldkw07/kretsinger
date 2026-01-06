import typing as t
from pathlib import Path
from typing import Any, Literal, TypedDict

if t.TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


class CSVLogger___init___TypedDict(TypedDict, total=False):
    save_dir: str | Path
    name: str | None
    version: int | str | None
    prefix: str
    flush_logs_every_n_steps: int


class WandbLogger___init___TypedDict(TypedDict, total=False):
    save_dir: str | Path
    name: str | None
    version: str | None
    offline: bool
    dir: str | Path | None
    id: str | None
    anonymous: bool | None
    project: str | None
    log_model: Literal["all"] | bool
    experiment: Run | RunDisabled | None
    prefix: str
    checkpoint_name: str | None
    add_file_policy: Literal["mutable", "immutable"]
    kwargs: Any


class TensorBoardLogger___init___TypedDict(TypedDict, total=False):
    save_dir: str | Path
    name: str | None
    version: int | str | None
    log_graph: bool
    default_hp_metric: bool
    prefix: str
    sub_dir: str | Path | None
    kwargs: Any

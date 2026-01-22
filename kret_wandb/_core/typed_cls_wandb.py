from pathlib import Path
from typing import Any, Literal, TypedDict
from collections.abc import Sequence

from wandb.sdk import Settings


class Wandb_Init_TypedDict(TypedDict, total=False):
    entity: str | None  # = None
    project: str | None  # = None
    dir: str | Path | None  # = None
    id: str | None  # = None
    name: str | None  # = None
    notes: str | None  # = None
    tags: Sequence[str] | None  # = None
    config: dict[str, Any] | str | None  # = None
    config_exclude_keys: list[str] | None  # = None
    config_include_keys: list[str] | None  # = None
    allow_val_change: bool | None  # = None
    group: str | None  # = None
    job_type: str | None  # = None
    mode: Literal["online", "offline", "disabled", "shared"] | None  # = None
    force: bool | None  # = None
    reinit: bool | Literal[None, "default", "return_previous", "finish_previous", "create_new"]  # = None
    resume: bool | Literal["allow", "never", "must", "auto"] | None  # = None
    resume_from: str | None  # = None
    fork_from: str | None  # = None
    save_code: bool | None  # = None
    tensorboard: bool | None  # = None
    sync_tensorboard: bool | None  # = None
    monitor_gym: bool | None  # = None
    settings: Settings | dict[str, Any] | None  # = None
    # anonymous: DoNotSet  # = object(...)

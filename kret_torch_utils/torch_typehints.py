from collections.abc import Callable, Iterable
from typing import Any, TypedDict, TypeVar

import torch
from torch.utils.data import Dataset, Sampler

_T = TypeVar("_T")

_collate_fn_t = Callable[[list[_T]], Any]
_worker_init_fn_t = Callable[[int], None]


class DataLoader___init___TypedDict(TypedDict, total=False):
    dataset: Dataset
    batch_size: int | None  # = 1
    shuffle: bool | None  # = None
    sampler: Sampler | Iterable | None  # = None
    batch_sampler: Sampler[list] | Iterable[list] | None  # = None
    num_workers: int  # = 0
    collate_fn: _collate_fn_t | None  # = None
    pin_memory: bool  # = False
    drop_last: bool  # = False
    timeout: float  # = 0
    worker_init_fn: _worker_init_fn_t | None  # = None
    multiprocessing_context: Any | None  # = None
    generator: torch.Generator | None  # = None
    prefetch_factor: int | None  # = None
    persistent_workers: bool  # = False
    pin_memory_device: str  # = ''
    in_order: bool  # = True

import typing as t

from optuna import pruners, samplers, storages
from optuna.study import Study, StudyDirection
from optuna.study.study import ObjectiveFuncType
from optuna.trial import FrozenTrial


class Create_study_TypedDict(t.TypedDict, total=False):
    storage: str | storages.BaseStorage | None  # = None
    sampler: samplers.BaseSampler | None  # = None
    pruner: pruners.BasePruner | None  # = None
    study_name: str | None  # = None
    direction: str | StudyDirection | None  # = None
    load_if_exists: bool  # = False
    directions: t.Sequence[str | StudyDirection] | None  # = None


class Study_Optimize_TypedDict(t.TypedDict, total=False):
    func: ObjectiveFuncType
    n_trials: int | None  # = None
    timeout: float | None  # = None
    n_jobs: int  # = 1
    catch: t.Iterable[type[Exception]] | type[Exception]  # = ()
    callbacks: t.Iterable[t.Callable[[Study, FrozenTrial], None]] | None  # = None
    gc_after_trial: bool  # = False
    show_progress_bar: bool  # = False

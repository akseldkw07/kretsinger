import typing as t
from typing import Any

from lightgbm.sklearn import _LGBM_ScikitCustomObjectiveFunction
from numpy.random import Generator, RandomState


class LGBMRegressor___init___TypedDict(t.TypedDict, total=False):
    boosting_type: str  # = 'gbdt'
    num_leaves: int  # = 31
    max_depth: int  # = -1
    learning_rate: float  # = 0.1
    n_estimators: int  # = 100
    subsample_for_bin: int  # = 200000
    objective: str | _LGBM_ScikitCustomObjectiveFunction | None  # None
    class_weight: dict | str | None  # = None
    min_split_gain: float  # = 0.0
    min_child_weight: float  # = 0.001
    min_child_samples: int  # = 20
    subsample: float  # = 1.0
    subsample_freq: int  # = 0
    colsample_bytree: float  # = 1.0
    reg_alpha: float  # = 0.0
    reg_lambda: float  # = 0.0
    random_state: int | RandomState | Generator | None  # = None
    n_jobs: int | None  # = None
    importance_type: str  # = 'split'
    kwargs: Any


class LGBMClassifier___init___TypedDict(t.TypedDict, total=False):
    boosting_type: str  # = 'gbdt'
    num_leaves: int  # = 31
    max_depth: int  # = -1
    learning_rate: float  # = 0.1
    n_estimators: int  # = 100
    subsample_for_bin: int  # = 200000
    objective: str | _LGBM_ScikitCustomObjectiveFunction | None  # None
    class_weight: dict | str | None  # = None
    min_split_gain: float  # = 0.0
    min_child_weight: float  # = 0.001
    min_child_samples: int  # = 20
    subsample: float  # = 1.0
    subsample_freq: int  # = 0
    colsample_bytree: float  # = 1.0
    reg_alpha: float  # = 0.0
    reg_lambda: float  # = 0.0
    random_state: int | RandomState | Generator | None  # = None
    n_jobs: int | None  # = None
    importance_type: str  # = 'split'
    kwargs: Any

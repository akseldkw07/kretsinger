import typing as t
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from lightgbm import Booster, LGBMModel
from lightgbm.basic import _LGBM_InitScoreType, _LGBM_WeightType
from lightgbm.sklearn import (
    _LGBM_ScikitCustomObjectiveFunction,
)
from numpy.random import Generator, RandomState

OBJECTIVE_LITERAL = t.Literal[
    "regression",
    "regression_l1",
    "huber",
    "fair",
    "poisson",
    "quantile",
    "mape",
    "gamma",
    "tweedie",
    "binary",
    "multiclass",
    "multiclassova",
    "cross_entropy",
    "cross_entropy_lambda",
    "lambdarank",
    "rank_xendcg",
]
EVAL_METRIC_LITERAL = t.Literal["rmse", "l2", "logloss", "ndcg"]


class LGBMRegressor_Fit_TypedDict(TypedDict, total=False):
    # X: _LGBM_ScikitMatrixLike
    # y: _LGBM_LabelType
    sample_weight: _LGBM_WeightType | None  # = None
    init_score: _LGBM_InitScoreType | None  # = None
    # eval_set: _LGBM_ScikitValidSet | None  # = None
    # eval_names: list[str] | None  # = None
    eval_sample_weight: _LGBM_WeightType | None  # = None
    eval_init_score: _LGBM_InitScoreType | None  # = None
    eval_metric: EVAL_METRIC_LITERAL | None  # = None
    feature_name: list[str] | t.Literal["auto"]  # = 'auto'
    categorical_feature: list[str] | list[int] | t.Literal["auto"]  # = 'auto'
    callbacks: list[Callable] | None  # = None
    init_model: str | Path | Booster | LGBMModel | None  # = None


class LGBM_API_Args(t.TypedDict, total=False):
    data_sample_strategy: t.Literal["goss", "bagging"]
    feature_fraction: float
    bagging_fraction: float
    min_data_in_leaf: int
    bagging_freq: int
    metric: t.Literal["rmse"]


class LGBMRegressor___init___TypedDict(LGBM_API_Args, total=False):
    boosting_type: t.Literal["gbdt", "dart", "rf"]  # = 'gbdt'
    num_leaves: int  # = 31
    max_depth: int  # = -1
    learning_rate: float  # = 0.1
    n_estimators: int  # = 100
    subsample_for_bin: int  # = 200000
    objective: OBJECTIVE_LITERAL | _LGBM_ScikitCustomObjectiveFunction | None  # None
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


class LGBMClassifier___init___TypedDict(LGBM_API_Args, total=False):
    boosting_type: str  # = 'gbdt'
    num_leaves: int  # = 31
    max_depth: int  # = -1
    learning_rate: float  # = 0.1
    n_estimators: int  # = 100
    subsample_for_bin: int  # = 200000
    objective: OBJECTIVE_LITERAL | _LGBM_ScikitCustomObjectiveFunction | None  # None
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

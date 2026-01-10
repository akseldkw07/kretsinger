import typing as t

import numpy as np


class MakeRegression_Params_TypedDict(t.TypedDict, total=False):
    # sklearn.datasets.make_regression kwargs

    n_samples: int  # = 100
    n_features: int  # = 100

    n_informative: int  # = 10
    n_targets: int  # = 1
    bias: float  # = 0.0
    effective_rank: int | None  # = None
    tail_strength: float  # = 0.5
    noise: float  # = 0.0
    shuffle: bool  # = True
    coef: bool  # = False
    random_state: int | np.random.RandomState | None  # = None


class MakeClassification_Params_TypedDict(t.TypedDict, total=False):
    # sklearn.datasets.make_classification kwargs

    n_samples: int  # = 100
    n_features: int  # = 20

    n_informative: int  # = 2
    n_redundant: int  # = 2
    n_repeated: int  # = 0
    n_classes: int  # = 2
    n_clusters_per_class: int  # = 2
    weights: list[float] | None  # = None
    flip_y: float  # = 0.01
    class_sep: float  # = 1.0
    hypercube: bool  # = True
    shift: float  # = 0.0
    scale: float  # = 1.0
    shuffle: bool  # = True
    random_state: int | np.random.RandomState | None  # = None
    # return_X_y: bool  # = True


class MakeMultilabelClassification_Params_TypedDict(t.TypedDict, total=False):
    # sklearn.datasets.make_multilabel_classification kwargs

    n_samples: int  # = 100
    n_features: int  # = 20

    n_classes: int  # = 5
    n_labels: int  # = 2
    length: int  # = 50
    allow_unlabeled: bool  # = True
    sparse: bool  # = False
    return_indicator: t.Literal["dense", "sparse"]  # = "dense"
    return_distributions: bool  # = False
    random_state: int | np.random.RandomState | None  # = None

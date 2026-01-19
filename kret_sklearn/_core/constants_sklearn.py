from .typed_cls_sklean import (
    MakeClassification_Params_TypedDict,
    MakeMultilabelClassification_Params_TypedDict,
    MakeRegression_Params_TypedDict,
)


class SklearnDefaults:
    MAKE_REGRESSION_PARAMS_SKLEARN: MakeRegression_Params_TypedDict = {
        "n_samples": 10_000,
        "n_features": 12,
        "n_informative": 8,
        "n_targets": 1,
        "noise": 0.2,
        "shuffle": True,
        "coef": True,
        "random_state": 0,
    }

    MAKE_CLASSIFICATION_PARAMS_SKLEARN: MakeClassification_Params_TypedDict = {
        "n_samples": 10_000,
        "n_features": 16,
        "n_informative": 10,
        "n_redundant": 3,
        "n_classes": 5,
        "shuffle": False,
        "random_state": 0,
        # "return_X_y": False,
    }

    MAKE_MULTILABEL_CLASSIFICATION_PARAMS_SKLEARN: MakeMultilabelClassification_Params_TypedDict = {
        "n_samples": 10_000,
        "n_features": 16,
        "n_classes": 5,
        "n_labels": 2,
        "allow_unlabeled": False,
        "random_state": 0,
        "return_distributions": True,
    }

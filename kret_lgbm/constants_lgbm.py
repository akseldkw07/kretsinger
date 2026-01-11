from lightgbm.callback import early_stopping, log_evaluation, record_evaluation

from kret_utils.constants_kret import KretConstants

from .typed_cls_lgbm import (
    LGBMRegressor___init___TypedDict,
    LGBMRegressor_Fit_TypedDict,
)


class LGBM_Constants:
    LGBM_MODEL_WEIGHT_DIR = KretConstants.DATA_DIR / "lgbm"


class LGBM_Defaults:
    LGBM_REGRESSOR_DEFAULTS: LGBMRegressor___init___TypedDict = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 7,
        "learning_rate": 0.1,
        "reg_alpha": 1e-3,  # L1 regularization
        "reg_lambda": 1e-3,  # L2 regularization
        "n_jobs": -2,  # everything except one core
        "bagging_fraction": 0.85,
        "feature_fraction": 0.85,
        "min_data_in_leaf": 10,
        # "verbose": 0, # this is handled in sitecustomize.py
    }
    EVAL_RECORDS = {}
    CALLBACK_FIT_LGBM = [early_stopping(stopping_rounds=50), log_evaluation(period=10), record_evaluation(EVAL_RECORDS)]

    LGBM_FIT_DEFAULTS: LGBMRegressor_Fit_TypedDict = {
        "eval_metric": "l2",
        "callbacks": CALLBACK_FIT_LGBM,
    }

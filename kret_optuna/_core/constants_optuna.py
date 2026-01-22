import optuna

from kret_utils._core.constants_kret import KretConstants

from .typed_cls_optuna import Create_study_TypedDict, Study_Optimize_TypedDict


class OptunaDefaults:
    HYPERBAND_PRUNER = optuna.pruners.HyperbandPruner()
    CREATE_STUDY_DEFAULTS: Create_study_TypedDict = {"pruner": HYPERBAND_PRUNER, "load_if_exists": True}

    STUDY_8_HOURS: Study_Optimize_TypedDict = {"n_trials": None, "timeout": 8 * 60 * 60}
    STUDY_100_TRIAL: Study_Optimize_TypedDict = {"n_trials": 100, "timeout": None}
    OPTIM_STUDY_DEF: Study_Optimize_TypedDict = {"n_jobs": -2, "gc_after_trial": True, "show_progress_bar": True}


class OptunaConstants:
    OPTUNA_STUDY_DIR = KretConstants.DATA_DIR / "optuna_studies"
    OPTUNA_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_STORAGE_DB = r"sqlite:////" + str(OPTUNA_STUDY_DIR / "optuna_studies.db")

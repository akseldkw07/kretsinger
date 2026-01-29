from kret_utils._core.constants_kret import KretConstants

from .typed_cls_optuna import Create_study_TypedDict, Study_Optimize_TypedDict


class OptunaConstants:
    OPTUNA_STUDY_DIR = KretConstants.DATA_DIR / "optuna_studies"
    OPTUNA_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_STORAGE_DB = r"sqlite:////" + str(OPTUNA_STUDY_DIR / "optuna_studies.db")


class OptunaDefaults:
    # Lazy — only imports optuna when first accessed
    @classmethod
    @property
    def HYPERBAND_PRUNER(cls):
        if not hasattr(cls, "_HYPERBAND_PRUNER"):
            import optuna

            cls._HYPERBAND_PRUNER = optuna.pruners.HyperbandPruner()
        return cls._HYPERBAND_PRUNER

    @classmethod
    @property
    def CREATE_STUDY_DEFAULTS(cls) -> Create_study_TypedDict:
        return {
            "pruner": cls.HYPERBAND_PRUNER,
            "load_if_exists": True,
            "storage": OptunaConstants.OPTUNA_STORAGE_DB,
        }

    @classmethod
    def study_n_hours(cls, hours: int) -> Study_Optimize_TypedDict:
        return {"n_trials": None, "timeout": hours * 60 * 60}

    @classmethod
    def study_n_trials(cls, n_trials: int) -> Study_Optimize_TypedDict:
        return {"n_trials": n_trials, "timeout": None}

    STUDY_8_HOURS: Study_Optimize_TypedDict = {"n_trials": None, "timeout": 8 * 60 * 60}
    STUDY_100_TRIAL: Study_Optimize_TypedDict = {"n_trials": 100, "timeout": None}
    OPTIM_STUDY_DEF: Study_Optimize_TypedDict = {
        "n_jobs": 1,  # KretConstants.CPU_COUNT - 2, TODO fix, bad errors
        "gc_after_trial": True,
        "show_progress_bar": True,
    }

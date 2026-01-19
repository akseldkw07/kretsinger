from kret_utils._core.constants_kret import KretConstants


class OptunaDefaults: ...


class OptunaConstants:
    OPTUNA_STUDY_DIR = KretConstants.DATA_DIR / "optuna_studies"
    OPTUNA_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_STORAGE_DB = r"sqlite:////" + str(OPTUNA_STUDY_DIR / "optuna_studies.db")

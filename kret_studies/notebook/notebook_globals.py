from __future__ import annotations

from .source_env_vars import source_zsh_env
from .nb_setup import load_dotenv_file
from kret_studies.low_prio.typed_cls import LBGMRegressor__init___TypedDict

# source env variables
load_dotenv_file()
source_zsh_env()


SEED = RANDOM_STATE = 1
LGBM_DEFAULT_PARAMS: LBGMRegressor__init___TypedDict = {
    "boosting_type": "gbdt",
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

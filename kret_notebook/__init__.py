# autoflake: skip_file

from .constants_FINAL import KretConstantsNB as UKS_CONSTANTS, KretDefaultsNB as UKS_DEFAULTS

from .nb_setup import NBSetupUtils


# source env variables
NBSetupUtils.load_dotenv_file()
NBSetupUtils.source_zsh_env()

DATA_DIR = UKS_CONSTANTS.DATA_DIR

# autoflake: skip_file

from .constants_FINAL import KretNotebookPaths as UKS_PATHS, KretNotebookConstants as UKS_CONSTANTS_FINAL

from .nb_setup import NBSetupUtils


# source env variables
NBSetupUtils.load_dotenv_file()
NBSetupUtils.source_zsh_env()

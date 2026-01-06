import os
from pathlib import Path

from kret_utils.constants_kret import KretConstants

from .nb_setup import NBSetupUtils


# source env variables
NBSetupUtils.load_dotenv_file()
NBSetupUtils.source_zsh_env()
DATA_DIR = KretConstants.DATA_DIR

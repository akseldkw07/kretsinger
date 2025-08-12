from __future__ import annotations

from .source_env_vars import source_zsh_env
from .nb_setup import load_dotenv_file

# source env variables
load_dotenv_file()
source_zsh_env()


SEED = RANDOM_STATE = 1

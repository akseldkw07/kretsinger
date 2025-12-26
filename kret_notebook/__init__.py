from __future__ import annotations

import os
from pathlib import Path

from .nb_setup import NBSetupUtils


# source env variables
NBSetupUtils.load_dotenv_file()
NBSetupUtils.source_zsh_env()
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

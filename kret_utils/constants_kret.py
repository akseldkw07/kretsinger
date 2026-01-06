from __future__ import annotations

import os
from pathlib import Path


class KretConstants:
    DATA_DIR = Path(os.getenv("DATA_DIR") or Path(__file__).resolve().parent.parent.parent / "data_kretsinger")

    @classmethod
    def data_dir_dynamic(cls) -> Path:
        return Path(__file__).resolve().parent.parent.parent / "data_kretsinger"

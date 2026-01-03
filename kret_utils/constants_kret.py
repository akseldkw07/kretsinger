from __future__ import annotations

import os
from pathlib import Path


class KretConstants:
    DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

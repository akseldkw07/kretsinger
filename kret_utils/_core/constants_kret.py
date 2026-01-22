import os
from pathlib import Path


class KretConstants:
    DATA_DIR = Path(os.getenv("DATA_DIR") or Path(__file__).resolve().parent.parent.parent / "data_kretsinger")
    CPU_COUNT = os.cpu_count() or 1

    @classmethod
    def data_dir_dynamic(cls) -> Path:
        return Path(__file__).resolve().parent.parent.parent / "data_kretsinger"

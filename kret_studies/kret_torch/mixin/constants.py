import os
import typing as t
from pathlib import Path

import torch

# PATHS
PYTORCH_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent.parent))
MODEL_WEIGHT_DIR = PYTORCH_DIR / "pytorch"

# DEVICE
DEVICE_LITERAL = t.Literal["cuda", "mps", "xpu", "cpu"]  # extend to include "xla", "xpu" if needed


def pick_device() -> DEVICE_LITERAL:
    if torch.cuda.is_available():
        return "cuda"
    # If you plan to use TPUs:
    if torch.backends.mps.is_available():
        return "mps"
    # If using Intel GPUs:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


DEVICE_TORCH_STR: DEVICE_LITERAL = pick_device()
DEVICE = torch.device(DEVICE_TORCH_STR)

"/Users/Akseldkw/coding/Columbia/COMS4776-Data/data"

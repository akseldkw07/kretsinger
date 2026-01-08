import typing as t

import torch

from kret_utils.constants_kret import KretConstants

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


class TorchConstants:
    TORCH_MODEL_WEIGHT_DIR = KretConstants.DATA_DIR / "pytorch_weights"
    DEVICE_TORCH_STR: DEVICE_LITERAL = pick_device()
    DEVICE = torch.device(DEVICE_TORCH_STR)
    HUGGING_FACE_DIR = KretConstants.DATA_DIR / "hugging_face"

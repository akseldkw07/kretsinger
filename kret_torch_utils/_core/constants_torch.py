import typing as t

from kret_decorators.class_property import classproperty
from kret_utils._core.constants_kret import KretConstants

from .typed_cls_torch import DataLoader___init___TypedDict

if t.TYPE_CHECKING:
    import torch
# DEVICE
DEVICE_LITERAL = t.Literal["cuda", "mps", "xpu", "cpu"]  # extend to include "xla", "xpu" if needed


def pick_device() -> DEVICE_LITERAL:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    # If you plan to use TPUs:
    if torch.backends.mps.is_available():
        return "mps"
    # If using Intel GPUs:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


class TorchDefaults:
    DATA_LOADER_INIT: DataLoader___init___TypedDict = {
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }


class TorchConstants:
    TORCH_MODEL_WEIGHT_DIR = KretConstants.DATA_DIR / "pytorch_weights"
    HUGGING_FACE_DIR = KretConstants.DATA_DIR / "hugging_face"
    KAGGLEHUB_DIR = (
        KretConstants.DATA_DIR / "kagglehub"
    )  # NOTE this doesn't actually work - must set OS env variable KAGGLEHUB_CACHE
    TORCH_MODEL_VIZ_DIR = KretConstants.DATA_DIR / "pytorch_viz"

    # Lazy — only imports torch when first accessed
    @classproperty
    def DEVICE_TORCH_STR(cls) -> DEVICE_LITERAL:
        if not hasattr(cls, "_DEVICE_TORCH_STR"):
            cls._DEVICE_TORCH_STR = pick_device()
        return cls._DEVICE_TORCH_STR

    @classproperty
    def DEVICE(cls) -> "torch.device":
        if not hasattr(cls, "_DEVICE"):
            import torch

            cls._DEVICE = torch.device(cls.DEVICE_TORCH_STR)
        return cls._DEVICE

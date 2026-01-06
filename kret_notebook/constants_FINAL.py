from kret_lightning.constants_lightning import LightningConstants
from kret_torch_utils.constants_torch import TorchConstants
from kret_utils.constants_kret import KretConstants
from kret_wandb.constants_wandb import WandbConstants


class KretNotebookPaths:
    DATA_DIR = KretConstants.DATA_DIR
    LIGHTNING_LOG_DIR = LightningConstants.LIGHTNING_LOG_DIR
    TORCH_MODEL_WEIGHT_DIR = TorchConstants.TORCH_MODEL_WEIGHT_DIR
    WANDB_LOG_DIR = WandbConstants.WANDB_LOG_DIR


class KretNotebookConstants:
    DEVICE_TORCH_STR = TorchConstants.DEVICE_TORCH_STR
    DEVICE_TORCH = TorchConstants.DEVICE

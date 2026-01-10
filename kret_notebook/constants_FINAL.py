from kret_lightning.constants_lightning import LightningConstants
from kret_matplotlib.constants_mpl import MPLConstants
from kret_polars.constants_polars import PolarsConstants
from kret_rosetta.constants_rosetta import RosettaConstants
from kret_torch_utils.constants_torch import TorchConstants
from kret_utils.constants_kret import KretConstants
from kret_wandb.constants_wandb import WandbConstants


class KretNotebookPaths(
    LightningConstants,
    TorchConstants,
    KretConstants,
    WandbConstants,
    PolarsConstants,
    RosettaConstants,
    MPLConstants,
): ...


class KretNotebookConstants:
    DEVICE_TORCH_STR = TorchConstants.DEVICE_TORCH_STR
    DEVICE_TORCH = TorchConstants.DEVICE

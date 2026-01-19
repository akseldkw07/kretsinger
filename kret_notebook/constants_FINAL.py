from kret_lgbm._core.constants_lgbm import LGBM_Constants, LGBM_Defaults
from kret_lightning._core.constants_lightning import LightningConstants
from kret_matplotlib._core.constants_mpl import MPLConstants, MPLDefaults
from kret_np_pd._core.constants_np_pd import NP_PD_Defaults
from kret_polars._core.constants_polars import PolarsConstants
from kret_rosetta._core.constants_rosetta import RosettaConstants
from kret_sklearn._core.constants_sklearn import SklearnDefaults
from kret_torch_utils._core.constants_torch import TorchConstants
from kret_tqdm._core.constants_tqdm import TQDMConstants, TQDMDefaults
from kret_utils._core.constants_kret import KretConstants
from kret_wandb._core.constants_wandb import WandbConstants


class KretConstantsNB(
    LightningConstants,
    TorchConstants,
    KretConstants,
    WandbConstants,
    PolarsConstants,
    RosettaConstants,
    MPLConstants,
    LGBM_Constants,
    TQDMConstants,
): ...


class KretDefaultsNB(
    MPLDefaults,
    SklearnDefaults,
    LGBM_Defaults,
    NP_PD_Defaults,
    TQDMDefaults,
): ...

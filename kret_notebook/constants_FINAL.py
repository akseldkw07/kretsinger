from kret_lgbm.constants_lgbm import LGBM_Constants, LGBM_Defaults
from kret_lightning.constants_lightning import LightningConstants
from kret_matplotlib.constants_mpl import MPLConstants, MPLDefaults
from kret_np_pd.constants_np_pd import NP_PD_Defaults
from kret_polars.constants_polars import PolarsConstants
from kret_rosetta.constants_rosetta import RosettaConstants
from kret_sklearn.constants_sklearn import SklearnDefaults
from kret_torch_utils.constants_torch import TorchConstants
from kret_tqdm.constants_tqdm import TQDMConstants, TQDMDefaults
from kret_utils.constants_kret import KretConstants
from kret_wandb.constants_wandb import WandbConstants


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

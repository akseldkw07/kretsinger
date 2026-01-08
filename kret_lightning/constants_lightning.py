from kret_utils.constants_kret import KretConstants
import typing as t

STAGE_LITERAL = t.Literal["fit", "validate", "predict", "test"]


class LightningConstants:
    LIGHTNING_LOG_DIR = KretConstants.DATA_DIR / "lightning_logs"

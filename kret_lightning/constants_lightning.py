import typing as t

from kret_utils.constants_kret import KretConstants

STAGE_LITERAL = t.Literal["fit", "validate", "predict", "test"]


class LightningConstants:
    LIGHTNING_LOG_DIR = KretConstants.DATA_DIR / "lightning_logs"

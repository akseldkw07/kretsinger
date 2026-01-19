import re
import typing as t

from kret_utils._core.constants_kret import KretConstants

STAGE_LITERAL = t.Literal["fit", "validate", "predict", "test"]


class LightningConstants:
    LIGHTNING_LOG_DIR = KretConstants.DATA_DIR / "lightning_logs"


class LightningDefaults:
    # Compiled once at import time (avoid re-compiling inside properties / callbacks).
    # Matches both:
    #   - best-03-0.12.ckpt  / best_03_0.12.ckpt
    #   - best-epoch=03-val_loss=0.12.ckpt (Lightning inserts field names)
    CKPT_BEST_PATTERN = re.compile(
        r"best"  # prefix
        r"[-_]"  # separator
        r"(?:epoch=)?(?P<epoch>\d+)"  # epoch or epoch=NN
        r"[-_]"  # separator
        r"(?:val_loss=)?(?P<loss>-?\d+(?:\.\d+)?)"  # loss or val_loss=NN.NN
        r"(?:\D|$)"  # tolerate suffix like .ckpt
    )
    CKPT_FILENAME = "best-{epoch:02d}-{val_loss:.2f}"  # NOTE don't change without update CKPT_BEST_PATTERN

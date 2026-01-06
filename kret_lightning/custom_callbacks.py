from __future__ import annotations

from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint


class CallbackConfig:
    CKPT_VAL_LOSS = ModelCheckpoint(
        monitor="val_loss",  # metric name you log
        mode="min",  # "min" for loss, "max" for accuracy
        save_top_k=1,  # keep only the best
        save_last=True,  # optional but very useful
    )

    CALLBACKS_DEFAULT: list[Callback] = [CKPT_VAL_LOSS]

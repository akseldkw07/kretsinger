from datetime import timedelta
from pathlib import Path
from typing import Literal, TypedDict

from lightning import Callback, LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint

from kret_lightning.abc_lightning import ABCLM


class CallbackConfig:
    @classmethod
    def trainer_dynamic_defaults(cls, nn: ABCLM, datamodule: LightningDataModule):

        CKPT_VAL_LOSS = ModelCheckpoint(
            # dirpath=nn.ckpt_path,
            filename="best",
            verbose=True,
            save_weights_only=False,
            monitor="val_loss",  # metric name you log
            mode="min",  # "min" for loss, "max" for accuracy
            save_top_k=1,  # keep only the best
            save_last=True,  # optional but very useful
        )
        ret: list[Callback] = [CKPT_VAL_LOSS]
        return ret


# region TypedDicts
class ModelCheckpoint___init___TypedDict(TypedDict, total=False):
    dirpath: str | Path | None  # = None
    filename: str | None  # = None
    monitor: str | None  # = None
    verbose: bool  # = False
    save_last: bool | Literal["link"] | None  # = None
    save_top_k: int  # = 1
    save_on_exception: bool  # = False
    save_weights_only: bool  # = False
    mode: str  # = 'min'
    auto_insert_metric_name: bool  # = True
    every_n_train_steps: int | None  # = None
    train_time_interval: timedelta | None  # = None
    every_n_epochs: int | None  # = None
    save_on_train_epoch_end: bool | None  # = None
    enable_version_counter: bool  # = True


# endregion

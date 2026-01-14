import typing as t

from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .abc_lightning import ABCLM, HPDict
from .constants_lightning import LightningDefaults


class CallbackMixin(ABCLM):
    @property
    def model_checkpoint(self):

        return ModelCheckpoint(
            dirpath=self.ckpt_path,
            filename=LightningDefaults.CKPT_FILENAME,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
            save_weights_only=False,
        )

    @property
    def early_stopping(self):
        hp = t.cast(HPDict, self.hparams_initial)
        return EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=hp.patience, verbose=True, mode="min")

    def configure_callbacks(self) -> t.Sequence[Callback] | Callback:
        ret: list[Callback] = [self.model_checkpoint, self.early_stopping]
        return ret

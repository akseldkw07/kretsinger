import re
import typing as t

from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .abc_lightning import ABCLM, HPDict


class CallbackMixin(ABCLM):
    @property
    def model_checkpoint(self):

        ABCLM._ckpt_pattern = re.compile(  # TODO improve this
            r"best(?:[-_](?:epoch=)?(?P<epoch>\d+))"  # epoch or epoch=NN
            r"(?:[-_](?:val_loss=)?(?P<loss>-?\d+(?:\.\d+)?))"  # loss or val_loss=NN.NN
            r"(?:\D|$)"  # tolerate suffix like .ckpt
        )

        return ModelCheckpoint(
            dirpath=self.ckpt_path,
            filename="best-{epoch:02d}-{val_loss:.2f}",  # NOTE don't change without update self.ckpt_pattern
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

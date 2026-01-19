import typing as t

from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from ._core.constants_lightning import LightningDefaults
from .abc_lightning import ABCLM, HPDict


class CallbackMixin(ABCLM):
    _ckpt_pattern_tuple = LightningDefaults.CKPT_TUPLE
    _sweep_mode: bool = False  # Set to True during sweeps, e.g. optuna

    def configure_callbacks(self) -> t.Sequence[Callback] | Callback:
        callbacks: list[Callback] = [self.early_stopping]
        if self.enable_checkpointing:
            callbacks.append(self.model_checkpoint)
        return callbacks

    @property
    def enable_checkpointing(self) -> bool:
        """
        Don't enable checkpointing during hyperparameter sweeps
        (to avoid saving too many checkpoints)
        """
        return not self._sweep_mode

    @property
    def model_checkpoint(self):
        return ModelCheckpoint(
            dirpath=self.ckpt_path,
            filename=self._ckpt_pattern_tuple.filename,
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

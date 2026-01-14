import typing as t
from pathlib import Path

import torch
import torch.nn as nn
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE

from kret_decorators.post_init import post_init
from kret_lightning.constants_lightning import LightningConstants  # type: ignore
from kret_lightning.utils_lightning import LightningModuleAssert
from kret_torch_utils.constants_torch import TorchConstants
from kret_torch_utils.priors import PriorLosses
from kret_utils.filename_utils import FileSearchUtils

from .abc_lightning import ABCLM, HPDict, SaveLoadLoggingDict


@post_init
class BaseLightningNN(ABCLM):
    """
    TODO

    1- finalize naming / filepath conventions
        how does load_from_checkpoint load hparams? how to save them?
    2- work on training loop defaults
    3- re-implement beijing nn with this base class
    4- add callbacks to assert correct behavior (tensor size? early stopping?)
    """

    _criterion: nn.Module
    ignore_hparams: tuple[str, ...] = ()

    # region Init
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.5,
        stepsize: int = 12,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        patience: int = 10,  # passed to class_callbacks.CallbackMixin.early_stopping
        **kwargs,
    ):
        """
        NOTE: don't call .to(device) here; Lightning handles device placement
        """
        super().__init__()
        print(f"Saving hparams, ignoring {self.ignore_hparams}")
        self.save_hyperparameters(ignore=self.ignore_hparams)

    def __post_init__(self) -> None:
        # super().__post_init__()
        LightningModuleAssert.initialization_check(self)
        self.to(TorchConstants.DEVICE_TORCH_STR)

    @property
    def criterion(self):
        return self._criterion.to(self.device)

    def configure_optimizers(self, *args, **kwargs):
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        """
        hp = t.cast(HPDict, self.hparams_initial)

        optimizer = torch.optim.Adam(self.parameters(), lr=hp.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.stepsize, gamma=hp.gamma)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # endregion

    # region Naming & Saving

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}__{self.hparams_str}"

    @property
    def root_dir(self) -> Path:
        val = self._load_dir_override if self._load_dir_override is not None else self._root_dir
        return Path(val)

    @property
    def hparams_str(self) -> str:
        hp = HPDict(self.hparams_initial)
        return hp.as_str_safe()

    @property
    def save_load_logging_dict(self) -> SaveLoadLoggingDict:
        ret: SaveLoadLoggingDict = {"save_dir": self.root_dir, "name": self.__class__.__name__, "version": self.version}
        return ret

    @property
    def ckpt_path(self) -> Path:
        return self.root_dir / self.__class__.__name__ / self.version

    @classmethod
    def ckpt_file_name(cls) -> Path:
        """Return the best checkpoint file by parsing val_loss from filenames.

        Expected filename format:
            best-{epoch:02d}-{val_loss:.2f}.ckpt
        Example:
            best-03-0.12.ckpt

        Searches both:
            <root>/<ModelName>/<version>/
            <root>/<ModelName>/<version>/checkpoints/
        """

        base_folder = Path(cls._root_dir) / cls.__name__ / cls.version
        folders = [base_folder, base_folder / "checkpoints"]

        candidates: list[tuple[float, int, Path]] = []
        matching_files = FileSearchUtils.find_matching_files(folders, cls._ckpt_pattern)

        for p in matching_files:
            name = p.name
            if "best-" not in name and "best_" not in name:
                continue

            m = cls._ckpt_pattern.search(name)
            if m is None:
                continue

            try:
                loss = float(m.group("loss"))
                epoch = int(m.group("epoch"))
            except ValueError:
                continue

            # store (-epoch) so "lowest" sorts to highest epoch when loss ties
            candidates.append((loss, -epoch, p))

        if not candidates:
            raise FileNotFoundError(
                f"No 'best' checkpoint with parsable loss found in {base_folder} or {base_folder / 'checkpoints'}"
            )

        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    # endregion

    # region Loss

    def forward(self, *args, **kwargs) -> t.Any:
        """
        Note - don't call .to(device) here; bad for memory
        """
        raise NotImplementedError("Subclasses must implement forward method.")

    def get_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss. If in training mode, includes prior regularization.
        """
        # Base loss (cross-entropy, MSE, etc.)
        base_loss = self.criterion(outputs, labels)

        # Prior regularization
        prior_loss = torch.tensor(0.0, device=self.device)
        if self.training:
            prior_loss = self.compute_prior_loss(outputs=outputs)

        return base_loss + prior_loss

    def compute_prior_loss(
        self, inputs: torch.Tensor | None = None, outputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Inputs and outputs are provided in case of more complex prior computations,
        such as entropic priors that depend on model outputs.
        """
        hp = t.cast(HPDict, self.hparams_initial)
        tot = torch.tensor(0.0, device=next(self.parameters()).device)

        # Check that penalty is non-zero before computing to save time
        if hp.l1_penalty > 0.0:
            tot += hp.l1_penalty * PriorLosses.compute_l1_loss(self)
        if hp.l2_penalty > 0.0:
            tot += hp.l2_penalty * PriorLosses.compute_l2_loss(self)
        return tot

    # endregion
    # region Training / Validation Steps
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """

        loss = ...
        return loss
        """
        x, y = batch
        outputs = self(x)
        loss = self.get_loss(outputs, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        val_loss = ...
        self.log('val_loss', val_loss)
        """
        x, y = batch
        outputs = self(x)
        val_loss = self.get_loss(outputs, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_extra_metrics(outputs, y, stage="validate")

    # endregion

    @classmethod
    def load_from_checkpoint_custom(
        cls,
        checkpoint_path: str | Path | t.IO | None = None,
        map_location: _MAP_LOCATION_TYPE = TorchConstants.DEVICE_TORCH_STR,
        hparams_file: str | Path | None = None,
        strict: bool | None = True,
        weights_only: bool | None = False,
        **kwargs: t.Any,
    ) -> t.Self:
        """
        Thin Wrapper around LightningModule.load_from_checkpoint to set default checkpoint, location, strict, weights_only
        """
        if checkpoint_path is None:
            checkpoint_path = cls.ckpt_file_name()
        return cls.load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, weights_only, **kwargs)

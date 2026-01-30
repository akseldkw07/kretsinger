import typing as t
from pathlib import Path

import torch
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from kret_decorators.class_property import classproperty
from kret_decorators.post_init import post_init
from kret_lightning._core.constants_lightning import LightningConstants  # type: ignore
from kret_lightning.utils_lightning import LightningModuleAssert
from kret_torch_utils._core.constants_torch import TorchConstants
from kret_torch_utils.priors import PriorLosses

from .abc_lightning import ABCLM, HPDict, SaveLoadLoggingDict
from .deprecated_funcs import DeprecatedLigthningRoutes


@post_init
class BaseLightningNN(DeprecatedLigthningRoutes, ABCLM):
    """
    TODO

    1- finalize naming / filepath conventions
        how does load_from_checkpoint load hparams? how to save them?
        IDEA: somehow hash model + hparams to get unique id for each model config?
    2- work on training loop defaults
    3- re-implement beijing nn with this base class
    4- add callbacks to assert correct behavior (tensor size? early stopping?)
    """

    ignore_hparams = []

    # region Init
    def __init__(
        self,
        lr: float = 1e-3,
        warmup_step_frac: float = 0.1,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        # gamma: float = 0.5,
        # stepsize: int = 12,
        patience: int = 10,  # passed to class_callbacks.CallbackMixin.early_stopping
        **kwargs,
    ):
        """
        NOTE: don't call .to(device) here; Lightning handles device placement
        """
        super().__init__()
        # Commented out for now - handled in child class (initialization_check asserts that this happened correctly)
        print(f"Saving hparams, ignoring {self.ignore_hparams}")
        self.save_hyperparameters(ignore=self.ignore_hparams)

    def __post_init__(self) -> None:
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

        # NOTE apparently Andrej Karpathy recommends eps=1e-10 for stability
        optimizer = torch.optim.Adam(self.parameters(), lr=hp.lr, eps=1e-10)

        # Learning rate scheduler: 10% warmup, then cosine annealing
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * hp.warmup_step_frac)

        warmup_sch = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_sch = CosineAnnealingLR(optimizer, T_max=int(total_steps - warmup_steps), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_steps])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    # endregion

    # region Naming & Saving

    @property
    def name_instance(self) -> str:
        return f"{self.classname}__{self.hparams_str}"

    @property
    def hparams_str(self) -> str:
        hp = HPDict(self.hparams_initial)
        return hp.as_str_safe()

    @classproperty
    def classname(cls):
        # cls is the class when called on class, but the instance when called on instance
        return cls.__name__ if isinstance(cls, type) else type(cls).__name__

    @classproperty
    def root_dir(cls) -> Path:
        val = cls._load_dir_override if cls._load_dir_override is not None else cls._root_dir
        return Path(val)

    @classproperty
    def save_load_logging_dict(cls) -> SaveLoadLoggingDict:
        ret: SaveLoadLoggingDict = {"save_dir": cls.root_dir, "name_cls": cls.classname, "version": cls.version}
        return ret

    @classproperty
    def ckpt_path(cls) -> Path:
        return cls.root_dir / cls.classname / cls.version

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

    def _compute_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared computation for training/validation steps.

        Returns:
            (outputs, targets, loss) tuple
        """
        x, y = batch
        outputs = self(x)
        loss = self.get_loss(outputs, y)
        return outputs, y, loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Minimal training step - just compute and return loss.
        Override in MetricMixin for logging.
        """
        _, _, loss = self._compute_step(batch)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Minimal validation step - compute and log val_loss for checkpointing.
        Override in MetricMixin for additional metrics.
        """
        _, _, val_loss = self._compute_step(batch)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Minimal test step - compute and log test_loss.
        Override in MetricMixin for additional metrics.
        """
        _, _, test_loss = self._compute_step(batch)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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
            model_ckpt = cls.create_model_saver(create_new_on_fail=False)
            checkpoint_path = model_ckpt.best_checkpoints[0][2]
            hparams_file = model_ckpt._yaml_path_for(checkpoint_path)
            print(f"Loading best checkpoint from {str(checkpoint_path)} with hparams from {hparams_file}")
        return cls.load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, weights_only, **kwargs)

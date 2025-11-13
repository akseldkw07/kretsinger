from __future__ import annotations
import json
import pprint
import logging
import pathlib
import typing as t

import torch
import torch.nn as nn

from kret_studies.kret_torch.abc_nn import (
    ABCNN,
    ModelPathDict,
    HyperParamDict,
    HyperParamTotalDict,
    ModelStateDict,
    FullStateDict,
)
from kret_studies.kret_torch.constants import DEVICE_TORCH_STR, MODEL_WEIGHT_DIR

LOAD_LTRL = t.Literal["assert", "try", "fresh"]


# Default training state
DEFAULT_HYPER_PARAMS = HyperParamTotalDict(lr=1e-3, gamma=0.1, stepsize=7, patience=25, improvement_tol=1e-4)
DEFAULT_MODEL_STATE = ModelStateDict(best_loss=float("inf"), epochs_trained=0)


class BaseNN(ABCNN, nn.Module):
    version: str = "v000"
    model: nn.Module  # nn.Sequential or other nn.Module TODO define in subclass
    optimizer: torch.optim.Optimizer  # NOTE: NOT SET in __init__
    scheduler: torch.optim.lr_scheduler.LRScheduler  # NOTE: NOT SET in __init__
    device: t.Literal["cuda", "mps", "xpu", "cpu"]  # DEVICE_TORCH_STR
    _criterion: nn.Module  # TODO define

    hparams: HyperParamTotalDict

    _load_weights_act: t.Literal["assert", "try", "fresh"]
    _post_init_done: bool = False
    model_state: ModelStateDict
    _log: bool
    # region INHERITED METHODS
    """INHERITED METHODS"""

    def __init__(self, log: bool = True, **hparams: t.Unpack[HyperParamDict]):
        super().__init__()

        # Initialize logging and model components
        self.logger = logging.getLogger(self.name())
        self.device = DEVICE_TORCH_STR

        # Initialize default Hyperparameters (may be overridden by load)
        hp = DEFAULT_HYPER_PARAMS.copy()
        hp.update(hparams)
        self.hparams = hp

        # Initialize default training state (may be overridden by load)
        self.model_state = DEFAULT_MODEL_STATE.copy()
        self._log = log

        self.to(self.device)

    # SAVING
    @property
    def root_dir(self) -> pathlib.Path:
        return MODEL_WEIGHT_DIR / f"{self.name()}"

    @property
    def model_paths(self):
        return ModelPathDict(
            {
                "weight_path": self.root_dir / "weights.pt",
                "model_path": self.root_dir / "model.txt",
                "state_path": self.root_dir / "state.json",
            }
        )

    @property
    def FullStateDict(self):
        return FullStateDict({"state": self.model_state, "hparams": self.hparams})

    @property
    def FullStateDictDisplay(self):
        ret = self.FullStateDict.copy()
        ret["state"]["best_loss"] = round(ret["state"]["best_loss"], 3)
        return ret

    def save_weights(self, increment_version: bool = False):
        """
        Save the model weights, summary, and training state to a file.
        """
        if increment_version:
            version_num = int(self.version.strip("v")) + 1
            self.version = f"v{version_num:03d}"
            self.logger.warning(f"Incremented model version to {self.version}.")

        if self._log:
            self.logger.info(
                f"Saving model weights to {self.root_dir}, view model summary at {self.model_paths['model_path']}"
            )

        self.root_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), self.model_paths["weight_path"])
        with open(self.model_paths["model_path"], "w", encoding="utf-8") as f:
            f.write(self._summary())

        with open(self.model_paths["state_path"], "w", encoding="utf-8") as f:
            json.dump(self.FullStateDict, f)

    def load_weights(self, load_weights: LOAD_LTRL = "try", override_load_hp: bool = False):
        """
        Load the model weights, summary, and training state from a file.

        Args:
            load_weights: Controls loading behavior - "assert" (must succeed), "try" (silent fail), or "fresh" (skip loading).
            override_load_hp: If True, replace current hyperparameters with loaded values from the state file.
        """
        if load_weights == "assert":
            self._load_weights(override_load_hp)
        elif load_weights == "try":
            try:
                self._load_weights(override_load_hp)
            except Exception as ex:
                self.logger.error(f"Failed to load state from {self.root_dir}: {ex}. Continuing with fresh weights.")
                self.model_state = DEFAULT_MODEL_STATE.copy()
                if override_load_hp:
                    self.hparams = DEFAULT_HYPER_PARAMS.copy()
        elif load_weights == "fresh":
            self.model_state = DEFAULT_MODEL_STATE.copy()
        else:
            raise ValueError("Invalid load_weights value.")

    def _load_weights(self, override_load_hp):
        """
        Attempt to load model weights and state from specified path.
        """
        self.load_state_dict(torch.load(self.model_paths["weight_path"], map_location=self.device))

        with open(self.model_paths["state_path"], encoding="utf-8") as f:
            full_state: FullStateDict = json.load(f)

        self.model_state = full_state["state"]
        if override_load_hp:
            self.hparams = full_state["hparams"]
        self.logger.info(f"Loaded model weights and state from {self.root_dir}.")

    # NAMING
    @classmethod
    def name(cls, include_version: bool = True) -> str:
        """
        Returns a string identifier combining the class name and version.
        """
        version = getattr(cls, "version", "v000")
        if include_version:
            return f"{cls.__name__}_{version}"
        return cls.__name__

    def _summary(self):
        """
        Prints a string summarizing training progress and best metrics.
        """
        name = self.name()
        model = self.__str__()
        return f"Model: {name=}\n\n{model}\n\n{pprint.pformat(self.FullStateDictDisplay)}\n"

    def summary(self):
        print(self._summary())

    # endregion
    # region LIKELY REDEFINED IN SUBCLASS
    """LIKELY REDEFINED IN SUBCLASS"""

    @property
    def criterion(self):
        return self._criterion.to(self.device)

    def post_init(self, load_weights: LOAD_LTRL = "try", override_load_hp: bool = False):
        """
        Specify actions that must be taken after the model architecture is defined.

        1) Resetting the optimizer
        2) Loading saved weights
        3) Restoring model performance
        """
        try:
            self.model.to(self.device)
        except AttributeError as ex:
            raise RuntimeError(
                "Model architecture must be defined using set_model() before calling post_init()."
            ) from ex
        self._reset_optimizer()

        self._load_weights_act = load_weights
        self.load_weights(load_weights, override_load_hp)

        self._post_init_done = True
        self.logger.info("Full State:")
        pprint.pprint(self.FullStateDictDisplay)

    def _reset_optimizer(self) -> None:
        """(Re)create optimizer and scheduler now that parameters exist."""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.hparams["stepsize"], gamma=self.hparams["gamma"]
        )

    def forward(self, x: torch.Tensor):
        h: torch.Tensor = self.model(x)
        return h

    def get_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        return self.criterion(outputs, labels)

    # endregion
    # region NOT IMPLEMENTED
    """NOT IMPLEMENTED"""

    def set_model(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement set_model().")

    def train_model(self, epochs: int, **kwargs) -> None:
        """
        Train the model using the provided data loaders and optimizer.
        Implements early stopping: if neither val_loss nor val accuracies improve by
        `improvement_tol` for `patience` consecutive epochs, stop training.
        """
        if not self._post_init_done:
            raise RuntimeError("post_init must be called before training the model.")

        self.device
        self.model_state["epochs_trained"] + epochs

        raise NotImplementedError("Subclasses must implement train_model().")

    def evaluate(self, **kwargs) -> float:
        """
        Evaluate the model on a validation set.

        Returns:
            tuple: (validation loss, validation accuracy (subclass), validation accuracy (superclass))
        """
        # raise NotImplementedError("Subclasses must implement evaluate().")
        self.eval()
        self.device

        raise NotImplementedError("Subclasses must implement evaluate().")

    # endregion

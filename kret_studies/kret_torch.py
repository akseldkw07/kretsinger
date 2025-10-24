import math
from collections.abc import Callable
from copy import deepcopy
from functools import cache
from math import ceil, log

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

LossSpec = str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
from kret_studies.helpers.float_utils import notable_number
from kret_studies.low_prio.typed_cls import TorchTrainResult

if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"
DEVICE = torch.device(_DEVICE)


@cache
def _exp_decay(required_len: int, initial_epsilon: float = 0.95, half_life: float = 1000, min_value: float = 0.01):
    t = np.arange(required_len)
    decay = np.exp(-np.log(2.0) * t / half_life)  # exp with half-life semantics
    arr: NDArray[np.float32] = np.maximum(min_value, initial_epsilon * decay).astype(np.float32)
    return arr


def exp_decay(episode: int, initial_epsilon: float = 0.95, half_life: float = 1000.0, min_value: float = 0.01):
    eff_episode = 2 ** (ceil(log(episode + 1, 2)))
    arr = _exp_decay(eff_episode, initial_epsilon, half_life, min_value)

    return float(arr[episode])


def train_regression(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    loss: LossSpec = "mse",  # "mse", "sse", or a callable
    target_loss: float = 5e-3,
    max_epochs: int = 20_000,  # -1 for no cap
    patience: int = 500,  # 0 disables early stopping
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,  # e.g. torch.optim.lr_scheduler.*
    clip_grad_norm: float | None = None,
    improvement_tol: float = 1e-6,  # how much lower to count as "improved"
    verbose: bool = True,
):
    """
    Trains until one of: loss <= target_loss, no improvement for `patience` epochs,
    or `max_epochs` reached. Restores the best weights before returning.

    Returns:
        {
          "best_loss": float,
          "epochs_run": int,
          "history": list[float],
          "stopped_reason": str
        }
    """
    # Build loss function
    if isinstance(loss, str):
        if loss.lower() == "mse":
            loss_fn = nn.MSELoss(reduction="mean")
        elif loss.lower() == "sse":
            loss_fn = nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Unknown loss spec '{loss}'. Use 'mse', 'sse', or a callable.")
    else:
        loss_fn = loss  # custom callable

    model.train()
    best_loss = math.inf
    best_state = deepcopy(model.state_dict())
    epochs_no_improve = 0
    # history: list[float] = []
    epoch = 0
    stopped_reason = "max_epochs_reached"  # default; may be overwritten
    y_hat: torch.Tensor = model(x)

    def should_continue(e: int) -> bool:
        return (max_epochs == -1) or (e < max_epochs)

    while should_continue(epoch):
        optimizer.zero_grad(set_to_none=True)

        y_hat = model(x)
        L = loss_fn(y_hat, y)

        L.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        current_loss = float(L.detach().item())
        # history.append(current_loss)
        if verbose and notable_number(epoch):
            print(f"Epoch {epoch:06d} | Loss = {current_loss:.6f}")

        # Improvement tracking
        if current_loss < best_loss - improvement_tol:
            best_loss = current_loss
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Stopping conditions
        if current_loss <= target_loss:
            stopped_reason = "target_loss_reached"
            break
        if patience > 0 and epochs_no_improve >= patience:
            stopped_reason = "early_stopping_no_improvement"
            break

        epoch += 1

    # Restore best model weights
    model.load_state_dict(best_state)

    return TorchTrainResult(
        best_loss=best_loss, epochs_run=epoch + 1, stopped_reason=stopped_reason, history=[], y_hat=y_hat
    )

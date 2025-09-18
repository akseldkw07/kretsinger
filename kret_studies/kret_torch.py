import torch
import math
from copy import deepcopy
from typing import Union
from collections.abc import Callable

LossSpec = Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]


def train_until(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    loss: LossSpec = "mse",  # "mse", "sse", or a callable
    target_loss: float = 1e-2,
    max_epochs: int = 10_000,  # -1 for no cap
    patience: int = 500,  # 0 disables early stopping
    scheduler: object | None = None,  # e.g. torch.optim.lr_scheduler.*
    clip_grad_norm: float | None = None,
    improvement_tol: float = 1e-12,  # how much lower to count as "improved"
    verbose: bool = True,
) -> dict[str, object]:
    """
    Trains until one of: loss <= target_loss, no improvement for `patience` epochs,
    or `max_epochs` reached. Restores the best weights before returning.

    Returns:
        {
          "best_loss": float,
          "epochs_run": int,
          "history": List[float],
          "stopped_reason": str
        }
    """
    # Build loss function
    if isinstance(loss, str):
        if loss.lower() == "mse":
            loss_fn = torch.nn.MSELoss(reduction="mean")
        elif loss.lower() == "sse":
            loss_fn = torch.nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Unknown loss spec '{loss}'. Use 'mse', 'sse', or a callable.")
    else:
        loss_fn = loss  # custom callable

    model.train()
    best_loss = math.inf
    best_state = deepcopy(model.state_dict())
    epochs_no_improve = 0
    history: list[float] = []
    epoch = 0
    stopped_reason = "max_epochs_reached"  # default; may be overwritten

    def should_continue(e: int) -> bool:
        return (max_epochs == -1) or (e < max_epochs)

    while should_continue(epoch):
        optimizer.zero_grad(set_to_none=True)

        y_hat: torch.Tensor = model(x)
        L: torch.Tensor = loss_fn(y_hat, y)

        L.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        current_loss = float(L.detach().item())
        history.append(current_loss)
        if verbose and (epoch % 100 == 0 or epoch < 10):
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

    return {
        "best_loss": best_loss,
        "epochs_run": epoch + 1 if len(history) > 0 else 0,
        "history": history,
        "stopped_reason": stopped_reason,
    }

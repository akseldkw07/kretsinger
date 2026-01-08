import torch
import torch.nn as nn


class PriorLosses:
    @classmethod
    def categorical_gnm_entropy(cls, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=-1))

    @classmethod
    def compute_l1_loss(cls, model: nn.Module) -> torch.Tensor:
        l1_loss = sum(
            (torch.abs(p).sum() for p in model.parameters()), torch.tensor(0.0, device=next(model.parameters()).device)
        )

        return l1_loss

    @classmethod
    def compute_l2_loss(cls, model: nn.Module) -> torch.Tensor:
        l2_loss = sum(
            (p.pow(2).sum() for p in model.parameters()), torch.tensor(0.0, device=next(model.parameters()).device)
        )
        return l2_loss

    @classmethod
    def compute_horseshoe_loss(cls, model: nn.Module, tau: float = 0.1, epsilon: float = 1e-8) -> torch.Tensor:
        # Keep your existing implementation if you already have one; placeholder here would be wrong.
        # This function should already exist in your repo.
        raise NotImplementedError("compute_horseshoe_loss should use your project implementation.")

    @classmethod
    def smoothness_prior_hutchinson(cls, model: nn.Module, x: torch.Tensor, lam: float, n_samples: int = 1):
        """
        Approximates the sum of squared second derivatives using Hutchinson's trick.

        Ngl, idk what this does fully, but the version that derives the full gradients
        is too memory intesive. Unclear if this is doing the right things tho
        """
        x = x.clone().detach().requires_grad_(True)
        B = x.shape[0]

        y = model(x).squeeze()  # shape [B]

        prior = torch.tensor(0.0, device=next(model.parameters()).device)
        for _ in range(n_samples):
            # Random vector v same shape as x
            v = torch.randint_like(x, low=0, high=2) * 2 - 1  # Rademacher ±1

            # First derivative
            grads = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[
                0
            ]  # shape [B, C, H, W]

            # Hessian-vector product
            Hv = torch.autograd.grad(grads, x, grad_outputs=v, retain_graph=True, create_graph=False)[0]

            prior += (Hv**2).sum()

        prior = (lam / (2 * B * n_samples)) * prior
        return prior

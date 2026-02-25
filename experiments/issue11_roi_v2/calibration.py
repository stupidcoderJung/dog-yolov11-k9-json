from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _to_prob_tensor(probabilities: torch.Tensor | list[float]) -> torch.Tensor:
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().to(dtype=torch.float32)
    else:
        probs = torch.tensor(probabilities, dtype=torch.float32)
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    return probs.clamp(1e-6, 1.0 - 1e-6)


def _to_target_tensor(targets: torch.Tensor | list[int] | list[bool]) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        out = targets.detach().to(dtype=torch.float32)
    else:
        out = torch.tensor(targets, dtype=torch.float32)
    if out.ndim != 1:
        out = out.reshape(-1)
    return out.clamp(0.0, 1.0)


def apply_temperature_to_probability(
    probabilities: torch.Tensor | list[float],
    temperature: float,
) -> torch.Tensor:
    """
    Apply temperature scaling to probabilities in logit space.
    """
    probs = _to_prob_tensor(probabilities)
    temp = max(float(temperature), 1e-3)
    logits = torch.logit(probs)
    return torch.sigmoid(logits / temp)


def fit_binary_temperature(
    probabilities: torch.Tensor | list[float],
    targets: torch.Tensor | list[int] | list[bool],
    *,
    max_iter: int = 50,
) -> float:
    """
    Fit a single temperature for binary correctness targets.
    probabilities: model confidence in [0,1]
    targets: 1 if prediction is correct else 0
    """
    probs = _to_prob_tensor(probabilities)
    y = _to_target_tensor(targets)
    if probs.numel() != y.numel():
        raise ValueError("probabilities and targets must have the same length")
    if probs.numel() == 0:
        return 1.0

    log_t = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_t],
        lr=0.1,
        max_iter=max(1, int(max_iter)),
        line_search_fn="strong_wolfe",
    )

    logits = torch.logit(probs)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_t).clamp(min=1e-3, max=100.0)
        scaled_logits = logits / temperature
        loss = F.binary_cross_entropy_with_logits(scaled_logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t.detach()).clamp(min=1e-3, max=100.0).item())


def expected_calibration_error(
    probabilities: torch.Tensor | list[float],
    targets: torch.Tensor | list[int] | list[bool],
    *,
    n_bins: int = 15,
) -> float:
    probs = _to_prob_tensor(probabilities)
    y = _to_target_tensor(targets)
    if probs.numel() != y.numel():
        raise ValueError("probabilities and targets must have the same length")
    if probs.numel() == 0:
        return 0.0

    n_bins = max(1, int(n_bins))
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = probs.new_zeros(())

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not torch.any(mask):
            continue
        bin_prob = probs[mask].mean()
        bin_acc = y[mask].mean()
        weight = mask.float().mean()
        ece = ece + torch.abs(bin_prob - bin_acc) * weight
    return float(ece.item())


def brier_score(
    probabilities: torch.Tensor | list[float],
    targets: torch.Tensor | list[int] | list[bool],
) -> float:
    probs = _to_prob_tensor(probabilities)
    y = _to_target_tensor(targets)
    if probs.numel() != y.numel():
        raise ValueError("probabilities and targets must have the same length")
    if probs.numel() == 0:
        return 0.0
    return float(torch.mean((probs - y) ** 2).item())

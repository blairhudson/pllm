"""Public offline float oracle only; never a serving or protected numeric kernel."""

from __future__ import annotations

import hashlib
from typing import Any


def fit_channel_polynomial(
    gate: Any, up: Any, *, degree: int, target: Any = None
) -> dict[str, Any]:
    """Batched least squares for an optimistic public per-channel SiLU fit.

    Torch supplies the independent floating-point oracle and bounded offline
    linear algebra. Runtime modular arithmetic belongs to pllm-garble.
    """
    import torch

    if type(degree) is not int or degree not in (1, 2, 4):
        raise ValueError("numeric screen supports degrees one, two or four")
    if (
        not isinstance(gate, torch.Tensor)
        or not isinstance(up, torch.Tensor)
        or gate.ndim != 2
        or gate.shape != up.shape
        or gate.device.type != "cpu"
        or up.device.type != "cpu"
        or not degree + 1 <= gate.shape[0] <= 256
        or not 1 <= gate.shape[1] <= 8192
    ):
        raise ValueError("polynomial calibration exceeds the public bounded tensor contract")
    with torch.inference_mode():
        g, u = gate.to(torch.float64), up.to(torch.float64)
        y = torch.nn.functional.silu(g) * u if target is None else target.to(torch.float64)
        if y.shape != g.shape or not all(bool(torch.isfinite(v).all()) for v in (g, u, y)):
            raise ValueError("polynomial calibration requires finite matching public arrays")
        center = g.mean(dim=0)
        scale = g.std(dim=0, correction=0).clamp_min(1e-3)
        z = (g - center) / scale
        design = torch.stack([u * z.pow(power) for power in range(degree + 1)], dim=-1)
        solution = torch.linalg.lstsq(design.transpose(0, 1), y.T.unsqueeze(-1), driver="gelsd")
        coefficients = solution.solution.squeeze(-1).T.contiguous()
        digest = hashlib.sha256(b"pllm.public_polynomial_float_oracle.v1\0" + bytes([degree]))
        for tensor in (center, scale, coefficients):
            digest.update(tensor.numpy().astype("<f8", copy=False).tobytes())
        return {
            "degree": degree,
            "center": center,
            "scale": scale,
            "coefficients": coefficients,
            "digest": digest.hexdigest(),
            "minimum_fit_rank": int(solution.rank.min()),
        }


def evaluate_channel_polynomial(gate: Any, profile: dict[str, Any] | None) -> Any:
    """Float oracle; ``None`` is the native numerator's idealized Q7 Taylor law."""
    if profile is None:
        return gate * 0.5 + gate.square() * 0.25
    z = (gate - profile["center"].to(gate.dtype)) / profile["scale"].to(gate.dtype)
    coefficients = profile["coefficients"].to(gate.dtype)
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * z + coefficient
    return value

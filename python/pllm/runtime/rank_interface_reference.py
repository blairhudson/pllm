"""Bounded offline rank diagnostics; no protected execution or runtime component.

Projecting a *computed* nonlinear output is an optimistic upper bound on a
learned narrow interface. It does not make the original nonlinear operator
executable by parties holding only additive shares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutputRankBasis:
    mean: np.ndarray
    directions: np.ndarray
    calibration_rows: int

    def project(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values)
        if raw.dtype != np.float32 or raw.ndim < 1 or raw.shape[-1] != self.mean.size:
            raise ValueError("rank projection requires finite float32 outputs of the calibrated width")
        if not np.all(np.isfinite(raw)):
            raise ValueError("rank projection rejects non-finite outputs")
        centered = raw - self.mean
        result = self.mean + (centered @ self.directions) @ self.directions.T
        return np.asarray(result, dtype=np.float32)


def fit_output_rank_basis(samples: np.ndarray, rank: int) -> OutputRankBasis:
    """Fit an affine rank-r *output* projector on public calibration rows."""
    raw = np.asarray(samples)
    if (
        raw.dtype != np.float32 or raw.ndim != 2
        or not 2 <= raw.shape[0] <= 512 or not 2 <= raw.shape[1] <= 1024
        or not np.all(np.isfinite(raw))
    ):
        raise ValueError("public calibration must be 2–512 finite float32 rows of width 2–1024")
    if type(rank) is not int or not 0 < rank < min(raw.shape[0], raw.shape[1]) or rank > 128:
        raise ValueError("rank exceeds public sample count, width or the 128-column bound")
    mean = np.mean(raw, axis=0, dtype=np.float64).astype(np.float32)
    _, _, right = np.linalg.svd(raw.astype(np.float64) - mean, full_matrices=False)
    directions = np.ascontiguousarray(right[:rank].T, dtype=np.float32)
    mean.setflags(write=False)
    directions.setflags(write=False)
    return OutputRankBasis(mean, directions, raw.shape[0])


@dataclass(frozen=True)
class AffineResidualBasis:
    """Public affine bypass and *oracle* projected nonlinear residual.

    The full nonlinear output is needed to compute this residual. A real narrow
    client cut needs a separate learned map of its rank-width input.
    """

    input_mean: np.ndarray
    output_mean: np.ndarray
    public_map: np.ndarray
    residual: OutputRankBasis

    def project(self, inputs: np.ndarray, full_outputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs)
        y = np.asarray(full_outputs)
        if (
            x.dtype != np.float32 or y.dtype != np.float32 or x.shape != y.shape
            or x.ndim != 2 or x.shape[1] != self.input_mean.size
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
        ):
            raise ValueError("affine residual requires finite paired float32 hidden rows")
        bypass = self.output_mean + (x - self.input_mean) @ self.public_map
        return np.asarray(bypass + self.residual.project(np.asarray(y - bypass, dtype=np.float32)), dtype=np.float32)


def fit_affine_residual_basis(
    inputs: np.ndarray, outputs: np.ndarray, rank: int, *, ridge_ratio: float = 1.0,
) -> AffineResidualBasis:
    x = np.asarray(inputs)
    y = np.asarray(outputs)
    if (
        x.dtype != np.float32 or y.dtype != np.float32 or x.shape != y.shape
        or x.ndim != 2 or not 2 <= x.shape[0] <= 512 or not 2 <= x.shape[1] <= 1024
        or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
        or type(rank) is not int or not 1 <= rank < min(x.shape) or rank > 128
        or type(ridge_ratio) is not float or not 0 < ridge_ratio <= 10
    ):
        raise ValueError("affine residual needs bounded paired float32 calibration rows")
    xmean = x.mean(axis=0, dtype=np.float64)
    ymean = y.mean(axis=0, dtype=np.float64)
    centered = x.astype(np.float64) - xmean
    target = y.astype(np.float64) - ymean
    gram = centered @ centered.T
    penalty = ridge_ratio * np.trace(gram) / x.shape[0]
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("calibration inputs have zero or invalid variation")
    solution = np.linalg.solve(gram + penalty * np.eye(x.shape[0]), target)
    public_map = np.asarray(centered.T @ solution, dtype=np.float32)
    input_mean = np.asarray(xmean, dtype=np.float32)
    output_mean = np.asarray(ymean, dtype=np.float32)
    fitted = output_mean + (x - input_mean) @ public_map
    residual = fit_output_rank_basis(np.asarray(y - fitted, dtype=np.float32), rank)
    for value in (input_mean, output_mean, public_map):
        value.setflags(write=False)
    return AffineResidualBasis(input_mean, output_mean, public_map, residual)


def certify_top1_from_error_bound(logits: np.ndarray, error_bound: float) -> bool:
    """Sufficient condition for preserving an exact top-1 under a *proven* bound.

    An observed error on an evaluation cohort is not a proven error bound for a
    later private request and cannot be used for live admission.
    """
    scores = np.asarray(logits)
    if (
        scores.ndim != 1 or scores.size < 2 or scores.dtype != np.float32
        or not np.all(np.isfinite(scores)) or not np.isfinite(error_bound)
        or error_bound < 0
    ):
        raise ValueError("certificate requires finite float32 logits and a nonnegative bound")
    winner = int(np.argmax(scores))
    runner_up = float(np.max(np.delete(scores, winner)))
    return float(scores[winner]) - runner_up > 2 * float(error_bound)

"""Offline extraction of selected existing gated-MLP channels (research only).

No pretrained weights are changed. Public calibration chooses channels and a
fixed Taylor or closed-form affine approximation for omitted channels. The selected channels retain their
original pretrained gate, up and down weights and exact SiLU×up arithmetic.
Only the resulting *local numeric surrogate* is executable; no protected
nonlinearity, provider placement or whole-model accuracy claim follows.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np


class ChannelExtractionError(ValueError):
    pass


def _silu(values: np.ndarray) -> np.ndarray:
    # Calibration and evaluation use the same finite float32 domain.
    v = values.astype(np.float64)
    return np.asarray(v * _sigmoid(v), dtype=np.float32)


def _sigmoid(v: np.ndarray) -> np.ndarray:
    sigmoid = np.empty_like(v)
    positive = v >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-v[positive]))
    exp = np.exp(v[~positive])
    sigmoid[~positive] = exp / (1.0 + exp)
    return sigmoid


@dataclass(frozen=True)
class ExtractedGatedMlp:
    indices: np.ndarray
    affine: np.ndarray
    offset: np.ndarray
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray
    gate_slope: np.ndarray
    up_slope: np.ndarray
    intercept: np.ndarray
    source_digest: str
    calibration_digest: str
    digest: str
    original_channels: int
    affine_method: str = "taylor"

    @property
    def hidden(self) -> int:
        return self.affine.shape[0]

    @property
    def width(self) -> int:
        return self.indices.size

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs)
        if (
            x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != self.hidden
            or not 1 <= x.shape[0] <= 512 or not np.all(np.isfinite(x))
        ):
            raise ChannelExtractionError("extracted MLP expects bounded finite float32 hidden rows")
        with np.errstate(over="ignore", invalid="ignore"):
            base = x @ self.affine.T + self.offset
        if not np.all(np.isfinite(base)):
            raise ChannelExtractionError("public affine result leaves finite float32 domain")
        if not self.width:
            return np.asarray(base, dtype=np.float32)
        gate = x @ self.gate.T
        up = x @ self.up.T
        with np.errstate(over="ignore", invalid="ignore"):
            residual = _silu(gate) * up - (
                gate * self.gate_slope + up * self.up_slope + self.intercept
            )
            result = base + residual @ self.down.T
        if not all(np.all(np.isfinite(item)) for item in (gate, up, residual, result)):
            raise ChannelExtractionError("selected channel execution leaves finite float32 domain")
        return np.asarray(result, dtype=np.float32)


@dataclass(frozen=True)
class PublicChannelCalibration:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray
    priority: np.ndarray
    affine: np.ndarray
    offset: np.ndarray
    gate_slope: np.ndarray
    up_slope: np.ndarray
    intercept: np.ndarray
    source_digest: str
    calibration_digest: str
    affine_method: str

    def select(self, width: int) -> ExtractedGatedMlp:
        if type(width) is not int or not 0 <= width <= self.gate.shape[0]:
            raise ChannelExtractionError("selected channels exceed public model dimensions")
        indices = np.sort(self.priority[:width]).astype(np.int32)
        owned = (
            np.ascontiguousarray(indices), np.ascontiguousarray(self.affine.copy()),
            np.ascontiguousarray(self.offset.copy()), np.ascontiguousarray(self.gate[indices]),
            np.ascontiguousarray(self.up[indices]), np.ascontiguousarray(self.down[:, indices]),
            np.ascontiguousarray(self.gate_slope[indices], dtype=np.float32),
            np.ascontiguousarray(self.up_slope[indices], dtype=np.float32),
            np.ascontiguousarray(self.intercept[indices], dtype=np.float32),
        )
        digest = hashlib.sha256(b"pllm.extracted_gated_mlp.v1\0")
        digest.update(bytes.fromhex(self.source_digest))
        digest.update(bytes.fromhex(self.calibration_digest))
        digest.update(struct.pack("<III", self.gate.shape[1], self.gate.shape[0], width))
        if self.affine_method != "taylor":
            digest.update(self.affine_method.encode("ascii") + b"\0")
        for item in owned:
            digest.update(item.tobytes())
            item.setflags(write=False)
        return ExtractedGatedMlp(
            *owned, self.source_digest, self.calibration_digest,
            digest.hexdigest(), self.gate.shape[0], self.affine_method,
        )


def calibrate_gated_mlp_channels(
    *, gate_weight: np.ndarray, up_weight: np.ndarray, down_weight: np.ndarray,
    calibration_inputs: np.ndarray, affine_method: str = "taylor",
) -> PublicChannelCalibration:
    """Change no model weights; choose public priorities and affine coefficients.

    The three matrices are copied and preflighted before any derived material
    is emitted. Channel priorities are the public rms linearization residual
    weighted by the original down-projection column norm.
    """
    gate, up, down, x = (
        np.asarray(gate_weight), np.asarray(up_weight),
        np.asarray(down_weight), np.asarray(calibration_inputs),
    )
    if type(affine_method) is not str or affine_method not in ("taylor", "least_squares"):
        raise ChannelExtractionError("unknown fixed public affine calibration contract")
    if (
        gate.dtype != np.float32 or up.dtype != np.float32 or down.dtype != np.float32
        or x.dtype != np.float32 or gate.ndim != 2 or up.ndim != 2
        or down.ndim != 2 or x.ndim != 2
        or gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0])
        or x.shape[1] != gate.shape[1] or not 2 <= x.shape[0] <= 512
        or not 2 <= gate.shape[1] <= 1024 or not 2 <= gate.shape[0] <= 8192
        or not all(np.all(np.isfinite(item)) for item in (gate, up, down, x))
    ):
        raise ChannelExtractionError("extraction requires bounded finite bias-free gated-MLP weights and public rows")
    projected_peak = (
        gate.size * 48 + x.shape[0] * gate.shape[0] * 40 + gate.shape[1] ** 2 * 8
    )
    if projected_peak > 512 * 1024 * 1024:
        raise ChannelExtractionError("public extraction working set exceeds the 512 MiB preflight")

    source_hash = hashlib.sha256(b"pllm.gated_mlp_source.v1\0")
    for item in (gate, up, down):
        source_hash.update(np.ascontiguousarray(item).tobytes())
    calibration_hash = hashlib.sha256(
        b"pllm.gated_mlp_calibration.v1\0" + np.ascontiguousarray(x).tobytes()
    ).hexdigest()

    gate_values = x @ gate.T
    up_values = x @ up.T
    if not np.all(np.isfinite(gate_values)) or not np.all(np.isfinite(up_values)):
        raise ChannelExtractionError("calibrated MLP projections leave the finite domain")
    gate_center = gate_values.mean(axis=0, dtype=np.float64)
    up_center = up_values.mean(axis=0, dtype=np.float64)
    if affine_method == "taylor":
        sigmoid = _sigmoid(gate_center)
        silu_center = gate_center * sigmoid
        gate_slope = (sigmoid + gate_center * sigmoid * (1 - sigmoid)) * up_center
        up_slope = silu_center
        intercept = silu_center * up_center - gate_slope * gate_center - up_slope * up_center
    else:
        # Solve a deterministic, independent 2×2 ridge system per channel.
        # The original pretrained matrices remain untouched, and only public
        # calibration rows enter the fixed affine representation.
        g = gate_values.astype(np.float64) - gate_center
        u = up_values.astype(np.float64) - up_center
        target = _silu(gate_values).astype(np.float64) * up_values
        target_center = target.mean(axis=0)
        centered_target = target - target_center
        gg = np.mean(g * g, axis=0)
        uu = np.mean(u * u, axis=0)
        gu = np.mean(g * u, axis=0)
        gy = np.mean(g * centered_target, axis=0)
        uy = np.mean(u * centered_target, axis=0)
        ridge = 0.001 * (gg + uu) / 2 + 1e-12
        determinant = (gg + ridge) * (uu + ridge) - gu * gu
        if not np.all(np.isfinite(determinant)) or np.any(determinant <= 0):
            raise ChannelExtractionError("public affine calibration has an invalid normal matrix")
        gate_slope = (gy * (uu + ridge) - uy * gu) / determinant
        up_slope = (uy * (gg + ridge) - gy * gu) / determinant
        intercept = target_center - gate_slope * gate_center - up_slope * up_center
    if not all(np.all(np.isfinite(item)) for item in (gate_slope, up_slope, intercept)):
        raise ChannelExtractionError("public affine coefficients leave finite domain")
    with np.errstate(over="ignore", invalid="ignore"):
        linear = gate_values * gate_slope + up_values * up_slope + intercept
        residual = _silu(gate_values) * up_values - linear
    if not np.all(np.isfinite(residual)):
        raise ChannelExtractionError("calibrated nonlinear residual leaves finite domain")
    weighted_error = (
        np.sqrt(np.mean(residual.astype(np.float64) ** 2, axis=0))
        * np.sqrt(np.sum(down.astype(np.float64) ** 2, axis=0))
    )
    priority = np.argsort(-weighted_error, kind="stable")
    # The full-channel affine is a fixed public matrix and constant. Summing
    # selected exact-minus-affine corrections makes selected channels exact.
    combined = gate.astype(np.float64) * gate_slope[:, None]
    combined += up.astype(np.float64) * up_slope[:, None]
    affine = np.asarray(down.astype(np.float64) @ combined, dtype=np.float32)
    offset = np.asarray(down.astype(np.float64) @ intercept, dtype=np.float32)
    if not np.all(np.isfinite(affine)) or not np.all(np.isfinite(offset)):
        raise ChannelExtractionError("public extracted affine leaves the finite domain")
    owned = (
        np.ascontiguousarray(gate.copy()), np.ascontiguousarray(up.copy()),
        np.ascontiguousarray(down.copy()), np.ascontiguousarray(priority),
        np.ascontiguousarray(affine), np.ascontiguousarray(offset),
        np.ascontiguousarray(gate_slope, dtype=np.float32),
        np.ascontiguousarray(up_slope, dtype=np.float32),
        np.ascontiguousarray(intercept, dtype=np.float32),
    )
    for item in owned:
        item.setflags(write=False)
    return PublicChannelCalibration(*owned, source_hash.hexdigest(), calibration_hash, affine_method)


def extract_gated_mlp_channels(
    *, gate_weight: np.ndarray, up_weight: np.ndarray, down_weight: np.ndarray,
    calibration_inputs: np.ndarray, selected_width: int, affine_method: str = "taylor",
) -> ExtractedGatedMlp:
    """One-shot bounded extraction; reuse calibration.select() for many widths."""
    if type(selected_width) is not int or selected_width < 0:
        raise ChannelExtractionError("selected channels need a bounded nonnegative width")
    return calibrate_gated_mlp_channels(
        gate_weight=gate_weight, up_weight=up_weight, down_weight=down_weight,
        calibration_inputs=calibration_inputs, affine_method=affine_method,
    ).select(selected_width)


def projected_channel_cut_bodies(
    *, layers: int, rows: int, channels_per_layer: int, word_bytes: int,
) -> dict[str, int]:
    """Two-worker client-cut bodies, not full wire or an executable topology."""
    if (
        type(layers) is not int or not 1 <= layers <= 24
        or type(rows) is not int or not 1 <= rows <= 70
        or type(channels_per_layer) is not int or not 0 <= channels_per_layer <= 8192
        or type(word_bytes) is not int or word_bytes not in (4, 8)
    ):
        raise ChannelExtractionError("channel cut exceeds bounded public workload")
    if not channels_per_layer:
        return {"workers_to_client_body_bytes": 0,
                "client_to_workers_body_bytes": 0, "all_link_body_bytes": 0}
    # Per row/layer: two party->client gate/up vectors of length 2r, then
    # two client->party fresh correction shares of length r. Body headers use
    # the test-local 50-byte session/stage/party frame envelope.
    frame = 50
    downloaded = 2 * rows * layers * (frame + 2 * channels_per_layer * word_bytes)
    uploaded = 2 * rows * layers * (frame + channels_per_layer * word_bytes)
    return {"workers_to_client_body_bytes": downloaded,
            "client_to_workers_body_bytes": uploaded,
            "all_link_body_bytes": downloaded + uploaded}

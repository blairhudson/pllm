"""Bounded, phase-local reuse of the existing unscaled float32 RoPE arithmetic."""
from __future__ import annotations

import numpy as np


MAX_ROTARY_COEFFICIENT_BYTES = 1 << 20
MAX_ROTARY_COEFFICIENT_TABLES = 16


def rotary_coefficient_bound(graph: dict) -> int:
    """Price retained tables for any input length within this public phase bound."""
    widths = set()
    for operation in graph["operations"]:
        if operation["operator"] != "rotary_embedding":
            continue
        attributes = operation["attributes"]
        if (attributes.get("output_dtype") == "bfloat16"
                or any(key in attributes for key in
                       ("frequency_scaling", "mrope_interleaved", "mrope_section"))):
            continue
        widths.add((int(attributes["rotary_dimensions"]), float(attributes["theta"])))
    return min(MAX_ROTARY_COEFFICIENT_BYTES,
               sum(8 * graph["query_sequence"] * width for width, _ in widths))


def default_rotary_coefficients(positions: np.ndarray, width: int, theta: float):
    # Preserve both expression order and vector shape from the uncached operator.
    # A full-context table can select a different NumPy SIMD/scalar libm path.
    frequency = 1.0 / (theta ** (np.arange(0, width, 2, dtype=np.float32) / width))
    angles = positions.astype(np.float32)[:, None] * frequency[None, :]
    cosine = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[None, None, :, :]
    sine = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[None, None, :, :]
    return cosine, sine


class PhaseRotaryCoefficients:
    """Only the current phase's client-local positions; no cross-request cache."""

    def __init__(self, positions: np.ndarray) -> None:
        self.positions: np.ndarray | None = positions
        self._tables: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
        self.retained_bytes = 0

    def get(self, positions: np.ndarray, width: int, theta: float):
        key = (width, theta)
        same_positions = positions is self.positions
        if same_positions and key in self._tables:
            return self._tables[key]
        result = default_rotary_coefficients(positions, width, theta)
        size = sum(table.nbytes for table in result)
        if (same_positions
                and len(self._tables) < MAX_ROTARY_COEFFICIENT_TABLES
                and self.retained_bytes + size <= MAX_ROTARY_COEFFICIENT_BYTES):
            for table in result:
                table.flags.writeable = False
            self._tables[key] = result
            self.retained_bytes += size
        return result

    def clear(self) -> None:
        self._tables.clear()
        self.positions = None
        self.retained_bytes = 0

"""Scoped SDK measurements for projected polynomial correlation experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectedPolynomialCostProbe:
    """Measure a bounded one-use modular MLP numerator, not a decoder.

    Layouts independently test coefficient derivation, early down-projection
    and seed compression. No input, key, mask or output share enters the report.
    """

    mode: str = "seeded"
    ring_bits: int = 24
    hidden: int = 8
    channels: int = 32
    outputs: int = 8
    rows: int = 4

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in (
            "dense",
            "derived",
            "contracted",
            "seeded",
        ):
            raise ValueError("unsupported projected polynomial layout")
        if type(self.ring_bits) is not int or self.ring_bits not in (24, 32, 64):
            raise ValueError("unsupported projected polynomial ring")
        if any(
            type(value) is not int or not 1 <= value <= bound
            for value, bound in (
                (self.hidden, 64),
                (self.channels, 128),
                (self.outputs, 64),
                (self.rows, 16),
            )
        ):
            raise ValueError("polynomial probe exceeds bounded reference dimensions")

    def run(self) -> dict[str, Any]:
        """Execute both local party references against an independent integer oracle."""
        import numpy as np

        from pllm import _native

        rng = np.random.default_rng(28103)  # Public fixture only, never correlation material.
        gate, up, down = (
            rng.integers(-2, 3, shape, dtype=np.int8)
            for shape in (
                (self.channels, self.hidden),
                (self.channels, self.hidden),
                (self.outputs, self.channels),
            )
        )
        x = rng.integers(-3, 4, (self.rows, self.hidden), dtype=np.int64)
        mask = np.uint64((1 << self.ring_bits) - 1)
        numeric_digest = hashlib.sha256(
            b"pllm.projected_polynomial_fixture.v1\0"
            + bytes([self.ring_bits, self.rows, self.hidden, self.channels, self.outputs])
            + gate.tobytes()
            + up.tobytes()
            + down.tobytes()
        ).hexdigest()
        started = time.process_time_ns()
        output, metadata = _native.projected_polynomial_probe(
            self.mode,
            self.ring_bits,
            self.rows,
            self.hidden,
            self.channels,
            self.outputs,
            gate.tobytes(),
            up.tobytes(),
            down.tobytes(),
            (x.astype("<u8") & mask).tobytes(),
            bytes.fromhex(numeric_digest),
        )
        total_cpu = time.process_time_ns() - started
        measured = json.loads(metadata)
        g = x.astype(object) @ gate.astype(object).T
        u = x.astype(object) @ up.astype(object).T
        expected = (((g * g + 256 * g) * u) @ down.astype(object).T) % (1 << self.ring_bits)
        actual = np.frombuffer(output, dtype="<u8").reshape(self.rows, self.outputs)
        if not np.array_equal(actual, expected.astype(np.uint64)):
            raise RuntimeError("projected polynomial reference differs from exact integer oracle")
        material_a, material_b = measured.pop("material_bytes")
        opening_a, opening_b = measured.pop("opening_bytes")
        links = [
            {
                "phase": "offline",
                "source": "dealer",
                "destination": "worker_a",
                "body_bytes": material_a,
            },
            {
                "phase": "offline",
                "source": "dealer",
                "destination": "worker_b",
                "body_bytes": material_b,
            },
            {
                "phase": "online",
                "source": "worker_a",
                "destination": "worker_b",
                "body_bytes": opening_a,
            },
            {
                "phase": "online",
                "source": "worker_b",
                "destination": "worker_a",
                "body_bytes": opening_b,
            },
        ]
        return {
            **measured,
            "schema": "pllm.projected_polynomial_cost_probe.v1",
            "scope": "bounded in-process one-use modular numerator; serialized key/opening bodies",
            "mode": self.mode,
            "ring_bits": self.ring_bits,
            "rows": self.rows,
            "hidden": self.hidden,
            "channels": self.channels,
            "outputs": self.outputs,
            "numeric_contract_digest": numeric_digest,
            "body_bytes_by_edge": links,
            "offline_body_bytes": material_a + material_b,
            "online_body_bytes": opening_a + opening_b,
            "covered_all_link_body_bytes": sum(edge["body_bytes"] for edge in links),
            "native_call_process_cpu_seconds": total_cpu / 1e9,
            "exact_modular_parity": True,
            "whole_decoder_executable": False,
            "full_wire_bytes": None,
            "peak_memory_bytes": None,
            "limitations": [
                "No source import, private input delivery or output-share transport measured",
                "No protected fixed-point rounding, SiLU quality or complete-layer execution",
                "AES256-counter seeded correlations, test-local dealer, no independent cryptographic review",
                "Array storage is not peak memory; local timings are not distributed full-response compute",
            ],
        }

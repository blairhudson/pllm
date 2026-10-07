"""Complete Compact polynomial block with an independent integer oracle."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import time
from collections.abc import Sequence

from .activations import _COMPACT, _round_even
from .wavelet_gated import _inputs
from .operator_benchmarks import (
    _OperatorBenchmark, _sdk_digest, OperatorBenchmarkQuality, OperatorBenchmarkReport,
    OperatorBenchmarkResources, OperatorBenchmarkSample, OperatorCitation,
)


def _profile(component):
    from pllm import _native
    params = component.get_params()
    native = _native.CompactProductReference(
        struct.pack("<1024I", *params["public_calibration_counts"]), params["max_pieces"], params["degree"])
    metadata = json.loads(native.metadata())
    # Independently bind the published coefficients, source and exact fitting
    # configuration. The arithmetic oracle evaluates these committed integers.
    digest = hashlib.sha256(b"pllm.numeric.compact_product_q18_q12.v1" + _COMPACT.source_sha256.encode())
    digest.update(bytes((params["max_pieces"], params["degree"], len(metadata["pieces"]))))
    digest.update(struct.pack("<1024I", *params["public_calibration_counts"]))
    for piece in metadata["pieces"]:
        digest.update(struct.pack("<ii4q", piece["lower"], piece["upper"], *piece["coefficients"]))
    if digest.hexdigest() != metadata["profile_digest"]:
        raise RuntimeError("Compact profile commitments differ")
    return native, metadata


def _oracle(profile, gate, up):
    gate = _round_even(gate, 64)
    up = _round_even(up, 64)
    if gate <= -(8 << 12):
        activated = 0
    elif gate >= (8 << 12):
        activated = gate
    else:
        piece = next(p for p in profile["pieces"] if p["lower"] <= gate < p["upper"])
        t = (2 * gate - piece["lower"] - piece["upper"]) * (1 << 20) // (piece["upper"] - piece["lower"])
        coefficients = piece["coefficients"][:profile["degree"] + 1]
        value = coefficients[-1]
        for coefficient in reversed(coefficients[:-1]):
            value = _round_even(value * t, 1 << 20) + coefficient
        activated = _round_even(value, 1 << 20)
    return _round_even(activated * up, 4096)


class _CompactProductBenchmark(_OperatorBenchmark):
    __slots__ = ()

    def run(self, *, gate_q18: Sequence[int] | None = None,
            up_q18: Sequence[int] | None = None) -> OperatorBenchmarkReport:
        from pllm import __version__
        gate, up, source = _inputs(gate_q18, up_q18, self.lanes)
        start = time.perf_counter_ns()
        native, profile = _profile(self.component)
        setup_seconds = (time.perf_counter_ns() - start) / 1e9
        expected = [_oracle(profile, g, u) for g, u in zip(gate, up, strict=True)]
        packed = [struct.pack(f"<{self.lanes}i", *values) for values in (gate, up)]
        samples, resources = [], None
        for index in range(self.samples):
            start = time.process_time_ns()
            output, raw = native.benchmark(*packed)
            cpu = (time.process_time_ns() - start) / 1e9
            measured = json.loads(raw)
            if (measured["profile_digest"] != profile["profile_digest"] or measured["lanes"] != self.lanes
                    or len(output) != 4 * self.lanes
                    or list(struct.unpack(f"<{self.lanes}i", output)) != expected):
                raise RuntimeError("Compact differs from independent complete integer oracle")
            timing = {key: measured.pop(key) for key in ("issuance_ns", "online_ns", "clear_ns")}
            if resources is not None and measured != resources:
                raise RuntimeError("Compact topology changed between fresh samples")
            resources = measured
            samples.append(OperatorBenchmarkSample(
                timing["issuance_ns"] / 1e9, timing["online_ns"] / 1e9,
                timing["clear_ns"] / 1e9, cpu, (), setup_seconds if index == 0 else 0.0))
        assert resources is not None
        keys = tuple(resources["party_key_payload_bytes"])
        parts = tuple(tuple(row) for row in resources["key_payload_bytes_by_operation"])
        if tuple(sum(row[i] for row in parts) for i in (1, 2)) != keys:
            raise RuntimeError("Compact complete material accounting does not conserve keys")
        errors = [abs(actual / 4096 - (g / (1 << 18)) / (1 + math.exp(-g / (1 << 18))) * (u / (1 << 18)))
                  for g, u, actual in zip(gate, up, expected, strict=True)]
        return OperatorBenchmarkReport(
            method="compact-silu-gated-q18-q12",
            configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest, numeric_contract_digest=profile["profile_digest"],
            native_implementation_digest=resources["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("compact_product.py"), input_source=source,
            resources=OperatorBenchmarkResources(
                keys, tuple(v * 8 for v in keys), (0, 0), 0, 0, 0,
                resources["protocol_allocation_estimate_bytes"], 0,
                resources["client_entry_exit_payload_estimate_bytes"],
                role_names=("garbler_client", "evaluator"),
                public_metadata_bytes_estimate=resources["public_metadata_bytes_estimate"],
                public_profile_bytes_estimate=4096 + 64 + 40 * len(profile["pieces"]),
                evaluator_ciphertext_bytes=resources["evaluator_ciphertext_bytes"],
                evaluator_constant_label_bytes=resources["evaluator_constant_label_bytes"],
                key_payload_bytes_by_operation=parts),
            quality=OperatorBenchmarkQuality(
                self.lanes * self.samples, self.lanes * self.samples,
                "Python float64 SiLU(gate) * up on supplied Q18 inputs; preceding float conversion excluded",
                max(errors), math.fsum(errors) / self.lanes, self.lanes * self.samples, 0),
            samples=tuple(samples), citations=(OperatorCitation(
                _COMPACT.paper_id, _COMPACT.title, _COMPACT.url, _COMPACT.source_sha256,
                "Sections 4.1–4.2 public-density-weighted Chebyshev interpolation; dyadic refinement, Q12 and ±8 tails are adaptations"),
                OperatorCitation("half-gates", "Two Halves Make a Whole", "https://eprint.iacr.org/2014/756",
                    "789ecaf8651b6dc8341d6033f951d76c0e6f4b04427cdde26cf91ecfb0af3083",
                    "One-use private selection, Horner arithmetic and complete product with exact ties-even rescaling")),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", resources["native_version"])),
            limitations=(
                "Public offline calibration only; fixed degree and dyadic refinement replace Compact's annealing and accuracy-constrained search",
                "The bounded Q12 profile and ±8 zero/identity tails differ from the original paper's numeric choices",
                "Only the trusted garbler-client decodes; one fresh serial circuit per lane, retired before the next",
                "Input ranges are client-enforced; float conversion and range enforcement are not protected operators",
                "Zero evaluator-peer frames exclude material and boundary delivery; public metadata and framing are estimates",
                "Public profile fitting is charged once in the first sample; native-call CPU excludes that public fitting",
                "No decoder tensor schedule, provider transport, complete wire, peak-RSS measurement or independent cryptographic review"),
        )

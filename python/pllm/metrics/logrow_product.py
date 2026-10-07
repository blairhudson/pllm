"""Complete LogRow SiLU×up operator on the locked uncompressed Q18/Q9 profile."""
from __future__ import annotations

from collections.abc import Sequence
import json
import math
import platform
import struct
import time

from .activations import _LOGROW, _round_even
from .wavelet_gated import _inputs, _profile
from .operator_benchmarks import (
    _OperatorBenchmark, _sdk_digest, OperatorBenchmarkQuality,
    OperatorBenchmarkReport, OperatorBenchmarkResources, OperatorBenchmarkSample,
    OperatorCitation,
)


class _LogRowProductBenchmark(_OperatorBenchmark):
    __slots__ = ()

    def run(self, *, gate_q18: Sequence[int] | None = None,
            up_q18: Sequence[int] | None = None) -> OperatorBenchmarkReport:
        from pllm import _native, __version__

        gate, up, source = _inputs(gate_q18, up_q18, self.lanes)
        table, digest = _profile(0)
        expected = [_round_even(table[(g + (64 << 18)) >> 13] * _round_even(u, 512), 512)
                    for g, u in zip(gate, up, strict=True)]
        packed = [struct.pack(f"<{self.lanes}i", *values) for values in (gate, up)]
        samples, resources = [], None
        for _ in range(self.samples):
            start = time.process_time_ns()
            output, raw = _native.logrow_product_benchmark(*packed)
            cpu = (time.process_time_ns() - start) / 1e9
            measured = json.loads(raw)
            if (measured["profile_digest"] != digest or measured["lanes"] != self.lanes
                    or len(output) != 4 * self.lanes
                    or list(struct.unpack(f"<{self.lanes}i", output)) != expected):
                raise RuntimeError("LogRow product differs from independent integer oracle")
            timings = {key: measured.pop(key) for key in ("issuance_ns", "online_ns", "clear_ns", "public_setup_ns")}
            if resources is not None and measured != resources:
                raise RuntimeError("LogRow public topology changed between one-use samples")
            resources = measured
            samples.append(OperatorBenchmarkSample(
                timings["issuance_ns"] / 1e9, timings["online_ns"] / 1e9,
                timings["clear_ns"] / 1e9, cpu, (), timings["public_setup_ns"] / 1e9,
            ))
        assert resources is not None
        keys = tuple(resources["party_key_payload_bytes"])
        parts = tuple(tuple(row) for row in resources["key_payload_bytes_by_operation"])
        if tuple(sum(row[i] for row in parts) for i in (1, 2)) != keys:
            raise RuntimeError("LogRow complete material accounting does not conserve keys")
        errors = [abs(actual / 512 - (g / (1 << 18)) / (1 + math.exp(-g / (1 << 18))) * (u / (1 << 18)))
                  for g, u, actual in zip(gate, up, expected, strict=True)]
        return OperatorBenchmarkReport(
            method="logrow-silu-gated-q18-q9",
            configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest, numeric_contract_digest=digest,
            native_implementation_digest=resources["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("logrow_product.py"), input_source=source,
            resources=OperatorBenchmarkResources(
                keys, tuple(v * 8 for v in keys), (0, 0), 0, 0, 0,
                resources["protocol_allocation_estimate_bytes"], 0,
                resources["client_entry_exit_payload_estimate_bytes"],
                role_names=("garbler_client", "evaluator"),
                public_metadata_bytes_estimate=resources["public_metadata_bytes_estimate"],
                public_profile_bytes_estimate=4096 * 4 + 64,
                evaluator_ciphertext_bytes=resources["evaluator_ciphertext_bytes"],
                evaluator_constant_label_bytes=resources["evaluator_constant_label_bytes"],
                key_payload_bytes_by_operation=parts,
            ),
            quality=OperatorBenchmarkQuality(
                self.lanes * self.samples, self.lanes * self.samples,
                "Python float64 SiLU(gate) * up on supplied Q18 inputs; preceding float conversion excluded",
                max(errors), math.fsum(errors) / self.lanes,
                self.lanes * self.samples, 0,
            ),
            samples=tuple(samples), citations=(_LOGROW, OperatorCitation(
                "half-gates", "Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits using Half Gates",
                "https://eprint.iacr.org/2014/756", "789ecaf8651b6dc8341d6033f951d76c0e6f4b04427cdde26cf91ecfb0af3083",
                "Figure 2 free-XOR half-gates for signed arithmetic; PLLM fixed-width ties-even composition")),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", resources["native_version"])),
            limitations=(
                "One fresh circuit per lane; serial material retirement; local trusted garbler",
                "LogRow uses the earlier two-row one-hot setup; exact arithmetic is a half-gates adaptation",
                "Index floor is protected wire selection; ties-even rescaling, signs and product stay opaque",
                "Fixed public input ranges are enforced by the trusted client, not by a malicious-input proof",
                "Same numeric profile as uncompressed Curl; different role ownership, material and execution contracts",
                "Zero evaluator-peer frames does not mean zero traffic: evaluator material and client labels are charged separately",
                "Public metadata and client-boundary framing are estimates; no external serialization or transport",
                "Float32-to-Q18 conversion stays client-local; allocation bound is serial-live, not measured RSS",
                "No decoder tensor schedule, complete response measurement or independent cryptographic review",
            ),
        )

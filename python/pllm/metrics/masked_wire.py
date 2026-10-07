"""Benchmark executor for the masked-wire gated-multiply component."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from numbers import Integral
import platform
import secrets
import struct
import time

from .operator_benchmarks import (
    _OperatorBenchmark,
    _sdk_digest,
    OperatorBenchmarkQuality,
    OperatorBenchmarkReport,
    OperatorBenchmarkResources,
    OperatorBenchmarkSample,
    OperatorCitation,
)

_PROFILE = "pllm.residual_silu_reference.sigma_gpu_order.q12.v1"
_CITATIONS = (
    OperatorCitation(
        "sigma",
        "SIGMA: Secure GPT Inference with Function Secret Sharing",
        "https://petsymposium.org/popets/2024/popets-2024-0107.php",
        "f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf",
        "Sections 2.5–2.6, 4.2.2, 5.1 and Figure 7: gap shifts, residual SiLU and masked wires; CPU/Q12 adaptation",
    ),
    OperatorCitation(
        "grotto",
        "Grotto: Screaming fast (2 + 1)-PC for Z2^n via (2, 2)-DPFs",
        "https://eprint.iacr.org/2023/108",
        "f476bca15ac55a0b8f2c1df86f0dd98e7b370bbafc8facc29c7dd9bd99be391c",
        "Sections 5–6: suffix-parity comparison using early-terminated Boolean DPFs",
    ),
    OperatorCitation(
        "pika",
        "Pika: Secure Computation using Function Secret Sharing over Rings",
        "https://eprint.iacr.org/2022/826",
        "52af6d2c46f366ed3756982c0f2c24831743de4b2a5e5be6f5c9ab51af0710ba",
        "Boolean-DPF signed lookup and sign correction; explicit correction shares instead of seeded distribution",
    ),
)
_FUSION_CITATION = OperatorCitation(
    "fusefss",
    "FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing",
    "https://arxiv.org/abs/2606.09551",
    "6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0",
    "Section 4.3, Appendix E and I.1: multi-view masked-input fusion of shifts and predicates; charged complete-block post-processing",
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _profile() -> tuple[tuple[int, ...], str]:
    # Independent public formula, not a native serving fallback.
    table = tuple(math.floor(4096 * (i / 64) / (1 + math.exp(i / 64))) for i in range(1024))
    h = hashlib.sha256(_PROFILE.encode() + b"\0")
    h.update(bytes((12, 40, 19)))
    h.update(struct.pack("<1026q", 48, 128, *table))
    return table, h.hexdigest()


def _public_values(lanes: int, bound: int) -> tuple[int, ...]:
    fixed = (
        -(bound << 24),
        -(16 << 24) - 1,
        -(16 << 24),
        -(1 << 24),
        -4097,
        -4096,
        -1,
        0,
        1,
        4095,
        4096,
        1 << 24,
        (16 << 24) - 1,
        16 << 24,
        bound << 24,
    )
    return tuple(
        fixed[i] if i < len(fixed) else i * 982451653 % (2 * (bound << 24) + 1) - (bound << 24)
        for i in range(lanes)
    )


def _values(value: Sequence[int], lanes: int, bound: int, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or len(value) != lanes:
        raise ValueError(f"{name} must contain exactly {lanes} Q24 integers")
    # Index only the admitted length; never exhaust an unbounded iterable.
    result = []
    for i in range(lanes):
        x = value[i]
        if isinstance(x, bool) or not isinstance(x, Integral) or abs(int(x)) > bound << 24:
            raise ValueError(f"{name} requires Q24 integers in [-{bound}, {bound}]")
        result.append(int(x))
    return tuple(result)


class _MaskedWireBenchmark(_OperatorBenchmark):
    __slots__ = ()

    def run(
        self,
        *,
        gate_q24: Sequence[int] | None = None,
        up_q24: Sequence[int] | None = None,
    ) -> OperatorBenchmarkReport:
        """Use the public fixture or caller-owned encoded input sequences.

        Inputs are signed Q24: gate in [-48, 48], up in [-128, 128]. A report
        retains aggregate errors but no values or data fingerprints. Replaying
        custom inputs requires supplying those inputs again.
        """
        from pllm import _native, __version__

        if (gate_q24 is None) != (up_q24 is None):
            raise ValueError("supply both gate_q24 and up_q24, or neither")
        if gate_q24 is None:
            gate, up = _public_values(self.lanes, 48), _public_values(self.lanes, 128)[::-1]
            source = "public-boundary-fixture-v1"
        else:
            gate = _values(gate_q24, self.lanes, 48, "gate_q24")
            up = _values(up_q24, self.lanes, 128, "up_q24")
            source = "caller-owned; supply inputs again for replay"
        table, profile_digest = _profile()
        layout = self.component.params.get("layout", "unfused")
        resource = json.loads(_native.masked_wire_resources(self.lanes, layout))
        if resource["profile_digest"] != profile_digest:
            raise RuntimeError("native residual table differs from the independent public profile")
        expected = []
        errors = []
        for g, u in zip(gate, up, strict=True):
            qg, qu = g // 4096, u // 4096
            result = ((max(qg, 0) - table[min(abs(qg // 64), 1023)]) * qu) // 4096
            expected.append(result)
            x, y = g / 2**24, u / 2**24
            errors.append(abs(result / 4096 - x / (1 + math.exp(-x)) * y))
        packed_gate, packed_up = (struct.pack(f"<{self.lanes}q", *x) for x in (gate, up))
        samples = []
        for _ in range(self.samples):
            # Correlation randomness remains OS-owned in Rust; this is only a public session ID.
            session = secrets.token_bytes(32)
            started = time.process_time_ns()
            output, raw = _native.masked_wire_benchmark(
                packed_gate,
                packed_up,
                bytes.fromhex(self.digest),
                session,
                layout,
            )
            cpu = (time.process_time_ns() - started) / 1e9
            measured = json.loads(raw)
            if any(measured[k] != v for k, v in resource.items()):
                raise RuntimeError("native benchmark resource or implementation identity changed")
            if (
                len(output) != 8 * self.lanes
                or list(struct.unpack(f"<{self.lanes}q", output)) != expected
            ):
                raise RuntimeError("protected block differs from independent integer oracle")
            phases = tuple(tuple(row) for row in measured["phase_frame_bytes"])
            if (
                len(phases) != resource["rounds"]
                or any(len(row) != 2 for row in phases)
                or sum(sum(row) for row in phases) != resource["peer_frame_bytes"]
            ):
                raise RuntimeError("native benchmark phase accounting differs")
            samples.append(
                OperatorBenchmarkSample(
                    measured["issuance_ns"] / 1e9,
                    measured["online_ns"] / 1e9,
                    measured["clear_ns"] / 1e9,
                    cpu,
                    phases,
                )
            )
        directions = tuple(
            sum(row[party] for row in samples[0].peer_frame_bytes_by_phase) for party in (0, 1)
        )
        if any(
            sample.peer_frame_bytes_by_phase != samples[0].peer_frame_bytes_by_phase
            for sample in samples
        ):
            raise RuntimeError("public frame layout changed between samples")
        native = OperatorBenchmarkResources(
            (resource["party_key_payload_bytes"],) * 2,
            (resource["party_key_bits"],) * 2,
            directions,
            resource["peer_payload_bits"],
            resource["rounds"],
            resource["public_table_bytes_per_party"],
            resource["protocol_allocation_estimate_bytes"],
            15 * self.lanes,
            25 * self.lanes,
        )
        return OperatorBenchmarkReport(
            method="masked-wire-gated-block",
            configuration_json=_canonical(self.to_dict()),
            configuration_digest=self.digest,
            numeric_contract_digest=profile_digest,
            native_implementation_digest=resource["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("masked_wire.py"),
            input_source=source,
            resources=native,
            quality=OperatorBenchmarkQuality(
                self.lanes * self.samples,
                self.lanes * self.samples,
                "Python float64 SiLU(gate) * up on supplied Q24 inputs; operator error only",
                max(errors),
                math.fsum(errors) / len(errors),
            ),
            samples=tuple(samples),
            citations=_CITATIONS + ((_FUSION_CITATION,) if layout == "fused" else ()),
            environment=(
                ("platform", platform.platform()),
                ("python", platform.python_version()),
                ("pllm", __version__),
                ("native", resource["native_version"]),
            ),
            limitations=(
                "Trusted local dealer/client and two in-process evaluators; no independent cryptographic review",
                "Fixed floor/Q12 approximation; scoped masked-input fusion, not the general FuseFSS compiler or paper GPU systems",
                "Raw key payloads exclude dealer metadata and transport framing; client-mask and entry/exit bytes are estimates",
                "Peer bytes are serialized local frames, not physical wire; public tables are separate storage",
                "Allocation estimate is not measured peak memory; phase wall timings include local roles and client boundary",
                "No protected float conversion, compiled tensor schedule, whole-model quality or serving recommendation",
                "Caller-owned inputs and outputs are omitted; reports contain derived aggregate approximation errors",
            ),
        )

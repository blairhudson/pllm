"""Benchmark executor for the shared-rescale nonlinear component."""

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

_NUMERIC = "pllm.signed_rescale_original_nonnegative.v1"
_CITATIONS = (
    OperatorCitation(
        "fusefss",
        "FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing",
        "https://arxiv.org/abs/2606.09551",
        "6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0",
        "Signed truncate/reduce and sign-extension fusion; bounded helper adaptation, not the general compiler or paper wire layout",
    ),
    OperatorCitation(
        "fss-mixed",
        "Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation",
        "https://eprint.iacr.org/2020/1392",
        "dc94c63a77a5cb52cfd12b186a327d045de4a1fc662060c1de6412b334f4fcf4",
        "Figure 1 compact DCF; Section 4.1/Figure 14 universal interval-key backend when selected",
    ),
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fixture(lanes: int, bits: int, shift: int) -> tuple[int, ...]:
    bound, divisor = 1 << (bits - 1), 1 << shift
    boundaries = (
        -bound,
        -bound + 1,
        -divisor - 1,
        -divisor,
        -divisor // 2 - 1,
        -divisor // 2,
        -1,
        0,
        1,
        divisor // 2,
        divisor // 2 + 1,
        divisor - 1,
        divisor,
        bound - 1,
    )
    fixed = tuple(x for x in boundaries if -bound <= x < bound)
    return tuple(
        fixed[i] if i < len(fixed) else (i * 982451653) % (2 * bound) - bound for i in range(lanes)
    )


class _SharedRescaleBenchmark(_OperatorBenchmark):
    __slots__ = ()

    def run(self, *, values: Sequence[int] | None = None) -> OperatorBenchmarkReport:
        """Benchmark public boundaries or caller-owned signed, already-encoded integers.

        Reports contain aggregate rounding errors, never input/output values or
        their hashes. Caller-owned replay requires supplying the same values again.
        """
        from pllm import _native, __version__

        bits, shift, rounding, layout, backend = (
            self.component.params[name] for name in ("bits", "shift", "rounding", "layout", "backend")
        )
        if values is None:
            inputs = _fixture(self.lanes, bits, shift)
            source = "public-signed-boundary-fixture-v1"
        else:
            if isinstance(values, (str, bytes, bytearray)) or len(values) != self.lanes:
                raise ValueError(f"values must contain exactly {self.lanes} signed integers")
            bound = 1 << (bits - 1)
            checked = []
            for i in range(self.lanes):
                x = values[i]
                if (
                    isinstance(x, bool)
                    or not isinstance(x, Integral)
                    or not -bound <= int(x) < bound
                ):
                    raise ValueError("values must be integers within the declared signed ring")
                checked.append(int(x))
            inputs = tuple(checked)
            source = "caller-owned; supply inputs again for replay"
        settings = (bits, shift, rounding, layout, backend)
        resource = json.loads(_native.shared_rescale_resources(settings, self.lanes))
        # Python arbitrary-precision divmod is independent of Rust's ring arithmetic.
        mask, divisor = (1 << bits) - 1, 1 << shift
        expected, errors = [], []
        for x in inputs:
            quotient, remainder = divmod(x, divisor)
            if rounding == "ties_even" and (
                2 * remainder > divisor or (2 * remainder == divisor and quotient % 2)
            ):
                quotient += 1
            expected.extend((quotient & mask, int(x >= 0)))
            errors.append(abs(quotient - x / divisor))
        packed = struct.pack(f"<{self.lanes}q", *inputs)
        samples = []
        for _ in range(self.samples):
            # Only a public identifier; Rust owns independent OS-random correlation material.
            session = secrets.token_bytes(32)
            start = time.process_time_ns()
            output, raw = _native.shared_rescale_benchmark(
                packed,
                settings,
                bytes.fromhex(self.digest),
                session,
            )
            cpu = (time.process_time_ns() - start) / 1e9
            measured = json.loads(raw)
            if any(measured[k] != v for k, v in resource.items()):
                raise RuntimeError("native benchmark resource or implementation identity changed")
            if (
                len(output) != self.lanes * 8
                or list(struct.unpack(f"<{2 * self.lanes}I", output)) != expected
            ):
                raise RuntimeError("shared rescale differs from independent integer oracle")
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
        if any(
            s.peer_frame_bytes_by_phase != samples[0].peer_frame_bytes_by_phase for s in samples
        ):
            raise RuntimeError("public frame layout changed between samples")
        directions = tuple(
            sum(row[p] for row in samples[0].peer_frame_bytes_by_phase) for p in (0, 1)
        )
        numeric = {
            "profile": _NUMERIC,
            "bits": bits,
            "shift": shift,
            "rounding": rounding,
        }
        return OperatorBenchmarkReport(
            method="shared-signed-rescale",
            configuration_json=_canonical(self.to_dict()),
            configuration_digest=self.digest,
            numeric_contract_digest=hashlib.sha256(_canonical(numeric).encode()).hexdigest(),
            native_implementation_digest=resource["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("shared_rescale.py"),
            input_source=source,
            resources=OperatorBenchmarkResources(
                (resource["party_key_payload_bytes"],) * 2,
                (8 * resource["party_key_payload_bytes"],) * 2,
                directions,
                resource["peer_payload_bits"],
                resource["rounds"],
                0,
                resource["protocol_allocation_estimate_bytes"],
                4 * self.lanes,
                24 * self.lanes,
            ),
            quality=OperatorBenchmarkQuality(
                2 * self.lanes * self.samples,
                2 * self.lanes * self.samples,
                "Float64 x / 2**shift in output integer units; original-input sign checked independently",
                max(errors),
                math.fsum(errors) / len(errors),
            ),
            samples=tuple(samples),
            citations=_CITATIONS[:1] if backend == "prefix_dpf" else _CITATIONS,
            environment=(
                ("platform", platform.platform()),
                ("python", platform.python_version()),
                ("pllm", __version__),
                ("native", resource["native_version"]),
            ),
            limitations=(
                "Trusted in-process dealer/client and two one-use evaluators; no independent security review or role transport",
                "Bounded truncate/reduce plus sign-extension fusion; no FuseFSS general compiler or GPU-system reproduction",
                "Floor is paper-derived; ties-even is PLLM's extension; original-input sign is a separate output",
                "Generic 32-bit additive comparison outputs; these are not SIGMA's Boolean-DPF GapARS costs",
                "Raw keys omit dealer metadata and offline framing; peer bytes include local frame headers, not physical wire",
                "Client estimates count a transient input share and 32-bit input/output shares; no measured client transport",
                "Online wall time includes input sharing and client reconstruction; issuance is separate; local roles run sequentially",
                "Allocation is an estimate; no measured peak memory, Qwen numeric mapping, whole-decoder execution or serving recommendation",
                "Inputs and outputs are omitted; aggregate rounding error is retained; custom inputs must be supplied for replay",
            ),
        )

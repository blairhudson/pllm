"""Complete Haar gated-block measurements with an independent integer oracle."""
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
    _OperatorBenchmark, _sdk_digest, OperatorBenchmarkQuality, OperatorBenchmarkReport,
    OperatorBenchmarkResources, OperatorBenchmarkSample, OperatorCitation,
)

_OPERATIONS = ("index_rescale", "up_rescale", "haar_lookup", "beaver_multiply", "output_rescale")
_ID = "pllm.numeric.haar_silu_gated.q18_q9.v1"


def _round_even(value: int, divisor: int) -> int:
    q, r = divmod(value, divisor)
    return q + int(2 * r > divisor or (2 * r == divisor and q % 2))


def _profile(levels: int) -> tuple[tuple[int, ...], str]:
    original = tuple(round((i / 32 - 64) / (1 + math.exp(-(i / 32 - 64))) * 512) for i in range(4096))
    width = 1 << levels
    table = tuple(_round_even(sum(original[i:i + width]), width) for i in range(0, 4096, width))
    digest = hashlib.sha256(_ID.encode() + bytes([18, 9, 5, levels])
        + struct.pack("<4i", 64, 48, 128, 4096) + struct.pack(f"<{len(table)}i", *table)).hexdigest()
    return table, digest


def _oracle(table, levels, gate, up):
    index = (gate + (64 << 18)) // (1 << (13 + levels))
    product = table[index] * _round_even(up, 512)
    if not -(1 << 31) <= product < 1 << 31:
        raise ValueError("gated product exceeds signed ring")
    return _round_even(product, 512)


def _fixture(lanes: int, bound: int) -> tuple[int, ...]:
    fixed = (-bound << 18, -8193, -8192, -8191, -769, -768, -257, -256, -1,
             0, 1, 256, 257, 768, 769, 8191, 8192, 8193, bound << 18)
    return tuple(fixed[i] if i < len(fixed) else (i * 982451653) % (2 * (bound << 18) + 1) - (bound << 18)
                 for i in range(lanes))


def _checked(values: Sequence[int], lanes: int, bound: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or len(values) != lanes:
        raise ValueError("Q18 inputs must contain exactly the admitted number of integers")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or abs(int(value)) > bound << 18:
            raise ValueError("Q18 inputs exceed their public range")
        result.append(int(value))
    return tuple(result)


class _WaveletGatedBenchmark(_OperatorBenchmark):
    __slots__ = ()

    def run(self, *, gate_q18: Sequence[int] | None = None, up_q18: Sequence[int] | None = None) -> OperatorBenchmarkReport:
        from pllm import _native, __version__

        levels = self.component.params["levels"]
        if gate_q18 is None and up_q18 is None:
            gate, up = _fixture(self.lanes, 48), _fixture(self.lanes, 128)[::-1]
            source = "public-q18-gated-boundary-fixture-v1"
        elif gate_q18 is None or up_q18 is None:
            raise ValueError("supply both gate_q18 and up_q18")
        else:
            gate, up = _checked(gate_q18, self.lanes, 48), _checked(up_q18, self.lanes, 128)
            source = "caller-owned; supply inputs again for replay"
        table, numeric = _profile(levels)
        original, _ = _profile(0)
        expected = [_oracle(table, levels, g, u) for g, u in zip(gate, up, strict=True)]
        base = [_oracle(original, 0, g, u) for g, u in zip(gate, up, strict=True)]
        resources = json.loads(_native.wavelet_gated_resources(levels, self.lanes))
        if resources["profile_digest"] != numeric:
            raise RuntimeError("native Haar profile differs from independent mathematical table")
        packed = tuple(struct.pack(f"<{self.lanes}i", *values) for values in (gate, up))
        samples = []
        for _ in range(self.samples):
            start = time.process_time_ns()
            output, raw = _native.wavelet_gated_benchmark(levels, *packed, bytes.fromhex(self.digest), secrets.token_bytes(32))
            cpu = (time.process_time_ns() - start) / 1e9
            measured = json.loads(raw)
            if any(measured[key] != value for key, value in resources.items()):
                raise RuntimeError("native Haar benchmark identity or resources changed")
            if len(output) != 4 * self.lanes or list(struct.unpack(f"<{self.lanes}i", output)) != expected:
                raise RuntimeError("protected Haar block differs from independent integer oracle")
            phases = tuple(tuple(row) for row in measured["phase_frame_bytes"])
            if phases != tuple((size, size) for size in resources["phase_frame_bytes_per_party"]):
                raise RuntimeError("Haar block phase conservation failed")
            samples.append(OperatorBenchmarkSample(measured["issuance_ns"] / 1e9,
                measured["online_ns"] / 1e9, measured["clear_ns"] / 1e9, cpu, phases,
                measured["public_setup_ns"] / 1e9))
        errors = [abs(y / 512 - (g / (1 << 18)) / (1 + math.exp(-g / (1 << 18))) * u / (1 << 18))
                  for g, u, y in zip(gate, up, expected, strict=True)]
        base_errors = [abs(a - b) for a, b in zip(base, expected, strict=True)]
        parts = resources["party_key_bytes_by_operation"]
        key_bytes = resources["party_key_payload_bytes"]
        if sum(parts) != key_bytes:
            raise RuntimeError("complete-block key attribution does not conserve material")
        return OperatorBenchmarkReport(
            method="curl-haar-gated-q9", configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest, numeric_contract_digest=numeric,
            native_implementation_digest=resources["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("wavelet_gated.py"), input_source=source,
            resources=OperatorBenchmarkResources(
                (key_bytes, key_bytes), (8 * key_bytes, 8 * key_bytes),
                (sum(resources["phase_frame_bytes_per_party"]),) * 2, resources["peer_payload_bits"],
                resources["rounds"], resources["public_table_bytes_per_party"],
                resources["protocol_allocation_estimate_bytes"], 8 * self.lanes, 24 * self.lanes,
                role_names=("worker_a", "worker_b"),
                public_metadata_bytes_estimate=resources["public_metadata_bytes_estimate"],
                public_profile_bytes_estimate=64,
                key_payload_bytes_by_operation=tuple((name, size, size) for name, size in zip(_OPERATIONS, parts, strict=True))),
            quality=OperatorBenchmarkQuality(self.lanes * self.samples, self.lanes * self.samples,
                "Python float64 SiLU(gate) * up on supplied Q18 inputs; preceding float conversion excluded",
                max(errors), math.fsum(errors) / self.lanes,
                sum(error == 0 for error in base_errors) * self.samples, max(base_errors)),
            samples=tuple(samples), citations=(
                OperatorCitation("curl", "Curl: Private LLMs through Wavelet-Encoded Look-Up Tables",
                    "https://eprint.iacr.org/2024/1127", "dc930fdb3219c989106f96eb9ae1f00f28aae16ec5d17eef0c5421eb4f241c1e",
                    "Sections 4.1.1/4.2.1, Figures 4/6: private one-hot lookup and Haar compression; exact truncation adaptation"),
                OperatorCitation("fss-mixed", "Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation",
                    "https://eprint.iacr.org/2020/1392", "dc94c63a77a5cb52cfd12b186a327d045de4a1fc662060c1de6412b334f4fcf4",
                    "Figure 14 universal interval keys for exact rescaling; ties-even is a PLLM extension")),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", resources["native_version"])),
            limitations=(
                "Trusted in-process dealer/client and two one-use evaluators; no independent security review or authenticated role transport",
                "Complete bounded Q18-to-Q9 block; exact floor index reduction, exact ties-even up/output shifts, fresh Beaver multiplication",
                "PLLM mathematical SiLU table, integer Haar means and exact rescaling differ from Curl's probabilistic-truncation system",
                "Input float conversion and range admission remain trusted-client work; secret float conversion and range proofs are absent",
                "Both-party keys and four phase/direction frames are charged; offline key framing and physical wire are unknown",
                "Client input-sharing/output-reconstruction payload estimates are separate; public tables are reusable, secret material is not",
                "Native CPU includes public fitting; local online time includes sharing/reconstruction; Python oracle/report costs are separate",
                "Combined allocation ceiling is an estimate, not measured peak memory; no tensor schedule or whole-decoder transport binding",
                "Private inputs, outputs and fingerprints are omitted; aggregate approximation error is retained; replay requires supplied inputs",
            ))

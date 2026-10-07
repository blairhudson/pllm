"""Installed bounded activation benchmarks with independent integer oracles."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from numbers import Integral
import platform
import struct
import time

from .operator_benchmarks import (
    _OperatorBenchmark, _sdk_digest, OperatorBenchmarkQuality,
    OperatorBenchmarkReport, OperatorBenchmarkResources,
    OperatorBenchmarkSample, OperatorCitation,
)

_COMPACT = OperatorCitation(
    "compact", "Compact: Approximating Complex Activation Functions for Secure Computation",
    "https://arxiv.org/abs/2309.04664",
    "e4497a81055e60cca1c462dac5ae04b4504c5f3fc76e392e170cc32bf8fee86c",
    "Public density-aware degree-two Chebyshev fitting; bounded Q7 adaptation, not the paper's MPC system",
)
_CURL = OperatorCitation(
    "curl", "Curl: Private LLMs through Wavelet-Encoded Look-Up Tables",
    "https://eprint.iacr.org/2024/1127",
    "dc930fdb3219c989106f96eb9ae1f00f28aae16ec5d17eef0c5421eb4f241c1e",
    "Section 4.1.1/Figure 4 additive-share lookup and public Haar low-pass compression; input reduction is client-local",
)
_LOGROW = OperatorCitation(
    "logrow", "Garbled Circuit Lookup Tables with Logarithmic Number of Ciphertexts",
    "https://eprint.iacr.org/2024/369",
    "ac61a6a77b1f21656fa6a90a2f550e59d854542e137cc0eaebb29e718de98371",
    "Masked-index lookup with the earlier two-row one-hot setup; includes masked table bits",
)


def _round_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(abs(numerator), denominator)
    quotient += 2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1)
    return -quotient if numerator < 0 else quotient


def _evaluate(pieces, value: int) -> int:
    """Independent integer evaluation of the committed public coefficients."""
    lower, upper, (constant, linear, quadratic) = next(
        piece for piece in pieces if piece[0] <= value <= piece[1])
    scale = 1 << 20
    t = _round_even((2 * value - lower - upper) * scale, upper - lower)
    square = _round_even(t * t, scale)
    result = constant + _round_even(linear * t, scale) + _round_even(quadratic * (2 * square - scale), scale)
    return _round_even(result, scale)


def _values(values: Sequence[int] | None, lanes: int) -> tuple[tuple[int, ...], str]:
    if values is None:
        boundaries = (-128, -127, -96, -65, -64, -1, 0, 1, 63, 64, 95, 127, 128)
        return tuple(boundaries[i] if i < len(boundaries) else i * 73 % 257 - 128
                     for i in range(lanes)), "public-q7-boundary-fixture-v1"
    if isinstance(values, (str, bytes, bytearray)) or len(values) != lanes:
        raise ValueError("values_q7 must contain exactly the admitted number of integers")
    result = []
    for index in range(lanes):
        value = values[index]
        if isinstance(value, bool) or not isinstance(value, Integral) or not -128 <= value <= 128:
            raise ValueError("values_q7 requires integers in [-128, 128]")
        result.append(int(value))
    return tuple(result), "caller-owned; supply inputs again for replay"


class _ActivationBenchmark(_OperatorBenchmark):
    __slots__ = ()

    def run(self, *, values_q7: Sequence[int] | None = None) -> OperatorBenchmarkReport:
        from pllm import _native, __version__
        from pllm.nonlinear import CompactPiecewiseActivation, CurlWaveletEncodedTable

        values, source = _values(values_q7, self.lanes)
        method = ("compact" if isinstance(self.component, CompactPiecewiseActivation)
                  else "curl" if isinstance(self.component, CurlWaveletEncodedTable) else "logrow")
        params = self.component.get_params()
        levels = params.get("levels", 0)
        counts = struct.pack("<257I", *params["public_calibration_counts"])
        profile = _native.fit_compact_silu_q7_reference(counts, params["max_pieces"])
        pieces = profile.pieces
        original = [_evaluate(pieces, min(i, 256) - 128) for i in range(512)]
        width = 1 << levels
        coefficients = [_round_even(sum(original[i:i + width]), width) for i in range(0, 512, width)]
        expected = [coefficients[(value + 128) >> levels] for value in values]
        numeric = profile.digest.hex()
        if levels:
            numeric = hashlib.sha256(b"pllm.haar_q7.profile.v1\0" + profile.digest + bytes([levels])
                                     + struct.pack(f"<{len(coefficients)}i", *coefficients)).hexdigest()
        packed = struct.pack(f"<{self.lanes}h", *values)
        samples, resources = [], None
        for _ in range(self.samples):
            started = time.process_time_ns()
            output, raw = _native.activation_benchmark(method, params["max_pieces"], counts, levels, packed)
            cpu = (time.process_time_ns() - started) / 1e9
            measured = json.loads(raw)
            if (measured["profile_digest"] != profile.digest.hex()
                    or measured["pieces"] != [[lo, hi, list(c)] for lo, hi, c in pieces]
                    or measured["lanes"] != self.lanes or len(output) != 2 * self.lanes
                    or list(struct.unpack(f"<{self.lanes}h", output)) != expected):
                raise RuntimeError("native activation differs from independent integer oracle or profile")
            timings = {key: measured.pop(key) for key in ("issuance_ns", "online_ns", "clear_ns", "public_setup_ns")}
            if resources is not None and measured != resources:
                raise RuntimeError("native public activation layout changed between fresh samples")
            resources = measured
            phases = tuple(tuple(frame) for frame in measured["phase_frame_bytes"])
            if phases != (((68 * self.lanes, 68 * self.lanes),) if method == "curl" else ()):
                raise RuntimeError("native activation peer frame conservation failed")
            samples.append(OperatorBenchmarkSample(
                timings["issuance_ns"] / 1e9, timings["online_ns"] / 1e9,
                timings["clear_ns"] / 1e9, cpu, phases, timings["public_setup_ns"] / 1e9,
            ))
        assert resources is not None
        keys = tuple(resources["party_key_payload_bytes"])
        directions = tuple(sum(frame[i] for frame in samples[0].peer_frame_bytes_by_phase) for i in (0, 1))
        errors = [abs(actual / 128 - (value / 128) / (1 + math.exp(-value / 128)))
                  for value, actual in zip(values, expected, strict=True)]
        base_errors = [abs(actual - original[value + 128]) for value, actual in zip(values, expected, strict=True)]
        return OperatorBenchmarkReport(
            method=f"{method}-activation-q7",
            configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest, numeric_contract_digest=numeric,
            native_implementation_digest=resources["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("activations.py"), input_source=source,
            resources=OperatorBenchmarkResources(
                keys, tuple(value * 8 for value in keys), directions, resources["peer_payload_bits"],
                len(samples[0].peer_frame_bytes_by_phase), resources["public_table_bytes_per_party"],
                resources["protocol_allocation_estimate_bytes"], 0,
                resources["client_entry_exit_payload_estimate_bytes"],
                role_names=("worker_a", "worker_b") if method == "curl" else ("garbler_client", "evaluator"),
                public_metadata_bytes_estimate=resources["public_metadata_bytes_estimate"],
                public_profile_bytes_estimate=257 * 4 + len(pieces) * 28 + 64,
                evaluator_ciphertext_bytes=resources["evaluator_ciphertext_bytes"] if method == "compact" else None,
                evaluator_constant_label_bytes=resources["evaluator_constant_label_bytes"] if method == "compact" else None,
            ),
            quality=OperatorBenchmarkQuality(
                self.lanes * self.samples, self.lanes * self.samples,
                "Python float64 SiLU on supplied signed Q7 codes; input quantization excluded",
                max(errors), math.fsum(errors) / self.lanes,
                sum(error == 0 for error in base_errors) * self.samples, max(base_errors),
            ),
            samples=tuple(samples), citations=(_COMPACT,) + ((_CURL,) if method == "curl" else (_LOGROW,) if method == "logrow" else ()),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", resources["native_version"])),
            limitations=(
                "One fresh issuance per lane; serial lanes retire material before the next; local garbler/dealer only",
                "Compact/LogRow key roles are garbler-client and evaluator; Curl key roles are two workers; ownership contracts differ",
                "Garbled constant labels and client encodings are charged; public topology/profile and boundary bytes are estimates",
                "Only Curl has serialized evaluator-peer frames; garbled input/output labels are client boundary payload estimates",
                "Public fitting/table construction is timed separately; native-call CPU includes it; Python oracle/report work is outside timings",
                "Clear timing is lookup in the committed public Q7 table; it is not a timed plaintext polynomial evaluator",
                "No protected input quantization or Curl index reduction, gated product, role transport or executable decoder composition",
                "Allocation is a conservative serial-live estimate, not measured peak memory; full wire and independent security review are absent",
                "Caller-owned values and outputs are omitted; aggregate approximation errors are retained",
            ),
        )

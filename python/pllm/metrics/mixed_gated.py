"""Fixed numeric profiles with party-local arithmetic/Boolean bridges."""
from __future__ import annotations

import json
import math
import platform
import secrets
import struct
import time

from .operator_benchmarks import (
    _OperatorBenchmark, _sdk_digest, OperatorBenchmarkQuality, OperatorBenchmarkReport,
    OperatorBenchmarkResources, OperatorBenchmarkSample, OperatorCitation,
)


class _MixedGatedBenchmark(_OperatorBenchmark):
    __slots__ = ()

    def run(self, *, gate_q18=None, up_q18=None):
        from pllm import _native, __version__
        from pllm.nonlinear import CompactGatedMultiplyQ12, CurlWaveletGatedMultiplyQ9
        from .activations import _COMPACT, _LOGROW, _CURL
        from .wavelet_gated import _inputs, _profile, _oracle

        gate, up, source = _inputs(gate_q18, up_q18, self.lanes)
        config = dict(method="logrow", levels=0, max_pieces=0, degree=0, counts=[])
        if isinstance(self.component, CompactGatedMultiplyQ12):
            from .compact_product import _profile as compact_profile, _oracle as compact_oracle
            _, profile = compact_profile(self.component)
            params = self.component.params
            config.update(method="compact", max_pieces=params["max_pieces"], degree=params["degree"],
                          counts=list(params["public_calibration_counts"]))
            numeric = profile["profile_digest"]
            expected = [compact_oracle(profile, g, u) for g, u in zip(gate, up, strict=True)]
            base = expected
            scale, citation = 4096, _COMPACT
            profile_bytes = 1024 * 4 + len(profile["pieces"]) * (8 + 8 * (params["degree"] + 1)) + 128
        else:
            if isinstance(self.component, CurlWaveletGatedMultiplyQ9):
                config.update(method="curl", levels=self.component.params["levels"])
            table, numeric = _profile(config["levels"])
            expected = [_oracle(table, config["levels"], g, u) for g, u in zip(gate, up, strict=True)]
            original, _ = _profile(0)
            base = [_oracle(original, 0, g, u) for g, u in zip(gate, up, strict=True)]
            scale, citation = 512, _CURL if config["method"] == "curl" else _LOGROW
            profile_bytes = len(table) * 4 + 64
        packed = [struct.pack(f"<{self.lanes}i", *values) for values in (gate, up)]
        samples, resources = [], None
        for _ in range(self.samples):
            start = time.process_time_ns()
            output, raw = _native.mixed_gated_benchmark(json.dumps(config), *packed,
                bytes.fromhex(self.digest), secrets.token_bytes(32))
            cpu = (time.process_time_ns() - start) / 1e9
            measured = json.loads(raw)
            if (measured["profile_digest"] != numeric or measured["lanes"] != self.lanes
                    or len(output) != 4 * self.lanes or list(struct.unpack(f"<{self.lanes}i", output)) != expected):
                raise RuntimeError("mixed gated block differs from independent profile oracle")
            timings = {key: measured.pop(key) for key in ("issuance_ns", "online_ns", "clear_ns", "public_setup_ns")}
            if resources is not None and measured != resources:
                raise RuntimeError("mixed gated public resources depend on private inputs")
            resources = measured
            samples.append(OperatorBenchmarkSample(timings["issuance_ns"] / 1e9, timings["online_ns"] / 1e9,
                timings["clear_ns"] / 1e9, cpu, tuple(map(tuple, measured["phase_frame_bytes"])), timings["public_setup_ns"] / 1e9))
        keys = tuple(resources["party_key_payload_bytes"])
        parts = tuple(map(tuple, resources["key_payload_bytes_by_operation"]))
        if tuple(sum(row[i] for row in parts) for i in (1, 2)) != keys:
            raise RuntimeError("mixed gated material attribution does not conserve owners")
        phases = resources["phase_frame_bytes"]
        directions = tuple(sum(row[i] for row in phases) for i in (0, 1))
        errors = [abs(y / scale - (g / 2**18) / (1 + math.exp(-g / 2**18)) * u / 2**18)
                  for g, u, y in zip(gate, up, expected, strict=True)]
        base_errors = [abs(a - b) for a, b in zip(base, expected, strict=True)]
        return OperatorBenchmarkReport(
            method=f"{config['method']}-mixed-arithmetic-gated",
            configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest, numeric_contract_digest=numeric,
            native_implementation_digest=resources["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("mixed_gated.py"), input_source=source,
            resources=OperatorBenchmarkResources(keys, tuple(v * 8 for v in keys), directions,
                resources["peer_payload_bits"],
                resources["peer_rounds"], resources["public_table_bytes_per_party"],
                resources["protocol_allocation_estimate_bytes"], resources["client_mask_payload_estimate_bytes"],
                resources["client_entry_exit_payload_estimate_bytes"], role_names=("garbler_worker", "evaluator_worker"),
                public_metadata_bytes_estimate=resources["public_metadata_bytes_estimate"],
                public_profile_bytes_estimate=profile_bytes,
                evaluator_ciphertext_bytes=resources["evaluator_ciphertext_bytes"],
                evaluator_constant_label_bytes=resources["evaluator_constant_label_bytes"],
                key_payload_bytes_by_operation=parts),
            quality=OperatorBenchmarkQuality(self.lanes * self.samples, self.lanes * self.samples,
                "Python float64 SiLU(gate) * up on supplied Q18 inputs; preceding float conversion excluded",
                max(errors), math.fsum(errors) / self.lanes,
                sum(error == 0 for error in base_errors) * self.samples, max(base_errors)),
            samples=tuple(samples), citations=(citation, OperatorCitation("half-gates",
                "Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits using Half Gates",
                "https://eprint.iacr.org/2014/756", "789ecaf8651b6dc8341d6033f951d76c0e6f4b04427cdde26cf91ecfb0af3083",
                "Figure 2 half-gates; PLLM dealer-aided random OT, hashed output decoding and arithmetic bridges")),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                ("pllm", __version__), ("native", resources["native_version"])),
            limitations=(
                "Trusted in-process dealer issues fresh random-OT correlations, garbling and Beaver triples; distributed setup absent",
                "Each worker owns one input/output share; no intermediate plaintext decode; garbler alone retains free-XOR delta",
                "Evaluator output hash commitments expose only freshly masked outputs, never output encoding pairs",
                "PLLM mixed arithmetic adaptation preserves the selected numeric profile; not a paper-system reproduction",
                "Serial lanes retire complete material before next issuance; peer rounds and framed directions charge every lane",
                "Fixed public input ranges and float conversion stay client-local; no malicious-input proof or independent review",
                "Offline material codecs, authenticated role transport, compiler tensor schedules and full response costs are absent",
                "Allocation bound prices all simultaneous native material; measured peak RSS and physical wire remain unknown",
                "Changed backend does not repair historical approximation errors; fresh checkpoint quality is a separate measurement",
            ))

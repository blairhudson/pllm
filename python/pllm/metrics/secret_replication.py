"""Full-field Secret Replication scalar/vector consumer with independent oracle."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import platform
import struct
import time

from pllm.configuration import ConfigurationError
from .operator_benchmarks import (
    _OperatorBenchmark, _sdk_digest, OperatorBenchmarkQuality, OperatorBenchmarkReport,
    OperatorBenchmarkResources, OperatorBenchmarkSample, OperatorCitation,
)

_SOURCE = OperatorCitation(
    "pcf-secret-replication",
    "Compressing Correlations via Secret Replication: PCFs from Symmetric Cryptography",
    "https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITC.2026.7",
    "6adaab779eae9ec559d735e0984de8d33e8d757b8ca7f1551057fb79a6eec9d2",
    "Sections 4.1–4.2 two-party line PCF/SVMT, Remark 25 GGM puncturing; local AES adaptation and full-field Beaver consumer",
)


class _SecretReplicationBenchmark(_OperatorBenchmark):
    __slots__ = ()

    def run(self, *, scalar: Sequence[int] | None = None,
            vector: Sequence[int] | None = None) -> OperatorBenchmarkReport:
        from pllm import _native, __version__

        params = self.component.get_params()
        field = params["field"]
        source = "public-fixture" if scalar is None and vector is None else "caller-owned"
        if source == "public-fixture":
            scalar, vector = [field - 1], [(i * 19) % field for i in range(self.lanes)]
        if (not isinstance(scalar, Sequence) or not isinstance(vector, Sequence)
                or isinstance(scalar, (str, bytes)) or isinstance(vector, (str, bytes))
                or len(scalar) != 1 or len(vector) != self.lanes
                or any(type(v) is not int or not 0 <= v < field for values in (scalar, vector) for v in values)):
            raise ConfigurationError("PCF requires one scalar and lanes vector entries, all canonical field integers")
        expected = [scalar[0] * v % field for v in vector]
        packed = struct.pack(f"<{self.lanes}I", *vector)
        samples, resources = [], None
        for _ in range(self.samples):
            start = time.process_time_ns()
            output, raw = _native.secret_replication_benchmark(field, params["encoding"], scalar[0], packed)
            cpu = (time.process_time_ns() - start) / 1e9
            measured = json.loads(raw)
            if (measured["field"] != field or measured["lanes"] != self.lanes
                    or len(output) != 4 * self.lanes
                    or list(struct.unpack(f"<{self.lanes}I", output)) != expected):
                raise RuntimeError("PCF differs from independent modular product oracle")
            timings = {key: measured.pop(key) for key in ("issuance_seconds", "expansion_seconds", "online_seconds", "clear_ns")}
            if resources is not None and measured != resources:
                raise RuntimeError("PCF public resources changed between one-use samples")
            resources = measured
            frames = tuple(measured["peer_frame_bytes_per_party"])
            samples.append(OperatorBenchmarkSample(
                timings["issuance_seconds"], timings["online_seconds"], timings["clear_ns"] / 1e9,
                cpu, (frames,), correlation_expansion_seconds=timings["expansion_seconds"],
            ))
        assert resources is not None
        keys = tuple(resources["party_key_payload_bytes"])
        numeric = {"schema": "pllm.scalar_vector.full_field.v1", "field": field,
                   "output": "scalar * vector modulo field", "scalar_count": 1}
        digest = "sha256:" + hashlib.sha256(json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return OperatorBenchmarkReport(
            method="secret-replication-scalar-vector",
            configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest, numeric_contract_digest=digest,
            native_implementation_digest=resources["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("secret_replication.py"), input_source=source,
            resources=OperatorBenchmarkResources(
                keys, tuple(v * 8 for v in keys), frames, 2 * 32 * (self.lanes + 1), 1, 0,
                resources["protocol_allocation_estimate_bytes"], 4 * (self.lanes + 1),
                16 * self.lanes + 8 + 4 * 78,
                expanded_correlation_bytes_per_party=tuple(resources["expanded_correlation_bytes_per_party"]),
                prf_field_evaluations_per_party=tuple(resources["prf_field_evaluations_per_party"]),
            ),
            quality=OperatorBenchmarkQuality(
                self.lanes * self.samples, self.lanes * self.samples,
                "Exact Python integer multiplication modulo field; floating approximation not applicable", 0.0, 0.0,
            ),
            samples=tuple(samples), citations=(_SOURCE,),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", resources["native_version"])),
            limitations=(
                "Two independent VOLE directions, one local dealer and two distinct non-cloneable party states",
                "One full-field scalar mask for the whole vector; each issuance and all coordinates consumed once",
                "Restricted scalar domains are rejected by this consumer; native references retain an exact opening-leakage witness",
                "Raw key payload counts seeds and scalar shares; there is no external key serialization or authenticated transport",
                "Worker correlation expansion is charged separately from dealer issuance and online openings",
                "PRF counts are field-output evaluations; rejection sampling and GGM tree expansion add AES work",
                "Expanded correlations remain resident through consumption; compact distribution does not remove that storage",
                "Client input/output sharing and framing are estimates; total physical wire and peak memory remain unknown",
                "Not ring32 elementwise triples, FSS-key compression, distributed setup or a Qwen nonlinear replacement",
                "Independent cryptographic review and a compiler-bound whole-decoder consumer remain unimplemented",
            ),
        )

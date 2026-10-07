"""Bounded RAA matvec costs with a Python signed-integer oracle."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
    "maverick", "Maverick: Private and Verifiable LLM Inference Made Practical via Matrix-Vector Multiplication Delegation",
    "https://arxiv.org/abs/2609.10264v1",
    "c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0",
    "Protocol 3, Eq. (1), Appendices A/B; BabyBear RAA algebra without distance or dual-LPN certification",
)


@dataclass(frozen=True, slots=True)
class _CodedDelegationBenchmark(_OperatorBenchmark):
    rows: int = 64
    columns: int = 64

    def __post_init__(self) -> None:
        _OperatorBenchmark.__post_init__(self)
        for name in ("rows", "columns"):
            if type(getattr(self, name)) is not int or not 64 <= getattr(self, name) <= 16384:
                raise ConfigurationError(f"Maverick {name} must be an integer in [64, 16384]")

    def to_dict(self) -> dict:
        return _OperatorBenchmark.to_dict(self) | {
            "schema": "pllm.operator_benchmark.v2", "geometry": {"rows": self.rows, "columns": self.columns},
        }

    def run(self, *, weights: Sequence[int] | None = None,
            vectors: Sequence[int] | None = None) -> OperatorBenchmarkReport:
        from pllm import _native, __version__

        projection = json.loads(_native.coded_delegation_projection(self.rows, self.columns, self.lanes))
        if not projection["admitted"]:
            raise ConfigurationError("Maverick benchmark exceeds 256 MiB allocation or 64M product bound")
        source = "public-fixture" if weights is None and vectors is None else "caller-owned"
        if source == "public-fixture":
            weights = [((i * 31 + i // self.columns * 7) & 255) - 128 for i in range(self.rows * self.columns)]
            vectors = [((i * 17 + i // self.columns * 19) & 255) - 128 for i in range(self.lanes * self.columns)]
        for values, size in ((weights, self.rows * self.columns), (vectors, self.lanes * self.columns)):
            if (not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != size
                    or any(type(v) is not int or not -128 <= v <= 127 for v in values)):
                raise ConfigurationError("Maverick inputs must be row-major signed-i8 weights and vectors of the declared shape")
        # Unbounded Python integer accumulation is independent of the native
        # field arithmetic and SIMD clear control. Never archive values or hashes.
        expected = [sum(weights[row * self.columns + col] * vectors[query * self.columns + col]
                        for col in range(self.columns))
                    for query in range(self.lanes) for row in range(self.rows)]
        packed_weights = bytes(v & 255 for v in weights)
        packed_vectors = bytes(v & 255 for v in vectors)
        seed = bytes.fromhex(self.component.get_params()["public_code_seed"])
        samples = []
        for _ in range(self.samples):
            start = time.process_time_ns()
            output, raw = _native.coded_delegation_benchmark(
                self.rows, self.columns, self.lanes, seed, packed_weights, packed_vectors,
            )
            cpu = (time.process_time_ns() - start) / 1e9
            measured = json.loads(raw)
            timings = {name: measured.pop(name) for name in (
                "public_setup_seconds", "issuance_seconds", "online_seconds",
                "evaluation_seconds", "verification_seconds", "clear_seconds",
            )}
            if measured != projection or list(struct.unpack(f"<{len(expected)}i", output)) != expected:
                raise RuntimeError("Maverick resources or outputs differ from independent signed-integer oracle")
            samples.append(OperatorBenchmarkSample(
                timings["issuance_seconds"], timings["online_seconds"], timings["clear_seconds"], cpu, (),
                public_setup_seconds=timings["public_setup_seconds"],
                evaluation_seconds=timings["evaluation_seconds"], verification_seconds=timings["verification_seconds"],
            ))
        numeric = {"schema": "pllm.matvec.signed_i8.v1", "field": projection["field"],
                   "output": "exact centered signed-i32 product; no bias, scaling or rounding"}
        digest = "sha256:" + hashlib.sha256(json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        query_bytes = projection["total_query_payload_bytes"]
        return OperatorBenchmarkReport(
            method="maverick-raa-matvec", configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest, numeric_contract_digest=digest,
            native_implementation_digest=projection["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("coded_delegation.py"), input_source=source,
            resources=OperatorBenchmarkResources(
                (query_bytes, 0), (8 * query_bytes, 0), (0, 0), 0, 0, 0,
                projection["protocol_allocation_estimate_bytes"], 0,
                projection["field_boundary_payload_estimate_bytes"], role_names=("client", "evaluator"),
                key_payload_bytes_by_operation=(("one-use-query-payload", query_bytes, 0),),
                public_table_bytes_by_party=(projection["persistent_client_payload_bytes"], projection["evaluator_weight_bytes"]),
            ),
            quality=OperatorBenchmarkQuality(
                len(expected) * self.samples, len(expected) * self.samples,
                "Exact Python signed-integer dot products; floating-point model fidelity not measured", 0.0, 0.0,
            ),
            samples=tuple(samples), citations=(_SOURCE,),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", projection["native_version"])),
            limitations=(
                "RAA distance and dual-LPN privacy are uncertified; conditional checks and fresh masks are not security proofs",
                "Appendix B's cited field range q > 2^31 excludes BabyBear; the seeded code ensemble has no distance certificate",
                "Privacy sparsity follows conditional Eq. (1); verification uses two fresh weight-41 challenges after the response",
                "Public P, Q and code payload stay client-owned; the evaluator's public i8 matrix is counted separately",
                "Key fields count total one-use masked-input/correction payload across serial queries, with only one query live at once",
                "Boundary bytes estimate raw field requests/replies; no wire codec, measured peer frames or authenticated transport exists",
                "Every sample builds fresh preprocessing; public setup, mask issuance, evaluation, verification and clear control are separate",
                "online_seconds includes evaluation plus verification; issuance requires the input and must be added to complete query time",
                "Native CPU includes setup/control; the independent Python oracle, import and distribution are outside those timings",
                "Native allocation estimate prices copied weights, inputs, outputs and scratch; Python objects, allocator/RSS and OS cache are unmeasured",
                "No public-seed input masks, reusable query handles, batch Protocol 2, decoder binding or checkpoint-quality result",
                "The historical Qwen2.5 whole-body P+Q lower bound is 22.90 GB before code payload, scratch and distribution",
            ),
        )

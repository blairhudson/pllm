"""Transfer-inclusive CKKS operator reference with independent float64 oracles."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import platform
import struct
import tempfile
import time
from pathlib import Path

from pllm.configuration import ConfigurationError
from pllm.runtime import ckks_reference as ckks
from .operator_benchmarks import (
    _OperatorBenchmark, _sdk_digest, OperatorBenchmarkQuality, OperatorBenchmarkReport,
    OperatorBenchmarkResources, OperatorBenchmarkSample, OperatorCitation,
)

_SOURCE = OperatorCitation(
    "nexus", "NEXUS: Secure Transformer Inference Made Non-interactive",
    "https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf",
    "c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2",
    "Section II RNS-CKKS SIMD operations only. PLLM's diagonal-slot matrix reference does not implement Section III coefficient packing, MOAI, bootstrapping or a NEXUS decoder",
)
_TOLERANCE = 0.0002


@dataclass(frozen=True, slots=True)
class _EncryptedLinearBenchmark(_OperatorBenchmark):
    rows: int = 8
    columns: int = 8

    def __post_init__(self):
        _OperatorBenchmark.__post_init__(self)
        if self.lanes > 64:
            raise ConfigurationError("encrypted linear admits at most 64 serial vectors")
        for name in ("rows", "columns"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 32:
                raise ConfigurationError(f"encrypted linear {name} must be in [1,32]")

    def to_dict(self):
        return _OperatorBenchmark.to_dict(self) | {
            "schema": "pllm.operator_benchmark.v2", "geometry": {"rows": self.rows, "columns": self.columns},
        }

    def run(self, *, weights: Sequence[float] | None = None,
            vectors: Sequence[float] | None = None) -> OperatorBenchmarkReport:
        from pllm import _native, __version__
        try:
            import _sealapi_cpp as seal
        except ImportError as error:
            raise RuntimeError("encrypted operators require pllm.run[he]") from error

        width = 1 << (max(self.rows, self.columns) - 1).bit_length()
        policy = self.component.params["rotation_keys"]
        steps = (() if width == 1 else (1,)) if policy == "single" else tuple(range(1, width))
        # All shape/material bounds are resolved before fitting, key generation
        # or decoding caller inputs. This is an estimate, not measured peak RSS.
        metadata = json.loads(_native.ckks_parameters(1, len(steps), False, 8))
        source = "public-fixture" if weights is None and vectors is None else "caller-owned"
        if source == "public-fixture":
            weights = [((i * 7 + i // self.columns * 3) % 19 - 9) / 16 for i in range(self.rows * self.columns)]
            vectors = [((i * 11 + i // self.columns * 5) % 31 - 15) / 16 for i in range(self.lanes * self.columns)]
        for values, size in ((weights, self.rows * self.columns), (vectors, self.lanes * self.columns)):
            if (not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != size
                    or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in values)):
                raise ConfigurationError("encrypted inputs require exact shape and finite [-1,1] numeric values")
        start = time.perf_counter()
        expected = [math.fsum(weights[r * self.columns + c] * vectors[q * self.columns + c]
                             for c in range(self.columns))
                    for q in range(self.lanes) for r in range(self.rows)]
        clear_seconds = time.perf_counter() - start
        start = time.perf_counter()
        diagonals = _native.ckks_diagonals(struct.pack(f"<{len(weights)}d", *weights), self.rows, self.columns)
        public_seconds = time.perf_counter() - start
        samples, errors, resource_records = [], [], []
        for _ in range(self.samples):
            cpu = time.process_time()
            with tempfile.TemporaryDirectory(prefix="pllm-ckks-linear-") as directory:
                client, provider, measured = ckks.setup(seal, _native, Path(directory), depth=1,
                    rotation_steps=steps, relinearization=False, live_ciphertexts=8, queries=self.lanes,
                    binding=bytes.fromhex(self.digest))
                frames, output = [], []
                started = time.perf_counter()
                evaluation_seconds = 0.0
                try:
                    for q in range(self.lanes):
                        vector = list(vectors[q * self.columns:(q + 1) * self.columns]) + [0.0] * (width - self.columns)
                        request = client.request(vector * (client.encoder.slot_count() // width))
                        start = time.perf_counter()
                        reply = provider.evaluate(request, lambda arithmetic, x: ckks.diagonal_linear(arithmetic, x, diagonals, policy))
                        evaluation_seconds += time.perf_counter() - start
                        output.extend(client.receive(reply, 1)[:self.rows])
                        frames.append((len(request), len(reply)))
                finally:
                    client.retire()
                    provider.retire()
                online = time.perf_counter() - started
                sample_errors = [abs(a - b) for a, b in zip(output, expected, strict=True)]
                if max(sample_errors) > _TOLERANCE:
                    raise RuntimeError("CKKS result exceeds the independent float64 tolerance")
                errors.extend(sample_errors)
                counts = tuple(sorted(provider.arithmetic.counts.items()))
                samples.append(OperatorBenchmarkSample(measured["setup_seconds"], online, clear_seconds,
                    time.process_time() - cpu, tuple(frames), public_setup_seconds=public_seconds,
                    evaluation_seconds=evaluation_seconds, homomorphic_operations=counts,
                    minimum_remaining_depth=provider.arithmetic.minimum_remaining_depth))
                setup = tuple((key, value, 0) for key, value in sorted(measured["setup_frame_bytes"].items()))
                resource_records.append((measured["parameters"], measured["secret_polynomial_bytes"], setup,
                                         (provider.request_bytes, provider.reply_bytes), counts))
        if any(record != resource_records[0] for record in resource_records[1:]):
            raise RuntimeError("CKKS public-shape resource counts changed across samples")
        parameters, secret, setup, directions, _ = resource_records[0]
        public_key = measured["setup_frame_bytes"]["public-key"]
        provider_keys = sum(value for name, value, _ in setup if name != "parameters")
        keys = (secret + public_key, provider_keys)
        contract = {"schema": "pllm.ckks.diagonal_linear.v1",
                    "parameters": {name: parameters[name] for name in ("degree", "coefficient_bits", "scale_bits", "depth")},
                    "rows": self.rows, "columns": self.columns,
                    "input_bound": 1, "weight_bound": 1, "float64_absolute_tolerance": _TOLERANCE}
        return OperatorBenchmarkReport(
            method="packed-ckks-linear", configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest,
            numeric_contract_digest="sha256:" + hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            native_implementation_digest=metadata["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("encrypted_linear.py"), input_source=source,
            resources=OperatorBenchmarkResources(keys, tuple(8*v for v in keys), directions, 8*sum(directions),
                self.lanes, 0, parameters["allocation_bytes_estimate"], 0, 0,
                role_names=("client-key-owner", "encrypted-evaluator"),
                public_table_bytes_by_party=(0, 8*self.rows*self.columns + 8*width*width),
                key_payload_bytes_by_operation=(("secret-polynomial-and-public-key", keys[0], 0),
                                                ("provider-evaluation-keys", 0, keys[1])),
                setup_frame_bytes_by_operation=setup, homomorphic_polynomial_degree=parameters["degree"],
                homomorphic_coefficient_bits=tuple(parameters["coefficient_bits"]),
                homomorphic_scale_bits=parameters["scale_bits"]),
            quality=OperatorBenchmarkQuality(len(errors), 0,
                "Independent float64 math.fsum dense dot products; CKKS is approximate, never exact integer parity",
                max(errors), math.fsum(errors)/len(errors), absolute_tolerance=_TOLERANCE,
                outputs_within_tolerance=len(errors)),
            samples=tuple(samples), citations=(_SOURCE,),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", metadata["native_version"])),
            limitations=(
                "Bounded 1..32 matrix dimensions, 1..64 serial vectors, public weights and hidden client vectors in [-1,1]",
                "N=8192, [50,40,50]-bit coefficient primes and 2^40 scale; SEAL TC128; one consumed rescale level",
                "Every sample creates a fresh client key; provider receives only parameters, public and Galois keys in its separate context",
                "Input/output frames include context/sequence headers and public padding; setup frames are separate and charged once per sample",
                "Raw-key fields include the secret polynomial and padded evaluation-key representations, not measured native pool residency",
                "Single-step rotation reuses one Galois key sequentially; full keys rotate from the original input; both costs and CKKS errors are measured",
                "Public-weight compilation, key setup, local online execution and provider evaluation are timed separately; no checkpoint transfer is measured",
                "Client alone decrypts final outputs; request lifecycle rejects foreign/replayed/malformed frames and retires on failure",
                "Framing is not authenticated transport, malicious-result verification, circuit privacy or independent cryptographic review",
                "No bootstrapping, compiler schedule, model quality, autoregressive service or MOAI/NEXUS system reproduction",
                "Allocation is a bounded estimate including key copies, contexts, scratch and live ciphertexts; process RSS and physical wire remain unknown",
            ),
        )

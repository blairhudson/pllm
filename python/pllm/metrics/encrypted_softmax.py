"""Bounded CKKS softmax with independent Decimal profile and exp oracles."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import time

from pllm.configuration import ConfigurationError
from pllm.runtime import ckks_reference as ckks, ckks_softmax as schedule
from .operator_benchmarks import (
    _OperatorBenchmark, _sdk_digest, OperatorBenchmarkQuality, OperatorBenchmarkReport,
    OperatorBenchmarkResources, OperatorBenchmarkSample, OperatorCitation,
)

_TOLERANCE = 0.0002
_SOURCES = {
    "repeated_squaring": OperatorCitation(
        "nexus", "NEXUS: Secure Transformer Inference Made Non-interactive",
        "https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf",
        "c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2",
        "Equation (4) repeated-squaring exponential; fixed public shift, narrow domain and three unrelaxed reciprocal iterations are PLLM adaptations; no secure maximum or bootstrap",
    ),
    "square_normalize": OperatorCitation(
        "thor", "THOR: Secure Transformer Inference with Homomorphic Encryption",
        "https://eprint.iacr.org/2024/1881",
        "ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd",
        "Appendix C degree-15 exponential and Algorithm 3 square/normalize with delta1=1, delta2=2; narrow domain and two unrelaxed reciprocal iterations replace the paper's relaxation schedule",
    ),
}


def _oracles(scores, width, method):
    """Horner/direct-power Decimal oracle, independent of ciphertext schedule."""
    profile, truth = [], []
    with localcontext() as context:
        context.prec = 60
        coefficients = [Decimal(str(v)) for v in schedule.THOR_EXP]
        def normalized(values, iterations, squared=False):
            scaled = [v * width if squared else v / width for v in values]
            total = sum(scaled)
            inverse = (1 - (1 - total) ** (2**iterations)) / total
            return [v * inverse for v in scaled]
        for offset in range(0, len(scores), width):
            row = [Decimal(str(v)) for v in scores[offset:offset + width]]
            exponentials = [v.exp() for v in row]
            denominator = sum(exponentials)
            truth.extend(float(v / denominator) for v in exponentials)
            if method == "repeated_squaring":
                values = [(1 + (v - Decimal("0.125")) / 256) ** 256 for v in row]
                result = normalized(values, 3)
            else:
                values = []
                for v in row:
                    value = coefficients[-1]
                    for coefficient in reversed(coefficients[:-1]):
                        value = value * v / 2 + coefficient
                    values.append(value)
                probabilities = normalized(values, 2)
                result = normalized([v*v for v in probabilities], 2, squared=True)
            profile.extend(float(v) for v in result)
    return profile, truth


@dataclass(frozen=True, slots=True)
class _EncryptedSoftmaxBenchmark(_OperatorBenchmark):
    width: int = 4

    def __post_init__(self):
        _OperatorBenchmark.__post_init__(self)
        if self.lanes > 64:
            raise ConfigurationError("encrypted softmax admits at most 64 serial rows")
        if type(self.width) is not int or self.width not in (1, 2, 4, 8, 16, 32):
            raise ConfigurationError("encrypted softmax width must be a power of two in [1,32]")

    def to_dict(self):
        return _OperatorBenchmark.to_dict(self) | {
            "schema": "pllm.operator_benchmark.v2", "geometry": {"width": self.width},
        }

    def run(self, *, scores: Sequence[float] | None = None) -> OperatorBenchmarkReport:
        from pllm import _native, __version__
        try:
            import _sealapi_cpp as seal
        except ImportError as error:
            raise RuntimeError("encrypted operators require pllm.run[he]") from error
        metadata = json.loads(_native.ckks_softmax_parameters(self.width))
        source = "public-fixture" if scores is None else "caller-owned"
        if scores is None:
            scores = [((i * 7 + i // self.width * 3) % 17 - 8) / 64
                      for i in range(self.width * self.lanes)]
        if (not isinstance(scores, Sequence) or isinstance(scores, (str, bytes))
                or len(scores) != self.width * self.lanes
                or any(type(v) not in (int, float) or abs(v) > schedule.BOUND or not math.isfinite(v)
                       for v in scores)):
            raise ConfigurationError("encrypted softmax requires exact shape and finite scores in [-1/8,1/8]")
        method = self.component.params["method"]
        started = time.perf_counter()
        profile, truth = _oracles(scores, self.width, method)
        oracle_seconds = time.perf_counter() - started
        errors, profile_errors, mass_errors, samples, resources = [], [], [], [], []
        for _ in range(self.samples):
            cpu = time.process_time()
            with tempfile.TemporaryDirectory(prefix="pllm-ckks-softmax-") as directory:
                client, provider, measured = ckks.setup(seal, _native, Path(directory),
                    depth=schedule.DEPTH, rotation_steps=(1,) if self.width > 1 else (),
                    relinearization=True, live_ciphertexts=32, queries=self.lanes,
                    binding=bytes.fromhex(self.digest), softmax_width=self.width)
                frames, output, evaluation_seconds = [], [], 0.0
                started = time.perf_counter()
                try:
                    for offset in range(0, len(scores), self.width):
                        row = list(scores[offset:offset + self.width])
                        request = client.request(row * (client.encoder.slot_count() // self.width))
                        before = time.perf_counter()
                        reply = provider.evaluate(request, lambda a, x: schedule.evaluate(a, x, method, self.width))
                        evaluation_seconds += time.perf_counter() - before
                        output.extend(client.receive(reply, schedule.DEPTH)[:self.width])
                        frames.append((len(request), len(reply)))
                finally:
                    client.retire()
                    provider.retire()
                online = time.perf_counter() - started
                e = [abs(a-b) for a,b in zip(output, truth, strict=True)]
                p = [abs(a-b) for a,b in zip(output, profile, strict=True)]
                if max(p) > _TOLERANCE:
                    raise RuntimeError("CKKS output exceeds the independent numeric-profile tolerance")
                errors.extend(e)
                profile_errors.extend(p)
                mass_errors.extend(abs(math.fsum(output[i:i + self.width]) - 1)
                                   for i in range(0, len(output), self.width))
                counts = tuple(sorted(provider.arithmetic.counts.items()))
                samples.append(OperatorBenchmarkSample(measured["setup_seconds"], online, oracle_seconds,
                    time.process_time() - cpu, tuple(frames), evaluation_seconds=evaluation_seconds,
                    homomorphic_operations=counts, minimum_remaining_depth=provider.arithmetic.minimum_remaining_depth))
                setup = tuple((name, size, 0) for name, size in sorted(measured["setup_frame_bytes"].items()))
                resources.append((measured["parameters"], measured["secret_polynomial_bytes"], setup,
                                  (provider.request_bytes, provider.reply_bytes), counts))
                del client, provider, request, reply
        if any(record != resources[0] for record in resources[1:]):
            raise RuntimeError("CKKS public-shape resources changed across samples")
        parameters, secret, setup, directions, _ = resources[0]
        keys = (secret + measured["setup_frame_bytes"]["public-key"],
                sum(size for name, size, _ in setup if name != "parameters"))
        contract = {"schema": "pllm.ckks.softmax.v1", "method": method, "width": self.width,
                    "public_score_bound": schedule.BOUND, "depth": schedule.DEPTH,
                    "degree": parameters["degree"], "coefficient_bits": parameters["coefficient_bits"],
                    "scale_bits": parameters["scale_bits"], "ckks_profile_tolerance": _TOLERANCE,
                    "thor_exponential": schedule.THOR_EXP if method == "square_normalize" else None,
                    "reciprocal_iterations": [3] if method == "repeated_squaring" else [2, 2]}
        return OperatorBenchmarkReport(
            method="packed-ckks-softmax", configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest,
            numeric_contract_digest="sha256:" + hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            native_implementation_digest=metadata["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("encrypted_softmax.py"), input_source=source,
            resources=OperatorBenchmarkResources(keys, tuple(8*v for v in keys), directions, 8*sum(directions),
                self.lanes, 0, parameters["allocation_bytes_estimate"], 0, 0,
                role_names=("client-key-owner", "encrypted-evaluator"),
                public_table_bytes_by_party=(0, 8*len(schedule.THOR_EXP) if method == "square_normalize" else 0),
                key_payload_bytes_by_operation=(("secret-polynomial-and-public-key", keys[0], 0),
                                                ("provider-evaluation-keys", 0, keys[1])),
                setup_frame_bytes_by_operation=setup, homomorphic_polynomial_degree=parameters["degree"],
                homomorphic_coefficient_bits=tuple(parameters["coefficient_bits"]),
                homomorphic_scale_bits=parameters["scale_bits"]),
            quality=OperatorBenchmarkQuality(len(errors), 0,
                "60-digit Decimal exp/sum softmax; separate Decimal Horner/direct-power profile error; no exact integer claim",
                max(errors), math.fsum(errors)/len(errors), absolute_tolerance=_TOLERANCE,
                outputs_within_tolerance=sum(e <= _TOLERANCE for e in errors),
                maximum_profile_absolute_error=max(profile_errors),
                mean_profile_absolute_error=math.fsum(profile_errors)/len(profile_errors),
                maximum_normalization_error=max(mass_errors)),
            samples=tuple(samples), citations=(_SOURCES[method],),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", metadata["native_version"])),
            limitations=(
                "Fixed public score range [-1/8,1/8], power-of-two widths 1..32, and 1..64 serial rows; no private range fitting or clipping",
                "SEAL TC128, degree 32768, [50]+[40]*14+[50] primes and scale 2^40; both complete schedules consume fourteen levels",
                "Source exponential and square/normalize primitives use narrow-domain unrelaxed reciprocals, not paper relaxation schedules or complete systems",
                "Every sample issues fresh keys; provider imports a separate public-only context and client alone decrypts final normalized outputs",
                "One Galois key executes width-1 rotations per reduction; input rows repeat across slots and all ciphertext products/relinearizations are charged",
                "Setup includes parameters, public, rotation and relinearization frames; full context/sequence padded request/reply frames remain separate",
                "Raw-key fields are payload accounting, not native pool residency; preflight includes serialized copies, context scratch and 32 live ciphertexts under 2 GiB",
                "Profile error isolates CKKS arithmetic from analytic softmax approximation and probability-mass error; reported outputs are aggregate only",
                "Local lifecycle framing is not authenticated transport, malicious-result verification, circuit privacy or independent cryptographic review",
                "No secure maximum, bootstrap, attention QK/value products, masking, model quality, autoregressive state or compiler-bound decoder",
                "Physical wire and measured peak memory are unknown; large evaluation keys are not a client-memory improvement",
            ),
        )

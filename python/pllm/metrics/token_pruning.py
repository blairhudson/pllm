"""Exact threshold/compaction benchmarks with explicit public-count leakage."""
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
    "cipherprune", "CipherPrune: Efficient and Scalable Private Transformer Inference",
    "https://arxiv.org/html/2502.16782v2",
    "cc6df60a5e12bcc65caec5a94a51a810d7b5cab25daf55c5b476949cb2c4dbb2",
    "Sections 3.2–3.3 and Appendix A Figures 13–14: exact score-numerator thresholds, protected count, stable compaction and disclosed compacted degree decisions; local half-gates adaptation",
)


@dataclass(frozen=True, slots=True)
class _TokenPruningBenchmark(_OperatorBenchmark):
    tokens: int = 4
    heads: int = 1

    def __post_init__(self):
        mixed = self.component.params.get("backend") == "mixed"
        if mixed:
            for name, maximum in (("lanes", 8192), ("samples", 16)):
                value = getattr(self, name)
                if type(value) is not int or not 1 <= value <= maximum:
                    raise ConfigurationError(f"{name} must be an integer in [1, {maximum}]")
        else:
            _OperatorBenchmark.__post_init__(self)
        if (type(self.tokens) is not int or not 1 <= self.tokens <= 256
                or type(self.heads) is not int or not 1 <= self.heads <= 32
                or self.heads * self.tokens * self.tokens > 1 << 20):
            raise ConfigurationError("pruning geometry exceeds bounded attention dimensions")
        if mixed:
            from pllm import _native
            params = self.component.get_params()
            factor = self.heads * self.tokens
            plan = {"rows": self.tokens, "words": self.lanes, "selection": {
                "method": "threshold", "threshold": params["threshold_q16"] * factor,
                "reduction": params["reduction_threshold_q16"] * factor,
            }}
            if not json.loads(_native.shared_selection_projection(json.dumps(plan), 0))["admitted"]:
                raise ConfigurationError("mixed pruning exceeds worst-count native allocation/material/work admission")

    def to_dict(self):
        result = _OperatorBenchmark.to_dict(self)
        result.update(schema="pllm.operator_benchmark.v2", geometry={"tokens": self.tokens, "heads": self.heads})
        return result

    def run(self, *, attention_q16: Sequence[int] | None = None,
            words: Sequence[int] | None = None) -> OperatorBenchmarkReport:
        from pllm import _native, __version__

        params = self.component.get_params()
        source = "public-fixture" if attention_q16 is None and words is None else "caller-owned"
        if source == "public-fixture":
            weights = [1 << (i % 4) for i in range(self.tokens)]
            row = [w * 65536 // sum(weights) for w in weights]
            row[-1] += 65536 - sum(row)
            attention_q16 = row * (self.heads * self.tokens)
            words = [(i * 0x9E3779B9) & 0xFFFFFFFF for i in range(self.tokens * self.lanes)]
        sizes = (self.heads * self.tokens * self.tokens, self.tokens * self.lanes)
        for name, values, size, maximum in zip(("attention_q16", "words"), (attention_q16, words), sizes, (65536, 0xFFFFFFFF), strict=True):
            if (not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != size
                    or any(type(v) is not int or not 0 <= v <= maximum for v in values)):
                raise ConfigurationError(f"{name} must contain {size} bounded unsigned integers")
        # Arbitrary-precision sums and stable Python filtering form an independent
        # oracle. No native scoring, sorting or comparison helper is reused.
        scores = [sum(attention_q16[i::self.tokens]) for i in range(self.tokens)]
        factor = self.heads * self.tokens
        selected = [i for i, score in enumerate(scores) if score > params["threshold_q16"] * factor]
        expected = [value for i in selected for value in words[i * self.lanes:(i + 1) * self.lanes]]
        degrees = ([scores[i] > params["reduction_threshold_q16"] * factor for i in selected]
                    if len(selected) < self.tokens else [])
        if params["backend"] == "mixed":
            from .shared_selection import run_pruning
            return run_pruning(self, scores, words, source, expected, degrees, len(selected), sizes)
        packed = [struct.pack(f"<{len(v)}I", *v) for v in (attention_q16, words)]
        samples, resources = [], None
        for _ in range(self.samples):
            start = time.process_time_ns()
            output, mask, raw = _native.token_pruning_benchmark(
                self.tokens, self.heads, self.lanes, params["threshold_q16"], params["reduction_threshold_q16"], *packed)
            cpu = (time.process_time_ns() - start) / 1e9
            measured = json.loads(raw)
            if (measured["retained"] != len(selected) or measured["tokens"] != self.tokens
                    or measured["features"] != self.lanes or measured["heads"] != self.heads
                    or measured["reduction_active"] != (len(selected) < self.tokens)
                    or output != struct.pack(f"<{len(expected)}I", *expected) or mask != bytes(degrees)):
                raise RuntimeError("private pruning differs from independent stable-filter oracle")
            timing = {key: measured.pop(key) for key in ("sharing_seconds", "issuance_seconds", "online_seconds", "clear_seconds")}
            if resources is not None and measured != resources:
                raise RuntimeError("pruning resources changed for the same public count")
            resources = measured
            samples.append(OperatorBenchmarkSample(
                sum(timing["issuance_seconds"]), sum(timing["online_seconds"]), timing["clear_seconds"], cpu, (),
                input_sharing_seconds=timing["sharing_seconds"],
            ))
        assert resources is not None
        phases = resources["phases"]
        keys = tuple(sum(p[name] for p in phases) for name in ("issuer_key_bytes", "evaluator_material_bytes"))
        numeric = {"schema": "pllm.numeric.token_pruning.q16.v1", "parameters": {
                       k: v for k, v in params.items() if k != "backend"},
                   "score": "exact nonnegative Q16 column-sum numerator", "comparison": "strict greater",
                   "records": "stable opaque 32-bit words", "degree_selection": "only after pruning"}
        digest = hashlib.sha256(json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        checked = (1 + len(expected) + len(degrees)) * self.samples
        return OperatorBenchmarkReport(
            method="cipherprune-threshold-stable-compaction",
            configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest, numeric_contract_digest=digest,
            native_implementation_digest=resources["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("token_pruning.py"), input_source=source,
            resources=OperatorBenchmarkResources(
                keys, tuple(v * 8 for v in keys), (0, 0), 0, 0, 0,
                resources["protocol_allocation_estimate_bytes"],
                4 * (sum(sizes) + len(expected)) + self.tokens,
                sum(p["boundary_payload_estimate_bytes"] for p in phases),
                role_names=("garbler_client", "evaluator"),
                public_metadata_bytes_estimate=sum(p["public_metadata_bytes_estimate"] for p in phases),
                evaluator_ciphertext_bytes=sum(p["evaluator_ciphertext_bytes"] for p in phases),
                evaluator_constant_label_bytes=sum(p["constant_label_bytes"] for p in phases),
                key_payload_bytes_by_operation=tuple(
                    (name, p["issuer_key_bytes"], p["evaluator_material_bytes"])
                    for name, p in zip(("threshold-count", "stable-compaction-degree"), phases, strict=True)),
            ),
            quality=OperatorBenchmarkQuality(
                checked, checked, "Exact Python integer thresholds and stable word filtering; no floating approximation or model quality measured", 0.0, 0.0),
            samples=tuple(samples), citations=(_SOURCE,),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", resources["native_version"])),
            limitations=(
                "Retained count determines public resources; compacted-order degree decisions are also disclosed; original keep positions are not separately opened",
                "Disclosed decisions and token words are checked locally but not archived; count reveals an aggregate threshold predicate even for normalized attention",
                "Public Q16 thresholds and exact numerator comparison replace learned thresholds and sharewise fixed-point division",
                "A dedicated marker bit preserves all signed/unsigned payload bits; complete score and token records move together",
                "Up to two fresh one-use half-gates circuits replace source OT primitives; count-dependent issuance follows worst-case 64 MiB preflight; empty/all-retained outputs skip compaction",
                "Only the trusted local garbler decodes; the in-process dealer holds fixture inputs and creates both independent share owners",
                "Local share generation has its own timing; no distributed input encoding, peer codec, authenticated provider transport or measured wire",
                "Keys include issuer encodings, evaluator ciphertexts and constant labels; boundary and public metadata sizes remain estimates",
                "Degree decisions do not execute selected polynomials; trained checkpoint, protected float conversion, causal state integration and decoder quality remain absent",
            ),
        )

"""Mixed-mode private selection, with public count/degree disclosure accounted."""
from __future__ import annotations

import hashlib
import json
import platform
import struct
import time

from .operator_benchmarks import (
    _sdk_digest, OperatorBenchmarkQuality, OperatorBenchmarkReport,
    OperatorBenchmarkResources, OperatorBenchmarkSample,
)
from .token_pruning import _SOURCE


def run_pruning(benchmark, scores, words, source, expected, degrees, retained, sizes):
    from pllm import _native, __version__

    params = benchmark.component.get_params()
    factor = benchmark.heads * benchmark.tokens
    specification = {"rows": benchmark.tokens, "words": benchmark.lanes, "selection": {
        "method": "threshold", "threshold": params["threshold_q16"] * factor,
        "reduction": params["reduction_threshold_q16"] * factor,
    }}
    specification_json = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    packed = [struct.pack(f"<{len(v)}I", *v) for v in (scores, words)]
    samples, resources = [], None
    for _ in range(benchmark.samples):
        start = time.process_time_ns()
        output, mask, raw = _native.shared_selection_benchmark(specification_json, *packed)
        cpu = (time.process_time_ns() - start) / 1e9
        measured = json.loads(raw)
        if (measured["retained"] != retained or measured["rows"] != benchmark.tokens
                or measured["words"] != benchmark.lanes
                or measured["degree_disclosed"] != (retained < benchmark.tokens)
                or output != struct.pack(f"<{len(expected)}I", *expected) or mask != bytes(degrees)):
            raise RuntimeError("mixed pruning differs from independent stable-filter oracle")
        timing = {key: measured.pop(key) for key in ("sharing_seconds", "issuance_seconds", "online_seconds")}
        if resources is not None and measured != resources:
            raise RuntimeError("shared-selection resources changed within one public-count cohort")
        resources = measured
        samples.append(OperatorBenchmarkSample(
            timing["issuance_seconds"], timing["online_seconds"], 0.0, cpu,
            tuple(tuple(p) for p in measured["resources"]["phases"]),
            input_sharing_seconds=timing["sharing_seconds"],
        ))
    assert resources is not None
    raw = resources["resources"]
    keys = tuple(raw["key_bytes"])
    frames = tuple(raw["frame_bytes"])
    numeric = {"schema": "pllm.numeric.token_pruning.q16.v1", "parameters": {
        k: v for k, v in params.items() if k != "backend"},
        "score": "exact nonnegative Q16 column-sum numerator", "comparison": "strict greater",
        "records": "stable opaque 32-bit words", "degree_selection": "only after pruning"}
    digest = hashlib.sha256(json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checked = (1 + len(expected) + len(degrees)) * benchmark.samples
    return OperatorBenchmarkReport(
        method="cipherprune-mixed-threshold-stable-compaction",
        configuration_json=json.dumps(benchmark.to_dict(), sort_keys=True, separators=(",", ":")),
        configuration_digest=benchmark.digest, numeric_contract_digest=digest,
        native_implementation_digest=resources["native_implementation_digest"],
        sdk_implementation_digest=_sdk_digest("shared_selection.py"), input_source=source,
        resources=OperatorBenchmarkResources(
            keys, tuple(v * 8 for v in keys), frames,
            8 * (sum(frames) - 2 * 78 * raw["rounds"]), raw["rounds"], 0,
            resources["allocation_estimate_bytes"], 4 * (sum(sizes) + len(expected)) + benchmark.tokens,
            8 * (len(scores) + len(words) + len(expected)),
            role_names=("party_0", "party_1"),
            public_metadata_bytes_estimate=raw["public_metadata_bytes_estimate"],
            evaluator_ciphertext_bytes=raw["ciphertext_bytes"],
            evaluator_constant_label_bytes=raw["constant_bytes"],
            key_payload_bytes_by_operation=tuple(tuple(row) for row in raw["key_parts"]),
        ),
        quality=OperatorBenchmarkQuality(checked, checked,
            "Exact independent Python threshold/filter oracle; no floating approximation or model quality measured", 0.0, 0.0),
        samples=tuple(samples), citations=(_SOURCE,),
        environment=(("platform", platform.platform()), ("python", platform.python_version()),
                     ("pllm", __version__), ("native", resources["native_version"])),
        limitations=(
            "Retained count and compacted-order degree predicates are disclosed; schedules depend only on public geometry/count",
            "One-shot garbled comparisons return arithmetic predicate shares; every full-record conditional move uses fresh Beaver triples",
            "Two party-local primitives consume independent shares; trusted in-process issuance supplies random OT and triples",
            "Client-local exact attention scoring precedes sharing; no provider scoring or trained thresholds are implemented",
            "Serial issuance preflights all possible counts under 64 MiB allocation, 8,388,608 products and 512 MiB cumulative keys",
            "Frames bind shape/context/role/issuance and burn on malformed input, replay or cancellation; transport authentication and malicious security are absent",
            "Both-party keys, conversions, constant labels, public metadata and all peer frames are charged; boundary bytes are estimates",
            "Final output opens only at the local benchmark oracle; no compiler-bound causal KV owner, degree polynomial execution or provider transport",
            "Checkpoint quality, measured process peak memory, full wire and independent cryptographic review remain unknown",
        ),
    )

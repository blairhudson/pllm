"""Opt-in timing wrappers for ordinary SDK response owners, never a runtime mode.

Records contain public timing labels and counters only. No arguments, return
values, exception text, token IDs, arrays, credentials or random state are kept.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import threading
import time
from unittest.mock import patch

IDENTITY = "pllm.private_boundary_decode_profile.selected3.v1"
NAMES = ("maintained", "coupled-serial-u32", "coupled-parallel-uniform_residues")
MAX_SPANS = 4096
EXTRA_MEMORY = 8 << 20
_LOCK = threading.Lock()


def sha256_identity(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

# Higher-priority nested owners remove their intervals from enclosing owners.
# Parallel child diagnostics are intentionally absent from this partition.
PARTITION = (
    ("head_quantization", "head_quantization"),
    ("paired_exchange", "paired_exchange"),
    ("head_codec_decode", "head_codec_decode"),
    ("head_dequantization", "head_dequantization"),
    ("head", "head_unclassified"),
    ("token_lookup", "token_lookup"),
    ("audit_and_sampler", "audit_and_sampler"),
    ("forward", "body_execution_excluding_lookup_and_head"),
)
DIAGNOSTICS = ("http_exchange", "http_exchange_a", "http_exchange_b", "frame_pack", "frame_unpack", "sampler", "lookup_exchange")


def union_ns(intervals):
    """Elapsed coverage, not the sum of overlapping worker durations."""
    total, end = 0, None
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError("negative timing interval")
        total += max(0, right - max(left, end if end is not None else left))
        end = max(right, end if end is not None else right)
    return total


def summarize(spans, window):
    """Clip to one monotonic window, then assign each segment exactly once."""
    left, right = window
    if type(left) is not int or type(right) is not int or right < left:
        raise ValueError("invalid timing window")
    clipped = [(row["label"], max(left, row["start_ns"]), min(right, row["end_ns"]))
               for row in spans if row["end_ns"] > left and row["start_ns"] < right]
    points = sorted({left, right, *(v for _, a, b in clipped for v in (a, b))})
    bins = {name: 0 for _, name in PARTITION}
    bins["unclassified"] = 0
    for a, b in zip(points, points[1:]):
        active = {label for label, start, end in clipped if start <= a and b <= end}
        name = next((name for label, name in PARTITION if label in active), "unclassified")
        bins[name] += b - a
    diagnostics = {}
    for label in DIAGNOSTICS:
        names = {"http_exchange", "http_exchange_a", "http_exchange_b"} if label == "http_exchange" else {label}
        diagnostics[label] = {"union_wall_ns": union_ns((a, b) for name, a, b in clipped if name in names),
            "clipped_call_count": sum(name in names for name, _, _ in clipped)}
    return {"elapsed_ns": right - left, "exclusive_wall_ns": bins,
            "diagnostic_nonadditive": diagnostics,
            "partition_reconciled": sum(bins.values()) == right - left}


class Recorder:
    def __init__(self):
        self.spans = []
        self.windows = []
        self.lock = threading.Lock()
        self.failed = False

    def wrap(self, label, function):
        def measured(*args, **kwargs):
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            try:
                return function(*args, **kwargs)
            finally:
                end, cpu_end = time.monotonic_ns(), time.thread_time_ns()
                with self.lock:
                    if len(self.spans) >= MAX_SPANS:
                        self.failed = True
                        raise RuntimeError("research timing span bound exceeded")
                    self.spans.append({"label": label, "start_ns": start, "end_ns": end,
                        "calling_thread_cpu_ns": cpu_end - cpu})
        return measured

    def capture(self, capture):
        fields = ("started_monotonic_ns", "finished_monotonic_ns",
                  "first_token_monotonic_ns", "last_token_monotonic_ns")
        values = [getattr(capture, key) for key in fields]
        if self.windows or any(type(value) is not int for value in values):
            self.failed = True
            raise RuntimeError("research profile requires one complete request window")
        self.windows.append(values)

    def report(self):
        if self.failed or len(self.windows) != 1:
            raise RuntimeError("incomplete research timing windows")
        start, finish, first, last = self.windows[0]
        if not start <= first <= last <= finish:
            raise RuntimeError("research timing windows out of order")
        # Relative timestamps disclose no wall-clock epoch or user data.
        spans = [{**row, "start_ns": row["start_ns"] - start,
                  "end_ns": row["end_ns"] - start} for row in self.spans]
        return {"schema": IDENTITY, "request": summarize(self.spans, (start, finish)),
            "after_first_output": summarize(self.spans, (first, last)), "spans": spans,
            "windows_relative_ns": {"request": [0, finish - start],
                                    "after_first_output": [first - start, last - start]},
            "role_cpu_attribution": None, "share_generation_attribution": None,
            "scope": "Client process only; calling-thread CPU spans overlap and are not additive. "
                "HTTP wait includes remote compute, network, pacing and client overhead; not role CPU. "
                "Head residual includes share generation, u32 reconstruction, checks and wiping. "
                "Audit bin includes sampler and audit wrapper; no overhead correction."}


def priced_admission(report, memory_bytes):
    """Price bounded profiling scratch in both probe and unprobed control."""
    import copy
    from pllm.runtime.benchmark_memory import admit_memory, host_memory
    estimate = copy.deepcopy(report["estimate"])
    estimate["client_peak_bytes"] += EXTRA_MEMORY
    estimate["native_total_peak_bytes"] += EXTRA_MEMORY
    estimate["components"]["private_boundary_profile_bytes"] = EXTRA_MEMORY
    return admit_memory(estimate, host_memory(), backend="native", enforced_wan=True,
                        memory_budget_bytes=memory_bytes)


@contextmanager
def instrument():
    """Wrap original functions once; no copied numerical or transport algorithm."""
    from pllm.runtime import benchmark_response_audit as audit
    from pllm.runtime import private_boundary_transport as transport
    from pllm.runtime.dashboard import DashboardRuntime
    from pllm.runtime.private_boundary_head_codec import HeadOutputCodec
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime

    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("concurrent research decode profile")
    record = Recorder()
    try:
        with ExitStack() as stack:
            def install(owner, name, label):
                stack.enter_context(patch.object(owner, name, record.wrap(label, getattr(owner, name))))
            install(SemanticDecoderRuntime, "_forward", "forward")
            install(SemanticDecoderRuntime, "_token_lookup", "token_lookup")
            install(transport.PrivateBoundaryTransport, "linear", "head")
            original_exchange = transport.PrivateBoundaryTransport._exchange
            def exchange(owner, operation, *args, **kwargs):
                label = "paired_exchange" if operation == "head" else "lookup_exchange"
                return record.wrap(label, original_exchange)(owner, operation, *args, **kwargs)
            stack.enter_context(patch.object(transport.PrivateBoundaryTransport, "_exchange", exchange))
            original_post = transport.PrivateBoundaryTransport._post
            def post(owner, index, *args, **kwargs):
                label = "http_exchange_a" if index == 0 else "http_exchange_b"
                return record.wrap(label, original_post)(owner, index, *args, **kwargs)
            stack.enter_context(patch.object(transport.PrivateBoundaryTransport, "_post", post))
            install(transport, "pack_frame", "frame_pack")
            install(transport, "unpack_frame", "frame_unpack")
            install(transport, "quantize_activation_per_row", "head_quantization")
            install(transport, "dequantize_matmul", "head_dequantization")
            install(HeadOutputCodec, "decode", "head_codec_decode")
            install(MaskedTransformerClientRuntime, "sample", "sampler")
            original_audit = audit.response_state_audit

            @contextmanager
            def timed_audit(salt):
                with original_audit(salt) as records:
                    with patch.object(MaskedTransformerClientRuntime, "sample",
                            record.wrap("audit_and_sampler", MaskedTransformerClientRuntime.sample)):
                        yield records
            stack.enter_context(patch.object(audit, "response_state_audit", timed_audit))
            original_finish = DashboardRuntime._finish_run

            def finish(runtime, capture, *args, **kwargs):
                record.capture(capture)
                return original_finish(runtime, capture, *args, **kwargs)
            stack.enter_context(patch.object(DashboardRuntime, "_finish_run", finish))
            yield record
    finally:
        _LOCK.release()


@contextmanager
def worker_bootstrap(mode, source_identity):
    """Research worker wraps the installed allowlisted response target executor."""
    from pllm import metrics
    from pllm._benchmark_contract import digest
    from pllm.runtime import benchmark_targets
    original = benchmark_targets.execute_target
    memory = metrics.benchmark_memory
    if not sha256_identity(source_identity):
        raise ValueError("research source identity must be SHA256")

    def admission(*args, **kwargs):
        return priced_admission(memory(*args, **kwargs), kwargs["memory_budget_bytes"])

    def execute(target, measurement):
        if target.kind != "response" or not target.to_spec()["parameters"].get("state_audit"):
            raise ValueError("research profile requires audited response target")
        if mode not in {"control", "probe"}:
            raise ValueError("invalid research profile mode")
        start = time.process_time_ns()
        if mode == "probe":
            with instrument() as record:
                original(target, measurement)
            profile = record.report()
            run, = measurement.details["report"]["runs"]
            for window, duration in (("request", "full_seconds"), ("after_first_output", "generation_seconds")):
                if abs(profile[window]["elapsed_ns"] / 1e9 - run["durations"][duration]) > 1e-6:
                    raise RuntimeError("research window differs from ordinary response metric")
            measurement.details["private_boundary_decode_profile"] = profile
        else:
            original(target, measurement)
        measurement.details["research_profile_identity"] = {"schema": IDENTITY, "mode": mode,
            "sources": source_identity, "executor_client_process_cpu_ns": time.process_time_ns() - start,
            "response_workload_digest": measurement.comparison["workload_digest"]}
        measurement.comparison["measurement_digest"] = digest({
            "ordinary": measurement.comparison["measurement_digest"], "research": IDENTITY,
            "mode": mode, "sources": source_identity})
        measurement.limitations += ["Opt-in research worker bootstrap; not untraced maintained SOTA evidence.",
            "Probe overhead remains charged; compare matched control/probe arms, never subtract a calibration."]

    with patch.object(metrics, "benchmark_memory", admission), patch.object(benchmark_targets, "execute_target", execute):
        yield


@contextmanager
def research_workers(script, source_identity):
    """Keep ordinary runner's watchdog, isolation and child cleanup unchanged.

    Only its exact worker command is redirected. No provider commands change.
    Launcher is serial, and this context must not be shared with another run.
    """
    from pllm.metrics import benchmark_runner as runner
    original_execute = runner._execute
    original_popen = runner.subprocess.Popen

    def execute(case, *args, **kwargs):
        mode = case.name.split("--", 1)[0]
        def popen(command, *p_args, **p_kwargs):
            if command[1:] == ["-I", "-m", "pllm._benchmark_worker"]:
                command = [command[0], "-I", str(script), "--worker", mode, source_identity]
            return original_popen(command, *p_args, **p_kwargs)
        with patch.object(runner.subprocess, "Popen", popen):
            return original_execute(case, *args, **kwargs)
    with patch.object(runner, "_execute", execute):
        yield


def matched_suite(**kwargs):
    from pllm.metrics import BenchmarkCase, BenchmarkSuite
    from .private_boundary_decode import matched_suite as base_suite
    base = base_suite(**kwargs)
    return BenchmarkSuite([BenchmarkCase(f"{mode}--{case.name}", case.target, case.build)
        for case in base.cases if case.name in NAMES for mode in ("control", "probe")], base.sampling, IDENTITY)


def ordinary_comparison(target, details):
    """Reconstruct unsalted response commitments without executing model work.

    The final worker workload HMAC cannot be recomputed without private salt.
    Its pre-HMAC digest is checked here; final HMAC must match cohort-wide.
    """
    from pllm import Experiment
    from pllm._benchmark_contract import digest
    from pllm._search_benchmark import _boundary
    from pllm.runtime.topology_accounting import ACCOUNTING_SCHEMA, accounting_coverage_identity
    p = target.to_spec()["parameters"]
    exp = Experiment.from_spec(p["model"]["experiment"]).with_params(pipeline__profile=p["model"]["profile"])
    report = details["report"]
    config, runs, scenario = report["configuration"], report["runs"], p["scenario"]
    boundary = _boundary(exp)
    measurement = digest({"schema": report["schema_version"], "scenario": scenario, "cached_checkpoint": True})
    coverage = accounting_coverage_identity(report["topology_accounting"])
    if coverage not in {ACCOUNTING_SCHEMA, "historical_coverage_not_recorded"}:
        measurement = digest({"response": measurement, "body_counter_coverage": coverage})
    measurement = digest({"response": measurement, "state_audit": "pllm.response_state_audit.v1"})
    return {"source_digest": config["source_lock_digest"],
        "numeric_digest": digest({"bodies": [r["model_fingerprint"] for r in runs], "contract": boundary["numeric"]}),
        "privacy_digest": digest(boundary["privacy"]), "state_digest": digest({"lifecycle": scenario}),
        "workload_digest": digest({"tokens": [r["tokens"] for r in runs], "scenario": scenario,
            "backend": config["provider_backend"], "wan": config["wan_emulation_digest"], "sampling": config["sampling"]}),
        "output_digest": digest({"output": digest([r["generation"]["output_text_digest"] for r in runs]),
            "state": details["response_state_audit"]["bindings"]}), "measurement_digest": measurement}


def required_metric_contracts(outputs):
    """Explicit metadata for ranked/overhead metrics; topology body scope separate."""
    from pllm._benchmark_evidence import measurement
    specs = {
        "client.peak_rss": dict(kind="memory", window="process_birth_through_scenario", scope="process_lifetime_high_water"),
        "client.cold_cpu": dict(kind="cpu", window="benchmark_start_through_first_response", scope="process_cpu"),
        "preparation.cold_cpu": dict(kind="cpu", role="preparation", window="role_birth_through_first_response", scope="process_cpu"),
        "preparation.peak_rss": dict(kind="memory", role="preparation", window="request_windows", scope="sampled_process_rss"),
        "response.ttft": dict(window="online_to_first_token", phase="online"),
        "response.decode_tps": dict(kind="throughput", window="after_first_output", phase="online",
            denominator={"value": outputs - 1, "unit": "generated_tokens"}),
        "request.0.full": dict(window="request_setup_through_completion", scope="ordinary_response_lifecycle"),
        "request.0.ttft": dict(window="online_to_first_token", phase="online"),
        "request.0.decode_tps": dict(kind="throughput", window="after_first_output", phase="online",
            denominator={"value": outputs - 1, "unit": "generated_tokens"}),
        "network.all_link_stream_bytes": dict(kind="bytes", role="all-links", statistic="sum",
            window="relay_start_through_scenario", scope="tcp_stream_payload"),
        "process.peak_rss": dict(kind="memory", window="process_birth_through_target", scope="process_lifetime_high_water"),
        "process.cpu": dict(kind="cpu", window="process_birth_through_target"),
        "target.cpu": dict(kind="cpu"), "target.latency": {},
    }
    return {name: measurement(name, 0, **params) for name, params in specs.items()}


def _validate_owner_sequence(spans, outputs, boundary, first, last, request_end):
    """One ordered forward/lookup/(head)/audited selection per emitted token."""
    from .sota_combinations import Rejected
    def rows(label):
        return sorted((s for s in spans if s["label"] == label), key=lambda s: s["start_ns"])
    def inside(child, parent):
        return parent["start_ns"] <= child["start_ns"] < child["end_ns"] <= parent["end_ns"]
    forwards, audits = rows("forward"), rows("audit_and_sampler")
    samplers, lookups, heads = rows("sampler"), rows("token_lookup"), rows("head")
    if len(forwards) != outputs or len(audits) != outputs:
        raise Rejected("profile_step_sequence_incomplete")
    for i, (forward, audit, sampler, lookup) in enumerate(zip(forwards, audits, samplers, lookups)):
        if (not 0 <= forward["start_ns"] < forward["end_ns"] <= audit["start_ns"] < audit["end_ns"] <= request_end
                or not inside(sampler, audit) or not inside(lookup, forward)
                or i and audits[i - 1]["end_ns"] > forward["start_ns"]
                or audit["end_ns"] > (first if i == 0 else last)):
            raise Rejected("profile_step_sequence_invalid")
        if boundary:
            head = heads[i]
            if not inside(head, forward):
                raise Rejected("profile_head_not_in_forward")
            labels = ["head_quantization", "paired_exchange"]
            if rows("head_codec_decode"):
                labels.append("head_codec_decode")
            labels.append("head_dequantization")
            children = []
            for label in labels:
                owned = [child for child in rows(label) if inside(child, head)]
                if len(owned) != 1:
                    raise Rejected("profile_head_children_not_owned")
                children.append(owned[0])
            # These owners run synchronously in the original linear() body.
            # Only HTTP/framing diagnostics inside exchange may overlap.
            if any(left["end_ns"] > right["start_ns"] for left, right in zip(children, children[1:])):
                raise Rejected("profile_head_child_order_invalid")


def preflight(suite):
    from .private_boundary_decode import preflight as ordinary
    return [{"case": row["case"], "admission": priced_admission(row["admission"], suite.sampling.memory_bytes)}
            for row in ordinary(suite)]


def audit(document, *, source_identity, tiny=False):
    from .sota_combinations import Rejected
    try:
        return _audit(document, source_identity=source_identity, tiny=tiny)
    except Rejected:
        raise
    except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
        raise Rejected("profile_missing_or_invalid_evidence") from exc


def _audit(document, *, source_identity, tiny=False):
    """Require full-state equality including unprobed arms; no historical pooling."""
    import math
    from pllm import Experiment
    from pllm.metrics import BenchmarkSuite
    from pllm.metrics.benchmark_runner import BenchmarkRun
    from pllm._benchmark_contract import digest
    from pllm._benchmark_evidence import metric_contract, measurement
    from . import private_boundary_decode as decode
    from . import private_boundary_tiers as tiers
    from .sota_combinations import Rejected
    if not sha256_identity(source_identity) or not sha256_identity(document["cohort"]):
        raise Rejected("profile_invalid_source_or_cohort_identity")
    # Validate actual v2 records, build/environment digests, admission, oracle,
    # case identity, comparison commitments and metric schema before using tiers.
    BenchmarkRun(document)
    suite = BenchmarkSuite.from_spec(document["suite"])
    expected = matched_suite(build=suite.cases[0].build, tiny=tiny,
        artifacts=_artifacts(suite), generated_source=_generated(suite) if tiny else None)
    if suite.to_spec() != expected.to_spec():
        raise Rejected("profile_suite_contract_changed")
    summary = tiers.audit(document, tiny=tiny)
    samples = {}
    common = None
    contracts_by_recipe = {}
    for row in document["results"]:
        execution = row["execution"]
        mode, name = execution["case"]["name"].split("--", 1)
        details = execution["details"]
        identity = details.get("research_profile_identity", {})
        if (identity.get("schema") != IDENTITY or identity.get("mode") != mode
                or identity.get("sources") != source_identity or execution["cohort"] != document["cohort"]
                or execution["suite_digest"] != suite.digest
                or execution["build"]["digest"] != suite.cases[0].build.expected_digest
                or execution["isolation"]["fresh_process"] is not True
                or execution["isolation"]["controller_included"] is not False
                or details["report"]["memory_guard"]["maximum_host_swap_growth_bytes"] != 0):
            raise Rejected("profile_identity_isolation_or_memory_changed")
        if (type(identity["executor_client_process_cpu_ns"]) is not int
                or identity["executor_client_process_cpu_ns"] < 0):
            raise Rejected("profile_executor_cpu_invalid")
        report = details["report"]
        if (row["schema_version"] != "pllm.benchmark_result.v2" or row["scope"] != "deployment"
                or row["warmups"] != 0 or row["repetitions"] != 1
                or len(row["samples"]) != 1 or row["samples"][0]["kind"] != "measurement"
                or row["samples"][0]["status"] != "completed"
                or report["schema_version"] != "pllm.loopback_benchmark.v1"
                or report["warmup_runs"] != [] or report["configuration"]["warmups"] != 0
                or report["configuration"]["repetitions"] != 1
                or report["runs"][0]["status"] != "completed" or report["runs"][0]["warm"] is not False
                or execution["failure_code"] is not None or execution["admission"]["admitted"] is not True
                or execution["admission"]["limit_bytes"] != suite.sampling.memory_bytes
                or execution["admission"]["required_bytes"] != 128 << 20
                or execution["admission"]["scope"] != "target_working_set"
                or details["topology_admission"]["admitted"] is not True
                or details["topology_admission"]["estimate"]["components"]["private_boundary_profile_bytes"] != EXTRA_MEMORY
                or details["topology_admission"]["estimate"]["components"]["response_state_audit_bytes"] != 32 << 20):
            raise Rejected("profile_result_lifecycle_or_admission_changed")
        state = details["response_state_audit"]
        binding, = state["bindings"]
        if (set(state["client_memory_owners"]) != {"sdk", "dashboard", "native_wan_relay", "audit", "role_supervision"}
                or binding["sampler_seed"] != 42 or binding["temperature"] != 0
                or binding["cryptographic_randomness_frozen"] is not False
                or execution["sampling"] != suite.sampling.to_spec()
                or execution["fixture"]["metadata"] != {"cached_checkpoint": True, "owned_client_cache": "fresh"}
                or execution["fixture"]["digest"] != report["configuration"]["source_lock_digest"]):
            raise Rejected("profile_state_owners_or_fixture_changed")
        for selection in binding["selections"]:
            if (any(selection.get(key) is None for key in ("position", "cache_lengths", "logits_and_active_kv",
                    "selected_token", "rng_before", "rng_after"))
                    or type(selection.get("checked_values")) is not int or selection["checked_values"] <= 0):
                raise Rejected("profile_full_state_missing")
        if execution["oracle"] != {"kind": "salted_full_logit_kv_sampler_commitments", "outcome": "passed",
                "checked_values": sum(s["checked_values"] for s in binding["selections"])}:
            raise Rejected("profile_oracle_changed")
        target = suite.cases[next(i for i, c in enumerate(suite.cases) if c.name == execution["case"]["name"])].target
        expected_comparison = ordinary_comparison(target, details)
        comparison = execution["comparison"]
        for key, value in expected_comparison.items():
            if key == "workload_digest":
                valid = identity["response_workload_digest"] == value
            elif key == "measurement_digest":
                valid = comparison[key] == digest({"ordinary": value, "research": IDENTITY,
                    "mode": mode, "sources": source_identity})
            else:
                valid = comparison[key] == value
            if not valid:
                raise Rejected("profile_comparison_not_reconstructed")
        current = {"environment": row["environment"], **{key: comparison[key] for key in (
            "source_digest", "numeric_digest", "state_digest", "workload_digest", "output_digest")}}
        if common is not None and current != common:
            raise Rejected("profile_environment_or_comparison_mismatch")
        common = current
        contracts = {m["id"]: metric_contract(m) for m in row["metrics"]}
        complete_contracts = {m["id"]: {key: value for key, value in m.items() if key != "value"}
            for m in row["metrics"]}
        if name in contracts_by_recipe and complete_contracts != contracts_by_recipe[name]:
            raise Rejected("profile_metric_metadata_changed")
        contracts_by_recipe[name] = complete_contracts
        for metric, expected_metric in required_metric_contracts(2 if tiny else 8).items():
            if contracts.get(metric) != metric_contract(expected_metric):
                raise Rejected("profile_required_metric_contract_changed")
            actual, = [m for m in row["metrics"] if m["id"] == metric]
            if actual["sample_count"] != 1 or type(actual["value"]) not in (int, float) or not math.isfinite(actual["value"]) or actual["value"] < 0:
                raise Rejected("profile_required_metric_unavailable")
        metrics = {m["id"]: m["value"] for m in row["metrics"]}
        durations = report["runs"][0]["durations"]
        for field in ("full_seconds", "online_seconds", "ttft_seconds", "generation_seconds"):
            value = durations[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise Rejected("profile_duration_invalid")
        if not 0 < durations["generation_seconds"] <= durations["online_seconds"] <= durations["full_seconds"]:
            raise Rejected("profile_duration_windows_invalid")
        expected_values = {"request.0.full": durations["full_seconds"],
            "request.0.ttft": durations["ttft_seconds"], "response.ttft": durations["ttft_seconds"],
            "request.0.decode_tps": ((2 if tiny else 8) - 1) / durations["generation_seconds"],
            "response.decode_tps": ((2 if tiny else 8) - 1) / durations["generation_seconds"]}
        if any(not math.isclose(metrics[key], value, rel_tol=1e-12, abs_tol=1e-12)
                for key, value in expected_values.items()):
            raise Rejected("profile_response_metric_value_differs")
        boundary = name != "maintained"
        from pllm.runtime.topology_accounting import ACCOUNTING_SCHEMA, accounting_coverage_identity
        coverage = accounting_coverage_identity(report["topology_accounting"])
        body_scope = "covered_application_bodies" + (f":{coverage or 'mixed_or_unknown'}"
            if coverage not in {ACCOUNTING_SCHEMA, "historical_coverage_not_recorded"} else "")
        if contracts["network.covered_body_bytes"] != metric_contract(measurement("network.covered_body_bytes", 0,
                kind="bytes", role="all-links", statistic="sum", window="setup_through_scenario", scope=body_scope)):
            raise Rejected("profile_body_metric_contract_changed")
        decode._wire(report, metrics, boundary)
        exp = Experiment.from_spec(execution["case"]["target"]["parameters"]["model"]["experiment"])
        decode._body(report, metrics, boundary,
            chunk_rows=exp.pipeline.components["linear"].get_params()["prefill_chunk_rows"])
        profile = details.get("private_boundary_decode_profile")
        if mode == "probe":
            if not profile or profile["schema"] != IDENTITY or len(profile["spans"]) > MAX_SPANS:
                raise Rejected("profile_spans_missing")
            if (set(profile) != {"schema", "request", "after_first_output", "spans", "windows_relative_ns",
                    "role_cpu_attribution", "share_generation_attribution", "scope"}
                    or profile["role_cpu_attribution"] is not None or profile["share_generation_attribution"] is not None):
                raise Rejected("profile_unknown_bins_relabelled")
            labels = [span["label"] for span in profile["spans"]]
            allowed = {label for label, _ in PARTITION} | set(DIAGNOSTICS)
            if any(set(span) != {"label", "start_ns", "end_ns", "calling_thread_cpu_ns"}
                    or span["label"] not in allowed
                    or any(type(span[key]) is not int for key in ("start_ns", "end_ns", "calling_thread_cpu_ns"))
                    or span["start_ns"] > span["end_ns"] or span["calling_thread_cpu_ns"] < 0
                    for span in profile["spans"]):
                raise Rejected("profile_span_schema_changed")
            outputs = 2 if tiny else 8
            if (labels.count("audit_and_sampler") != outputs or labels.count("sampler") != outputs
                    or labels.count("forward") != outputs or labels.count("token_lookup") != outputs
                    or labels.count("head") != (outputs if boundary else 0)
                    or labels.count("paired_exchange") != (outputs if boundary else 0)):
                raise Rejected("profile_owner_coverage_incomplete")
            if labels.count("head_codec_decode") != (outputs if "uniform_residues" in name else 0):
                raise Rejected("profile_codec_coverage_incomplete")
            if boundary and (labels.count("head_quantization") != outputs
                    or labels.count("head_dequantization") != outputs
                    or labels.count("http_exchange_a") < outputs
                    or labels.count("http_exchange_b") < outputs
                    or labels.count("frame_pack") < 2 * outputs
                    or labels.count("frame_unpack") < 2 * outputs
                    or labels.count("head_codec_decode") != (outputs if "uniform_residues" in name else 0)):
                raise Rejected("profile_boundary_children_incomplete")
            if set(profile["windows_relative_ns"]) != {"request", "after_first_output"}:
                raise Rejected("profile_windows_changed")
            request_start, request_end = profile["windows_relative_ns"]["request"]
            first, last = profile["windows_relative_ns"]["after_first_output"]
            durations = report["runs"][0]["durations"]
            expected_first = durations["full_seconds"] - durations["online_seconds"] + durations["ttft_seconds"]
            if (request_start != 0 or not 0 <= first < last <= request_end
                    or not math.isfinite(expected_first) or abs(first / 1e9 - expected_first) > 1e-6):
                raise Rejected("profile_window_anchors_changed")
            _validate_owner_sequence(profile["spans"], outputs, boundary, first, last, request_end)
            for window, bounds in profile["windows_relative_ns"].items():
                if summarize(profile["spans"], bounds) != profile[window]:
                    raise Rejected("profile_partition_changed")
                duration = "full_seconds" if window == "request" else "generation_seconds"
                seconds = report["runs"][0]["durations"][duration]
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0 or abs(
                        profile[window]["elapsed_ns"] / 1e9 - seconds) > 1e-6:
                    raise Rejected("profile_metric_window_differs")
        elif profile is not None:
            raise Rejected("profile_control_instrumented")
        samples[mode, name, execution["replicate"]] = {"metrics": metrics, "profile": profile}
    overhead = {}
    for name in NAMES:
        overhead[name] = [{"replicate": rep, "metric_deltas_probe_minus_control": {
            metric: samples["probe", name, rep]["metrics"][metric] - samples["control", name, rep]["metrics"][metric]
            for metric in ("request.0.full", "response.decode_tps", "client.cold_cpu", "process.peak_rss")}}
            for rep in range(3)]
    return {"schema": IDENTITY, "clients": summary["clients"], "complete": True,
        "all_logits_active_kv_positions_selections_rng_and_outputs_match": True,
        "state_commitment": summary["state_commitment"], "overhead_pairs": overhead,
        "profiles": [{"case": name, "replicate": rep, "timing": value["profile"]}
            for (mode, name, rep), value in samples.items() if mode == "probe"],
        "scope": "Research-only timings; probe overhead charged, not corrected. "
            "No historical SOTA update or causal attribution from comparator subtraction."}


def _generated(suite):
    from pllm import Experiment
    return Experiment.from_spec(suite.cases[0].target.to_spec()["parameters"]["model"]["experiment"]).pipeline.model


def _artifacts(suite):
    from pathlib import Path
    from pllm import Experiment
    exp = Experiment.from_spec(suite.cases[0].target.to_spec()["parameters"]["model"]["experiment"])
    tokenizer = exp.pipeline.components.get("tokenizer")
    return Path(tokenizer.get_params()["path"]).parent if tokenizer else Path("target/prepared-frontier-artifacts")

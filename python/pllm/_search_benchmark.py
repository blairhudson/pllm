"""Ordinary role benchmarks adapted to typed, cohort-safe search evidence."""

from __future__ import annotations

import gc
import hashlib
import math
import os
import platform
import secrets
from datetime import datetime, timezone

from pllm._version import __version__
from pllm.configuration import ConfigurationError
from pllm.evidence import BenchmarkResult, EvidenceReport, _canonical, environment_digest
from pllm.search import (
    BeamSearch, CandidateRejected, GridSearch, RandomSearch, SearchError,
    SearchOutcome, SearchRejection, evaluate_search,
)
from pllm.search.beam import _rank
from pllm.search import _plain


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _boundary(experiment):
    resolved = experiment.resolve()
    return {
        "source": experiment.pipeline.model.to_spec(),
        "profile": resolved.profile,
        "numeric": {name: value.to_spec() for name, value in experiment.pipeline.components.items()
                    if name in {"quantization", "nonlinear", "scheduler"}},
        "privacy": {
            "mode": resolved.privacy_mode,
            "protocol": resolved.privacy_protocol,
            "graph": resolved.role_graph.to_spec(),
            "verification": resolved.verification_component,
            "failure_bits": resolved.verification_target_failure_bits,
        },
    }


def _measurement(candidate, report, environment, max_output_tokens):
    """Reject incomplete observations before any value may guide the next trial."""
    experiment = candidate.experiment
    config = report.get("configuration", {})
    measured = report.get("experiment", {})
    if (report.get("schema_version") != "pllm.loopback_benchmark.v1"
            or report.get("checks", {}).get("passed") is not True
            or len(report.get("runs", ())) != 1 or report.get("warmup_runs")
            or config.get("warmups") != 0 or config.get("repetitions") != 1
            or measured.get("configuration_digest") != candidate.configuration_digest
            or measured.get("pipeline_digest") != experiment.pipeline.digest()):
        raise SearchError("search benchmark lacks exact checked Experiment evidence")
    run = report["runs"][0]
    tokens = run.get("tokens", {})
    if (run.get("status") != "completed" or tokens.get("authoritative") is not True
            or type(tokens.get("input_tokens")) is not int or tokens["input_tokens"] < 1
            or type(tokens.get("output_tokens")) is not int or tokens["output_tokens"] < 1
            or run.get("warm") is not False):
        raise SearchError("search benchmark needs one authoritative cold response")
    for value in (run.get("model_fingerprint"), config.get("source_lock_digest"),
                  config.get("prompt_digest"), run.get("generation", {}).get("output_text_digest")):
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise SearchError("search benchmark lacks immutable source, body, prompt or output identity")
    sampling = config.get("sampling", {})
    if (sampling.get("effective_temperature") != 0 or sampling.get("mode") != "greedy"
            or sampling.get("top_p") is not None or run.get("sampling", sampling) != sampling):
        raise SearchError("automatic search requires matched greedy sampling")
    full = run.get("durations", {}).get("full_seconds")
    if type(full) not in {int, float} or not math.isfinite(full) or full <= 0:
        raise SearchError("search benchmark request duration is invalid")
    boundary = _boundary(experiment)
    workload = {
        "prompt": config["prompt_digest"], "tokens": {
            "input": tokens["input_tokens"], "output": tokens["output_tokens"],
        },
        "cap": run.get("max_output_tokens"), "warm": run["warm"], "sampling": sampling,
        "output": run["generation"]["output_text_digest"],
        "backend": config.get("provider_backend"),
        "wan": config.get("wan_emulation_digest"), "links": config.get("link_conditions_digest"),
    }
    if workload["cap"] != max_output_tokens or workload["cap"] != config.get("max_output_tokens") or tokens["output_tokens"] > max_output_tokens:
        raise SearchError("search benchmark output cap changed")
    common = {"role": "client", "origin": "external_measured", "sample_count": 1,
              "unavailable_reason": None, "phase": "full"}
    metrics = [
        dict(common, id="full_seconds", component="pllm/latency", value=full,
             parameters={"statistic": "median", "phase": "full"}, unit="seconds",
             unit_detail=None, statistic="median"),
        dict(common, id="request_tps", component="pllm/throughput", value=tokens["output_tokens"] / full,
             parameters={"basis": "tokens"}, unit="per_second", unit_detail="tokens_per_second",
             statistic="median"),
    ]
    for identity, key, phase in (
        ("covered_bytes", "setup_inclusive_mb_per_output_token", "full"),
        ("online_bytes", "online_mb_per_output_token", "online"),
    ):
        per_token = report.get("communication_per_token", {}).get("summary", {}).get(key)
        available = type(per_token) in {int, float} and math.isfinite(per_token) and per_token >= 0
        if per_token is not None and not available:
            raise SearchError("search benchmark communication counter is invalid")
        metrics.append(dict(common, id=identity, component="pllm/communication",
            parameters={"direction": "total", "phase": phase}, phase=phase, role="all-links",
            value=per_token * tokens["output_tokens"] * 1e6 if available else None,
            unit="bytes", unit_detail="covered_application_bodies", statistic="sum",
            origin="external_measured" if available else "not_available", sample_count=1 if available else 0,
            unavailable_reason=None if available else "covered body measurement unavailable"))
    return BenchmarkResult.from_dict({
        "schema_version": "pllm.benchmark_result.v1", "id": candidate.trial_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "completed",
        "scope": "deployment", "profile": experiment.pipeline.profile,
        "component_ids": sorted({value.component for value in experiment.pipeline.components.values()}),
        "model": {"id": run["model_id"], "checkpoint_digest": None,
                  "source_lock_digest": config["source_lock_digest"]},
        "plan_lock_digest": None, "configuration_digest": candidate.configuration_digest,
        "workload_digest": _digest(workload), "privacy_cohort": _digest(boundary["privacy"]),
        "numeric_cohort": _digest({"body": run["model_fingerprint"], "contract": boundary["numeric"]}),
        "environment": environment, "warmups": 0, "repetitions": 1,
        "samples": [{"index": 0, "kind": "measurement", "status": "completed",
                     "duration_seconds": full, "artifact_digest": _digest(report)}],
        "metrics": metrics,
        "limitations": [
            "Single cold response with cached checkpoints; adaptive observations are exploratory.",
            "Co-located role processes; no independent-provider or full physical wire claim.",
            "Client lifetime memory is shared across trials and cannot rank isolated peaks.",
            "Publisher work and distribution of pre-positioned public artifacts are separate costs.",
        ],
    })


def benchmark_search(search, prompt, *, max_output_tokens, objective=None, direction=None,
                     timeout_seconds=900, backend="native", progress=None):
    from pllm.metrics import benchmark_memory
    from pllm.model_loader import expected_model_id
    from pllm.runtime.benchmark_cli import LoopbackBenchmarkError, run_loopback_benchmark

    if not isinstance(search, (BeamSearch, GridSearch, RandomSearch)):
        raise TypeError("search must be BeamSearch, GridSearch or RandomSearch")
    if isinstance(search, BeamSearch):
        if (objective is not None and objective != search.objective) or (direction is not None and direction != search.direction):
            raise SearchError("benchmark objective conflicts with BeamSearch")
        objective, direction = search.objective, search.direction
    else:
        objective, direction = objective or "request_tps", direction or "max"
    if objective not in {"request_tps", "full_seconds", "covered_bytes", "online_bytes"} or direction not in {"min", "max"}:
        raise SearchError("unsupported measured search objective or direction")
    if type(prompt) is not str or not prompt.strip():
        raise ValueError("search benchmark prompt must be nonempty text")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if backend not in {"native", "docker", "auto"}:
        raise ValueError("backend must be native, docker or auto")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    base = search.space.base
    boundary = _boundary(base)
    attributes = {"platform": platform.platform(), "machine": platform.machine(),
                  "python": platform.python_version(), "pllm": __version__, "cpu_count": os.cpu_count()}
    environment = {"digest": environment_digest(attributes), "attributes": attributes}
    salt = secrets.token_bytes(32)
    records = []
    admissions = []

    def admit(candidate):
        # Retire unreachable dashboard/client cycles before pricing the next trial.
        # Their native buffers are not reflected in Python's collection threshold.
        gc.collect()
        experiment = candidate.experiment
        try:
            comparison = _boundary(experiment)
        except (ConfigurationError, ValueError):
            return "component composition is incompatible"
        if comparison != boundary:
            return "model, numeric, topology or verification comparison boundary changed"
        if experiment.budget is None or experiment.budget.requests < 1 or max_output_tokens > experiment.budget.max_new_tokens:
            return "benchmark exceeds the Experiment execution budget"
        memory = benchmark_memory(experiment, backend=backend)
        admissions.append({"trial_id": candidate.trial_id, "memory": memory})
        if not memory["admitted"]:
            return "host memory preflight rejected candidate"
        return None

    def evaluate(candidate):
        experiment = candidate.experiment
        try:
            raw = run_loopback_benchmark(
                model=experiment.pipeline.model.source, model_id=expected_model_id(experiment.pipeline.model),
                tiny=False, experiment=experiment, prompt=prompt,
                max_output_tokens=max_output_tokens, warmups=0, repetitions=1,
                timeout_seconds=timeout_seconds, backend=backend, _cohort_salt=salt,
                temperature=0, capture_output_digest=True,
                progress=(lambda message: progress(candidate, message)) if progress else None,
            )
        except LoopbackBenchmarkError as exc:
            if isinstance(exc.__cause__, KeyboardInterrupt):
                raise exc.__cause__
            raise CandidateRejected("role benchmark failed; no completed measurement") from exc
        result = _measurement(candidate, raw, environment, max_output_tokens)
        records.append({"name": candidate.trial_id, "configuration_digest": candidate.configuration_digest,
                        "pipeline": experiment.pipeline.to_spec(), "report": raw,
                        "python_source": candidate.python_source()})
        return result

    if isinstance(search, BeamSearch):
        outcome = search.evaluate(evaluate, admit=admit)
    else:
        values, rejected = [], []
        candidates = search.candidates()
        for candidate in candidates:
            reason = admit(candidate)
            if reason is not None:
                rejected.append(SearchRejection(candidate.parameters, reason, candidate))
                continue
            try:
                observation, = evaluate_search((candidate,), evaluate)
            except CandidateRejected as exc:
                rejected.append(SearchRejection(candidate.parameters, str(exc), candidate))
                continue
            values.append(observation)
            _rank(values, objective, direction)
        ranked = _rank(values, objective, direction)
        outcome = SearchOutcome(tuple(values), tuple(rejected), ranked[0] if ranked else None,
                                objective, direction, "candidates_exhausted", len(candidates))
    return EvidenceReport(_canonical({
        "schema_version": "pllm.search_benchmark.v1", "strategy": type(search).__name__,
        "policy": {
            "identity": search.identity,
            "width": search.width if isinstance(search, BeamSearch) else None,
            "max_trials": search.max_trials if isinstance(search, BeamSearch) else None,
            "seed": search.seed if isinstance(search, RandomSearch) else None,
            "count": search.count if isinstance(search, RandomSearch) else None,
            "space": {
                "base": search.space.base.to_spec(), "parameters": _plain(search.space.parameters),
                "cardinality": math.prod(len(axis) for axis in search.space.parameters.values()),
                "constraints": [{"path": c.path, "operator": c.operator, "value": _plain(c.value)}
                                for c in search.space.constraints],
            },
        },
        "search": outcome.to_dict(), "candidates": records, "admissions": admissions,
        "environment": environment,
    }))

"""Thin selected-execution adapter for ordinary benchmark records and gateway clients.

No alternate inference implementation: each attempt owns the existing SDK client
and one ExecutionLease. Public artifacts may survive; protected sessions may not.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import secrets
import time
from contextlib import contextmanager
from dataclasses import replace

from pllm.compiler import plan
from pllm.deployment import LivePartyOffer, NetworkError, discover, open_execution

from .benchmark_cli import _comparison_key, build_loopback_report
from .benchmark_history import BenchmarkRun
from .dashboard import _sampling_choice, _validate_request_temperature, client_body_placement_snapshot
from .http_gateway import resource_dict


def _worker_metrics(client, lease=None):
    """Authenticated existing worker CPU endpoint; values are process-wide samples."""
    import httpx

    values = {}
    connections = {role: (http, key, "/v1/offset-reference/metrics")
                   for role, (http, key) in client._core._offset_workers.items()}
    if lease is not None:
        for role, (root, key) in getattr(lease._topology, "_connections", {}).items():
            if role in {"inference", "preparation"}:
                connections[role] = (httpx.Client(base_url=root, timeout=5), key, "/v1/party/metrics")
    for role, (http, key, path) in connections.items():
        try:
            value = http.get(path, headers={"authorization": f"Bearer {key}"}).raise_for_status().json()
            values[role] = {name: value.get(name) for name in ("cpu_ns", "peak_rss_bytes")}
        except (httpx.HTTPError, ValueError):
            values[role] = {"cpu_ns": None, "peak_rss_bytes": None}
        finally:
            if role in {"inference", "preparation"}:
                http.close()
    return values


class SelectedClientFactory:
    """Fixed plan: revalidate/admit. Request: bounded fresh planning before response."""

    def __init__(self, network, *, result=None, request=None, credentials=None, ttl_seconds=60):
        if (result is None) == (request is None):
            raise ValueError("choose exactly one plan or planning request")
        self.network, self.result, self.request = network, result, request
        self.credentials, self.ttl_seconds = credentials, ttl_seconds
        candidates = (result.experiment,) if result is not None else request.candidates
        if any(item is None for item in candidates):
            raise NetworkError("NO_FEASIBLE_PLACEMENT")
        self.model_ids = tuple(sorted({item.resolve().model for item in candidates}))
        self.default_model = self.model_ids[0] if len(self.model_ids) == 1 else None

    def select(self):
        if self.result is not None:
            return self.result
        snapshot = discover(self.network, credentials=self.credentials)
        request = replace(self.request, policy=replace(
            self.request.policy, evaluated_at_ms=time.time_ns() // 1_000_000
        ))
        result = plan(request, snapshot=snapshot)
        if result.status != "feasible":
            raise NetworkError("SEARCH_LIMIT" if result.status == "inconclusive" else "NO_FEASIBLE_PLACEMENT")
        return result

    @contextmanager
    def __call__(self):
        from pllm.runtime.client import OpenAI

        result = self.select()
        with open_execution(result, network=self.network, credentials=self.credentials,
                            ttl_seconds=self.ttl_seconds) as lease:
            with OpenAI(execution=lease) as client:
                yield client


def feasible_controls(result, *, maximum=32):
    """Bounded native-planned controls. Filtering only removes offered authority.

    Live controls enumerate ordered graph-role host assignments, keeping source,
    composition, workload and installed descriptor commitments unchanged. Local
    backend controls compare candidate compositions, not physical assignments.
    """
    request, snapshot = result.request, result.snapshot
    live = any(isinstance(offer, LivePartyOffer) for offer in snapshot.offers)
    if live and len(snapshot.offers) > 9:
        raise ValueError("feasible controls support at most eight remote hosts")
    controls = {}
    for experiment in request.candidates:
        single = replace(request, candidates=(experiment,), cost_evidence=tuple(
            row for row in request.cost_evidence
            if row.configuration_digest == experiment.configuration_digest()
        ), artifact_evidence=tuple(row for row in request.artifact_evidence
                                  if row.configuration_digest == experiment.configuration_digest()))
        snapshots = [snapshot]
        if live:
            roles = tuple(role.id for role in experiment.resolve().role_graph.roles if role.id != "client")
            choices = [tuple(offer.party_id for offer in snapshot.offers
                              if isinstance(offer, LivePartyOffer) and role in offer.role_ids)
                       for role in roles]
            import math
            if math.prod(map(len, choices)) * len(request.candidates) > maximum:
                raise ValueError("feasible controls exceed explicit 32-assignment bound")
            snapshots = []
            for parties in itertools.product(*choices):
                assignment = tuple(zip(roles, parties, strict=True))
                offers = tuple(
                    replace(offer, role_ids=tuple(role for role, party in
                            assignment if party == offer.party_id))
                    if isinstance(offer, LivePartyOffer) else offer
                    for offer in snapshot.offers
                    if not isinstance(offer, LivePartyOffer) or offer.party_id in set(parties)
                )
                ids = {offer.party_id for offer in offers}
                snapshots.append(replace(snapshot, offers=offers, links=tuple(
                    link for link in snapshot.links
                    if {link.source_party_id, link.target_party_id} <= ids
                )))
        for subset in snapshots:
            decision = plan(single, snapshot=subset)
            if decision.status != "feasible":
                continue
            key = _decision_key(decision)
            controls[key] = decision
    controls[_decision_key(result)] = result
    if len(controls) > maximum:
        raise ValueError("feasible controls exceed bounded comparison")
    return [controls[key] for key in sorted(controls)]


def _decision_key(result):
    return (result.experiment.configuration_digest(), tuple(
        (row["role_id"], row["party_id"]) for row in result.native_placement["roles"]
    ))


def _refresh_control(decision, network, credentials):
    """Before-response bounded replanning under control's removal-only allowlist."""
    snapshot = discover(network, credentials=credentials)
    previous = {offer.party_id: offer for offer in decision.snapshot.offers}
    offers = tuple(replace(offer, role_ids=tuple(
        role for role in offer.role_ids if role in previous[offer.party_id].role_ids
    )) if isinstance(offer, LivePartyOffer) else offer
        for offer in snapshot.offers if offer.party_id in previous)
    ids = {offer.party_id for offer in offers}
    snapshot = replace(snapshot, offers=offers, links=tuple(
        link for link in snapshot.links if {link.source_party_id, link.target_party_id} <= ids
    ))
    request = replace(decision.request, policy=replace(
        decision.request.policy, evaluated_at_ms=time.time_ns() // 1_000_000
    ))
    fresh = plan(request, snapshot=snapshot)
    if fresh.status != "feasible" or _decision_key(fresh) != _decision_key(decision):
        raise NetworkError("feasible benchmark control no longer admitted")
    return fresh


def _attempt(result, network, *, credentials, prompt, cap, temperature, timeout,
             capture_output_digest):
    from pllm.runtime.client import OpenAI

    started_at = time.time_ns()
    start = time.monotonic()
    cpu_start = time.process_time()
    first = last = None
    terminal = None
    with open_execution(result, network=network, credentials=credentials,
                        ttl_seconds=min(300, max(5, int(timeout)))) as lease:
        with OpenAI(execution=lease) as client:
            worker_before = _worker_metrics(client, lease)
            online = time.monotonic()
            source = client.responses.create(input=prompt, max_output_tokens=cap, stream=True,
                                             **({"temperature": temperature} if temperature is not None else {}))
            try:
                for event in source:
                    now = time.monotonic()
                    if now - start > timeout:
                        raise TimeoutError("network benchmark bounded response timeout")
                    value = resource_dict(event)
                    if value["type"] == "response.output_text.delta":
                        first = first or now
                        last = now
                    if value["type"] in {"response.completed", "response.incomplete"}:
                        terminal = value["response"]
            finally:
                source.close()
            audit = client.privacy_audit.to_dict()
            worker_after = _worker_metrics(client, lease)
            # Exact locked body fingerprint, not a prompt/output fingerprint.
            state = client._core._transformer_states.get(result.experiment.resolve().model)
            fingerprint = state.bundle.privacy.get("body_fingerprint") if state is not None else None
            body_placement = client_body_placement_snapshot(client, result.experiment.resolve().model)
            if body_placement is not None:
                body_placement["sample_boundary"] = "after-bounded-response-before-client-disposal"
    end = time.monotonic()
    if terminal is None:
        raise RuntimeError("response stream ended without authoritative usage")
    roles = tuple(row["role_id"] for row in result.native_placement["roles"])
    cache_hit = audit.get("bundle_cache_hits", 0) > 0
    run = BenchmarkRun(
        run_id=f"network-{secrets.token_hex(12)}", status="completed",
        model_id=result.experiment.resolve().model, model_fingerprint=fingerprint,
        max_output_tokens=cap, cold=not cache_hit, warm=cache_hit,
        started_at_ns=started_at, finished_at_ns=time.time_ns(),
        full_seconds=end - start, online_seconds=end - online,
        ttft_seconds=None if first is None else first - online,
        generation_seconds=None if first is None or last is None else last - first,
        input_tokens=terminal["usage"]["input_tokens"],
        output_tokens=terminal["usage"]["output_tokens"], token_usage_authoritative=True,
        privacy_delta=audit, process_metrics={role: {
            "cpu_seconds": time.process_time() - cpu_start if role == "client" else None,
            "rss_peak_bytes": None,
        } for role in roles},
    ).to_dict()
    run["sampling"] = _sampling_choice(temperature)
    run["generation"] = {"response_status": terminal["status"]}
    if capture_output_digest:
        text = "".join(content.get("text", "") for item in terminal.get("output", [])
                       for content in item.get("content", []) if content.get("type") == "output_text")
        run["generation"]["output_text_digest"] = hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()
    run["network_audit"] = {
        "placement": [{"role_id": row["role_id"], "party_id": row["party_id"]}
                      for row in result.native_placement["roles"]],
        "prediction": result.to_spec()["selection"]["costs"],
        "controller_request_count": None, "controller_response_body_bytes": None,
        "controller_request_body_bytes": None, "controller_retry_body_bytes": None,
        "full_wire_bytes": None, "full_response_cpu_seconds": None,
        "client_cpu_scope": "shared process delta including admission, response and release",
        "full_seconds_scope": "fresh admission through response and release; model host startup excluded",
        "warm_scope": "public bundle cache hit; fresh protected session and disposable client each attempt",
        "worker_process_samples": {role: {"before": worker_before.get(role), "after": value}
                                   for role, value in worker_after.items()},
        "worker_cpu_scope": "authenticated process-wide samples; shared processes may overlap, never sum as full-response CPU",
        "client_body_placement": body_placement,
        "lease_closed": lease.closed,
    }
    return run


def run_network_benchmark(*, network, result=None, request=None, credentials=None,
                          prompt, max_output_tokens, warmups=0, repetitions=1,
                          timeout_seconds=120, temperature=None, compare_feasible=False,
                          capture_output_digest=False, progress=None):
    """Registered-party driver using BenchmarkRun and existing workload cohort rules."""
    temperature = _validate_request_temperature(temperature)
    if type(capture_output_digest) is not bool:
        raise ValueError("capture_output_digest must be boolean")
    if not 0 <= warmups <= 100 or not 1 <= repetitions <= 100 or not 1 <= timeout_seconds <= 3600:
        raise ValueError("invalid bounded benchmark repetitions/warmups/timeout")
    factory = SelectedClientFactory(network, result=result, request=request, credentials=credentials)
    selected = factory.select()
    if not 1 <= max_output_tokens <= selected.experiment.budget.max_new_tokens:
        raise ValueError("output cap exceeds selected budget")
    controls = feasible_controls(selected) if compare_feasible else [selected]
    salt = secrets.token_bytes(32)
    prompt_digest = hashlib.sha256(b"pllm.benchmark.prompt.v1\0" + salt + prompt.encode()).hexdigest()
    warm_records = [[] for _ in controls]
    records = [[] for _ in controls]
    order = []
    # Public deterministic order; never reused as protocol mask or sampling seed.
    ordering = random.Random(20261001)
    for index in range(warmups + repetitions):
        indices = list(range(len(controls)))
        ordering.shuffle(indices)
        for control_index in indices:
            decision = controls[control_index]
            if request is not None:
                decision = (_refresh_control(decision, network, credentials) if compare_feasible
                            else factory.select())  # Replan only before each response.
            if progress is not None:
                progress(f"Network attempt {index + 1}/{warmups + repetitions}, control {control_index + 1}/{len(controls)}")
            record = _attempt(decision, network, credentials=credentials, prompt=prompt,
                              cap=max_output_tokens, temperature=temperature,
                              timeout=timeout_seconds, capture_output_digest=capture_output_digest)
            (warm_records if index < warmups else records)[control_index].append(record)
            order.append({"phase": "warmup" if index < warmups else "measurement",
                          "repetition": index if index < warmups else index - warmups,
                          "control_index": control_index})
    reports = []
    for decision, warm, runs in zip(controls, warm_records, records, strict=True):
        roles = tuple(row["role_id"] for row in decision.native_placement["roles"])
        report = build_loopback_report(
            model_id=decision.experiment.resolve().model, tiny=False,
            max_output_tokens=max_output_tokens, warmup_runs=warm, runs=runs,
            roles=roles, prompt_digest=prompt_digest, warmup_prompt_digest=prompt_digest,
            source_lock_digest=decision.request.source_lock_digest, temperature=temperature,
            capture_output_digest=capture_output_digest,
        )
        report["scope"] = "selected-network-bounded-response-diagnostic"
        if network.backend == "http":
            report["privacy_admission"] = {
                "independent_operators_verified": False,
                "reason": "authenticated party identities and declared operator separation do not establish physical independence",
            }
        report["limitations"] = [
            "declared operators do not prove physical independence",
            "full wire, controller costs and aggregate whole-response CPU unavailable",
            "transfer-only prediction excludes RTT, framing, compute and critical-path overlap",
        ]
        report["network"] = {"backend": network.backend, "network_digest": network.digest,
                             "selected": _decision_key(decision) == _decision_key(selected),
                             "placement": runs[0]["network_audit"]["placement"]}
        if not report["checks"]["passed"]:
            raise RuntimeError("network benchmark privacy/runtime checks failed")
        reports.append(report)
    if not compare_feasible:
        return reports[0]
    keys = [_comparison_key(report) for report in reports]
    matched = all(key is not None for key in keys) and len(set(keys)) == 1
    values = [report["summary"]["median_full_seconds"] for report in reports]
    selected_index = next(index for index, report in enumerate(reports) if report["network"]["selected"])
    regret = values[selected_index] - min(values) if matched else None
    cohort = None if not matched else {
        "model_fingerprint": keys[0][0],
        "input_tokens": keys[0][1], "output_tokens": keys[0][2],
        "max_output_tokens": keys[0][3], "warm": keys[0][4],
        "effective_sampling": reports[0]["configuration"]["sampling"],
        "source_lock_digest": selected.request.source_lock_digest,
        "prompt_digest": prompt_digest,
    }
    return {
        "schema_version": "pllm.loopback_benchmark_comparison.v1",
        "scope": "selected-network-feasible-controls-diagnostic",
        "checks": {"passed": matched, "matched_workload": matched,
                   "all_candidates_passed": True},
        "candidates": [{"name": f"control-{index}", "configuration_digest": decision.experiment.configuration_digest(),
                        "report": report} for index, (decision, report) in enumerate(zip(controls, reports, strict=True))],
        "summary": {"selected_control_index": selected_index,
                    "selected_plan_latency_regret_seconds": regret,
                    "latency_regret_scope": "matched median measured admission-through-release; not transfer prediction regret",
                    "full_cost_regret": None},
        "comparison_key": cohort,
        "measurement_order": order,
        "control_coverage": {"count": len(controls), "maximum": 32,
                             "scope": "live ordered two-offset host assignments" if network.backend == "http" else "local candidate compositions",
                             "planner_exhaustive": selected.exhaustive},
    }

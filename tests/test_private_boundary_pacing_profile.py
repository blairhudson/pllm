"""Pacing decisions, opaque HTTP hooks, counter conservation and bounded owners."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
import threading
import time

import httpx
import pytest

from benchmarks.research import private_boundary_pacing_profile as p
from pllm.deployment import WanConditions
from pllm.runtime.native_wan import SharedPacer
from test_private_boundary_role_profile import role_cohort, profile_cohort, cohort  # noqa: F401


def interval(start, end, **kwargs):
    return {"start_ns": start, "end_ns": end, "completed": True, **kwargs}


def fixture():
    roles = ("client", "inference", "preparation", *p.ROLES)
    pacer = SharedPacer(WanConditions(), roles)
    profile = {"windows_relative_ns": {"request": [0, 1_000_000_000]}, "spans": []}
    heads = {role: [] for role in p.ROLES}
    totals = {f"{a}->{b}": 0 for role in p.ROLES for a, b in (("client", role), (role, "client"))}
    for ordinal in range(2):
        start = 100_000_000 * (ordinal + 1)
        profile["spans"].append({"label": "paired_exchange", "start_ns": start - 1000, "end_ns": start + 20_000_000})
        for role, label in zip(p.ROLES, ("http_exchange_a", "http_exchange_b")):
            profile["spans"].append({"label": label, "start_ns": start + 2, "end_ns": start + 16_999_999})
            chunks = []
            for direction, count in (("upload", 1000), ("download", 8192)):
                key = f"client->{role}" if direction == "upload" else f"{role}->client"
                totals[key] += count
                offset = 20_000 if direction == "upload" else 4_000_000
                sleeps = [] if direction == "upload" else [interval(start + offset + 1000,
                    start + offset + 2_000_000, requested_seconds=8192 / 5_000_000,
                    source_tokens=0., destination_tokens=16384., source_rate=5_000_000., destination_rate=12_500_000.)]
                chunks.append(interval(start + offset, start + offset + 2_001_000,
                    direction=direction, bytes=count, counter_after=totals[key], sleeps=sleeps))
            http = [interval(start + i * 2_000_000 + 1000, start + (i + 1) * 2_000_000,
                label=label) for i, label in enumerate(p.HTTP_REQUIRED)]
            heads[role].append(interval(start + 1, start + 17_000_000,
                request_body_bytes=800, response_body_bytes=8000, chunks=chunks, http_spans=http))
    pacer.links.update(totals)
    for edge, value in totals.items():
        source, destination = edge.split("->")
        pacer.buckets[source, "upload"]["bytes"] += value
        pacer.buckets[destination, "download"]["bytes"] += value
    pacer.started -= 1
    relay = pacer.snapshot()
    trace = {"schema": p.IDENTITY, "clock": "same_client_process_monotonic_ns",
        "heads": heads, "boundary_stream_bytes": totals, "final_relay": relay}
    report = {"native_network_accounting": {"samples": {"runs": [{"after": {"wan_emulation": relay}}]}},
        "runs": [{"privacy": {"private_boundary_head_upload_bytes": 3200,
            "private_boundary_head_download_bytes": 32000}}]}
    return trace, profile, report


def test_counter_replay_sleep_ownership_and_parallel_unions():
    result = p.summarize_trace(*fixture(), 2)
    assert result["decode_head_ordinals"] == [1]
    assert result["paired_sleep_union_ns"] == [1_999_000] * 2
    head = result["roles"]["boundary_a"][1]
    assert head["links"]["download"]["sleep_ns_by_limiting_bucket"] == {
        "source_upload": 1_999_000, "destination_download": 0, "both": 0}
    assert head["links"]["download"]["http_framing_bytes"] == 192
    assert head["reservation_union_ns"] + head["http_outside_reservation_ns"] == head["http_ns"]
    assert sum(result["head_stream_bytes"].values()) == 4 * 9192


@pytest.mark.parametrize("mutation", ["source_rate", "tokens", "sleep", "nonfinite", "sleep_order",
    "overlap", "chunk_counter", "chunk_bytes", "head_scope", "body", "stream", "role", "unknown",
    "http_missing", "http_order", "http_secret", "head_failed", "chunk_failed", "sleep_failed", "clock", "bounds"])
def test_replay_rejects_structurally_rebound_tampering(mutation, monkeypatch):
    trace, profile, report = deepcopy(fixture())
    head = trace["heads"]["boundary_a"][0]
    chunk = head["chunks"][1]
    sleep = chunk["sleeps"][0]
    if mutation == "source_rate": sleep["source_rate"] = 50_000_000
    elif mutation == "tokens": sleep["source_tokens"] = 16385
    elif mutation == "sleep": sleep["requested_seconds"] *= 2
    elif mutation == "nonfinite": sleep["source_tokens"] = float("nan")
    elif mutation == "sleep_order": sleep["end_ns"] = chunk["end_ns"] + 1
    elif mutation == "overlap": head["chunks"].append(deepcopy(chunk))
    elif mutation == "chunk_counter": chunk["counter_after"] = 1
    elif mutation == "chunk_bytes": chunk["bytes"] = 16384
    elif mutation == "head_scope": head["end_ns"] += 10_000_000
    elif mutation == "body": head["response_body_bytes"] += 1
    elif mutation == "stream": trace["boundary_stream_bytes"]["boundary_a->client"] += 1
    elif mutation == "role": del trace["heads"]["boundary_b"]
    elif mutation == "unknown": chunk["direction"] = "socket"
    elif mutation == "http_missing": head["http_spans"].pop()
    elif mutation == "http_order": head["http_spans"].reverse()
    elif mutation == "http_secret": head["http_spans"][0]["headers"] = "PRIVATE"
    elif mutation == "head_failed": head["completed"] = False
    elif mutation == "chunk_failed": chunk["completed"] = False
    elif mutation == "sleep_failed": sleep["completed"] = False
    elif mutation == "clock": trace["clock"] = "cross_process_aligned"
    else: monkeypatch.setattr(p, "MAX_CHUNKS", 1)
    with pytest.raises((ValueError, KeyError)):
        p.summarize_trace(trace, profile, report, 2)


def test_limiting_destination_and_tie_are_recomputed():
    sleep = fixture()[0]["heads"]["boundary_a"][0]["chunks"][1]["sleeps"][0]
    sleep.update(source_tokens=16384., destination_tokens=0., requested_seconds=8192 / 12_500_000)
    assert p._sleep_owner(sleep, 8192) == "destination_download"
    sleep.update(source_tokens=8192 - sleep["requested_seconds"] * 5_000_000)
    assert p._sleep_owner(sleep, 8192) == "both"


def test_pacing_wrap_preserves_original_decisions_and_cancellation(monkeypatch):
    from pllm.runtime import native_wan as wan
    # Compare original and wrapped execution against exactly the same fake clock.
    original_asyncio = wan.asyncio
    now = [0.]
    async def advance(delay):
        now[0] += delay + .0001
    proxy = SimpleNamespace(sleep=advance)
    monkeypatch.setattr(wan, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(wan, "asyncio", proxy)
    async def exercise(pacer):
        for _ in range(4):
            await pacer.reserve("boundary_a", "client", 8192)
    baseline = SharedPacer(WanConditions(), ("client", *p.ROLES))
    asyncio.run(exercise(baseline))
    reference = deepcopy(baseline.buckets), dict(baseline.links)
    now[0] = 0.
    with p.instrument() as record:
        row = record.begin("boundary_a", 1)
        candidate = SharedPacer(WanConditions(), ("client", *p.ROLES))
        asyncio.run(exercise(candidate))
        record.finish("boundary_a", row, (200, b"result"))
    assert (candidate.buckets, candidate.links) == reference
    assert len(row["chunks"]) == 4 and record.sleeps > 0
    assert wan.asyncio is proxy and original_asyncio.sleep is asyncio.sleep
    monkeypatch.setattr(wan, "asyncio", original_asyncio)
    # A pending cancellation cannot make a pacing reservation count as sent.
    async def cancel():
        pacer = SharedPacer(WanConditions(upload_mbps=.01), ("client", *p.ROLES))
        with p.instrument() as record:
            row = record.begin("boundary_a", 1)
            task = asyncio.create_task(pacer.reserve("boundary_a", "client", 8192))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert record.failed and not row["chunks"][0]["completed"]
        assert not pacer.links
    asyncio.run(cancel())


def test_trace_never_keeps_callback_info_and_bounds_owners(monkeypatch):
    record = p.Recorder()
    row = record.begin("boundary_a", 1)
    callback = record.trace(row)
    for label in p.HTTP_REQUIRED:
        callback(label + ".started", {"authorization": "PRIVATE"})
        callback(label + ".complete", {"return_value": object()})
    assert "PRIVATE" not in json.dumps(row)
    with pytest.raises(RuntimeError): record.begin("boundary_a", 1)
    record.finish("boundary_a", row, (200, b"result"))
    monkeypatch.setattr(p, "MAX_HEADS", 1)
    with pytest.raises(RuntimeError): record.begin("boundary_a", 1)
    record = p.Recorder()
    row = record.begin("boundary_a", 1)
    callback = record.trace(row)
    with pytest.raises(RuntimeError): callback("http11.receive_response_body.complete", {})


def test_relay_and_real_http_hooks_keep_payloads_and_retire(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pllm.runtime.native_wan import NativePartyNetwork
    from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
    payload, response = b"private request", b"opaque response" * 4000
    observed = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *args): pass
        def do_POST(self):
            observed.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    network = NativePartyNetwork(WanConditions(), ("client", *p.ROLES))
    try:
        with p.instrument() as record:
            network.start([("client", "boundary_a", 0, server.server_port)])
            port = network.servers[0].sockets[0].getsockname()[1]
            with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
                owner = SimpleNamespace(_state_lock=threading.Lock(), _bodies={"boundary_a": {
                    "online_upload_bytes": 0, "online_download_bytes": 0}},
                    _diagnostics={"head": {"upload_bytes": 0, "download_bytes": 0}},
                    _clients=(client,), _keys=("PRIVATE_CREDENTIAL",), _timeout=5, _coverage_complete=True)
                status, actual = PrivateBoundaryTransport._post(owner, 0, "/execute", payload,
                    "online", limit=1 << 20, operation="head")
            network.close()
        assert status == 200 and actual == response and observed == [payload]
        assert not record.failed and not network.thread.is_alive()
        row, = record.heads["boundary_a"]
        assert {s["label"] for s in row["http_spans"]} >= set(p.HTTP_REQUIRED)
        assert sum(c["bytes"] for c in row["chunks"] if c["direction"] == "download") > len(response)
        assert any(c["sleeps"] for c in row["chunks"])
        assert "PRIVATE" not in json.dumps(row)
        assert not network.connections and not network.writers
    finally:
        network.close()
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_new_admission_prices_traces_on_both_arms_and_rejects_tampering():
    from test_private_boundary_role_profile import ordinary_estimate, admission
    engine = p._engine()
    baseline = ordinary_estimate()
    charged = engine._charged_estimate(baseline, True, "research_then_audit")
    assert charged["client_peak_bytes"] == baseline["client_peak_bytes"] + (72 << 20)
    assert charged["native_total_peak_bytes"] == baseline["native_total_peak_bytes"] + (80 << 20)
    details = {"topology_admission": admission(charged), "report": {"memory_preflight": admission(baseline)}}
    engine.validate_admission(details)
    for key in ("client_peak_bytes", "native_total_peak_bytes"):
        changed = deepcopy(details)
        changed["topology_admission"]["estimate"][key] -= 1
        with pytest.raises(ValueError): engine.validate_admission(changed)
    # Creating a specialized engine must not rewrite the old reader's identity.
    assert p.roles.IDENTITY != p.IDENTITY and p.roles.EXTRA_MEMORY == 16 << 20


def test_wrapper_restoration_on_failure_and_concurrency():
    from pllm.runtime import native_wan as wan
    original = wan.SharedPacer.reserve, wan.asyncio, httpx.Client.stream
    with pytest.raises(ValueError):
        with p.instrument():
            with pytest.raises(RuntimeError):
                with p.instrument(): pass
            raise ValueError("fixture")
    assert (wan.SharedPacer.reserve, wan.asyncio, httpx.Client.stream) == original


def test_driver_keeps_fresh_exclusive_paths(tmp_path):
    from scripts.profile_private_boundary_roles import reserve_output_paths
    output = tmp_path / "fresh.json"
    reserve_output_paths(output)
    with pytest.raises(FileExistsError): reserve_output_paths(output)


@pytest.fixture
def pacing_cohort(role_cohort):
    from pllm._benchmark_contract import digest
    from pllm.metrics import BenchmarkSuite
    from pllm.metrics.benchmark_runner import _result
    engine = p._engine()
    old = BenchmarkSuite.from_spec(role_cohort["suite"])
    suite = p.matched_suite(build=old.cases[0].build)
    result = {**role_cohort, "suite": suite.to_spec(), "results": []}
    for row in role_cohort["results"]:
        execution = deepcopy(row["execution"])
        details = execution["details"]
        mode = details["research_profile_identity"]["mode"]
        report = details["report"]
        case, = [c for c in suite.cases if c.name == execution["case"]["name"]]
        admission = details["topology_admission"]
        old_estimate = admission["estimate"]
        admission["estimate"] = engine._charged_estimate(
            old_estimate[engine.BASELINE_KEY], True, old_estimate[engine.ORDER_KEY])
        admission["candidates"]["native"]["required_host_bytes"] += p.EXTRA_TRACE_MEMORY
        details["research_profile_identity"]["schema"] = p.IDENTITY
        if mode == "probe":
            details["private_boundary_role_profile"]["schema"] = p.IDENTITY
            profile = details["private_boundary_decode_profile"]
            profile["schema"] = p.IDENTITY
            relay = deepcopy(report["native_network_accounting"]["samples"]["runs"][0]["after"]["wan_emulation"])
            relay["read_chunk_bytes"] = 8192
            totals = {edge: count for edge, count in relay["directed_stream_bytes"].items() if "boundary" in edge}
            heads = {}
            privacy = report["runs"][0]["privacy"]
            for index, (role, label) in enumerate(zip(p.ROLES, ("http_exchange_a", "http_exchange_b"))):
                heads[role] = []
                before = report["native_network_accounting"]["samples"]["runs"][0]["before"]["wan_emulation"]["directed_stream_bytes"]
                counters = {"upload": before[f"client->{role}"], "download": before[f"{role}->client"]}
                for ordinal, outer in enumerate(s for s in profile["spans"] if s["label"] == label):
                    start, end = outer["start_ns"] - 1, outer["end_ns"] + 1
                    bodies = {direction: total // 16 + int(ordinal * 2 + index < total % 16)
                        for direction, total in (("upload", privacy["private_boundary_head_upload_bytes"]),
                            ("download", privacy["private_boundary_head_download_bytes"]))}
                    chunks = []
                    for direction in bodies:
                        counters[direction] += bodies[direction] + 1
                        chunks.append(interval(start + 1, start + 2, direction=direction,
                            bytes=bodies[direction] + 1, counter_after=counters[direction], sleeps=[]))
                    heads[role].append(interval(start, end, request_body_bytes=bodies["upload"],
                        response_body_bytes=bodies["download"], chunks=chunks,
                        http_spans=[interval(start + i * 10 + 1, start + i * 10 + 2, label=label)
                            for i, label in enumerate(p.HTTP_REQUIRED)]))
            details[p.TRACE_KEY] = {"schema": p.IDENTITY, "clock": "same_client_process_monotonic_ns",
                "heads": heads, "boundary_stream_bytes": totals, "final_relay": relay}
        comparison = engine.client.ordinary_comparison(case.target, details)
        commitment = engine.telemetry_digest(details)
        pacing = digest(details.get(p.TRACE_KEY))
        details["research_profile_identity"].update(telemetry_digest=commitment, pacing_digest=pacing)
        base = digest({"ordinary": comparison["measurement_digest"], "research": p.IDENTITY,
            "mode": mode, "sources": "c" * 64})
        role_bound = digest({"ordinary_research": base, "role_telemetry": commitment})
        execution["comparison"]["measurement_digest"] = digest({"role_research": role_bound, "pacing": pacing})
        payload = {key: execution[key] for key in ("failure_code", "details", "comparison", "fixture", "admission", "oracle")}
        payload.update({key: row[key] for key in ("status", "metrics", "samples", "environment")})
        result["results"].append(_result(case, suite, execution["replicate"], result["cohort"], execution["build"], payload).to_dict())
    return result


def test_whole_cohort_replay_binds_original_gates_and_new_trace(pacing_cohort):
    result = p.audit(pacing_cohort, source_identity="c" * 64)
    assert result["clients"] == 6 and len(result["pacing_profiles"]) == 3
    assert result["admission_contract"]["research_client_bytes"] == 40 << 20


@pytest.mark.parametrize("mutation", ["trace", "pacing_digest", "metric", "build", "admission", "control"])
def test_whole_cohort_rejects_tampering(pacing_cohort, mutation):
    from benchmarks.research.sota_combinations import Rejected
    row = pacing_cohort["results"][-1]
    details = row["execution"]["details"]
    if mutation == "trace": details[p.TRACE_KEY]["heads"]["boundary_a"][0]["end_ns"] += 1
    elif mutation == "pacing_digest": details["research_profile_identity"]["pacing_digest"] = "d" * 64
    elif mutation == "metric": next(m for m in row["metrics"] if m["id"] == "response.decode_tps")["value"] += 1
    elif mutation == "build": row["execution"]["build"]["attributes"]["changed"] = True
    elif mutation == "admission": details["topology_admission"]["estimate"]["client_peak_bytes"] -= 1
    else: pacing_cohort["results"][0]["execution"]["details"][p.TRACE_KEY] = details[p.TRACE_KEY]
    with pytest.raises(Rejected): p.audit(pacing_cohort, source_identity="c" * 64)


def test_hashed_archive_replays_and_rejects_forged_summary_or_path(pacing_cohort, tmp_path):
    import gzip
    import hashlib
    from pllm._benchmark_contract import digest
    from scripts.summarize_private_boundary_pacing import archive, replay
    sources = {f"source-{i}.py": digest(i) for i in range(12)}
    source_id = digest({"schema": p.IDENTITY, "sources": sources})
    engine = p._engine()
    for row in pacing_cohort["results"]:
        execution = row["execution"]
        details = execution["details"]
        identity = details["research_profile_identity"]
        identity["sources"] = source_id
        ordinary = engine.client.ordinary_comparison(p._target(execution), details)["measurement_digest"]
        base = digest({"ordinary": ordinary, "research": p.IDENTITY, "mode": identity["mode"], "sources": source_id})
        role = digest({"ordinary_research": base, "role_telemetry": engine.telemetry_digest(details)})
        execution["comparison"]["measurement_digest"] = digest({"role_research": role, "pacing": identity["pacing_digest"]})
    plan = {"schema": p.IDENTITY, "suite": pacing_cohort["suite"], "tiny": False,
        "research_source_identity": source_id, "frozen_sources": sources}
    summary = p.audit(pacing_cohort, source_identity=source_id)
    manifest = {"schema": p.IDENTITY + ".archive", "files": {}, "sha256": {}}
    for key, value in (("report", pacing_cohort), ("plan", plan), ("summary", summary)):
        name = key + ".json.gz"
        raw = gzip.compress(json.dumps(value).encode(), mtime=0)
        (tmp_path / name).write_bytes(raw)
        manifest["files"][key] = name
        manifest["sha256"][key] = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(manifest))
    assert archive(path) == summary
    changed = deepcopy(summary)
    changed["pacing_decode_medians"]["paired_sleep_union_ns"] += 1
    with pytest.raises(ValueError): replay(pacing_cohort, plan, changed)
    changed = deepcopy(plan)
    changed["frozen_sources"]["source-0.py"] = "d" * 64
    with pytest.raises(ValueError): replay(pacing_cohort, changed, summary)
    manifest["files"]["report"] = "../report.json.gz"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): archive(path)
    manifest["files"]["report"] = "report.json.gz"
    path.write_text(json.dumps(manifest))
    (tmp_path / "report.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError): archive(path)


@pytest.mark.parametrize("mutation", ["counter_reuse", "outside_scenario", "short_sleep"])
def test_independent_reader_rejects_counter_and_timer_counterexamples(pacing_cohort, mutation):
    from scripts.summarize_private_boundary_pacing import validate_trace_order
    validate_trace_order(pacing_cohort)
    row = next(row for row in pacing_cohort["results"] if row["execution"]["case"]["name"].startswith("probe--"))
    details = row["execution"]["details"]
    heads = details[p.TRACE_KEY]["heads"]["boundary_a"]
    first, second = heads[0]["chunks"][1], heads[1]["chunks"][1]
    if mutation == "counter_reuse": second["counter_after"] = first["counter_after"]
    elif mutation == "outside_scenario":
        first["counter_after"] = details["report"]["native_network_accounting"]["samples"]["runs"][0]["after"]["wan_emulation"]["directed_stream_bytes"]["boundary_a->client"] + 1
    else:
        first["sleeps"] = [interval(first["start_ns"], first["end_ns"], requested_seconds=1)]
    with pytest.raises(ValueError): validate_trace_order(pacing_cohort)


@pytest.mark.parametrize("name", ["tiny", "qwen25"])
def test_committed_pacing_evidence_replays(name):
    from pathlib import Path
    from scripts.summarize_private_boundary_pacing import archive
    path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / f"private-boundary-pacing-{name}.manifest.json"
    summary = archive(path)
    assert summary["clients"] == 6
    assert summary["all_logits_active_kv_positions_selections_rng_and_outputs_match"]

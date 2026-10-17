"""Independent peak/conservation controls for publication resource accounting."""
import copy

import pytest

from benchmarks.research.publication_lifecycle import InvalidObservation, summarize


def trace():
    owners = {"c": {"role": "client", "host": "h", "client": True},
              "a": {"role": "worker", "host": "h", "client": False},
              "b": {"role": "worker", "host": "h", "client": False}}
    vectors = [(10, 0, 0), (11, 100, 2), (12, 3, 200), (13, 9, 9), (14, 0, 0)]
    return {"schema": "pllm.publication_lifecycle.v1", "complete": True,
        "dropped_samples": 0, "nominal_sample_interval_ns": 100,
        "owners": owners, "samples": [{"start_ns": i * 100, "end_ns": i * 100 + 5,
            "rss_bytes": dict(zip(owners, vector, strict=True)),
            "alive": ["c"] if i in (0, 4) else list(owners),
            "directed_stream_bytes": {"client->worker": i * 20, "worker->client": i * 30}}
            for i, vector in enumerate(vectors)],
        "events": [{"sample": 0, "phase": "public_setup"},
            {"sample": 1, "phase": "prefill"}, {"sample": 2, "phase": "decode"},
            {"sample": 3, "phase": "cleanup"}, {"sample": 4, "phase": "end"}]}


def test_simultaneous_non_client_peak_is_not_sum_of_individual_peaks():
    result = summarize(trace())
    peak = result["lifecycle_memory"]["non_client"]
    assert peak["individual_peak_bytes"] == {"a": 100, "b": 200}
    assert peak["simultaneous_peak_bytes"] == 203
    assert peak["vector_at_peak"] == {"a": 3, "b": 200}
    assert peak["peak_sample"] == 2
    assert result["lifecycle_memory"]["roles"]["worker"] == peak
    assert result["lifecycle_memory"]["hosts"]["h"]["simultaneous_peak_bytes"] == 215


def test_phases_conserve_traffic_but_share_boundary_memory_observations():
    result = summarize(trace())
    assert result["all_link_stream_bytes"] == 200
    assert sum(p["all_link_stream_bytes"] for p in result["phases"].values()) == 200
    for phase in ("public_setup", "prefill", "decode", "cleanup"):
        assert result["phases"][phase]["all_link_stream_bytes"] == 50
        assert result["phases"][phase]["client_upload_bytes"] == 20
        assert result["phases"][phase]["client_download_bytes"] == 30
    assert result["phases"]["private_preparation"]["memory"] is None
    assert result["phases"]["private_preparation"]["all_link_stream_bytes"] == 0
    assert result["phases"]["prefill"]["memory"]["non_client"]["simultaneous_peak_bytes"] == 203
    assert result["phases"]["decode"]["memory"]["non_client"]["simultaneous_peak_bytes"] == 203
    assert result["sampling"]["maximum_within_sample_skew_ns"] == 5
    assert result["sampling"]["maximum_between_sample_gap_ns"] == 95


@pytest.mark.parametrize("mutation,reason", [
    (lambda d: d.update(complete=False), "incomplete_trace"),
    (lambda d: d.update(dropped_samples=1), "incomplete_trace"),
    (lambda d: d["samples"][2]["rss_bytes"].pop("a"), "incomplete_rss_vector"),
    (lambda d: d["samples"][2]["rss_bytes"].update(a=True), "rss_bytes"),
    (lambda d: d["samples"][2]["rss_bytes"].update(a=-1), "rss_bytes"),
    (lambda d: d["samples"][2]["directed_stream_bytes"].update({"client->worker": 1}), "stream_counter_regression"),
    (lambda d: d["samples"][0]["directed_stream_bytes"].update({"client->worker": 1}), "missing_initial_traffic"),
    (lambda d: d["samples"][-1].update(alive=["c", "a"]), "providers_not_reaped"),
    (lambda d: d["events"][-1].update(sample=3), "phase_interval"),
    (lambda d: d["events"][-2].update(phase="decode"), "missing_cleanup"),
    (lambda d: d["events"][1].update(phase="invented"), "phase_identity"),
    (lambda d: d["samples"][2].update(start_ns=50), "sample_clock_order"),
    (lambda d: d["owners"]["a"].update(client=True), "mixed_client_provider_role"),
    (lambda d: d["samples"][0]["rss_bytes"].update(a=1), "inactive_process_has_rss"),
])
def test_incomplete_or_inconsistent_measurements_reject(mutation, reason):
    document = copy.deepcopy(trace())
    mutation(document)
    with pytest.raises(InvalidObservation, match=reason):
        summarize(document)


def trace_v2():
    from benchmarks.research.publication_lifecycle import SCHEMA, recorder_capacity
    from benchmarks.research.publication_lifecycle_live import digest, observation_clock
    roles = ["client", "inference", "preparation"]
    identity = {"runtime": "independent-test-build"}
    source_map = {"observer.py": "a" * 64}
    binding = {"nonce": "b" * 64, "source_identity": digest(source_map),
               "target_digest": "c" * 64, "prompt_manifest_digest": "d" * 64}
    owners = {role: {"role": role, "client": role == "client", "host": "local_host",
        "pid": 100 + i, "create_time_ns": 1000 + i, "first_sample": 0 if i == 0 else 1}
        for i, role in enumerate(roles)}
    for role in roles[1:]:
        owners[role].update(returncode=0, reaped_ns=755, command_digest="e" * 64,
            child_receipt={"pid": owners[role]["pid"], "create_time_ns": owners[role]["create_time_ns"],
                "ppid": 100, "role": role, "nonce": binding["nonce"], "runtime": identity,
                "source_identity": binding["source_identity"], "command_digest": "e" * 64})
    vectors = [(10, 0, 0), (11, 100, 2), (12, 9, 9), (12, 3, 200),
               (13, 9, 9), (13, 9, 9), (13, 20, 20), (14, 9, 9), (14, 0, 0)]
    samples = []
    for i, vector in enumerate(vectors):
        links = {"client->inference": 20 * i, "inference->client": 30 * i}
        network = {"directed_stream_bytes": links, "total_stream_bytes": 50 * i,
            "parties": {"client": {"upload": 20 * i, "download": 30 * i},
                        "inference": {"upload": 30 * i, "download": 20 * i},
                        "preparation": {"upload": 0, "download": 0}}}
        samples.append({"start_ns": i * 100, "end_ns": i * 100 + 5,
            "host_swap_used_bytes": 0,
            "network_elapsed_seconds": None if i == 0 else (i * 100 + 2 - 50) / 1e9,
            "rss_bytes": dict(zip(roles, vector, strict=True)), "network": network,
            "alive": ["client"] if i in (0, 8) else list(roles), "directed_stream_bytes": links})
    return {"schema": SCHEMA, "complete": True, "target_succeeded": True, "dropped_samples": 0,
        "expected_roles": roles, "owners": owners, "samples": samples, "sample_count": len(samples),
        "capacity": recorder_capacity(1200), "nominal_sample_interval_ns": 50_000_000,
        "clock": observation_clock(), "relay_started_monotonic_seconds": 50 / 1e9,
        "request_workers": [{"ident": 11, "native_id": 12, "entered_ns": 120,
            "returned_ns": 770, "joined_ns": 785, "outcome": "returned"}],
        "errors": [], "binding": binding, "runtime_provenance": identity,
        "network_conditions": {"download_mbps": 100.0, "upload_mbps": 40.0},
        "terminal_received": True,
        "cleanup_health": {**dict.fromkeys(("dashboard_joined", "client_close_returned", "client_workers_joined",
            "memory_guard_joined", "ordinary_sampler_joined", "observer_joined", "topology_closed", "relay_closed",
            "relay_joined", "relay_empty", "relay_loop_closed", "relay_listeners_closed", "no_live_children"), True), "relay_error": None},
        "semantic_work": {"prefill_forwards": 1, "decode_forwards": 1, "selections": 2,
            "terminal_eos": True, "text_events": 1, "preparation_calls": 1},
        "request": {"input_tokens": 103, "output_tokens": 1, "admitted_ns": 110,
            "completed_ns": 750, "first_output_ns": 550, "last_output_ns": 550,
            "preparation_started_ns": 150, "preparation_finished_ns": 350, "online_started_ns": 390},
        "final_network": copy.deepcopy(samples[-1]["network"]),
        "events": [{"sample": i, "phase": phase} for i, phase in enumerate(("public_setup", "request_setup",
            "private_preparation", "request_setup", "prefill", "unclassified", "decode", "cleanup", "end"))]}


def test_semantic_eos_decode_is_not_n_minus_one_or_cleanup():
    result = summarize(trace_v2())
    assert result["emission"]["output_tokens"] == 1
    assert result["emission"]["decode_tokens_per_second"] is None
    assert result["emission"]["decode_seconds"] is None
    assert result["phases"]["decode"]["all_link_stream_bytes"] == 50
    assert result["phases"]["cleanup"]["all_link_stream_bytes"] == 50
    assert result["phases"]["request_setup"]["all_link_stream_bytes"] == 100
    assert result["sampled_window_memory"]["non_client"]["simultaneous_peak_bytes"] == 203
    assert result["sampled_window_memory"]["non_client"]["largest_individual_peak_bytes"] == 200
    assert "lifecycle_memory" not in result
    assert result["scope"]["provider_lifetime_high_water"] is None


@pytest.mark.parametrize("mutation,reason", [
    (lambda d: d["expected_roles"].pop(), "expected_role_graph"),
    (lambda d: d["owners"]["inference"].update(host="other"), "client_ownership"),
    (lambda d: d["owners"]["inference"].update(pid=100, create_time_ns=1000), "process_incarnation"),
    (lambda d: d["owners"]["inference"].update(first_sample=2), "process_lifecycle"),
    (lambda d: d["samples"][3]["alive"].remove("inference"), "process_lifecycle"),
    (lambda d: [s["alive"].remove("inference") for s in d["samples"] if "inference" in s["alive"]], "process_lifecycle"),
    (lambda d: d["owners"]["inference"].pop("returncode"), "providers_not_reaped"),
    (lambda d: d["owners"]["inference"]["child_receipt"].update(ppid=999), "child_provenance"),
    (lambda d: d["owners"]["inference"]["child_receipt"].update(runtime={}), "child_provenance"),
    (lambda d: d["owners"]["inference"]["child_receipt"].update(nonce="wrong"), "child_provenance"),
    (lambda d: d["cleanup_health"].update(client_workers_joined=False), "cleanup_health"),
    (lambda d: d["cleanup_health"].update(relay_error="failure"), "cleanup_health"),
    (lambda d: d["cleanup_health"].update(relay_empty=False), "cleanup_health"),
    (lambda d: d["cleanup_health"].update(memory_guard_joined=False), "cleanup_health"),
    (lambda d: d["cleanup_health"].update(relay_loop_closed=False), "cleanup_health"),
    (lambda d: d["cleanup_health"].update(relay_listeners_closed=False), "cleanup_health"),
    (lambda d: d["samples"][2]["network"]["parties"]["client"].update(upload=1), "party_conservation"),
    (lambda d: d["samples"][2]["network"].update(total_stream_bytes=101), "network_total_conservation"),
    (lambda d: d["final_network"].update(total_stream_bytes=401), "final_network_binding"),
    (lambda d: d["events"][4].update(phase="request_setup"), "missing_required_phase"),
    (lambda d: d["events"][5].update(phase="cleanup"), "semantic_phase_order"),
    (lambda d: d["semantic_work"].update(decode_forwards=0), "semantic_work"),
    (lambda d: d["semantic_work"].update(terminal_eos=False), "semantic_work"),
    (lambda d: d["semantic_work"].update(text_events=2), "semantic_work"),
    (lambda d: d.update(terminal_received=False), "semantic_work"),
    (lambda d: d["capacity"].update(maximum_samples=10), "recorder_capacity"),
    (lambda d: d["errors"].append({"where": "close"}), "lifecycle_failed"),
    (lambda d: d["request"].update(completed_ns=1000), "request_worker_retirement"),
    (lambda d: d["samples"][-1].update(host_swap_used_bytes=1), "host_swap_grew_in_observed_window"),
])
def test_v2_invalid_evidence_fails_closed(mutation, reason):
    document = trace_v2()
    mutation(document)
    with pytest.raises(InvalidObservation, match=reason):
        summarize(document)


@pytest.mark.parametrize("outputs,first,last,expected", [(0, None, None, None), (1, 20, 20, None), (4, 20, 80, 50_000_000)])
def test_emission_denominator_zero_one_and_n_minus_one(outputs, first, last, expected):
    from benchmarks.research.publication_lifecycle import emission_summary
    result = emission_summary({"input_tokens": 10, "output_tokens": outputs, "admitted_ns": 0,
        "completed_ns": 100, "first_output_ns": first, "last_output_ns": last})
    assert result["decode_intervals"] == max(0, outputs - 1)
    if expected is None:
        assert result["decode_tokens_per_second"] is None
    else:
        assert result["decode_tokens_per_second"] == pytest.approx(expected)


def test_capacity_admits_timeout_and_shutdown_for_every_supported_topology():
    import math
    from benchmarks.research.publication_lifecycle import recorder_capacity, MAX_SAMPLES, MAX_TRACE_FILE_BYTES
    for timeout in (1, 1200, 86400):
        capacity = recorder_capacity(timeout)
        needed = math.ceil((timeout + 30) * 1e9 / capacity["interval_ns"]) + capacity["boundary_sample_allowance"]
        assert needed <= capacity["maximum_samples"] <= MAX_SAMPLES
        assert capacity["maximum_trace_disk_bytes"] <= MAX_TRACE_FILE_BYTES
        assert capacity["client_allocation_admission_bytes"] == 64 << 20
    assert recorder_capacity(1200)["interval_ns"] == 50_000_000


def capture_configuration():
    document = trace_v2()
    return {"expected_roles": document["expected_roles"], "timeout_seconds": 1200, "output_cap": 2,
        "binding": document["binding"], "runtime": document["runtime_provenance"],
        "wan": document["network_conditions"], "clock": document["clock"]}


@pytest.fixture
def live_capture(tmp_path):
    from types import SimpleNamespace
    from benchmarks.research.publication_lifecycle_live import Capture
    runtime = SimpleNamespace(_topology=None, _worker=None, _background_threads=[], _memory_guard=None)
    capture = Capture(runtime, capture_configuration(), tmp_path)
    capture.start()
    yield capture
    capture.stop.set()
    capture.thread.join(timeout=10)
    capture.file.close()


def test_recorder_streams_rows_without_retaining_trace(live_capture):
    from scripts.benchmark_publication_lifecycle import read_samples
    live_capture.stop.set()
    live_capture.thread.join()
    for _ in range(300):
        live_capture.sample()
    rows, sha, size, truncated = read_samples(live_capture.directory)
    assert len(rows) == live_capture.count
    assert sha == live_capture.hasher.hexdigest()
    assert size == live_capture.disk_bytes
    assert not truncated and "samples" not in live_capture.document
    assert live_capture.last == rows[-1]


def test_exit_between_poll_and_rss_only_allowed_during_cleanup(live_capture, monkeypatch):
    import threading
    from types import SimpleNamespace
    import psutil
    real_process, pid = psutil.Process, 987654321
    live_capture.stop.set()
    live_capture.thread.join()
    live_capture.target_done = True  # Isolate the poll/RSS shutdown race from helper census.
    child = SimpleNamespace(pid=pid, poll=lambda: None)
    topology = SimpleNamespace(_native_network=None, _process_lock=threading.Lock(),
        _role_ids=("inference", "preparation"), _processes={"inference": child})
    live_capture.runtime._topology = topology
    live_capture.document["owners"]["inference"].update(pid=pid, create_time_ns=1)
    def process(value=None):
        if value == pid:
            raise psutil.NoSuchProcess(value)
        return real_process(value)
    monkeypatch.setattr(psutil, "Process", process)
    with pytest.raises(psutil.NoSuchProcess):
        live_capture._observe()
    live_capture.phase = "cleanup"
    observed = live_capture._observe()
    assert observed["rss_bytes"]["inference"] == 0
    assert observed["alive"] == ["client"]
    monkeypatch.setattr(psutil, "Process", lambda *_: (_ for _ in ()).throw(psutil.AccessDenied(pid)))
    with pytest.raises(psutil.AccessDenied):
        live_capture._observe()


@pytest.mark.parametrize("field,value", [("relay_error", "failure"), ("client_workers_joined", False),
    ("relay_joined", False), ("no_live_children", False), ("ordinary_sampler_joined", False)])
def test_finish_cannot_certify_failed_retirement(live_capture, monkeypatch, field, value):
    import json
    from benchmarks.research.publication_lifecycle_live import FINAL_FILE
    health = trace_v2()["cleanup_health"]
    health[field] = value
    monkeypatch.setattr(live_capture, "health", lambda: health)
    live_capture.document["target_succeeded"] = True
    live_capture.finish(after_serialization=True)
    receipt = json.loads((live_capture.directory / FINAL_FILE).read_text())
    assert receipt["document"]["complete"] is False
    assert receipt["document"]["errors"][-1]["code"] == "cleanup_health"


def test_recorder_failure_cannot_mask_primary_error(live_capture, monkeypatch):
    monkeypatch.setattr(live_capture, "checkpoint", lambda: (_ for _ in ()).throw(OSError("disk full")))
    error = RuntimeError("primary execution error")
    live_capture.error(error, "target")
    assert live_capture.document["errors"][-1] == {
        "where": "target", "exception": "RuntimeError", "code": "execution_failed"}


def receipt_fixture(tmp_path, document=None):
    from benchmarks.research.publication_lifecycle_live import canonical, digest, write_private, STATE_FILE, SAMPLES_FILE, FINAL_FILE
    import hashlib
    from benchmarks.research.publication_lifecycle import request_durations
    document = copy.deepcopy(document) if document is not None else trace_v2()
    configuration = {**capture_configuration(), "prompt_manifest": {"input_tokens": 103, "source_lock_digest": "lock"}}
    document["binding"]["prompt_manifest_digest"] = digest(configuration["prompt_manifest"])
    configuration["binding"] = copy.deepcopy(document["binding"])
    rows = document.pop("samples")
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    (tmp_path / SAMPLES_FILE).write_bytes(raw)
    write_private(tmp_path / STATE_FILE, document)
    receipt = {"schema": "pllm.publication_finalization.v2", "document": document,
        "after_worker_serialization": True, "samples_sha256": hashlib.sha256(raw).hexdigest(),
        "samples_bytes": len(raw), "process_self_high_water_bytes": 2100, "observed_ns": rows[-1]["end_ns"] + 95,
        "high_water_scope": "test RUSAGE_SELF window"}
    write_private(tmp_path / FINAL_FILE, receipt)
    for role, owner in document["owners"].items():
        if role != "client":
            write_private(tmp_path / f"publication-child-{role}.json", owner["child_receipt"])
    durations = {**request_durations(document["request"]), "tokens_per_second": None}
    details = {"publication_lifecycle_binding": document["binding"],
        "publication_lifecycle_sources": {"observer.py": "a" * 64}, "report": {
            "configuration": {"source_lock_digest": "lock"},
            "checks": {"passed": True},
            "wan_readiness": {"emulation": {"native_stream_rate_snapshots_checked": True}},
            "runs": [{"tokens": {"authoritative": True, "input_tokens": 103, "output_tokens": 1},
                      "durations": durations}],
            "memory_guard": {"tripped": False, "maximum_host_swap_growth_bytes": 0}}}
    def ordinary_snapshot(row, at_ns=None):
        net = copy.deepcopy(row["network"])
        net["parties"] = {role: {direction: {"admitted_bytes": value} for direction, value in values.items()}
            for role, values in net["parties"].items()}
        net.update(conditions=configuration["wan"], role_parties={role: role for role in document["expected_roles"]},
            elapsed_seconds=row["network_elapsed_seconds"] if at_ns is None else
                at_ns / 1e9 - document["relay_started_monotonic_seconds"])
        return {"wan_emulation": net}
    details["report"]["native_network_accounting"] = {"samples": {"startup": ordinary_snapshot(rows[1]),
        "runs": [{"before": ordinary_snapshot(rows[1]),
            "after": ordinary_snapshot(rows[document["events"][-2]["sample"]], document["request"]["completed_ns"] + 10)}]}}
    assert digest(details["publication_lifecycle_sources"]) == configuration["binding"]["source_identity"]
    payload = {"status": "completed", "details": details, "metrics": [
        {"id": "process.peak_rss", "value": 2000}, {"id": "client.peak_rss", "value": 1900},
        {"id": "network.all_link_stream_bytes", "value": 350},
        {"id": "response.decode_tps", "value": None}, {"id": "request.0.decode_tps", "value": None},
        {"id": "request.0.full", "value": durations["full_seconds"]},
        {"id": "request.0.preparation", "value": durations["preparation_seconds"]},
        {"id": "request.0.ttft", "value": durations["ttft_seconds"]},
        {"id": "response.ttft", "value": durations["ttft_seconds"]}]}
    return payload, configuration, {"worker_identity": {"pid": 100, "create_time_ns": 1000},
        "launch_started_ns": 0, "timeout_started_ns": 0, "supervisor_returned_ns": receipt["observed_ns"] + 100}


def accept_fixture(payload, configuration, parent, directory):
    from scripts.benchmark_publication_lifecycle import reserve_output, begin_trial, retain_trial, accept_receipt
    output = reserve_output(directory / "retained" / "cohort.json")
    durable = begin_trial(output.with_suffix(".traces"), configuration)
    reference = retain_trial(directory, durable, configuration, parent, payload)
    return accept_receipt(payload, configuration, parent, reference)


def test_parent_accepts_only_bound_receipt_and_preserves_ordinary_peak(tmp_path):
    payload, configuration, identity = receipt_fixture(tmp_path)
    result = accept_fixture(payload, configuration, identity, tmp_path)
    assert result["client_high_water"]["tier_observation_bytes"] == 2000
    assert result["client_high_water"]["post_serialization_atexit_self_peak_bytes"] == 2100
    assert result["sampled_window_memory"]["client"]["simultaneous_peak_bytes"] == 14
    assert result["post_scenario_stream_bytes"] == 50
    assert payload["details"]["publication_lifecycle"]["partial"] is False


@pytest.mark.parametrize("change,reason", [
    (lambda p, c, i: i["worker_identity"].update(pid=999), "parent_child_receipt_binding"),
    (lambda p, c, i: c["prompt_manifest"].update(input_tokens=102), "prompt_manifest_binding"),
    (lambda p, c, i: p["details"]["report"]["runs"][0]["tokens"].update(input_tokens=102), "ordinary_request_reconciliation"),
    (lambda p, c, i: p["details"]["report"]["configuration"].update(source_lock_digest="other"), "ordinary_request_reconciliation"),
    (lambda p, c, i: p["details"]["publication_lifecycle_sources"].clear(), "parent_child_receipt_binding"),
    (lambda p, c, i: p["metrics"][0].update(value=2200), "client_high_water_reconciliation"),
    (lambda p, c, i: p["details"]["report"]["memory_guard"].update(maximum_host_swap_growth_bytes=1), "ordinary_traffic_or_pressure_reconciliation"),
])
def test_parent_rejects_forged_or_mismatched_receipts(tmp_path, change, reason):
    payload, configuration, identity = receipt_fixture(tmp_path)
    change(payload, configuration, identity)
    with pytest.raises(InvalidObservation, match=reason):
        accept_fixture(payload, configuration, identity, tmp_path)


def test_missing_exit_receipt_retains_partial_trace(tmp_path):
    from scripts.benchmark_publication_lifecycle import load_trace
    from benchmarks.research.publication_lifecycle_live import FINAL_FILE
    payload, configuration, identity = receipt_fixture(tmp_path)
    (tmp_path / FINAL_FILE).unlink()
    payload.update(status="failed", failure_code="original_target_failure")
    with pytest.raises(InvalidObservation, match="incomplete_durable_receipt"):
        accept_fixture(payload, configuration, identity, tmp_path)
    assert payload["failure_code"] == "original_target_failure"
    assert payload["details"]["publication_lifecycle"]["partial"]
    trace, _, _ = load_trace(payload["details"]["publication_lifecycle"]["trace_ref"], allow_partial=True)
    assert not trace["complete"]
    assert len(trace["samples"]) == 9


def test_truncated_journal_tail_is_visible(tmp_path):
    from scripts.benchmark_publication_lifecycle import read_samples
    from benchmarks.research.publication_lifecycle_live import SAMPLES_FILE
    (tmp_path / SAMPLES_FILE).write_bytes(b'{"sample":0}\n{"partial":')
    rows, _, size, truncated = read_samples(tmp_path)
    assert rows == [{"sample": 0}] and truncated and size == 24


@pytest.mark.parametrize("suffix", [".json", ".plan.json", ".progress.json", ".summary.json", ".model", ".prompt.txt", ".reservation", ".traces"])
def test_all_existing_companions_are_preserved(tmp_path, suffix):
    from scripts.benchmark_publication_lifecycle import reserve_output
    output = tmp_path / "cohort.json"
    old = output.with_suffix(suffix)
    old.write_bytes(b"historical evidence")
    with pytest.raises(FileExistsError):
        reserve_output(output)
    assert old.read_bytes() == b"historical evidence"
    assert list(tmp_path.iterdir()) == [old]


def test_dangling_companion_symlink_is_preserved(tmp_path):
    from scripts.benchmark_publication_lifecycle import reserve_output
    output = tmp_path / "cohort.json"
    old = output.with_suffix(".plan.json")
    old.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        reserve_output(output)
    assert old.is_symlink()


def test_live_wrappers_keep_eos_compute_before_cleanup(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from pathlib import Path
    from pllm.runtime import benchmark_resources, benchmark_response_audit
    from pllm.runtime.client import ResponseStream
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime
    from benchmarks.research.publication_lifecycle_live import instrument

    monkeypatch.setattr(benchmark_resources.ProcessMemorySampler, "start", lambda _: None)
    monkeypatch.setattr(SemanticDecoderRuntime, "_forward", lambda *_: None)
    selections = iter([42, 99])
    monkeypatch.setattr(MaskedTransformerClientRuntime, "sample", lambda *_: next(selections))
    @contextmanager
    def audit(_):
        yield []
    monkeypatch.setattr(benchmark_response_audit, "response_state_audit", audit)
    runtime = SimpleNamespace(_topology=None)
    capture = None
    try:
        with instrument(capture_configuration(), tmp_path, Path(__file__)) as session:
            benchmark_resources.ProcessMemorySampler.start(SimpleNamespace(runtime=runtime))
            capture = session["capture"]
            capture.mark("request_setup")
            decoder = SimpleNamespace(position=0, cfg={"eos_token_id": 99})
            with benchmark_response_audit.response_state_audit(b"unused"):
                SemanticDecoderRuntime._forward(decoder, [])
                assert MaskedTransformerClientRuntime.sample(decoder, []) == 42
                stream = ResponseStream(iter([{"type": "response.output_text.delta"},
                                              {"type": "response.completed"}]))
                next(stream)
                assert capture.phase == "unclassified"
                decoder.position = 103
                SemanticDecoderRuntime._forward(decoder, [])
                assert capture.phase == "decode"
                assert MaskedTransformerClientRuntime.sample(decoder, []) == 99
                assert capture.phase == "cleanup"
                next(stream)
                assert capture.document["semantic_work"] == {"prefill_forwards": 1, "decode_forwards": 1,
                    "selections": 2, "terminal_eos": True, "text_events": 1, "preparation_calls": 0}
    finally:
        if capture is not None:
            capture.stop.set()
            capture.thread.join(timeout=10)
            capture.file.close()


def test_live_close_preserves_original_failure_and_partial_journal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from pllm.runtime import benchmark_resources, benchmark_cli
    from benchmarks.research.publication_lifecycle_live import instrument
    from scripts.benchmark_publication_lifecycle import read_samples
    original_error = RuntimeError("original cleanup failure")
    def broken_close(_):
        raise original_error
    monkeypatch.setattr(benchmark_resources.ProcessMemorySampler, "start", lambda _: None)
    monkeypatch.setattr(benchmark_cli._DashboardHandle, "close", broken_close)
    capture = None
    try:
        with pytest.raises(RuntimeError) as caught:
            with instrument(capture_configuration(), tmp_path, Path(__file__)) as session:
                benchmark_resources.ProcessMemorySampler.start(SimpleNamespace(runtime=SimpleNamespace(_topology=None)))
                capture = session["capture"]
                benchmark_cli._DashboardHandle.close(SimpleNamespace())
        assert caught.value is original_error
        assert capture.document["errors"][0]["where"] == "dashboard_close"
        assert read_samples(tmp_path)[0]
        assert not capture.document["complete"]
    finally:
        if capture is not None:
            capture.stop.set()
            capture.thread.join(timeout=10)
            capture.file.close()


def test_provider_bootstrap_receipt_binds_actual_child_before_delegation(tmp_path, monkeypatch):
    import os
    import sys
    import json
    from scripts import benchmark_publication_lifecycle as driver
    from benchmarks.research.publication_lifecycle_live import REQUEST_FILE, write_private
    configuration = capture_configuration()
    source_map = {"observer.py": "a" * 64}
    configuration["build"] = {"test": "build"}
    write_private(tmp_path / REQUEST_FILE, configuration)
    monkeypatch.setattr(driver, "sources", lambda: source_map)
    monkeypatch.setattr(driver, "runtime_provenance", lambda build: configuration["runtime"])
    delegated = []
    monkeypatch.setattr(driver.runpy, "run_module", lambda *args, **kwargs: delegated.append((args, kwargs, list(sys.argv))))
    monkeypatch.setattr(sys, "argv", ["original"])
    command = ["-m", "pllm", "serve", "inference", "--port", "1234"]
    driver.provider_main("inference", tmp_path, command)
    receipt = json.loads((tmp_path / "publication-child-inference.json").read_text())
    assert receipt["pid"] == os.getpid() and receipt["ppid"] == os.getppid()
    assert receipt["nonce"] == configuration["binding"]["nonce"]
    assert receipt["command_digest"] == driver.digest(command)
    assert delegated == [(("pllm",), {"run_name": "__main__", "alter_sys": True}, command[1:])]
    with pytest.raises(InvalidObservation, match="provider_entry_point"):
        driver.provider_main("preparation", tmp_path, command)
    monkeypatch.setattr(driver, "sources", lambda: {})
    with pytest.raises(InvalidObservation, match="provider_observer_source"):
        driver.provider_main("inference", tmp_path, command)


def test_parent_timeout_preserves_original_failure_and_partial_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import benchmark_publication_lifecycle as driver
    from pllm.metrics import benchmark_runner
    from benchmarks.research.publication_lifecycle_live import STATE_FILE, SAMPLES_FILE, write_private
    def failed_execute(case, sampling, inputs, salt, directory):
        document = trace_v2()
        rows = document.pop("samples")
        write_private(directory / STATE_FILE, document)
        (directory / SAMPLES_FILE).write_bytes(b"".join(driver.canonical(row) + b"\n" for row in rows))
        return {"status": "timeout", "failure_code": "original_timeout"}
    monkeypatch.setattr(benchmark_runner, "_execute", failed_execute)
    monkeypatch.setattr(driver, "role_ids", lambda _: trace_v2()["expected_roles"])
    target = SimpleNamespace(digest="c" * 64, to_spec=lambda: {"parameters": {
        "scenario": {"max_output_tokens": 2}, "wan": capture_configuration()["wan"]}})
    output = driver.reserve_output(tmp_path / "retained" / "cohort.json")
    with driver.publication_workers(identity="a" * 64, build={}, runtime={}, manifest={}, store=output.with_suffix(".traces")):
        payload = benchmark_runner._execute(SimpleNamespace(target=target),
            SimpleNamespace(timeout_seconds=1200), {}, b"", tmp_path)
    assert payload["status"] == "timeout" and payload["failure_code"] == "original_timeout"
    assert payload["details"]["publication_lifecycle"]["partial"]
    assert len(driver.load_trace(payload["details"]["publication_lifecycle"]["trace_ref"], allow_partial=True)[0]["samples"]) == 9


@pytest.mark.parametrize("padding", ("", " ", "\n", "\r\n", " \t\n"))
def test_public_fixture_count_matches_real_dashboard_request(tmp_path, monkeypatch, padding):
    """Use the actual route and tokenizer without launching roles or loading weights."""
    import asyncio
    import json
    from types import SimpleNamespace
    from starlette.requests import Request
    from pllm import Model
    from pllm.deployment.execution import _TokenizerView
    from pllm.runtime import dashboard
    from pllm.runtime.responses import normalize_input
    from pllm.sources import _tiny_model_config
    from scripts import benchmark_publication_lifecycle as driver

    text = ("abcde " * 14).strip()
    raw = text + padding
    received = []
    runtime = SimpleNamespace(begin=lambda prompt, *a, **k: received.append(prompt) or "test-run")
    monkeypatch.setattr(dashboard, "DashboardRuntime", lambda *a: runtime)
    config = dashboard.DashboardConfig("127.0.0.1", 8999, None, "test", 2,
        history_path=tmp_path / "history.sqlite")
    app = dashboard.create_dashboard_app(config)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/api/run")

    async def receive():
        return {"type": "http.request", "body": json.dumps({"prompt": raw}).encode()}

    request = Request({"type": "http", "headers": [(b"content-type", b"application/json")]}, receive)
    try:
        asyncio.run(endpoint(request))
    finally:
        app.state.benchmark_history.close()
    assert received == [text]
    descriptor = {"kind": "byte", "chat_template":
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant: "}
    view = _TokenizerView(descriptor, _tiny_model_config(),
        SimpleNamespace(chat_template=descriptor["chat_template"]))
    messages = [row.to_prompt_dict() for row in normalize_input(received[0])]
    actual = len(view.tokenizer().encode(view.render_prompt(messages), add_bos=True))
    counts, _ = driver.public_request_counts(Model.tiny(), None, raw, tiny=True)
    assert counts == {"tiny": actual}
    assert raw == text + padding


def test_source_gate_survives_main_file_retirement_at_exit():
    """Exercise the real source recheck after CPython retires __main__.__file__."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_publication_lifecycle.py"
    code = '''
import sys
from pathlib import Path
__file__ = sys.argv[1]
source = Path(__file__).read_text().rsplit('\\nif __name__ == "__main__":', 1)[0]
exec(compile(source, __file__, "exec"), globals())
expected = sources()
@atexit.register
def check():
    # CPython 3.13 does this itself; retain the regression on other interpreters.
    globals().pop("__file__", None)
    print(json.dumps({"matches": sources() == expected, "files": len(expected)}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(script)],
        check=True, capture_output=True, text=True, timeout=20)
    assert json.loads(result.stdout) == {"matches": True, "files": 10}


def test_atexit_receipt_observes_post_serialization_self_rss(tmp_path):
    """Bounded subprocess control: no runtime/model work, only Python allocation."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = '''
import sys, atexit, json
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from benchmarks.research.publication_lifecycle_live import Capture, observation_clock
from benchmarks.research.publication_lifecycle import CLEANUP_FLAGS
runtime = SimpleNamespace(_topology=None)
config = {"expected_roles":["client"], "timeout_seconds":2, "output_cap":1,
          "binding":{}, "runtime":{}, "wan":{}, "clock":observation_clock()}
c = Capture(runtime, config, sys.argv[2])
c.start()
c.mark("cleanup")
c.target_done = True
c.document["target_succeeded"] = True
c.health = lambda: {**dict.fromkeys(CLEANUP_FLAGS, True), "relay_error": None}
done = False
atexit.register(lambda: c.finish(after_serialization=done))
# Allocation and stdout serialization intentionally occur after target return.
payload = bytearray(8 << 20)
sys.stdout.write(json.dumps({"result": "x" * (1 << 20)}))
sys.stdout.flush()
done = True
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root), str(tmp_path)],
        check=True, stdout=subprocess.DEVNULL, timeout=20)
    receipt = json.loads((tmp_path / "publication-final.json").read_text())
    assert receipt["after_worker_serialization"] is True
    assert receipt["process_self_high_water_bytes"] >= 8 << 20
    assert receipt["document"]["events"][-1]["phase"] == "end"
    assert "later receipt serialization" in receipt["high_water_scope"]


@pytest.mark.parametrize("outputs,eos", [(0, True), (1, False)])
def test_complete_zero_or_one_output_trace_has_no_fabricated_decode(outputs, eos):
    document = trace_v2()
    document["events"] = [event for event in document["events"] if event["phase"] not in {"decode", "unclassified"}]
    document["semantic_work"].update(decode_forwards=0, selections=1, terminal_eos=eos, text_events=outputs)
    document["request"].update(output_tokens=outputs, first_output_ns=550 if outputs else None,
                               last_output_ns=550 if outputs else None)
    result = summarize(document)
    assert result["emission"]["decode_tokens_per_second"] is None
    assert result["phases"]["decode"]["status"] == "not_entered"
    assert result["phases"]["decode"]["memory"] is None
    assert result["phases"]["decode"]["all_link_stream_bytes"] == 0


def test_undeclared_real_helper_is_detected(live_capture):
    import subprocess
    import sys
    live_capture.stop.set()
    live_capture.thread.join()
    child = subprocess.Popen([sys.executable, "-I", "-B", "-c", "import time; time.sleep(20)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with pytest.raises(InvalidObservation, match="undeclared_serving_helper"):
            live_capture._observe()
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_live_process_identity_cannot_be_reused(live_capture):
    live_capture.stop.set()
    live_capture.thread.join()
    live_capture.document["owners"]["client"]["create_time_ns"] += 1
    with pytest.raises(InvalidObservation, match="process_replaced"):
        live_capture._observe()


def test_closed_relay_error_cannot_bypass_snapshot_validation(live_capture):
    import threading
    from types import SimpleNamespace
    live_capture.stop.set()
    live_capture.thread.join()
    live_capture.runtime._topology = SimpleNamespace(_role_ids=("inference", "preparation"),
        _processes={}, _process_lock=threading.Lock(),
        _native_network=SimpleNamespace(closed=True, error="late relay error", pacer=SimpleNamespace(started=0.0)))
    with pytest.raises(InvalidObservation, match="relay_error"):
        live_capture._observe()


def test_trace_disk_admission_precedes_allocation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.research import publication_lifecycle_live as live
    monkeypatch.setattr(live.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1, f_frsize=4096))
    with pytest.raises(InvalidObservation, match="trace_disk_admission"):
        live.Capture(SimpleNamespace(_topology=None), capture_configuration(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_oversized_row_rejects_without_dropping_previous_samples(live_capture, monkeypatch):
    live_capture.stop.set()
    live_capture.thread.join()
    before, size = live_capture.count, live_capture.disk_bytes
    monkeypatch.setattr(live_capture, "_observe", lambda: {**live_capture.last, "unexpected": "x" * 8192})
    with pytest.raises(InvalidObservation, match="trace_row_bound"):
        live_capture.sample()
    assert live_capture.count == before and live_capture.disk_bytes == size


@pytest.mark.parametrize("roles", [("inference", "preparation"), ("inference", "preparation", "boundary_a", "boundary_b")])
def test_observer_admission_prices_each_owner_without_mutating_baseline(monkeypatch, roles):
    from pllm.runtime import benchmark_memory as memory
    from benchmarks.research.publication_lifecycle_live import priced_admission, PROVENANCE_ADMISSION_BYTES
    from benchmarks.research.publication_lifecycle import TRACE_ADMISSION_BYTES
    report = {"estimate": {"client_peak_bytes": 100, "native_total_peak_bytes": 1000,
        "provider_peak_bytes": dict.fromkeys(roles, 200), "components": {}}}
    baseline = copy.deepcopy(report)
    monkeypatch.setattr(memory, "host_memory", lambda: None)
    monkeypatch.setattr(memory, "admit_memory", lambda estimate, *args, **kwargs: estimate)
    result = priced_admission(report, 8 << 30)
    assert report == baseline
    assert result["client_peak_bytes"] == 100 + TRACE_ADMISSION_BYTES
    assert result["native_total_peak_bytes"] == 1000 + TRACE_ADMISSION_BYTES + len(roles) * PROVENANCE_ADMISSION_BYTES
    assert result["provider_peak_bytes"] == dict.fromkeys(roles, 200 + PROVENANCE_ADMISSION_BYTES)


def long_trace():
    """900 seconds of synthetic observations, no waiting or model execution."""
    document = trace_v2()
    final = copy.deepcopy(document["samples"][-1])
    rows = document["samples"][:-1]
    for i in range(1, 18001):
        at = i * 900_000_000_000 // 18001
        row = copy.deepcopy(final)
        row.update(start_ns=at, end_ns=at + 5,
            network_elapsed_seconds=(at + 2) / 1e9 - document["relay_started_monotonic_seconds"],
            network=copy.deepcopy(rows[7]["network"]), directed_stream_bytes=dict(rows[7]["directed_stream_bytes"]))
        rows.append(row)
    final.update(start_ns=900_000_000_000, end_ns=900_000_000_005,
        network_elapsed_seconds=900.000000002 - document["relay_started_monotonic_seconds"])
    rows.append(final)
    document.update(samples=rows, sample_count=len(rows))
    document["events"][-1]["sample"] = len(rows) - 1
    assert len(rows) == 18009
    return document


def ordinary_result_with_details(details, *, invalid_build=False):
    """Use the installed result validator, without running a benchmark executor."""
    from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling
    from pllm.metrics.benchmark_runner import _result
    from pllm._benchmark_contract import digest
    case = BenchmarkCase("metadata-control", BenchmarkTarget.runtime())
    suite = BenchmarkSuite([case], Sampling(processes=1))
    build = {"attributes": {}, "digest": "0" * 64 if invalid_build else digest({})}
    return _result(case, suite, 0, "a" * 64, build, {
        "status": "failed", "failure_code": "synthetic_validation_control", "details": details})


def test_900_second_trace_uses_durable_reference_and_fits_real_result_contract(tmp_path):
    import shutil
    from scripts import benchmark_publication_lifecycle as driver
    document = long_trace()
    assert len(driver.canonical(document)) > driver.DETAILS_LIMIT
    source = tmp_path / "worker"
    source.mkdir()
    payload, configuration, parent = receipt_fixture(source, document)
    output = driver.reserve_output(tmp_path / "cohort.json")
    durable = driver.begin_trial(output.with_suffix(".traces"), configuration)
    reference = driver.retain_trial(source, durable, configuration, parent, payload)
    driver.accept_receipt(payload, configuration, parent, reference)
    shutil.rmtree(source)  # The ordinary runner does this before constructing BenchmarkResult.
    assert "trace" not in payload["details"]["publication_lifecycle"]
    assert len(driver.canonical(payload["details"])) < driver.LIFECYCLE_DETAILS_LIMIT
    assert ordinary_result_with_details(payload["details"]).to_dict()["execution"]["details"] == payload["details"]
    replay, _, _ = driver.load_trace(reference)
    assert len(replay["samples"]) == 18009
    assert summarize(replay)["elapsed_seconds"] == 900.0
    with pytest.raises(ValueError, match="benchmark details exceed bounded metadata"):
        ordinary_result_with_details({"publication_lifecycle": {"trace": replay}})


@pytest.mark.parametrize("alias_scratch", [False, True])
def test_result_construction_failure_after_tempdir_removal_keeps_raw_evidence(tmp_path, monkeypatch, alias_scratch):
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from pllm.metrics import benchmark_runner
    from scripts import benchmark_publication_lifecycle as driver
    from benchmarks.research.publication_lifecycle_live import SAMPLES_FILE
    def fake_execute(case, sampling, inputs, salt, directory):
        assert directory == directory.resolve(strict=True)
        payload, _, _ = receipt_fixture(directory)
        payload.update(status="failed", failure_code="synthetic_control")
        return payload
    monkeypatch.setattr(benchmark_runner, "_execute", fake_execute)
    monkeypatch.setattr(driver, "role_ids", lambda _: trace_v2()["expected_roles"])
    target = SimpleNamespace(digest="c" * 64, to_spec=lambda: {"parameters": {
        "scenario": {"max_output_tokens": 2}, "wan": capture_configuration()["wan"]}})
    output = driver.reserve_output(tmp_path / "cohort.json")
    scratch = tmp_path / "private-scratch"
    scratch.mkdir()
    alias = tmp_path / "scratch-alias"
    if alias_scratch:
        alias.symlink_to(scratch, target_is_directory=True)
    with driver.publication_workers(identity="a" * 64, build={}, runtime={}, manifest={}, store=output.with_suffix(".traces")):
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            if alias_scratch:
                directory = alias / Path(directory).name
            payload = benchmark_runner._execute(SimpleNamespace(target=target),
                SimpleNamespace(timeout_seconds=1200), {}, b"", Path(directory))
    assert not Path(directory).exists()
    with pytest.raises(ValueError, match="observed build digest differs"):
        ordinary_result_with_details(payload["details"], invalid_build=True)
    reference = payload["details"]["publication_lifecycle"]["trace_ref"]
    retained = Path(reference["path"]).parent
    assert (retained / SAMPLES_FILE).exists() and (retained / "ordinary-payload.json").exists()
    assert driver.read_bounded(retained / "acceptance.json")["failure_code"] == "synthetic_control"
    assert len(driver.load_trace(reference, allow_partial=True)[0]["samples"]) == 9


def test_ordinary_details_budget_failure_retains_full_original_payload(tmp_path):
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    payload, configuration, parent = receipt_fixture(tmp_path)
    payload["details"]["synthetic_metadata"] = "x" * driver.DETAILS_LIMIT
    with pytest.raises(InvalidObservation, match="ordinary_plus_lifecycle_details_bound"):
        accept_fixture(payload, configuration, parent, tmp_path)
    reference = payload["details"]["publication_lifecycle"]["trace_ref"]
    driver.bound_details(payload, reference)
    assert payload["status"] == "failed" and payload["failure_code"] == "publication_details_bound"
    assert len(driver.canonical(payload["details"])) < 4096
    raw = driver.read_bounded(Path(reference["path"]).parent / "ordinary-payload.json", 16 << 20)
    assert len(raw["details"]["synthetic_metadata"]) == driver.DETAILS_LIMIT
    ordinary_result_with_details(payload["details"])


def test_archive_is_exclusive_hash_bound_and_detects_tampering(tmp_path):
    import stat
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    payload, configuration, parent = receipt_fixture(tmp_path)
    accept_fixture(payload, configuration, parent, tmp_path)
    reference = payload["details"]["publication_lifecycle"]["trace_ref"]
    directory = Path(reference["path"]).parent
    samples = directory / driver.SAMPLES_FILE
    assert stat.S_IMODE(samples.stat().st_mode) == 0o400
    with pytest.raises(FileExistsError):
        driver.begin_trial(directory.parent, configuration)
    samples.chmod(0o600)
    with samples.open("ab") as file:
        file.write(b"\n")
    with pytest.raises(InvalidObservation, match="trace_file_binding"):
        driver.load_trace(reference)


def test_unreserved_or_symlinked_store_cannot_acquire_trial_ownership(tmp_path):
    from scripts import benchmark_publication_lifecycle as driver
    configuration = capture_configuration()
    output = driver.reserve_output(tmp_path / "cohort.json")
    alias = tmp_path / "alias"
    alias.symlink_to(output.with_suffix(".traces"), target_is_directory=True)
    with pytest.raises(InvalidObservation, match="evidence_store_owner"):
        driver.begin_trial(alias, configuration)


def test_inline_ordinary_report_must_match_retained_payload(tmp_path):
    from scripts import benchmark_publication_lifecycle as driver
    payload, configuration, parent = receipt_fixture(tmp_path)
    output = driver.reserve_output(tmp_path / "retained" / "cohort.json")
    durable = driver.begin_trial(output.with_suffix(".traces"), configuration)
    reference = driver.retain_trial(tmp_path, durable, configuration, parent, payload)
    payload["details"]["report"]["configuration"]["source_lock_digest"] = "replaced-after-archiving"
    with pytest.raises(InvalidObservation, match="ordinary_payload_binding"):
        driver.accept_receipt(payload, configuration, parent, reference)


def test_missing_actual_request_thread_receipt_is_unqualified():
    document = trace_v2()
    document["request_workers"] = []
    with pytest.raises(InvalidObservation, match="request_worker_retirement"):
        summarize(document)


@pytest.mark.parametrize("teardown_fails", [False, True])
def test_real_execute_run_thread_survives_dashboard_clear_and_captures_teardown(tmp_path, monkeypatch, teardown_fails):
    import threading
    from contextlib import contextmanager
    from pathlib import Path
    from types import SimpleNamespace
    from pllm.runtime import benchmark_resources
    from pllm.runtime.dashboard import DashboardRuntime
    from benchmarks.research.publication_lifecycle_live import instrument
    monkeypatch.setattr(benchmark_resources.ProcessMemorySampler, "start", lambda _: None)
    teardown, release, joining, joined = (threading.Event() for _ in range(4))
    escaped = []
    original_error = OSError("tokenizer scope teardown failed")
    @contextmanager
    def tokenizer_scope():
        yield
        teardown.set()
        assert release.wait(5)
        if teardown_fails:
            raise original_error
    runtime = SimpleNamespace(_topology=None, _worker=None, _background_threads=[], _memory_guard=None,
        _client=SimpleNamespace(tokenizer_scope=tokenizer_scope), record_status=None)
    def scoped_run(*_):
        runtime.record_status = "completed"
        runtime._worker = None  # Actual DashboardRuntime._finish_run behavior.
    runtime._execute_scoped_run = scoped_run
    capture = worker = joiner = None
    try:
        with instrument(capture_configuration(), tmp_path, Path(__file__)) as session:
            benchmark_resources.ProcessMemorySampler.start(SimpleNamespace(runtime=runtime,
                thread=SimpleNamespace(is_alive=lambda: False)))
            capture = session["capture"]
            def execute():
                try:
                    DashboardRuntime._execute_run(runtime, "test-run", "fixture", 2)
                except BaseException as error:
                    escaped.append(error)
            worker = threading.Thread(target=execute)
            runtime._worker = worker
            worker.start()
            assert teardown.wait(5)
            assert runtime.record_status == "completed" and runtime._worker is None
            assert capture.request_workers[0][0] is worker
            assert not capture.health()["client_workers_joined"]
            def join_request():
                joining.set()
                capture.join_request_workers()
                joined.set()
            joiner = threading.Thread(target=join_request)
            joiner.start()
            assert joining.wait(5) and not joined.wait(0.02)
            release.set()
            assert joined.wait(5)
            assert not worker.is_alive()
            assert capture.health()["client_workers_joined"] is (not teardown_fails)
            if teardown_fails:
                assert escaped == [original_error]
                assert capture.document["errors"][-1]["where"] == "request_worker_teardown"
                assert capture.document["request_workers"][0]["outcome"] == "failed"
            else:
                assert not escaped
                assert capture.document["request_workers"][0]["returned_ns"] <= capture.document["request_workers"][0]["joined_ns"]
    finally:
        release.set()
        for thread in (worker, joiner):
            if thread is not None:
                thread.join(timeout=5)
        if capture is not None:
            capture.stop.set()
            capture.thread.join(timeout=5)
            capture.file.close()


def mutate_receipt(directory, mutation):
    import json
    from benchmarks.research.publication_lifecycle_live import FINAL_FILE
    path = directory / FINAL_FILE
    receipt = json.loads(path.read_text())
    mutation(receipt)
    path.write_text(json.dumps(receipt))


@pytest.mark.parametrize("field", ["capacity", "clock"])
def test_parent_binds_capacity_and_clock_to_requested_contract(tmp_path, field):
    from benchmarks.research.publication_lifecycle import recorder_capacity
    payload, configuration, parent = receipt_fixture(tmp_path)
    def mutation(receipt):
        if field == "capacity":
            receipt["document"]["capacity"] = recorder_capacity(86400)
            receipt["document"]["nominal_sample_interval_ns"] = recorder_capacity(86400)["interval_ns"]
        else:
            receipt["document"]["clock"]["resolution_ns"] = 100_000
    mutate_receipt(tmp_path, mutation)
    with pytest.raises(InvalidObservation, match="parent_observation_contract"):
        accept_fixture(payload, configuration, parent, tmp_path)


def test_summary_rejects_8000_second_trace_under_1200_second_capacity():
    document = trace_v2()
    document["samples"][-1]["end_ns"] = 8000_000_000_000
    with pytest.raises(InvalidObservation, match="trace_duration_exceeds_capacity"):
        summarize(document)


@pytest.mark.parametrize("mutation", ["late_launch", "early_finish", "missing_timeout"])
def test_receipt_must_be_within_independent_parent_window(tmp_path, mutation):
    from benchmarks.research.publication_lifecycle import clock_tolerance_ns
    payload, configuration, parent = receipt_fixture(tmp_path)
    if mutation == "late_launch":
        parent["launch_started_ns"] = 50
    elif mutation == "early_finish":
        parent["supervisor_returned_ns"] = 900 - clock_tolerance_ns(configuration["clock"]) - 1
    else:
        parent["timeout_started_ns"] = None
    with pytest.raises(InvalidObservation, match="parent_observation_window"):
        accept_fixture(payload, configuration, parent, tmp_path)


@pytest.mark.parametrize("field", ["full_seconds", "online_seconds", "ttft_seconds", "preparation_seconds", "transition_seconds"])
def test_ordinary_duration_cannot_disagree_with_observed_endpoints(tmp_path, field):
    payload, configuration, parent = receipt_fixture(tmp_path)
    payload["details"]["report"]["runs"][0]["durations"][field] = 123.0
    with pytest.raises(InvalidObservation, match="ordinary_request_duration"):
        accept_fixture(payload, configuration, parent, tmp_path)


def test_request_metric_cannot_disagree_with_its_report_window(tmp_path):
    payload, configuration, parent = receipt_fixture(tmp_path)
    next(row for row in payload["metrics"] if row["id"] == "request.0.full")["value"] = 123.0
    with pytest.raises(InvalidObservation, match="ordinary_request_metric"):
        accept_fixture(payload, configuration, parent, tmp_path)


def scaled_counters(snapshot, factor):
    snapshot["total_stream_bytes"] *= factor
    snapshot["directed_stream_bytes"] = {key: value * factor for key, value in snapshot["directed_stream_bytes"].items()}
    for party in snapshot["parties"].values():
        for bucket in party.values():
            bucket["admitted_bytes"] *= factor


@pytest.mark.parametrize("mutation,reason", [
    (lambda samples: scaled_counters(samples["startup"]["wan_emulation"], 9), "ordinary_network_chronology"),
    (lambda samples: scaled_counters(samples["runs"][0]["before"]["wan_emulation"], 5), "ordinary_network_sample_bracket"),
    (lambda samples: samples["runs"][0]["before"]["wan_emulation"].update(elapsed_seconds=30 / 1e9), "ordinary_network_chronology"),
    (lambda samples: samples["runs"][0]["after"]["wan_emulation"].update(elapsed_seconds=600 / 1e9), "ordinary_network_request_bounds"),
])
def test_individually_conserved_snapshots_must_obey_time_and_counter_bounds(tmp_path, mutation, reason):
    payload, configuration, parent = receipt_fixture(tmp_path)
    mutation(payload["details"]["report"]["native_network_accounting"]["samples"])
    with pytest.raises(InvalidObservation, match=reason):
        accept_fixture(payload, configuration, parent, tmp_path)


def test_different_read_endpoints_are_bracketed_not_forced_equal(tmp_path):
    payload, configuration, parent = receipt_fixture(tmp_path)
    report = payload["details"]["report"]
    report["runs"][0]["durations"]["full_seconds"] += configuration["clock"]["resolution_ns"] / 1e9
    snapshots = report["native_network_accounting"]["samples"]
    snapshots["runs"][0]["before"]["wan_emulation"]["elapsed_seconds"] = 57 / 1e9  # Absolute t=107ns.
    snapshots["runs"][0]["after"]["wan_emulation"]["elapsed_seconds"] = 725 / 1e9  # Absolute t=775ns.
    result = accept_fixture(payload, configuration, parent, tmp_path)
    assert result["ordinary_network_reconciliation"]["maximum_clock_tolerance_ns"] <= 2 * configuration["clock"]["resolution_ns"] + 1


def test_main_checks_lexical_dangling_output_and_historical_companion_before_resolution(tmp_path, monkeypatch):
    import sys
    from contextlib import nullcontext
    import filelock
    from pllm import Model
    from scripts import benchmark_publication_lifecycle as driver
    requested = tmp_path / "cohort.json"
    redirected = tmp_path / "elsewhere" / "missing.json"
    requested.symlink_to(redirected)
    companion = requested.with_suffix(".plan.json")
    companion.write_bytes(b"historical plan")
    monkeypatch.setattr(sys, "argv", ["driver", "--output", str(requested), "--tiny", "--preflight-only"])
    monkeypatch.setattr(filelock, "FileLock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(Model, "tiny", lambda: (_ for _ in ()).throw(AssertionError("model setup must not run")))
    with pytest.raises(FileExistsError):
        driver.main()
    assert requested.is_symlink() and companion.read_bytes() == b"historical plan"
    assert not redirected.parent.exists()
    assert set(tmp_path.iterdir()) == {requested, companion}


def archive_fixture(tmp_path):
    from scripts import benchmark_publication_lifecycle as driver
    payload, configuration, parent = receipt_fixture(tmp_path)
    output = driver.reserve_output(tmp_path / "retained" / "cohort.json")
    directory = driver.begin_trial(output.with_suffix(".traces"), configuration)
    reference = driver.retain_trial(tmp_path, directory, configuration, parent, payload)
    return payload, configuration, parent, reference


def rewrite_test_archive(reference, files=None, edit_manifest=None):
    """Deliberately forge a test-only archive while updating its outer hashes."""
    import hashlib
    import json
    from pathlib import Path
    from benchmarks.research.publication_lifecycle_live import canonical
    reference = copy.deepcopy(reference)
    path = Path(reference["path"])
    manifest = json.loads(path.read_bytes())
    for name, value in (files or {}).items():
        raw = value if isinstance(value, bytes) else canonical(value)
        file = path.parent / name
        file.chmod(0o600)
        file.write_bytes(raw)
        file.chmod(0o400)
        manifest["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if edit_manifest is not None:
        edit_manifest(manifest)
    raw = canonical(manifest)
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(0o400)
    reference.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return reference


def test_parent_and_standalone_replay_share_full_qualification(tmp_path):
    from scripts import benchmark_publication_lifecycle as driver
    payload, configuration, parent, reference = archive_fixture(tmp_path)
    accepted = driver.accept_receipt(payload, configuration, parent, reference)
    trace, _, replayed = driver.load_trace(reference)
    assert trace["complete"] and replayed["qualified"]
    assert accepted == replayed["summary"]
    assert "ordinary-envelope" in accepted["scope"]["archive_qualification"]
    assert "observation structure only" in summarize(trace)["scope"]["qualification"]


def test_manifest_nonce_a_cannot_carry_configuration_and_receipt_b(tmp_path):
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    payload, configuration, parent, reference = archive_fixture(tmp_path)
    old = Path(reference["path"]).parent
    other = old.with_name("a" * 64)
    old.rename(other)
    reference["path"] = str(other / driver.MANIFEST_FILE)
    reference = rewrite_test_archive(reference, edit_manifest=lambda m: m["binding"].update(nonce="a" * 64))
    reference["binding"]["nonce"] = "a" * 64
    with pytest.raises(InvalidObservation, match="archive_binding"):
        driver.load_trace(reference)
    with pytest.raises(InvalidObservation, match="archive_binding"):
        driver.accept_receipt(payload, configuration, parent, reference)


@pytest.mark.parametrize("name", ["publication-request.json", "publication-state.json", "publication-final.json",
    "publication-samples.jsonl", "parent.json", "ordinary-payload.json",
    "publication-child-inference.json", "publication-child-preparation.json"])
def test_complete_archive_cannot_omit_any_consumed_file(tmp_path, name):
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    _, _, _, reference = archive_fixture(tmp_path)
    reference = rewrite_test_archive(reference, edit_manifest=lambda m: m["files"].pop(name))
    if name == driver.SAMPLES_FILE:
        # Reproduce the independent audit: an unlisted but present journal with
        # changed RSS must never reach summarize() as a complete observation.
        rows = trace_v2()["samples"]
        rows[3]["rss_bytes"]["client"] = 123
        path = Path(reference["path"]).parent / name
        path.chmod(0o600)
        path.write_bytes(b"".join(driver.canonical(row) + b"\n" for row in rows))
    for partial in (False, True):
        with pytest.raises(InvalidObservation, match="trace_manifest_(files|coverage)"):
            driver.load_trace(reference, allow_partial=partial)


@pytest.mark.parametrize("field", ["sample_count", "samples_sha256", "samples_bytes"])
def test_archive_reconciles_receipt_with_actual_sample_count_hash_and_size(tmp_path, field):
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    _, _, _, reference = archive_fixture(tmp_path)
    receipt = driver.read_bounded(Path(reference["path"]).parent / driver.FINAL_FILE)
    if field == "sample_count":
        receipt["document"][field] += 1
    elif field == "samples_sha256":
        receipt[field] = "0" * 64
    else:
        receipt[field] += 1
    reference = rewrite_test_archive(reference, {driver.FINAL_FILE: receipt})
    with pytest.raises(InvalidObservation, match="receipt_sample_binding"):
        driver.load_trace(reference)


def test_rehashed_sample_mutation_cannot_bypass_final_receipt(tmp_path):
    from scripts import benchmark_publication_lifecycle as driver
    _, _, _, reference = archive_fixture(tmp_path)
    rows = trace_v2()["samples"]
    rows[3]["rss_bytes"]["client"] = 123
    reference = rewrite_test_archive(reference, {driver.SAMPLES_FILE:
        b"".join(driver.canonical(row) + b"\n" for row in rows)})
    with pytest.raises(InvalidObservation, match="receipt_sample_binding"):
        driver.load_trace(reference)


@pytest.mark.parametrize("control,reason", [("duration", "ordinary_request_duration"),
    ("pressure", "ordinary_traffic_or_pressure_reconciliation"), ("target", "ordinary_target_failed")])
def test_standalone_archive_rejects_ordinary_failures_and_only_recovers_unqualified_data(tmp_path, control, reason):
    from scripts import benchmark_publication_lifecycle as driver
    payload, _, _, reference = archive_fixture(tmp_path)
    if control == "duration":
        payload["details"]["report"]["runs"][0]["durations"]["full_seconds"] = 123.0
    elif control == "pressure":
        payload["details"]["report"]["memory_guard"]["maximum_host_swap_growth_bytes"] = 1
    else:
        payload.update(status="failed", failure_code="original_target_failure")
    reference = rewrite_test_archive(reference, {"ordinary-payload.json": payload})
    with pytest.raises(InvalidObservation, match=reason):
        driver.load_trace(reference)
    partial, receipt, retained = driver.load_trace(reference, allow_partial=True)
    assert not partial["complete"] and not retained["qualified"]
    assert receipt is not None and retained["qualification_failure"]["code"] == reason
    with pytest.raises(InvalidObservation, match="incomplete_trace"):
        summarize(partial)


def test_failed_finalization_receipt_survives_archive_diagnostic_recovery(tmp_path):
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    _, _, _, reference = archive_fixture(tmp_path)
    receipt = driver.read_bounded(Path(reference["path"]).parent / driver.FINAL_FILE)
    receipt["document"].update(complete=False, errors=[{"where": "finalization", "code": "cleanup_health"}])
    reference = rewrite_test_archive(reference, {driver.FINAL_FILE: receipt, driver.STATE_FILE: receipt["document"]})
    with pytest.raises(InvalidObservation, match="lifecycle_failed"):
        driver.load_trace(reference)
    trace, recovered, retained = driver.load_trace(reference, allow_partial=True)
    assert not trace["complete"] and not retained["qualified"]
    assert recovered == receipt and trace["errors"] == receipt["document"]["errors"]


def test_legacy_inline_replay_never_inherits_archive_guarantees(tmp_path):
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    legacy = summarize(trace())
    assert legacy["schema"] == "pllm.publication_lifecycle_summary.v1"
    assert "archive_qualification" not in legacy["scope"] and "qualification" not in legacy["scope"]
    _, _, _, reference = archive_fixture(tmp_path)
    receipt = driver.read_bounded(Path(reference["path"]).parent / driver.FINAL_FILE)
    receipt["document"]["schema"] = "pllm.publication_lifecycle.v1"
    reference = rewrite_test_archive(reference, {driver.FINAL_FILE: receipt, driver.STATE_FILE: receipt["document"]})
    with pytest.raises(InvalidObservation, match="archive_trace_schema"):
        driver.load_trace(reference)


def bounded_reader_subprocess(mode, value):
    """An unsafe FIFO reader fails this test by timeout; always kill and reap it."""
    import json
    from pathlib import Path
    import subprocess
    import sys
    code = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from scripts import benchmark_publication_lifecycle as driver
mode, value = sys.argv[2], json.loads(sys.argv[3])
try:
    if mode == "archive":
        driver.load_trace(value)
    elif mode == "samples":
        driver.read_samples(value)
    else:
        driver.read_bounded(value)
except (driver.InvalidObservation, OSError) as error:
    print(type(error).__name__ + ":" + str(error))
else:
    raise AssertionError("unsafe file was accepted")
'''
    process = subprocess.Popen([sys.executable, "-I", "-B", "-c", code,
        str(Path(__file__).resolve().parents[1]), mode, json.dumps(value)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=2)
        assert process.returncode == 0, stderr
        return stdout
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=2)


@pytest.mark.parametrize("name", ["manifest.json", "publication-request.json", "publication-state.json",
    "publication-final.json", "publication-samples.jsonl", "parent.json", "ordinary-payload.json",
    "publication-child-inference.json", "publication-child-preparation.json"])
def test_every_archive_reader_rejects_fifo_without_blocking(tmp_path, name):
    import hashlib
    import os
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    _, _, _, reference = archive_fixture(tmp_path)
    path = Path(reference["path"]).parent / name
    if name == driver.MANIFEST_FILE:
        reference.update(bytes=0, sha256=hashlib.sha256(b"").hexdigest())
    else:
        changes = {}
        if name == driver.SAMPLES_FILE:
            # Exercise the reader itself, rather than fail earlier geometry.
            receipt = driver.read_bounded(path.parent / driver.FINAL_FILE)
            receipt["document"]["complete"] = False
            changes[driver.FINAL_FILE] = receipt
        reference = rewrite_test_archive(reference, changes, edit_manifest=lambda m:
            m["files"].update({name: {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}}))
    path.unlink()
    os.mkfifo(path)
    assert "sidecar_not_regular" in bounded_reader_subprocess("archive", reference)


@pytest.mark.parametrize("mode,name", [("samples", "publication-samples.jsonl"),
    ("json", "publication-child-inference.json"), ("json", "publication-request.json")])
def test_direct_partial_and_worker_json_readers_reject_fifo(tmp_path, mode, name):
    import os
    os.mkfifo(tmp_path / name)
    value = str(tmp_path if mode == "samples" else tmp_path / name)
    assert "sidecar_not_regular" in bounded_reader_subprocess(mode, value)


@pytest.mark.parametrize("name", ["manifest.json", "publication-request.json", "publication-state.json",
    "publication-final.json", "publication-samples.jsonl", "parent.json", "ordinary-payload.json",
    "publication-child-inference.json", "publication-child-preparation.json"])
def test_sparse_inputs_exceeding_filename_cap_are_rejected_before_hash_or_read(tmp_path, monkeypatch, name):
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    from benchmarks.research import publication_lifecycle_live as live
    _, _, _, reference = archive_fixture(tmp_path)
    path = Path(reference["path"]).parent / name
    cap = driver.artifact_limit(name)
    path.chmod(0o600)
    with path.open("wb") as file:
        file.truncate(cap + 1)  # Sparse; no large allocation or linear hash pass.
    original = live.SidecarReader._read
    def no_read_before_bounds(reader, size, **kwargs):
        assert reader.name != name, "over-limit file was read before its size check"
        return original(reader, size, **kwargs)
    monkeypatch.setattr(live.SidecarReader, "_read", no_read_before_bounds)
    with pytest.raises(InvalidObservation, match="sidecar_size"):
        driver.load_trace(reference)
    assert "sidecar_size" in bounded_reader_subprocess("archive", reference)


def test_sample_geometry_is_checked_before_hashing_sparse_journal(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    from benchmarks.research import publication_lifecycle_live as live
    _, _, _, reference = archive_fixture(tmp_path)
    directory = Path(reference["path"]).parent
    receipt = driver.read_bounded(directory / driver.FINAL_FILE)
    receipt.update(samples_bytes=16 << 20, samples_sha256="0" * 64)
    reference = rewrite_test_archive(reference, {driver.FINAL_FILE: receipt}, edit_manifest=lambda m:
        m["files"].update({driver.SAMPLES_FILE: {"bytes": 16 << 20, "sha256": "0" * 64}}))
    path = directory / driver.SAMPLES_FILE
    path.chmod(0o600)
    with path.open("wb") as file:
        file.truncate(16 << 20)
    original = live.SidecarReader._read
    def guarded(reader, size, **kwargs):
        assert reader.name != driver.SAMPLES_FILE
        return original(reader, size, **kwargs)
    monkeypatch.setattr(live.SidecarReader, "_read", guarded)
    with pytest.raises(InvalidObservation, match="receipt_sample_binding"):
        driver.load_trace(reference)
    assert "receipt_sample_binding" in bounded_reader_subprocess("archive", reference)


@pytest.mark.parametrize("kind", ["leaf_symlink", "directory_symlink", "escape", "unlisted_escape"])
def test_archive_paths_cannot_escape_or_follow_symlinks(tmp_path, kind):
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    _, _, _, reference = archive_fixture(tmp_path)
    directory = Path(reference["path"]).parent
    if kind == "leaf_symlink":
        path = directory / "parent.json"
        outside = tmp_path / "outside.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif kind == "directory_symlink":
        alias = tmp_path / directory.name
        alias.symlink_to(directory, target_is_directory=True)
        reference["path"] = str(alias / driver.MANIFEST_FILE)
    elif kind == "escape":
        reference["path"] = str(directory / ".." / directory.name / driver.MANIFEST_FILE)
    else:
        reference = rewrite_test_archive(reference, edit_manifest=lambda m:
            m["files"].update({"../outside": {"bytes": 0, "sha256": "0" * 64}}))
    with pytest.raises(InvalidObservation, match="sidecar_|trace_manifest_files"):
        driver.load_trace(reference)


@pytest.mark.parametrize("change", ["replace_inode", "modify_in_place", "grow", "replace_directory"])
def test_same_fd_hash_parse_and_late_stability_checks_reject_races(tmp_path, monkeypatch, change):
    import os
    from pathlib import Path
    from scripts import benchmark_publication_lifecycle as driver
    _, _, _, reference = archive_fixture(tmp_path)
    directory = Path(reference["path"]).parent
    path = directory / driver.REQUEST_FILE
    original = driver.sidecar_json
    def raced(reader):
        result = original(reader)
        if reader.name == driver.REQUEST_FILE:
            if change == "replace_inode":
                replacement = directory / "replacement.json"
                replacement.write_bytes(path.read_bytes())  # Same bytes, different inode.
                os.replace(replacement, path)
            elif change == "replace_directory":
                directory.rename(directory.with_name("retired"))
                directory.mkdir()
            else:
                path.chmod(0o600)
                with path.open("ab" if change == "grow" else "r+b") as file:
                    file.write(b"x")
        return result
    monkeypatch.setattr(driver, "sidecar_json", raced)
    with pytest.raises(InvalidObservation, match="sidecar_(changed|directory_changed)"):
        driver.load_trace(reference)


def test_live_failed_fifo_receipt_is_persisted_without_masking_primary_failure(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from scripts import benchmark_publication_lifecycle as driver
    os.mkfifo(tmp_path / "publication-child-inference.json")
    code = r'''
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from benchmarks.research.publication_lifecycle import CLEANUP_FLAGS
from benchmarks.research.publication_lifecycle_live import Capture, observation_clock
configuration = {"expected_roles":["client","inference","preparation"], "timeout_seconds":2,
    "output_cap":2, "binding":{}, "runtime":{}, "wan":{}, "clock":observation_clock()}
capture = Capture(SimpleNamespace(_topology=None), configuration, sys.argv[2])
capture.start()
capture.health = lambda: {**dict.fromkeys(CLEANUP_FLAGS, True), "relay_error":None}
capture.document["target_succeeded"] = True
capture.error(RuntimeError("primary failure"), "target")
capture.finish(after_serialization=True)
'''
    process = subprocess.Popen([sys.executable, "-I", "-B", "-c", code,
        str(Path(__file__).resolve().parents[1]), str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _, stderr = process.communicate(timeout=2)
        assert process.returncode == 0, stderr.decode()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=2)
    receipt = driver.read_bounded(tmp_path / driver.FINAL_FILE)
    assert receipt["document"]["complete"] is False
    assert receipt["document"]["errors"][0]["exception"] == "RuntimeError"
    assert receipt["document"]["errors"][-1]["code"] == "sidecar_not_regular"

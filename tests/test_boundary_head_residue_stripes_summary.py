"""Model-free mocked reports and forgery tests. No benchmark output discovery."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stripe_summary", Path(__file__).parents[1] / "scripts/summarize_boundary_head_residue_stripes.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def guard():
    return {"host": {"total": 64 << 30, "available": 32 << 30, "swap_used": 0, "disk_free": 100 << 30},
            "reserve_bytes": 8 << 30, "allocation_allowance_bytes": 6 * summary.WEIGHTS + (512 << 20),
            "minimum_available_bytes": 30 << 30, "maximum_swap_growth_bytes": 0, "sample_interval_seconds": .1}


def network(bodies):
    online = {link: size + 500 for role, pair in bodies.items()
              for link, size in zip((f"client->{role}", f"{role}->client"), pair)}
    links = {link: size + 3000 for link, size in online.items()}
    conditions = {"schema": "pllm.wan_conditions.v1", "download_mbps": 100., "upload_mbps": 40.,
                  "parties": [], "role_parties": []}
    net = {"schema": "pllm.native_wan.v1", "backend": "native-shared-tcp-pacer", "enforced": True,
           "conditions": conditions, "conditions_digest": summary.sha(b"pllm.wan_conditions.v1\0" + summary.canonical(conditions)),
           "role_parties": {r: r for r in ("client", *summary.ROLES)}, "burst_bytes": 16384,
           "read_chunk_bytes": 8192, "max_connections": 32, "elapsed_seconds": 100.,
           "directed_stream_bytes": links, "total_stream_bytes": sum(links.values()),
           "client_stream_bytes": sum(links.values()), "parties": {}}
    for party in ("client", *summary.ROLES):
        net["parties"][party] = {direction: {"bytes_per_second": rate, "admitted_bytes": sum(
            size for link, size in links.items() if link.split("->")[side] == party)}
            for direction, rate, side in (("upload", 5_000_000., 0), ("download", 12_500_000., 1))}
    return online, net


def row(mode, read, worker, repeat, position, descriptor):
    baseline = mode == "parallel"
    bodies = {role: summary.body_sizes(mode, role) for role in summary.ROLES}
    ledgers, timings = {}, []
    for role in summary.ROLES:
        request, reply = bodies[role]
        ledgers[role] = ({"installed_online_upload_bytes": request, "installed_online_download_bytes": reply}
            if baseline else {"attempted_request_body_bytes": request, "generated_reply_body_bytes": reply,
                              "observed_reply_body_bytes": reply})
        timing = {"native_seconds": 0. if baseline else .1, "encode_seconds": 0. if baseline else .01,
                  "first_asgi_body_seconds": .2, "asgi_response_body_bytes": reply}
        if not baseline:
            timing.update(first_yield_seconds=.2, first_client_body_seconds=.3,
                          first_paired_decoded_stripe_seconds=.4)
        timings.append(timing)
    online, net = network(bodies)
    return {"mode": mode, "repeat": repeat, "position": position, "success": True, "error": None,
            "exchange_seconds": float(worker + repeat + summary.MODES.index(mode)),
            "exchange_cpu_seconds": float(worker * 10 + repeat + summary.MODES.index(mode)),
            "setup_seconds": .01, "geometry": summary.codec_geometry(descriptor),
            "candidate_read_chunk_bytes": read, "remaining_native_tasks": 0, "online_coverage_complete": True,
            "replay_status": [] if baseline else [409, 409], "per_role_bodies": ledgers,
            "timing_breakdown": timings, "owners": None if baseline else [
                {"attempted": True, "retired": True, "native_calls": 1 if mode == "sliced" else 19,
                 "drained_native_calls": 0, "peak_output_slots": 1 if mode == "sliced" else 2,
                 "peak_output_bytes": 4 * summary.V if mode == "sliced" else 8 * summary.STRIPE,
                 "peak_pending_native": 1, "replay_rejected": True} for _ in summary.ROLES],
            "baseline_ownership": "installed worker/session and paired executor" if baseline else None,
            "candidate_operation_counters": None if baseline else {role: {"head_calls_reserved": 1,
                "sequence_reserved": 0, "complete_stripes_expected": 19} for role in summary.ROLES},
            "installed_operation_counters": {"lookup": {"calls": 0, "rows": 0, "upload_bytes": 0, "download_bytes": 0},
                "head": {"calls": int(baseline), "rows": int(baseline),
                         "upload_bytes": sum(v[0] for v in bodies.values()) if baseline else 0,
                         "download_bytes": sum(v[1] for v in bodies.values()) if baseline else 0}},
            "online_tcp_payload_bytes": online, "all_phases_network": net, "logits_sha256": "c" * 64}


def documents(report):
    core = {k: v for k, v in report.items() if k not in ("plan", "host_guard", "progress")}
    fixture = {"kind": "fixture", "setup": report["setup"], "descriptor_sha256": report["descriptor_sha256"],
               "geometry_sha256": report["codec_geometry_sha256"], "source_sha256": report["source_sha256"],
               "weight_sha256": report["weight_sha256"], "scales_sha256": report["scales_sha256"]}
    events = [fixture, *({"kind": "run", "report": r} for r in report["runs"]), {"kind": "complete", "report": core}]
    return [json.dumps(report).encode(), json.dumps(report["progress"]).encode(),
            ("\n".join(json.dumps(event) for event in events) + "\n").encode()]


def write_report(path, report):
    paths = [path, Path(str(path) + ".progress.json"), Path(str(path) + ".runs.jsonl")]
    for target, raw in zip(paths, documents(report)):
        target.write_bytes(raw)


@pytest.fixture
def cohort(tmp_path):
    checkout = tmp_path / "checkout"
    sources = {}
    for key, relative in summary.SOURCE_PATHS.items():
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("mock source: " + relative)
        sources[key] = summary.sha(target.read_bytes())
    pins = {"sources_sha256": summary.sha(summary.canonical(sources)), "native_sha256": "a" * 64,
            "build": {"expected_digest": "b" * 64, "interpreter_sha256": "d" * 64}}
    provenance = {**pins, "sources": sources, "python_version": "mock", "python_executable_name": "python"}
    paths, reports = [], []
    for read in (8192, 1024):
        for worker in (1, 2, 3):
            descriptor = "e" * 64
            geometry = summary.geometry(read)
            geometry_hash = summary.sha(summary.canonical(geometry))
            progress = {"schema": "pllm.research.head-residue-stripes-progress.v1", "status": "completed",
                        "measurement_complete": True, "planned_runs": 18, "completed_runs": 18,
                        "geometry_sha256": geometry_hash, "sources_sha256": pins["sources_sha256"]}
            report = {"schema": "pllm.research.head-residue-stripes.v1", "source_sha256": summary.sha(
                summary.canonical({"generated": 17, "vocabulary": summary.V, "hidden": summary.H})),
                "descriptor_sha256": descriptor, "codec_geometry_sha256": summary.sha(summary.canonical(summary.codec_geometry(descriptor))),
                "weight_sha256": "f" * 64, "scales_sha256": "1" * 64, "native_sha256": pins["native_sha256"],
                "native_filename": "_native.abi3.so", "installed_matrix_methods": ["shape", "weight_bytes"],
                "installed_executor_methods": ["wrap32"], "repeat_orders": summary.ORDERS,
                "production_ready": False, "setup": {"wall_seconds": .5, "generated_weight_bytes": summary.WEIGHTS,
                    "full_native_weight_bytes": summary.WEIGHTS, "additional_snapshot_weight_bytes": summary.WEIGHTS,
                    "codec_layout_bytes": summary.V}, "progress": progress, "host_guard": guard(),
                "plan": {"sampled_qwen": True, "sampled_worker": worker, "geometry": geometry,
                         "geometry_sha256": geometry_hash, "repeat_orders": summary.ORDERS,
                         "provenance": copy.deepcopy(provenance), "host_admission": guard()},
                "runs": [row(mode, read, worker, repeat, position, descriptor)
                         for repeat, order in enumerate(summary.ORDERS) for position, mode in enumerate(order)]}
            path = tmp_path / f"report-{read}-{worker}.json"
            write_report(path, report)
            paths.append(path)
            reports.append(report)
    return paths, reports, pins, checkout


def test_deterministic_read_only_summary_and_two_level_medians(cohort):
    paths, reports, pins, checkout = cohort
    before = {p: p.read_bytes() for path in paths for p in (path, Path(str(path) + ".progress.json"), Path(str(path) + ".runs.jsonl"))}
    result = summary.summarize(paths, pins, checkout=checkout)
    same = summary.summarize(paths[::-1], pins, checkout=checkout)
    assert summary.canonical(result) == summary.canonical(same)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert set(result["read_cohorts"]) == {"8192", "1024"}
    read = result["read_cohorts"]["8192"]
    assert read["workers"][0]["arms"]["parallel"]["elapsed_seconds_median"] == 3.5
    assert read["across_worker_medians_of_worker_medians"]["parallel"]["elapsed_seconds_median"] == 4.5
    assert read["workers"][0]["paired_delta_medians"]["snapshots_minus_parallel"]["elapsed_seconds"] == 2.
    assert read["workers"][0]["paired_delta_medians"]["snapshots_minus_sliced"]["application_body_bytes"] == 0
    assert result["production_ready"] is False
    assert result["read_cohorts"]["8192"]["workers"][0]["validated_exposed_counts"] == {
        "successful_exchange_rows": 18, "candidate_retired_owners": 24, "candidate_native_calls": 240,
        "replay_409_rejections": 24, "remaining_native_tasks": 0,
        "baseline_retired_owner_count": None, "context_fault_injection_count": None}
    assert result["evidence_limits"]["integer_parity"]["parallel"].startswith("unknown")
    assert result["evidence_limits"]["import_and_process_startup"].startswith("unknown")
    assert read["workers"][0]["arms"]["parallel"]["encoding_elapsed_seconds_median_by_role"] is None


def test_across_worker_statistic_never_pools_repeat_samples(cohort):
    paths, reports, pins, checkout = cohort
    samples = ([1.] * 6, [2., 3., 4., 100., 101., 102.], [5., 6., 7., 8., 9., 10.])
    for path, report, values in zip(paths[:3], reports[:3], samples):
        for row_value, elapsed in zip((r for r in report["runs"] if r["mode"] == "parallel"), values):
            row_value["exchange_seconds"] = elapsed
            row_value["all_phases_network"]["elapsed_seconds"] = 1000.
        write_report(path, report)
    result = summary.summarize(paths, pins, checkout=checkout)
    actual = result["read_cohorts"]["8192"]["across_worker_medians_of_worker_medians"]["parallel"]["elapsed_seconds_median"]
    assert actual == 7.5
    assert actual != summary.med(value for worker_samples in samples for value in worker_samples)


@pytest.mark.parametrize("size", [32, 255, 256, 3584, 25674, 474874])
def test_independent_msgpack_body_geometry(size):
    import msgpack
    row_value = {"schema": "pllm.private_boundary.v1.reply", "role": "boundary_a", "session": "c" * 32,
                 "epoch": 0, "sequence": 0, "operation": "head", "frames": [bytes(size)]}
    assert summary.frame_bytes("boundary_a", size, response=True) == len(msgpack.packb(row_value, use_bin_type=True))
    assert summary.RANGES[-1] == 4480 and len(summary.RANGES) == 19


@pytest.mark.parametrize("mutation", [
    lambda r: r["runs"].pop(),
    lambda r: r["runs"][0].update(repeat=1),
    lambda r: r["runs"][0].update(success=False),
    lambda r: r["runs"][0].update(logits_sha256="9" * 64),
    lambda r: r["runs"][0].update(exchange_seconds=float("nan")),
    lambda r: r["runs"][0].update(exchange_cpu_seconds=-1),
    lambda r: r["runs"][0].update(online_coverage_complete=False),
    lambda r: r["runs"][0].update(remaining_native_tasks=1),
    lambda r: r["runs"][1]["owners"][0].update(retired=False),
    lambda r: r["runs"][1]["owners"][0].update(peak_pending_native=2),
    lambda r: r["runs"][2]["owners"][0].update(peak_output_bytes=1000000),
    lambda r: r["runs"][1].update(replay_status=[200, 409]),
    lambda r: r["runs"][1]["per_role_bodies"]["boundary_a"].update(observed_reply_body_bytes=1),
    lambda r: r["runs"][0]["all_phases_network"].update(total_stream_bytes=1),
    lambda r: r["runs"][0]["all_phases_network"]["parties"]["client"]["download"].update(admitted_bytes=1),
    lambda r: r["runs"][0]["all_phases_network"]["role_parties"].update(boundary_a="client"),
    lambda r: r["runs"][0]["timing_breakdown"][0].update(asgi_response_body_bytes=1),
    lambda r: r["host_guard"].update(maximum_swap_growth_bytes=1),
    lambda r: r["setup"].update(additional_snapshot_weight_bytes=0),
    lambda r: r["progress"].update(measurement_complete=False),
    lambda r: r["plan"].update(sampled_worker=2),
    lambda r: r["plan"]["geometry"].update(stripe_rows=4096),
    lambda r: r.update(native_sha256="0" * 64),
    lambda r: r.update(weight_sha256="0" * 64),
    lambda r: r["plan"]["provenance"]["build"].update(expected_digest="0" * 64),
    lambda r: r["runs"][1].pop("owners"),
    lambda r: r["runs"][0]["timing_breakdown"][0].pop("first_asgi_body_seconds"),
    lambda r: r["runs"][0].update(exchange_seconds=True),
    lambda r: r["runs"][0].update(exchange_seconds=10 ** 400),
])
def test_rejects_self_consistent_forged_or_partial_report(cohort, mutation):
    paths, reports, pins, checkout = cohort
    report = copy.deepcopy(reports[0])
    mutation(report)
    write_report(paths[0], report)  # Forge every sidecar consistently; cross-check independent invariants.
    with pytest.raises((summary.Rejected, KeyError, TypeError, IndexError)):
        summary.summarize(paths, pins, checkout=checkout)


@pytest.mark.parametrize("suffix", ["", ".progress.json", ".runs.jsonl"])
def test_missing_any_artifact_rejected(cohort, suffix):
    paths, _, pins, checkout = cohort
    Path(str(paths[0]) + suffix).unlink()
    with pytest.raises(summary.Rejected):
        summary.summarize(paths, pins, checkout=checkout)


def test_mismatched_jsonl_progress_and_duplicate_json_keys_rejected(cohort):
    paths, reports, pins, checkout = cohort
    sidecar = Path(str(paths[0]) + ".runs.jsonl")
    original = sidecar.read_bytes()
    sidecar.write_bytes(original.rstrip(b"\n"))
    with pytest.raises(summary.Rejected, match="partial_jsonl"):
        summary.summarize(paths, pins, checkout=checkout)
    sidecar.write_bytes(original)
    progress = Path(str(paths[0]) + ".progress.json")
    progress.write_text('{"status":"completed","status":"rejected"}')
    with pytest.raises(summary.Rejected, match="duplicate_json_key"):
        summary.summarize(paths, pins, checkout=checkout)


def test_wrong_original_pins_and_changed_source_refused(cohort):
    paths, _, pins, checkout = cohort
    changed = copy.deepcopy(pins)
    changed["native_sha256"] = "0" * 64
    with pytest.raises(summary.Rejected, match="original_native"):
        summary.summarize(paths, changed, checkout=checkout)
    (checkout / "scripts/probe_boundary_head_residue_stripes.py").write_text("changed source")
    with pytest.raises(summary.Rejected, match="original_source_file"):
        summary.summarize(paths, pins, checkout=checkout)


def test_sidecar_only_forgery_refused(cohort):
    paths, _, pins, checkout = cohort
    path = Path(str(paths[0]) + ".runs.jsonl")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[1]["report"]["exchange_seconds"] = 123.
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(summary.Rejected, match="run_sidecar"):
        summary.summarize(paths, pins, checkout=checkout)


def test_nonfinite_overflow_and_duplicate_json_rejected():
    for raw in (b'{"x":NaN}', b'{"x":1e999}', b'{"x":1,"x":2}'):
        with pytest.raises(summary.Rejected):
            summary.parse(raw)


def test_no_duplicate_paths_or_missing_read_cohort(cohort):
    paths, _, pins, checkout = cohort
    for bad in (paths[:5], paths[:5] + [paths[0]]):
        with pytest.raises(summary.Rejected):
            summary.summarize(bad, pins, checkout=checkout)


def test_cli_rejection_is_deterministic_and_sanitized(tmp_path, capsys):
    args = [str(tmp_path / f"private-{i}.json") for i in range(6)]
    for flag in ("--expected-sources-sha256", "--expected-native-sha256", "--expected-build-digest", "--expected-interpreter-sha256"):
        args.extend([flag, "a" * 64])
    assert summary.cli(args) == 1
    out = capsys.readouterr().out
    assert json.loads(out)["status"] == "rejected"
    assert str(tmp_path) not in out
    assert not list(tmp_path.iterdir())


def test_cli_success_uses_explicit_paths_and_stdout_only(cohort, monkeypatch, capsys):
    paths, _, pins, checkout = cohort
    real = summary.summarize
    expected = real(paths, pins, checkout=checkout)
    monkeypatch.setattr(summary, "summarize", lambda paths, pins: real(paths, pins, checkout=checkout))
    args = [str(p) for p in paths]
    args += ["--expected-sources-sha256", pins["sources_sha256"],
             "--expected-native-sha256", pins["native_sha256"],
             "--expected-build-digest", pins["build"]["expected_digest"],
             "--expected-interpreter-sha256", pins["build"]["interpreter_sha256"]]
    before = {p: p.read_bytes() for path in paths for p in (path, Path(str(path) + ".progress.json"), Path(str(path) + ".runs.jsonl"))}
    assert summary.cli(args) == 0
    out = capsys.readouterr().out
    assert json.loads(out) == expected
    assert out == json.dumps(expected, sort_keys=True, indent=2, allow_nan=False) + "\n"
    assert all(path.read_bytes() == raw for path, raw in before.items())

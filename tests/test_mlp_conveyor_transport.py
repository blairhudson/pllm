"""Real loopback TCP, independent wire arithmetic and owner lifecycle regressions."""
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_mlp_conveyor_transport.py"
SPEC = importlib.util.spec_from_file_location("mlp_conveyor_transport", PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("rows,tile", [(1, 7), (7, 7), (17, 7), (17, 16)])
def test_three_controls_complete_output_and_real_wire(rows, tile):
    results = [asyncio.run(probe.run_case(rows=rows, tile=tile, mode=mode)) for mode in probe.MODES]
    assert len({r["output_sha256"] for r in results}) == 1
    for r in results:
        assert r["success"] and r["exact_bits"] and r["scalar_integer_oracle"]
        assert r["wire_reconciled"] and not r["errors"]
        assert r["client_claimed_rows"] == r["provider_claimed_rows"] == 2 * rows
        assert r["client_burned_rows"] == r["provider_burned_rows"] == 0
        assert r["terminal_product_owners"] == 0 and r["weakref_retirement"]
        assert r["session_replay_rejected"]
        assert r["state"]["peak_tile_owners"] <= (2 if r["mode"] == "conveyor" else 1)
        assert r["state"]["peak_frame_credits"] <= 2
        assert r["state"]["peak_request_queue"] <= 1
        assert r["state"]["request_frames"] == r["state"]["reply_frames"] == r["frames_expected"]
        # Closed form independent of envelope and descriptor constructors.
        h, m = 8, 13
        assert r["ledger"]["client_tx"] == rows * (h + m) * 4 + r["frames_expected"] * probe.HEADER.size
        assert r["ledger"]["client_rx"] == rows * (2 * m + h) * 4 + r["frames_expected"] * probe.HEADER.size
        assert r["wire_parts"]["client_tx"]["body"] == rows * (h + m) * 4
        assert r["wire_parts"]["client_rx"]["body"] == rows * (2 * m + h) * 4
        for direction, parts in r["wire_parts"].items():
            assert sum(parts.values()) == r["ledger"][direction]
        assert r["retained_source_output_residual_bytes"] == 3 * rows * h * 4
        assert r["remote_fixture_body_mac_fraction"] == 1
        assert r["sdk_remote_body_mac_fraction"] is None and not r["whole_client_admitted"]


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5])
def test_cancellation_drains_submitted_frames_then_burns_both_stage_leases(cut):
    r = asyncio.run(probe.run_case(cancel_after_stage=cut))
    assert r["canceled"] and not r["success"] and r["exact_bits"] is None
    assert r["wire_reconciled"]
    assert r["state"]["submitted_frames"] == r["state"]["reply_frames"]
    assert r["client_claimed_rows"] + r["client_burned_rows"] == 34
    assert r["provider_claimed_rows"] + r["provider_burned_rows"] == 34
    assert r["weakref_retirement"] and r["session_replay_rejected"]


@pytest.mark.parametrize("fault", ["session", "sequence", "row", "truncate", "provider", "kernel",
                                   "request_session", "request_sequence", "request_row"])
def test_actual_corrupt_reply_or_provider_failure_closes_and_burns(fault):
    r = asyncio.run(probe.run_case(fault=fault))
    assert not r["success"] and r["errors"] and r["exact_bits"] is None
    assert r["client_claimed_rows"] + r["client_burned_rows"] == 34
    assert r["provider_claimed_rows"] + r["provider_burned_rows"] == 34
    assert r["terminal_product_owners"] == 0 and r["session_replay_rejected"]
    assert r["unread_request_bytes"] >= 0 and r["unread_reply_bytes"] >= 0


def test_public_stage_order_and_per_stage_contiguous_one_use_rows():
    r = asyncio.run(probe.run_case(rows=25))
    sends = [e for e in r["events"] if e["event"] == "client_send_start"]
    assert [(e["stage"], e["row"]) for e in sends] == [
        ("G", 0), ("G", 7), ("D", 0), ("G", 14), ("D", 7),
        ("G", 21), ("D", 14), ("D", 21)]
    assert [e["sequence"] for e in sends] == list(range(8))
    # G1 is sent without waiting for G0's full reply: actual cross-stage lead.
    send_g1 = next(e for e in sends if e["stage"] == "G" and e["row"] == 7)
    reply_g0 = next(e for e in r["events"] if e["event"] == "client_reply_done" and e["sequence"] == 0)
    assert send_g1["seconds"] < reply_g0["seconds"]


def test_header_rejects_every_context_field_before_body_allocation():
    c = probe.geometry(17, 8, 13, 7, "conveyor")
    desc = probe.descriptors(c)[0]
    packet = probe.envelope(c, b"s" * 16, 0, desc, False, b"\0" * (7 * 8 * 4))
    raw = packet[:probe.HEADER.size]
    for field in range(len(probe.HEADER.unpack(raw))):
        values = list(probe.HEADER.unpack(raw))
        old = values[field]
        values[field] = b"x" * len(old) if isinstance(old, bytes) else old + 1
        with pytest.raises(ValueError):
            probe.validate_header(probe.HEADER.pack(*values), c, b"s" * 16, 0, desc, False)


def test_external_task_cancellation_joins_tcp_handlers_and_kernel():
    async def scenario():
        baseline = set(asyncio.all_tasks())
        task = asyncio.create_task(probe.run_case(rows=17, tile=1))
        await asyncio.sleep(.002)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(.01)
        leaked = [t for t in asyncio.all_tasks() - baseline if not t.done()]
        assert not leaked
    asyncio.run(scenario())


def test_full288_is_untiled_not_production_tile288():
    c = probe.geometry(288, 896, 4864, 64, "full")
    assert c.tiles() == [(0, 288)] and c.depth == 1
    assert len(probe.descriptors(c)) == 36


@pytest.mark.parametrize("rows,hidden,intermediate,tile", [(17, 8, 13, 7),
                                                         (150, 896, 4864, 64),
                                                         (288, 896, 4864, 64)])
def test_installed_native_public_geometry_eligibility_no_tensor_loading(rows, hidden, intermediate, tile):
    pytest.importorskip("pllm")
    bound = probe.compiler_binding(probe.geometry(rows, hidden, intermediate, tile, "conveyor"))
    region, = bound["contract"]["regions"]
    assert (region["hidden_width"], region["intermediate_width"]) == (hidden, intermediate)
    assert len(region["operation_ids"]) == 5


def test_native_eligibility_digest_is_bound_into_frame_context():
    pytest.importorskip("pllm")
    bound = probe.compiler_binding(probe.geometry(17, 8, 13, 7, "conveyor"))
    r = asyncio.run(probe.run_case(compiler_contract_sha256=bound["sha256"]))
    assert r["exact_bits"] and r["compiler_contract_sha256"] == bound["sha256"]


def test_tiny_shared100_40_pacer_counts_every_directed_envelope_and_body():
    pytest.importorskip("pllm")
    r = asyncio.run(probe.run_case(paced=True))
    assert r["success"] and r["wire_reconciled"]
    assert r["pacer"]["total_stream_bytes"] == r["ledger"]["client_tx"] + r["ledger"]["server_tx"]
    assert r["pacer"]["directed_stream_bytes"] == {
        "client->provider": r["ledger"]["client_tx"], "provider->client": r["ledger"]["server_tx"]}


@pytest.mark.parametrize("sequence", [1, 2, 5])
def test_executing_kernel_exception_never_escapes_tcp_callback_and_retires_every_owner(sequence):
    async def scenario():
        loop = asyncio.get_running_loop()
        callbacks = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: callbacks.append(context))
        try:
            r = await probe.run_case(fault="kernel", kernel_fail_sequence=sequence)
            await asyncio.sleep(0)
            assert not callbacks
            assert any("executing kernel failure" in e for e in r["errors"])
            assert r["provider_claimed_rows"] > 0 and r["provider_claimed_rows"] + r["provider_burned_rows"] == 34
            assert r["terminal_resources"] == {
                "provider_handlers": 0, "provider_writers": 0, "native_jobs": 0,
                "tile_slots": 0, "queued_requests": 0,
                "client_lease_closed": True, "provider_lease_closed": True}
        finally:
            loop.set_exception_handler(previous)
    asyncio.run(scenario())


@dataclass
class FakeHost:
    available: int = 4 << 30
    reserve: int = 1 << 30
    swap_used: int = 64 << 20


class FakeGuard:
    error = None
    minimum_available_bytes = 4 << 30
    maximum_swap_growth_bytes = 123

    def check(self):
        pass


def tiny_driver_dependencies(output, mutate=None, fail_second=False):
    """Real tiny TCP workers, mocked host/build/launcher. Never Qwen computation."""
    pytest.importorskip("pllm")
    hashes, build = probe.source_hashes(), {"test_build": "tiny-only-not-a-measurement"}
    payloads = [{"rows": 17, "hidden": 8, "intermediate": 13, "tile": 7, "mode": mode}
                for mode in ("serial", "conveyor")]
    calls = []

    @contextmanager
    def workers(script, *, memory_bytes, timeout):
        def run(payload):
            checkpoint = json.loads(output.read_text())
            assert checkpoint["status"] == "incomplete" and checkpoint["build"] == build
            assert checkpoint["attempts"][-1]["status"] == "running"
            assert len(checkpoint["cases"]) == len(calls)
            calls.append(payload)
            if fail_second and len(calls) == 2:
                raise RuntimeError("later-arm failure " + "x" * 5000)
            bound = probe.compiler_binding(probe.geometry(**payload))
            case = asyncio.run(probe.run_case(**payload, paced=True,
                                             compiler_contract_sha256=bound["sha256"]))
            case.update(source_sha256=hashes, build=build, compiler_eligibility=bound)
            if mutate:
                mutate(case)
            return case
        run.build = build
        yield run, FakeGuard(), FakeHost()
    return payloads, hashes, {"workers": workers, "sample": FakeHost, "capture_build": lambda: build}, calls


def test_later_arm_failure_preserves_accepted_checkpoint_identity_and_guard(tmp_path):
    output = tmp_path / "partial.json"
    payloads, hashes, deps, calls = tiny_driver_dependencies(output, fail_second=True)
    result = probe.measure_cohort(output, payloads, hashes, **deps)
    saved = json.loads(output.read_text())
    assert saved == result and saved["status"] == "failed"
    assert len(calls) == 2 and len(saved["cases"]) == 1
    assert saved["cases"][0]["acceptance_status"] == "accepted"
    assert [a["status"] for a in saved["attempts"]] == ["accepted", "failed"]
    assert saved["failure"]["accepted_arms"] == 1 and len(saved["failure"]["message"]) == 4096
    assert saved["guard_observations"][-1]["maximum_swap_growth_bytes"] == 123
    assert saved["guard_observations"][-1]["current_host"]["swap_used"] == 64 << 20
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        probe.measure_cohort(output, payloads, hashes, **deps)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("field", ["source", "build", "hash", "geometry", "wire", "frames", "compiler", "pacer", "order"])
def test_acceptance_rejects_changed_identity_hash_case_or_wire_before_status(tmp_path, field):
    def mutate(case):
        if field == "source":
            case["source_sha256"] = {}
        elif field == "build":
            case["build"] = {}
        elif field == "hash":
            case["output_sha256"] = "0" * 64
        elif field == "geometry":
            case["geometry"]["rows"] += 1
        elif field == "wire":
            case["wire_parts"]["server_rx"]["body"] += 1
        elif field == "frames":
            case["state"]["request_frames"] -= 1
        elif field == "compiler":
            case["wire_binding_sha256"] = "0" * 64
        elif field == "pacer":
            case["pacer"]["conditions"]["upload_mbps"] = 100
        elif field == "order":
            next(e for e in case["events"] if e["event"] == "client_send_start")["row"] += 1
    output = tmp_path / "rejected.json"
    payloads, hashes, deps, _ = tiny_driver_dependencies(output, mutate)
    result = probe.measure_cohort(output, payloads[:1], hashes, **deps)
    assert result["status"] == "failed" and not result["cases"]
    assert result["attempts"][0]["status"] == "rejected"
    assert "case validation" in result["failure"]["message"]


def test_successful_tiny_driver_atomically_accepts_controls_and_records_guard(tmp_path):
    output = tmp_path / "complete.json"
    payloads, hashes, deps, _ = tiny_driver_dependencies(output)
    result = probe.measure_cohort(output, payloads, hashes, **deps)
    assert result["status"] == "complete" and result["controls_equal_outputs"]
    assert len(result["cases"]) == 2 and result["guard_observations"]
    assert json.loads(output.read_text()) == result


def test_foreign_reservation_never_overwritten(tmp_path):
    output = tmp_path / "foreign.json"
    probe.reserve_output(output, {"attempt_id": "original", "status": "incomplete"})
    before = output.read_bytes()
    with pytest.raises(RuntimeError, match="ownership changed"):
        probe.checkpoint_output(output, {"attempt_id": "foreign", "status": "complete"})
    assert output.read_bytes() == before


def test_admission_failure_is_durable_before_any_worker(tmp_path):
    @contextmanager
    def refused(*args, **kwargs):
        raise RuntimeError("test admission refusal")
        yield  # pragma: no cover
    output = tmp_path / "admission.json"
    result = probe.measure_cohort(output, [], probe.source_hashes(), workers=refused,
                                 sample=FakeHost, capture_build=lambda: {"test": "before-work"})
    assert result["status"] == "failed" and result["failure"]["phase"] == "admission"
    assert result["source_sha256"] and result["build"] == {"test": "before-work"}
    assert json.loads(output.read_text()) == result


def test_overlap_field_matches_only_named_adjacent_gate_write_intersections():
    r = asyncio.run(probe.run_case())
    starts = {(e["event"], e["sequence"]): e["seconds"] for e in r["events"]}
    c = probe.geometry(17, 8, 13, 7, "conveyor")
    frames = probe.descriptors(c)
    expected = 0.
    for i, a in enumerate(frames):
        for j, b in enumerate(frames):
            if a[0] == b[0] == "G" and a[1] == b[1] + 1:
                left = max(starts["client_send_start", i], starts["provider_reply_start", j])
                right = min(starts["client_send_end", i], starts["provider_reply_end", j])
                expected += max(0., right - left)
    assert r["next_gate_upload_vs_previous_gate_reply_write_overlap_seconds"] == expected
    assert "cross_stage_upload_reply_write_overlap_seconds" not in r
    assert "zero excludes neither" in r["overlap_scope"]

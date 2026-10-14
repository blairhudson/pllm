"""Independent numeric, dependency, resource and retirement probe oracles."""
import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_mlp_conveyor.py"
SPEC = importlib.util.spec_from_file_location("mlp_conveyor_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("rows", [1, 7, 17, 150, 288])
@pytest.mark.parametrize("tile", [7, 64])
@pytest.mark.parametrize("depth", [1, 2])
def test_scalar_projection_and_complete_float32_bits(rows, tile, depth):
    result = probe.numeric_probe(rows, tile, depth)
    assert result["exact_bits"]
    assert result["claimed"] == 2 * rows
    assert result["burned"] == result["terminal_product_owners"] == 0
    assert result["weakref_retirement"]
    assert result["peak_product_owners"] <= depth


@pytest.mark.parametrize("cancel", [1, 2, 3, 5])
def test_cancellation_burns_all_reserved_rows(cancel):
    result = probe.numeric_probe(17, 7, 2, cancel_after=cancel)
    assert result["canceled"] and result["exact_bits"] is None
    assert result["claimed"] + result["burned"] == 34
    assert result["replay_rejected"] and result["terminal_product_owners"] == 0


def test_live_replay_and_out_of_order_rejected():
    lease = probe.Lease(17)
    lease.claim("G", 0, 7, 0)
    for claim in [("G", 0, 7, 1), ("D", 7, 14, 1), ("D", 0, 7, 0)]:
        with pytest.raises(ValueError):
            lease.claim(*claim)
    lease.claim("D", 0, 7, 1)
    assert lease.close() == 20


@pytest.mark.parametrize("rows", [1, 17, 150, 288])
@pytest.mark.parametrize("depth", [1, 2])
def test_independent_resource_dependency_and_byte_oracle(rows, depth):
    c = probe.Contract(rows, tile=64, depth=depth)
    r = probe.simulate(c, local_stall_seconds=0.05)
    previous_upload = previous_reply = 0
    by_stage = {"G": [], "D": []}
    gate_done = {}
    down_done = {}
    for event in r["events"]:
        stage, tile = event["stage"], event["tile"]
        assert event["upload_start"] >= previous_upload
        assert event["compute_start"] >= max(previous_reply, event["upload_end"])
        assert event["reply_end"] >= event["compute_end"]
        if stage == "D":
            start, end = c.tiles()[tile]
            assert event["upload_start"] >= gate_done[tile] + (end - start) * 0.00002 + 0.05 - 1e-12
            down_done[tile] = event["reply_end"]
        else:
            if tile >= depth:
                assert event["upload_start"] >= down_done[tile - depth]
            gate_done[tile] = event["reply_end"]
        by_stage[stage] += list(range(event["row"], event["row"] + event["rows"]))
        previous_upload, previous_reply = event["upload_end"], event["reply_end"]
    assert by_stage == {"G": list(range(rows)), "D": list(range(rows))}
    # Closed-form conservation independent of frame_bytes/simulator.
    assert r["tx_bytes"] == rows * (c.hidden + c.intermediate) * 4 + r["frames"] * 1024
    assert r["rx_bytes"] == rows * (2 * c.intermediate + c.hidden) * 4 + r["frames"] * 1024
    assert r["seconds"] >= max(r["tx_bytes"], r["rx_bytes"]) * 8 / 40e6
    assert r["peak_tile_owners"] <= depth


def test_public_schedule_backpressure_and_conservative_owners():
    c = probe.Contract(288)
    normal = probe.simulate(c)
    stalled = probe.simulate(c, local_stall_seconds=5, upload_mbps=1, download_mbps=1)
    assert [(e["stage"], e["row"], e["rows"]) for e in normal["events"]] == [
        (e["stage"], e["row"], e["rows"]) for e in stalled["events"]]
    assert stalled["seconds"] > normal["seconds"]
    assert stalled["peak_tile_owners"] <= 2
    ledger = probe.ownership(c)
    assert ledger["whole_client_bytes"] is ledger["low_256_mib_fits"] is None
    assert ledger["priced_region_bytes"] == 2 * sum(ledger["per_slot"].values()) + sum(ledger["shared"].values())
    assert ledger["provider_inflight_bytes"] == sum(ledger["provider_inflight"].values())
    assert ledger["provider_original_weights_inventory_runtime_bytes"] is None
    assert not probe.ownership(c, 256 << 20)["low_256_mib_fits"]


def test_no_overlap_claim_on_zero_local_time_and_reject_private_geometry():
    serial = probe.simulate(probe.Contract(150, depth=1), local_seconds_per_row=0)
    pipeline = probe.simulate(probe.Contract(150), local_seconds_per_row=0)
    # Cross-stage sender may hide upload behind an earlier reply, but cannot
    # eliminate shared-link/provider service or change arithmetic traffic.
    assert serial["tx_bytes"] == pipeline["tx_bytes"]
    assert serial["rx_bytes"] == pipeline["rx_bytes"]
    for kwargs in ({"rows": True}, {"rows": 385}, {"rows": 17, "depth": 3},
                   {"rows": 17, "input_bits": 24}):
        with pytest.raises(ValueError):
            probe.Contract(**kwargs)
    with pytest.raises(ValueError):
        probe.simulate(probe.Contract(17), download_mbps=100)

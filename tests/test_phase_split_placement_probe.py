"""Small independent cost/numeric/lifecycle controls; no source values."""
import importlib.util
from pathlib import Path
import random
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_phase_split_placement.py"
SPEC = importlib.util.spec_from_file_location("phase_split_placement_probe", PATH)
p = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p
SPEC.loader.exec_module(p)


def stage(name, macs, *, terminal=False, traffic=100, weight=10):
    return {"id": name, "in": 2, "out": 3, "macs": macs, "terminal": terminal,
            "weight_bytes": weight, "scale_bytes": 12, "remote_row_bytes": traffic,
            "correction_row_bytes": traffic // 2, "identity": {"source": "toy", "group": name}}


def test_dp_matches_independent_brute_force():
    rng = random.Random(921)
    for _ in range(35):
        items = [stage(str(i), rng.randrange(1, 20), terminal=bool(i % 2),
                       traffic=rng.randrange(30, 400), weight=rng.randrange(5, 900)) for i in range(5)]
        items.append(stage("remote-body", 250, traffic=40))
        events = [p.Event("initial", 15, 3), p.Event("continuation", 4, 2, True)]
        for fixed in (False, True):
            kwargs = dict(extra_memory_limit=1500, fixed=fixed)
            got = p.choose(items, events, [str(i) for i in range(5)], p.Owners(), **kwargs)
            want = p.brute_force(items, events, [str(i) for i in range(5)], p.Owners(), **kwargs)
            assert got["selection"] == want["selection"]
            assert got["net_arithmetic_byte_saving"] == want["net_arithmetic_byte_saving"]


def test_split_can_choose_different_groups_and_union_cold_cost():
    items = [stage("terminal", 19, terminal=True, traffic=500),
             stage("dense", 10, traffic=300), stage("remote", 171, traffic=10)]
    events = [p.Event("initial", 100, 20)]
    kwargs = dict(extra_memory_limit=200)
    split = p.choose(items, events, ["terminal", "dense"], p.Owners(), **kwargs)
    fixed = p.choose(items, events, ["terminal", "dense"], p.Owners(), fixed=True, **kwargs)
    assert split["net_arithmetic_byte_saving"] > fixed["net_arithmetic_byte_saving"]
    assert split["selection"] == {"terminal": 3, "dense": 1}
    report = p.replay(items, events, split["selection"], p.Owners(), **kwargs)
    assert report["client_cold_raw_weight_scale_payload_bytes"] == 44
    # Dense group remains provider-owned in decode: no delete/reload fiction.
    assert report["provider_union_weight_scale_payload_bytes_per_provider"] == 44
    assert report["events"][0]["phases"]["decode"]["fresh_correction_rows"] == 40
    assert report["complete_all_link_body_bytes"] is None
    assert report["events"][0]["transition_cpu_seconds"] is None
    assert not report["compiled_phase_split_executor_implemented"]


def test_continuation_disables_terminal_pruning_and_independent_caps():
    items = [stage("terminal", 30, terminal=True), stage("remote", 170)]
    initial = [p.Event("initial", 100, 1)]
    # Initial-only permits this group's prefill, but decode never does.
    p.replay(items, initial, {"terminal": 1}, p.Owners(), extra_memory_limit=100)
    with pytest.raises(ValueError, match="per-phase"):
        p.replay(items, [*initial, p.Event("suffix", 10, 1, True)], {"terminal": 1},
                 p.Owners(), extra_memory_limit=100)
    with pytest.raises(ValueError, match="per-phase"):
        p.replay(items, initial, {"terminal": 2}, p.Owners(), extra_memory_limit=100)
    hit = p.replay(items, [p.Event("hit", 0, 3)], {}, p.Owners(), extra_memory_limit=0)
    assert hit["events"][0]["phases"]["prefill"]["fresh_correction_rows"] == 0


def test_union_snapshot_paged_and_unknown_costs():
    item = stage("chosen", 1, weight=100)
    items = [item, stage("remote", 99)]
    events = [p.Event("cold", 3, 2)]
    a = p.replay(items, events, {"chosen": 1}, p.Owners(), extra_memory_limit=300)
    b = p.replay(items, events, {"chosen": 3}, p.Owners(), extra_memory_limit=300)
    assert a["union_client_resident_owner_bytes"] == b["union_client_resident_owner_bytes"] == 224
    assert b["provider_union_weight_scale_payload_bytes_per_provider"] == 22
    assert b["whole_client_peak_owner_floor_bytes"] is None
    assert b["events"][0]["phases"]["decode"]["client_cpu_seconds"] is None
    with pytest.raises(ValueError, match="windows unknown"):
        p.choose(items, events, ["chosen"], p.Owners(storage="paged"), extra_memory_limit=10000)
    paged = p.replay(items, events, {"chosen": 3},
                     p.Owners(storage="paged", paged_window_bytes=4096), extra_memory_limit=10000)
    assert paged["client_paged_shared_file_bytes"] == 100
    assert paged["client_native_i8_snapshot_bytes"] == 0
    assert paged["union_client_resident_owner_bytes"] > 4096
    with pytest.raises(ValueError, match="union-owner"):
        p.replay(items, events, {"chosen": 3}, p.Owners(), extra_memory_limit=223)


def test_bounds_and_invalid_ownership():
    items = [stage("a", 1), stage("b", 99)]
    events = [p.Event("cold", 1, 1)]
    with pytest.raises(ValueError, match="state bound"):
        p.choose(items, events, ["a"], p.Owners(), extra_memory_limit=100, max_states=1)
    with pytest.raises(ValueError, match="invalid bounded"):
        p.choose(items, events, ["absent"], p.Owners(), extra_memory_limit=100)
    with pytest.raises(ValueError, match="ownership mask"):
        p.replay(items, events, {"a": True}, p.Owners(), extra_memory_limit=100)


def test_synthetic_exact_integer_switch_and_fresh_material():
    """Independent clear dot vs fresh masked correction; not native parity.

    Preserve ordered group concatenation, signed modular reconstruction, and
    identical float scaling/local bias AFTER exact integer result. Tiny values
    bound sums below signed ring limit. No real weights or source values.
    """
    rng = random.Random(87)
    groups = {"qkv": [[rng.randrange(-8, 8) for _ in range(5)] for _ in range(6)],
              "down": [[rng.randrange(-8, 8) for _ in range(5)] for _ in range(3)]}
    masks = {"qkv": 2, "down": 1}
    used = set()
    outputs = []
    for event, rows, decode in (("initial", 4, 2), ("suffix", 3, 2), ("hit", 0, 2)):
        for phase, count, bit in (("prefill", rows, 1), ("decode", decode, 2)):
            for row in range(count):
                x = [rng.randrange(-12, 12) for _ in range(5)]
                for name, weights in groups.items():
                    expected = [sum(a * b for a, b in zip(x, w)) for w in weights]
                    if masks[name] & bit:
                        actual = expected[:]
                    else:
                        ticket = (event, phase, name, row)
                        assert ticket not in used
                        used.add(ticket)
                        mask = [rng.randrange(1 << 16) for _ in x]
                        modulus = 1 << 24
                        actual = []
                        for w in weights:
                            returned = sum(((a + m) % modulus) * b for a, m, b in zip(x, mask, w)) % modulus
                            correction = sum(m * b for m, b in zip(mask, w)) % modulus
                            value = (returned - correction) % modulus
                            actual.append(value - modulus if value >= modulus // 2 else value)
                    assert actual == expected
                    assert [v * 0.125 + 0.25 for v in actual] == [v * 0.125 + 0.25 for v in expected]
                    outputs.append(actual)
    assert outputs and len(used) == 13  # Seven remote prefill + six remote decode rows.


def test_installed_public_schedule_geometry_and_pruning():
    items, context = p.load_public_problem()
    assert len(items) == 96
    assert any(i["terminal"] for i in items)
    assert context["retained_measured_control"]["scope"].startswith("context only")
    assert all(i["identity"]["weight_sha256"] for i in items)

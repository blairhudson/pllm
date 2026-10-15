"""Frozen v3 replay only: no native builds, models or timing workers."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_boundary_head_dotprod.py"
PATH = ROOT / "docs/evidence/boundary-head-dotprod-v3-2026-10-06.json"
SUMMARY = PATH.with_suffix(".summary.json")
SPEC = importlib.util.spec_from_file_location("byte_dot_summary", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
ARCHIVE = json.loads(PATH.read_bytes())


def alter(name, mutate):
    archive = copy.deepcopy(ARCHIVE)
    item = next(r for r in archive["files"] if r["name"] == name)
    if name == "results.jsonl":
        data = [json.loads(s) for s in item["utf8"].splitlines()]
        mutate(data)
        item["utf8"] = "".join(json.dumps(r) + "\n" for r in data)
    else:
        data = json.loads(item["utf8"])
        mutate(data)
        item["utf8"] = json.dumps(data)
    item["sha256"] = hashlib.sha256(item["utf8"].encode()).hexdigest()
    return archive


def test_exact_retained_replay_and_cli():
    expected = SUMMARY.read_text()
    assert probe.replay(PATH) == expected
    actual = subprocess.check_output([sys.executable, str(SCRIPT), str(PATH)], text=True)
    assert actual == expected
    result = json.loads(actual)
    assert result["workers"] == 6 and result["repeats_per_worker"] == 4
    assert result["maximum_process_rss_bytes"] == 281673728
    assert result["maximum_observed_swap_growth_bytes"] == 0
    assert result["whole_client_memory_tier"] is None
    medians = result["worker_medians"]
    assert medians["timings"]["candidate_allocating_including_packing"]["cpu_seconds"]["median"] == pytest.approx(.004228)
    assert medians["timings"]["baseline"]["cpu_seconds"]["median"] == pytest.approx(.00654875)
    assert medians["cpu_reduction_percent"] == pytest.approx(35.43806069860673)
    assert result["worker_four_call_totals"]["cpu_reduction_percent"] == pytest.approx(31.325278207257256)
    assert result["archive_sha256"] == hashlib.sha256(PATH.read_bytes()).hexdigest()
    assert "2026-10-15" in result["build_timestamp_utc"]


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop(),
    lambda r: r[0].update(schema="boundary-head-dotprod-v2"),
    lambda r: r[0].update(rows=97),
    lambda r: r[0].update(cols=897),
    lambda r: r[0].update(repeats=3),
    lambda r: r[0].update(exact=False),
    lambda r: r[0]["allocation_matched"].update(exact=False),
    lambda r: r[0]["samples"][0]["allocation_matched"].update(exact=False),
    lambda r: r[0]["samples"][0].update(exact=False),
    lambda r: r[0]["samples"].pop(),
    lambda r: r[0].update(order_offset=1),
    lambda r: r[0]["process_schedule"].update(index=1),
    lambda r: r[0]["samples"][0].update(order="baseline-first"),
    lambda r: r[0]["samples"][0].update(cohort_order="allocation-matched-first"),
    lambda r: r[0]["samples"][0]["allocation_matched"].update(order="baseline-first"),
    lambda r: r[0]["provenance"].update(binary_sha256="f" * 64),
    lambda r: r[0]["provenance"]["source_sha256"].update({"Cargo.lock": "f" * 64}),
    lambda r: r[0].update(compiled_source_sha256="f" * 64),
    lambda r: r[0].update(compiled_kernels_sha256="f" * 64),
    lambda r: r[0]["build_provenance"].update(rustc_verbose="different compiler"),
    lambda r: r[0]["provenance"]["native_command"].__setitem__(4, "2"),
    lambda r: r[0].update(selected="scalar-balanced-limbs"),
    lambda r: r[0]["fresh_process"].update(peak_rss_bytes=1 << 30),
    lambda r: r[0]["fresh_process"].update(peak_rss_bytes=0),
    lambda r: r[0]["allocation_matched"]["baseline"].update(cpu_seconds=.1),
    lambda r: r[0]["samples"][0]["allocation_matched"]["packing"].update(cpu_seconds=float("nan")),
    lambda r: r[0]["samples"][0]["allocation_matched"]["baseline"].update(cpu_seconds=float("inf")),
    lambda r: r[0]["samples"][0]["allocation_matched"]["baseline"].update(cpu_seconds=-1),
    lambda r: r[0]["samples"][0]["allocation_matched"]["baseline"].update(elapsed_seconds=61),
    lambda r: r[0]["samples"][0]["allocation_matched"]["candidate_allocating_including_packing"].update(cpu_seconds=.2),
    lambda r: r[0]["samples"][0]["candidate_including_packing"].update(cpu_seconds=.2),
])
def test_rejects_worker_tampering_even_with_rehashed_raw(mutate):
    with pytest.raises(ValueError):
        probe.summarize(alter("results.jsonl", mutate))


@pytest.mark.parametrize("mutate", [
    lambda g: g.update(exit_code=1),
    lambda g: g.update(guard_error="failed"),
    lambda g: g.update(maximum_swap_growth_bytes=4096),
    lambda g: g.update(minimum_available_bytes=1),
    lambda g: g["host"].update(available=1),
    lambda g: g["command"].__setitem__(-3, "5"),
])
def test_rejects_guard_tampering_even_with_rehashed_raw(mutate):
    with pytest.raises(ValueError):
        probe.summarize(alter("guard.json", mutate))


def test_rejects_raw_corruption_missing_duplicate_and_frozen_lock_changes():
    for kind in ("corrupt", "missing", "duplicate", "frozen"):
        archive = copy.deepcopy(ARCHIVE)
        if kind == "corrupt":
            archive["files"][0]["utf8"] += " "
        elif kind == "missing":
            archive["files"].pop()
        elif kind == "duplicate":
            archive["files"].append(archive["files"][0])
        else:
            archive["frozen_source_sha256"]["Cargo.lock"] = "f" * 64
        with pytest.raises(ValueError):
            probe.summarize(archive)


def test_archive_size_bound(tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * ((2 << 20) + 1))
    with pytest.raises(ValueError, match="exceeds"):
        probe.replay(path)

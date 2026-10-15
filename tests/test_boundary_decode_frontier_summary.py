"""Retained evidence rejects missing, corrupted, unmatched or failed observations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "boundary_decode_summary", ROOT / "scripts/summarize_boundary_decode_frontier.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
ARCHIVE = json.loads((ROOT / "docs/evidence/boundary-decode-frontier-2026-10-15.json").read_bytes())


def alter(name, mutate):
    archive = copy.deepcopy(ARCHIVE)
    item = next(row for row in archive["files"] if row["name"] == name)
    data = json.loads(item["utf8"])
    mutate(data)
    item["utf8"] = json.dumps(data)
    item["sha256"] = hashlib.sha256(item["utf8"].encode()).hexdigest()
    return archive


def test_retained_screen_reconciles_without_claiming_client_tier():
    result = probe.summarize(ARCHIVE)
    assert result["successful_monitored_jobs"] == 13
    assert result["whole_client_memory_tier"] is None
    assert result["duplex"]["serial"]["application_stream_bytes"] == 1_223_276
    assert result["duplex"]["parallel"]["application_stream_bytes"] == 1_223_276
    assert result["duplex"]["striped"]["application_stream_bytes"] == 1_229_176
    assert result["uniform_residues"]["costs"]["raw_payload_minus_new_frames_and_cold_delivery_bytes"] == 2_122_510
    assert result["runahead_equal_service_gain_ms"] == 0


@pytest.mark.parametrize("name,mutate", [
    ("duplex-0.json", lambda r: r["runs"].pop()),
    ("duplex-0.json", lambda r: r["runs"][0].update(full_logits_sha256="changed")),
    ("duplex-0.json", lambda r: r["runs"][0]["wan"].update(total_stream_bytes=1)),
    ("duplex-0.json", lambda r: r["provenance"].update(native_extension_sha256="changed")),
    ("duplex-0.json", lambda r: r["runs"][0]["owners"][0].update(retired=False)),
    ("manifest.json", lambda r: r[0].update(maximum_swap_growth_bytes=4096)),
    ("manifest.json", lambda r: r[0].update(status=1)),
    ("residues.json", lambda r: r["uniform_geometry_profile"]["certificate"].update(bias=True)),
    ("fusion-4096-0.json", lambda r: r["native_fused_reference"].update(correct=False)),
])
def test_rejects_ineligible_evidence_even_with_recomputed_container_hash(name, mutate):
    with pytest.raises(ValueError):
        probe.summarize(alter(name, mutate))


def test_rejects_raw_corruption():
    archive = copy.deepcopy(ARCHIVE)
    archive["files"][0]["utf8"] += " "
    with pytest.raises(ValueError, match="digest"):
        probe.summarize(archive)

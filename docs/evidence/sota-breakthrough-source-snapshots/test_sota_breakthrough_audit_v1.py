"""Archive corruption gates only: no target tensors, training or timing cohorts."""
import json
from pathlib import Path
import shutil

import pytest

from scripts import audit_sota_breakthrough as audit


@pytest.fixture
def archive(tmp_path):
    folder = tmp_path / "archive"
    folder.mkdir()
    for name in audit.FILES.values():
        shutil.copyfile(audit.ROOT / "docs/evidence" / name, folder / name)
    return folder


def read(folder, key):
    return json.loads((folder / audit.FILES[key]).read_text())


def write(folder, key, value):
    (folder / audit.FILES[key]).write_text(json.dumps(value) + "\n")


def rebind_attempt(folder):
    """Rehash edited report to test semantic checks, not just outer checksums."""
    attempt = read(folder, "attempt")
    attempt["report_sha256"] = audit.file_hash(folder / audit.FILES["pilot"])
    write(folder, "attempt", attempt)


def test_complete_bounded_replay_native():
    result = audit.audit()
    assert result["status"] == "verified_bounded_evidence"
    assert result["head"]["native_replayed"]
    assert result["head"]["checked_integer_outputs"] == 5 * 151936
    assert result["pilot"]["refit_exact"]
    assert result["pilot"]["splits"]["dev"]["screen_pass"]
    assert not result["pilot"]["splits"]["heldout"]["screen_pass"]
    assert result["pilot"]["splits"]["heldout"]["arithmetic_ratio"] > 1
    assert len(result["conveyor"]["cases"]) == 6
    assert result["correlation"]["normalized_reproduction"]
    assert not result["serving_qualified"] and not result["memory_tier_qualified"]


def test_skip_native_is_deterministic_and_never_imports_native_validator(monkeypatch):
    def forbidden(*_):
        pytest.fail("skip-native must not import installed native/source modules")
    monkeypatch.setattr(audit.importlib, "import_module", forbidden)
    first = audit.audit(native=False)
    assert first == audit.audit(native=False)
    assert first["head"]["native_replayed"] is False
    assert first["head"]["checked_integer_outputs"] == 0
    assert "after launch" in first["pilot"]["dependency_caveat"]


def test_altered_pilot_summary_rejected_after_rebinding(archive):
    report = read(archive, "pilot")
    report["summary"]["heldout"]["gate"]["screen_pass"] = True
    write(archive, "pilot", report)
    rebind_attempt(archive)
    with pytest.raises(ValueError, match="summary reconciliation"):
        audit.audit(archive, native=False)


def test_fallback_cannot_erase_compressed_attempt_cost(archive):
    report = read(archive, "pilot")
    row = next(r for r in report["evaluation"] if r["results"]["learned_then_full"]["fallback"])
    final = row["results"]["learned_then_full"]
    final["projected_arithmetic_bytes"] = final["retry"]["projected_arithmetic_bytes"]
    write(archive, "pilot", report)
    rebind_attempt(archive)
    with pytest.raises(ValueError, match="fallback complete cost"):
        audit.audit(archive, native=False)


def test_report_artifact_mismatch_rejected(archive):
    artifact = read(archive, "helper")
    artifact["weights"][1] += 1
    write(archive, "helper", artifact)
    with pytest.raises(ValueError, match="artifact (file mismatch|measured length)"):
        audit.audit(archive, native=False)


def test_resigned_helper_still_requires_original_teacher_refit(archive):
    artifact = read(archive, "helper")
    artifact["weights"][1] += 1
    artifact["artifact_digest"] = audit.digest({k: v for k, v in artifact.items() if k != "artifact_digest"})
    write(archive, "helper", artifact)
    report = read(archive, "pilot")
    path = archive / audit.FILES["helper"]
    report["artifact_sha256"] = audit.file_hash(path)
    report["artifact_bytes"] = report["cold_additional_client_artifact_bytes"] = path.stat().st_size
    write(archive, "pilot", report)
    rebind_attempt(archive)
    with pytest.raises(ValueError, match="original refit/artifact mismatch"):
        audit.audit(archive, native=False)


def test_missing_conveyor_case_rejected(archive):
    report = read(archive, "conveyor")
    report["cases"].pop()
    write(archive, "conveyor", report)
    with pytest.raises(ValueError, match="six-case inventory"):
        audit.audit(archive, native=False)


def test_matching_but_false_conveyor_source_locks_rejected(archive):
    for key in ("conveyor", "conveyor_lock"):
        report = read(archive, key)
        report["source_sha256"]["scripts/probe_mlp_conveyor_transport.py"] = "0" * 64
        write(archive, key, report)
    with pytest.raises(ValueError, match="archived source binding"):
        audit.audit(archive, native=False)


@pytest.mark.parametrize("field", ["owner", "wire", "overlap", "guard"])
def test_conveyor_counter_and_owner_corruption_rejected(archive, field):
    report = read(archive, "conveyor")
    if field == "owner":
        report["cases"][0]["terminal_resources"]["native_jobs"] = 1
    elif field == "wire":
        report["cases"][0]["ledger"]["client_tx"] += 1
    elif field == "overlap":
        report["cases"][1]["next_gate_upload_vs_previous_gate_reply_write_overlap_seconds"] = 1
    else:
        report["guard_observations"][-1]["maximum_swap_growth_bytes"] = 4096
    write(archive, "conveyor", report)
    with pytest.raises(ValueError):
        audit.audit(archive, native=False)


def test_head_cost_corruption_rejected(archive):
    report = read(archive, "head")
    report["costs"]["net_saved_bytes_at_head_calls_estimate"] += 1
    write(archive, "head", report)
    native = read(archive, "native")
    native["scan_sha256"] = audit.file_hash(archive / audit.FILES["head"])
    write(archive, "native", native)
    with pytest.raises(ValueError, match="head complete cost ledger"):
        audit.audit(archive, native=False)


def test_correlation_corrected_prg_bill_cannot_drift(archive):
    report = read(archive, "correlation")
    report["ledgers"]["stage_analytic_only"]["zero_tape"]["preparation_PRG_output_bytes"] += 4
    write(archive, "correlation", report)
    with pytest.raises(ValueError, match="correlation normalized reproduction"):
        audit.audit(archive, native=False)


def test_untrusted_snapshot_never_executes(tmp_path, monkeypatch):
    (tmp_path / "checkpoint_tiny_model.py").write_text("raise AssertionError('must not execute')")
    monkeypatch.setattr(audit, "SNAPSHOTS", tmp_path)
    with pytest.raises(ValueError, match="snapshot source lock"):
        audit.snapshot("checkpoint_tiny_model.py", execute=True)


def test_cli_exclusive_output_and_failure_placeholder(tmp_path, monkeypatch):
    path = tmp_path / "audit.json"
    path.write_text("existing")
    with pytest.raises(FileExistsError):
        audit.main(["--output", str(path)])
    assert path.read_text() == "existing"

    def fail(*_args, **_kwargs):
        raise ValueError("injected bounded rejection")
    monkeypatch.setattr(audit, "audit", fail)
    failed = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="bounded rejection"):
        audit.main(["--output", str(failed)])
    assert json.loads(failed.read_text()) == {"status": "incomplete"}


def test_json_bound_rejects_oversize(tmp_path):
    path = tmp_path / "oversize.json"
    path.write_bytes(b" " * 100)
    with pytest.raises(ValueError, match="oversize audit input"):
        audit.bounded(path, limit=99)

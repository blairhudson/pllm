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
    shutil.copyfile(audit.ROOT / "docs/evidence" / audit.PILOT_LOG, folder / audit.PILOT_LOG)
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
    (folder / audit.PILOT_LOG).write_text(json.dumps(read(folder, "pilot")["summary"]))


def portable(folder=audit.ROOT / "docs/evidence"):
    return audit.audit(folder, native=False, primary_sources=False)


def test_complete_portable_bounded_replay():
    result = portable()
    assert result["status"] == "verified_bounded_evidence"
    assert not result["head"]["native_replayed"]
    assert not result["correlation"]["primary_sources_reverified"]
    assert result["pilot"]["refit_within_tolerance"]
    assert result["pilot"]["splits"]["dev"]["screen_pass"]
    assert not result["pilot"]["splits"]["heldout"]["screen_pass"]
    assert result["pilot"]["splits"]["heldout"]["arithmetic_ratio"] > 1
    assert len(result["conveyor"]["cases"]) == 6
    assert result["correlation"]["normalized_reproduction"]
    assert not result["serving_qualified"] and not result["memory_tier_qualified"]


def test_original_build_native_replay_when_dependencies_match():
    head = read(audit.ROOT / "docs/evidence", "head")
    try:
        audit.original_native_modules(head)
    except (ImportError, OSError, ValueError) as error:
        pytest.skip(f"requires original installed native/NumPy/SDK lock: {error}")
    result = audit.audit(native=True, primary_sources=False)
    assert result["head"]["native_replayed"]
    assert result["head"]["checked_integer_outputs"] == 5 * 151936
    assert result["pilot"]["refit_exact"]


def test_skip_native_is_deterministic_and_never_imports_native_validator(monkeypatch):
    def forbidden(*_):
        pytest.fail("skip-native must not import installed native/source modules")
    monkeypatch.setattr(audit.importlib, "import_module", forbidden)
    first = portable()
    assert first == portable()
    assert first["head"]["native_replayed"] is False
    assert first["head"]["checked_integer_outputs"] == 0
    assert "after launch" in first["pilot"]["dependency_caveat"]


def test_altered_pilot_summary_rejected_after_rebinding(archive):
    report = read(archive, "pilot")
    report["summary"]["heldout"]["gate"]["screen_pass"] = True
    write(archive, "pilot", report)
    rebind_attempt(archive)
    with pytest.raises(ValueError, match="summary reconciliation"):
        portable(archive)


def test_fallback_cannot_erase_compressed_attempt_cost(archive):
    report = read(archive, "pilot")
    row = next(r for r in report["evaluation"] if r["results"]["learned_then_full"]["fallback"])
    final = row["results"]["learned_then_full"]
    final["projected_arithmetic_bytes"] = final["retry"]["projected_arithmetic_bytes"]
    write(archive, "pilot", report)
    rebind_attempt(archive)
    with pytest.raises(ValueError, match="fallback complete cost"):
        portable(archive)


def test_report_artifact_mismatch_rejected(archive):
    artifact = read(archive, "helper")
    artifact["weights"][1] += 1
    write(archive, "helper", artifact)
    with pytest.raises(ValueError, match="artifact (file mismatch|measured length)"):
        portable(archive)


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
        portable(archive)


def test_missing_conveyor_case_rejected(archive):
    report = read(archive, "conveyor")
    report["cases"].pop()
    write(archive, "conveyor", report)
    with pytest.raises(ValueError, match="six-case inventory"):
        portable(archive)


def test_matching_but_false_conveyor_source_locks_rejected(archive):
    for key in ("conveyor", "conveyor_lock"):
        report = read(archive, key)
        report["source_sha256"]["scripts/probe_mlp_conveyor_transport.py"] = "0" * 64
        write(archive, key, report)
    with pytest.raises(ValueError, match="archived source binding"):
        portable(archive)


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
        portable(archive)


def test_head_cost_corruption_rejected(archive):
    report = read(archive, "head")
    report["costs"]["net_saved_bytes_at_head_calls_estimate"] += 1
    write(archive, "head", report)
    native = read(archive, "native")
    native["scan_sha256"] = audit.file_hash(archive / audit.FILES["head"])
    write(archive, "native", native)
    with pytest.raises(ValueError, match="head complete cost ledger"):
        portable(archive)


def test_correlation_corrected_prg_bill_cannot_drift(archive):
    report = read(archive, "correlation")
    report["ledgers"]["stage_analytic_only"]["zero_tape"]["preparation_PRG_output_bytes"] += 4
    write(archive, "correlation", report)
    with pytest.raises(ValueError, match="correlation normalized reproduction"):
        portable(archive)


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


def test_portable_replay_never_accesses_cached_primary_sources(monkeypatch):
    original_open, original_stat = Path.open, Path.stat

    def cached(path):
        return path.is_relative_to(audit.ROOT / "papers") or path.is_relative_to(audit.ROOT / "target/compact-setup-fulltext")

    def no_cached_open(path, *args, **kwargs):
        assert not cached(path), f"portable replay read cached primary source: {path}"
        return original_open(path, *args, **kwargs)

    def no_cached_stat(path, *args, **kwargs):
        assert not cached(path), f"portable replay stat cached primary source: {path}"
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_cached_open)
    monkeypatch.setattr(Path, "stat", no_cached_stat)
    result = portable()
    assert not result["correlation"]["primary_sources_reverified"]
    assert "NOT reverified" in result["correlation"]["primary_source_scope"]


@pytest.mark.parametrize("field", ["identity", "hash", "page", "excerpt"])
def test_malformed_retained_provenance_rejected_without_cached_sources(archive, field):
    report = read(archive, "correlation")
    paper = report["source_provenance"]["papers"][0]
    if field == "identity":
        paper["id"] = "untrusted/paper"
    elif field == "hash":
        paper["pdf_sha256"] = "0" * 64
    elif field == "page":
        paper["reviewed_excerpts"][0]["pdf_page"] = 0
    else:
        paper["reviewed_excerpts"][0]["excerpt"] = "invented source claim"
    write(archive, "correlation", report)
    with pytest.raises(ValueError, match="retained provenance metadata"):
        portable(archive)


def test_frozen_v3_metadata_does_not_follow_live_test_hash(archive):
    old = audit.CORRELATION_IMPLEMENTATION["tests/test_fixed_weight_correlation_gate.py"]
    assert audit.snapshot("test_fixed_weight_correlation_gate_v3.py") == old
    report = read(archive, "correlation")
    report["implementation_hashes"]["tests/test_fixed_weight_correlation_gate.py"] = "0" * 64
    write(archive, "correlation", report)
    with pytest.raises(ValueError, match="original implementation metadata lock"):
        portable(archive)


def test_archived_log_fixed_txt_basename_reconciles_summary(archive):
    # Ignore an unrelated historical .log spelling in the archive. Only the
    # explicitly renamed fixed .log.txt artifact supplies evidence.
    (archive / audit.PILOT_LOG.removesuffix(".txt")).write_text("unrelated")
    result = portable(archive)
    assert result["pilot_log"] == dict(status="archived_summary_reconciled", basename=audit.PILOT_LOG,
                                       sha256=audit.file_hash(archive / audit.PILOT_LOG))
    summary = json.loads((archive / audit.PILOT_LOG).read_text())
    summary["heldout"]["gate"]["screen_pass"] = True
    (archive / audit.PILOT_LOG).write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="archived log/summary mismatch"):
        portable(archive)


def test_portable_cli_flags_write_honest_fresh_summary(tmp_path):
    output = tmp_path / "portable.json"
    assert audit.main(["--skip-native", "--skip-primary-sources", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert not report["head"]["native_replayed"]
    assert not report["correlation"]["primary_sources_reverified"]

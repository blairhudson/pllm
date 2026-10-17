"""Historical archive relocation and whole-campaign evidence failures."""
import json
import shutil

import pytest

from scripts.audit_publication_lifecycle import DEFAULT, InvalidObservation, audit


def test_original_and_relocated_archive_replay_without_original_absolute_paths(tmp_path):
    expected = audit(DEFAULT)
    relocated = tmp_path.resolve() / "archive"
    shutil.copytree(DEFAULT, relocated)
    assert audit(relocated) == expected
    assert expected["trials"] == 3
    assert expected["samples"] == 289
    assert expected["process_incarnations"] == 11
    assert expected["state_equal"] is True
    assert expected["historical_rejections"] == 2


@pytest.mark.parametrize("missing", [
    "v3/cohort.json", "sources/scripts/benchmark_publication_lifecycle.py",
    "v1/executed-sources/scripts/benchmark_publication_lifecycle.py",
])
def test_unbound_campaign_or_source_is_not_qualified(tmp_path, missing):
    relocated = tmp_path.resolve() / "archive"
    shutil.copytree(DEFAULT, relocated)
    manifest_path = relocated / "archive.json"
    manifest = json.loads(manifest_path.read_bytes())
    del manifest["files"][missing]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(InvalidObservation, match="coverage"):
        audit(relocated)


def test_corrupted_samples_cannot_inherit_historical_pass(tmp_path):
    relocated = tmp_path.resolve() / "archive"
    shutil.copytree(DEFAULT, relocated)
    samples = next((relocated / "v3/cohort.traces").glob("*/publication-samples.jsonl"))
    samples.chmod(0o600)
    raw = samples.read_bytes()
    samples.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(InvalidObservation, match="trace_file_binding"):
        audit(relocated)

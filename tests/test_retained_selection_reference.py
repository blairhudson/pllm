"""Independent selected-record parity and complete owner/material evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research import retained_selection
    return retained_selection


@pytest.fixture
def report():
    return json.loads((ROOT / "docs/evidence/research-retained-selection.json").read_bytes())


def test_recorded_native_cohort_has_exact_records_and_complete_material(diagnostic, report):
    assert report["schema"] == "pllm.retained_selection_reference.v1"
    assert report["samples_per_case"] == 3
    assert len(report["cases"]) == len(diagnostic.CASES)
    assert set(report["source_sha256"]) == set(diagnostic.SOURCE_FILES)
    for path, digest in report["source_sha256"].items():
        assert len(digest) == 64 and (ROOT / path).is_file()
        if path.startswith("benchmarks/"):
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    total = 0
    changed_private_choices = False
    for case, expected in zip(report["cases"], diagnostic.CASES, strict=True):
        assert tuple(case["geometry"][k] for k in ("rows", "width", "keep", "recent")) == expected
        assert len(case["samples"]) == 3
        assert len({r["public_schedule_sha256"] for r in case["samples"]}) == 1
        for sample in case["samples"]:
            total += diagnostic.validate_case(sample)
            assert sample["material"] == case["samples"][0]["material"]
            assert sample["native_process_cpu_seconds"] > 0
            assert sample["native_process_peak_rss_bytes"] > 0
        # Clear selected addresses demonstrably reveal input-dependent choices,
        # although the garbled public access schedule above stays fixed.
        if expected[2] < expected[0]:
            changed_private_choices |= len({r["output_sha256"] for r in case["samples"]}) > 1
    assert changed_private_choices
    assert total == report["checked_output_words"] == 474
    for claim in ("mpcache_distributed_scoring", "provider_resident_state", "compiler_validated_context",
                  "whole_decoder", "transport", "independent_cryptographic_review"):
        assert report["scope"][claim] is False
    assert report["scope"]["full_wire_bytes"] is None


@pytest.mark.parametrize("field", ["constant_label_bytes", "input_label_bytes", "output_label_bytes",
                                  "client_encoding_bytes", "evaluator_ciphertext_bytes"])
def test_material_omissions_reject(diagnostic, report, field):
    sample = deepcopy(report["cases"][0]["samples"][0])
    sample["material"][field] = 0
    with pytest.raises(ValueError, match="conserve"):
        diagnostic.validate_case(sample)


def test_wrong_kv_and_uncharged_owners_reject(diagnostic, report):
    sample = deepcopy(report["cases"][0]["samples"][0])
    sample["output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="oracle"):
        diagnostic.validate_case(sample)
    sample = deepcopy(report["cases"][0]["samples"][0])
    sample["allocation"]["state_after_bytes"] -= 8
    with pytest.raises(ValueError, match="allocation"):
        diagnostic.validate_case(sample)


def test_private_score_order_recent_suffix_and_original_positions(diagnostic):
    assert diagnostic.oracle(8, 4, 3, 1, 0)["positions"] == [1, 3, 15]
    assert diagnostic.oracle(8, 4, 3, 1, 1)["positions"] == [9, 11, 15]
    assert diagnostic.oracle(8, 4, 8, 0, 2)["positions"] == list(range(1, 16, 2))
    with pytest.raises(ValueError, match="samples"):
        diagnostic.run(0)

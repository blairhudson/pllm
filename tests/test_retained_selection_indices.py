"""Compact sorting must pay for oblivious movement and the complete boundary."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research import retained_selection_indices
    return retained_selection_indices


@pytest.fixture
def report():
    return json.loads((ROOT / "docs/evidence/research-retained-selection-indices.json").read_bytes())


def test_matched_native_words_include_full_width_state_and_fresh_material(diagnostic, report):
    assert report["schema"] == "pllm.retained_selection_indices.v1"
    assert set(report["source_sha256"]) == set(diagnostic.SOURCE_FILES)
    for path, digest in report["source_sha256"].items():
        assert len(digest) == 64 and (ROOT / path).is_file()
        if path.startswith("benchmarks/"):
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    geometries = [(geometry, layout) for geometry in (*diagnostic.MATCHED, *diagnostic.INDEX_ONLY)
                  for layout in ((0, 1) if geometry in diagnostic.MATCHED else (1,))]
    assert len(report["cases"]) == len(geometries) == 12
    checked = 0
    samples = 0
    paired = {}
    for case, (geometry, layout) in zip(report["cases"], geometries, strict=True):
        rows, width, keep, recent = geometry
        assert case["geometry"] == dict(rows=rows, width=width, keep=keep, recent=recent)
        assert case["layout"] == layout
        projection = case["projection"]
        diagnostic.validate_projection(projection)
        assert projection["admitted"] is True
        assert len(case["samples"]) == report["samples_per_case"] == 3
        assert len({s["public_schedule_sha256"] for s in case["samples"]}) == 1
        for variant, sample in enumerate(case["samples"]):
            assert sample["layout"] == layout and sample["variant"] == variant
            checked += diagnostic.validate_case(sample)
            samples += 1
            assert sample["native_process_cpu_seconds"] > 0
            assert sample["native_process_peak_rss_bytes"] > 0
            assert projection["actual_wires"] == sample["allocation"]["actual_wires"]
            for key in ("and_gates", "constant_label_bytes", "input_label_bytes", "output_label_bytes",
                        "client_encoding_bytes", "public_metadata_bytes_estimate"):
                assert projection[key] == sample["material"][key]
        outputs = [s["output_sha256"] for s in case["samples"]]
        if geometry in paired:
            assert paired[geometry] == outputs
        paired[geometry] = outputs
    assert checked == report["checked_output_words"] == 4041
    assert samples == 36


def test_full_scan_gather_does_not_inherit_compact_record_savings(diagnostic, report):
    for geometry in diagnostic.MATCHED:
        cases = [c for c in report["cases"] if tuple(c["geometry"][k]
                 for k in ("rows", "width", "keep", "recent")) == geometry]
        assert [c["layout"] for c in cases] == [0, 1]
        full, compact = [c["samples"][0]["material"] for c in cases]
        assert compact["client_to_evaluator_material_bytes"] > full["client_to_evaluator_material_bytes"]
        assert compact["evaluator_to_client_material_bytes"] == full["evaluator_to_client_material_bytes"]
        assert compact["client_encoding_bytes"] == full["client_encoding_bytes"]
        assert compact["public_metadata_bytes_estimate"] < full["public_metadata_bytes_estimate"]
    for gate in report["full_width_projections"]:
        diagnostic.validate_projection(gate)
        assert gate["admitted"] is False
        assert gate["combined_preflight_bound_bytes"] > 64 << 20
        assert gate["width"] == report["geometry_source"]["kv_width"] == 128
        assert gate["whole_body_label_and_ciphertext_body_bytes"] == (
            gate["label_and_ciphertext_body_bytes"] * report["geometry_source"]["layers"])
    assert len(report["full_width_projections"]) == 8
    full, compact = [g for g in report["full_width_projections"] if g["rows"] == 256]
    assert full["label_and_ciphertext_body_bytes"] > 1_760_000_000
    assert compact["label_and_ciphertext_body_bytes"] > 8_700_000_000
    source = report["geometry_source"]
    assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    assert report["scope"]["full_wire_bytes"] is None
    for claim in ("provider_resident_state", "compiler_validated_context", "transport",
                  "whole_decoder", "independent_cryptographic_review"):
        assert report["scope"][claim] is False
    assert report["scope"]["scores"] == "supplied by trusted client; protected scoring absent"


@pytest.mark.parametrize("field", ["and_gates", "constant_label_bytes", "input_label_bytes",
                                   "output_label_bytes", "client_encoding_bytes"])
def test_exact_geometry_counts_reject_omissions(diagnostic, report, field):
    projection = deepcopy(report["cases"][1]["projection"])
    projection[field] = 0
    with pytest.raises(ValueError, match="projection"):
        diagnostic.validate_projection(projection)


def test_compact_full_width_gather_cannot_hide_wrong_records(diagnostic, report):
    case = next(c for c in report["cases"] if c["geometry"]["width"] == 128 and c["layout"] == 1)
    sample = deepcopy(case["samples"][0])
    sample["output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="oracle"):
        diagnostic.validate_case(sample)


def test_unpriced_geometry_cannot_authorize_issuance(diagnostic, report):
    projection = deepcopy(report["full_width_projections"][0])
    projection["admitted"] = True
    with pytest.raises(ValueError, match="admission"):
        diagnostic.validate_projection(projection)

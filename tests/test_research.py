from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from pllm.research import (
    MethodRecord,
    PublicationAssessment,
    RecipeRecord,
    ResearchMetadataError,
    SourceRecord,
    assess_publication,
    get_method,
    get_recipe,
    get_source,
    list_methods,
    list_recipes,
    list_sources,
    render_agents_guide,
)
from pllm.research._validation import decode_json, validate_payload

ROOT = Path(__file__).resolve().parents[1]


def test_public_research_api_identity_aliases_and_immutability() -> None:
    import pllm.research as research

    assert set(research.__all__) == {
        "MethodRecord",
        "PublicationAssessment",
        "RecipeRecord",
        "ResearchMetadataError",
        "ResearchMetadataIOError",
        "SourceRecord",
        "assess_publication",
        "get_method",
        "get_recipe",
        "get_source",
        "list_methods",
        "list_recipes",
        "list_sources",
        "render_agents_guide",
    }
    assert research.SourceRecord is SourceRecord
    assert research.MethodRecord is MethodRecord
    assert research.RecipeRecord is RecipeRecord
    assert research.PublicationAssessment is PublicationAssessment
    assert all(type(record) is SourceRecord for record in list_sources())
    assert all(type(record) is MethodRecord for record in list_methods())
    assert all(type(record) is RecipeRecord for record in list_recipes())
    assert [record.identifier for record in list_sources()] == sorted(
        record.identifier for record in list_sources()
    )
    assert get_source("R23").identifier == "pllm.source.mpcache.arxiv-2501.06807v2"
    assert get_method("R23").identifier == "pllm.method.mpcache-structural-adaptation.v1"
    assert get_recipe("R23").identifier == "pllm.recipe.mpcache-reproduction.v1"
    with pytest.raises(TypeError):
        list_sources()[0].payload["title"] = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        list_sources()[0].aliases = ()
    with pytest.raises(KeyError) as error:
        get_source("secret-value-must-not-appear")
    assert "secret-value-must-not-appear" not in str(error.value)


def test_agents_guide_is_deterministic_and_states_research_boundaries() -> None:
    first = render_agents_guide()
    assert first == render_agents_guide()
    assert first.startswith("# PLLM Research Agent Guide\n")
    assert first.endswith("\n")
    assert "Find, Add, Test, Compare" in first
    assert "Do not build, link, vendor" in first
    assert "Paper tables and figures are not PLLM measurements" in first
    assert "no generic command that runs a full-model matched benchmark" in first
    assert "Human review is required" in first
    assert "uv run pytest" in first


def test_public_research_import_is_metadata_only() -> None:
    code = """
import sys
import pllm.research as research
assert research.list_recipes()
banned = {'numpy', 'fastapi', 'httpx', 'cryptography', 'pllm._native', 'pllm.provider'}
assert not banned.intersection(sys.modules), banned.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "python")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":"x","schema_version":"y"}',
        b'{"value":NaN}',
        b'{"value":1e999}',
    ],
)
def test_strict_json_rejects_duplicates_and_non_finite_numbers(raw: bytes) -> None:
    with pytest.raises(ResearchMetadataError):
        decode_json(raw)


def test_catalog_validation_rejects_wrong_schema_and_missing_relationship() -> None:
    document = json.loads((ROOT / "python/pllm/research/_catalog.json").read_text())
    payload = document["payload"]
    payload["records"]["methods"][0]["schema_version"] = "wrong"
    with pytest.raises(ResearchMetadataError):
        validate_payload(payload)

    document = json.loads((ROOT / "python/pllm/research/_catalog.json").read_text())
    payload = document["payload"]
    payload["records"]["methods"][0]["source_record_ids"] = ["missing"]
    with pytest.raises(ResearchMetadataError):
        validate_payload(payload)


def test_catalog_validation_rejects_duplicate_and_ambiguous_aliases() -> None:
    document = json.loads((ROOT / "python/pllm/research/_catalog.json").read_text())
    payload = document["payload"]
    duplicate = dict(payload["records"]["recipes"][0])
    duplicate["id"] = "another-recipe"
    payload["records"]["recipes"].append(duplicate)
    with pytest.raises(ResearchMetadataError):
        validate_payload(payload)


def publication_request(*, publication_class: str = "documentation") -> dict[str, Any]:
    kind = "scope" if publication_class == "documentation" else "reproduction"
    return {
        "schema_version": "pllm.publication_assessment_request.v1",
        "claim": {
            "schema_version": "pllm.claim_record.v1",
            "id": "claim.example",
            "kind": kind,
            "statement": "The adapter represents the cited cache-selection structure.",
            "scope": "semantic decoder plan only",
            "limitations": ["No protected-execution claim."],
            "source_record_ids": ["pllm.source.mpcache.arxiv-2501.06807v2"],
            "method_record_ids": (
                []
                if publication_class == "documentation"
                else ["pllm.method.mpcache-structural-adaptation.v1"]
            ),
            "plan_lock_digests": [] if publication_class == "documentation" else ["a" * 64],
            "publication_class": publication_class,
        },
        "plan_lineages": [],
        "evidence_bundles": [],
    }


def test_publication_assessment_is_pure_deterministic_and_preserves_claim_text() -> None:
    request = publication_request()
    first = assess_publication(request)
    second = assess_publication(json.dumps(request))
    assert type(first) is PublicationAssessment
    assert first.decision == "ready_for_human_review"
    assert first.digest == second.digest
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.to_dict()["claim"]["statement"] == request["claim"]["statement"]
    assert first.to_dict()["reason_codes"] == ["HUMAN_REVIEW_REQUIRED"]
    assert first.to_dict()["citations"][0]["authors_as_supplied"][0] == "Wenxuan Zeng"
    with pytest.raises(TypeError):
        first.data["decision"] = "published"


def test_publication_assessment_fails_closed_on_missing_lineage_and_evidence() -> None:
    assessment = assess_publication(publication_request(publication_class="reproduction"))
    assert assessment.decision == "blocked"
    assert set(assessment.to_dict()["reason_codes"]) == {
        "ASSURANCE_EVIDENCE_MISSING",
        "BENCHMARK_EVIDENCE_MISSING",
        "METHOD_LIFECYCLE_UNCHECKED",
        "PLAN_LINEAGE_MISSING",
    }


def test_publication_assessment_rejects_claim_class_mismatch_and_duplicate_json() -> None:
    request = publication_request()
    request["claim"]["publication_class"] = "scientific"
    with pytest.raises(ResearchMetadataError):
        assess_publication(request)
    with pytest.raises(ResearchMetadataError):
        assess_publication(
            '{"schema_version":"x","schema_version":"y","claim":{},"plan_lineages":[],"evidence_bundles":[]}'
        )

    document = json.loads((ROOT / "python/pllm/research/_catalog.json").read_text())
    payload = document["payload"]
    colliding_source = dict(payload["records"]["sources"][0])
    colliding_source["id"] = "R23"
    payload["records"]["sources"].append(colliding_source)
    with pytest.raises(ResearchMetadataError):
        validate_payload(payload)

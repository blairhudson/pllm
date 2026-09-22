from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import pllm

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "docs/data/research/sources"


def source_document() -> dict:
    return json.loads((SOURCES / "R01.json").read_text())


def method_document(*, promoted: bool = False) -> dict:
    status = "passed" if promoted else "unchecked"
    return {
        "schema_version": "pllm.method_record.v1",
        "id": "pllm.method.dash-affine-label",
        "version": "1",
        "registry_aliases": ["R01"],
        "source_record_ids": ["R01"],
        "upstream_artifact_lock_ids": ["dash-upstream"],
        "classification": "structural_pllm_adaptation" if not promoted else "engineering_improvement",
        "semantic_scope": "bounded affine-label arithmetic fixture",
        "input_representations": ["signed-q10"],
        "output_representations": ["affine-label"],
        "online_roles": ["evaluator"],
        "preparation_phase": "offline",
        "native_owner": "Rust",
        "implementation_status": "native_operator" if promoted else "planned",
        "assurance_obligations": ["single-use labels", "source-independent clean-room code"],
        "lifecycle_status": {
            "fidelity": status,
            "protected_execution": status,
            "assurance": status,
            "benchmark": status,
            "promotion": status,
        },
        "eligible_components": ["pllm/arithmetic-garbling-silu-q7/v1"] if promoted else [],
    }


def artifact(*, verified: bool = False) -> pllm.ArtifactLock:
    value = pllm.ArtifactLock(
        id="dash-upstream",
        source_record_id="R01",
        revision=source_document()["artifact_commit"],
        artifact_digest="1" * 64,
        license_review="external_reproduction_only",
        isolation="external_process",
    )
    return value.verify(evidence_digest="2" * 64) if verified else value


def test_existing_source_records_validate_canonically_and_keep_attribution() -> None:
    schema = json.loads((ROOT / "schemas/source-record.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    paths = sorted(path for path in SOURCES.glob("*.json") if path.name != "tooling-sources.json")
    records = []
    for path in paths:
        document = json.loads(path.read_text())
        if document.get("schema_version") != "pllm.source_record.v1":
            continue
        Draft202012Validator(schema).validate(document)
        record = pllm.SourceRecord.from_dict(document)
        records.append(record)
        assert record.to_dict()["authors"] == document["authors"]
        assert record.to_dict()["primary_url"] == document["primary_url"]
        assert len(record.digest) == 64
        assert record.canonical_bytes() == json.dumps(
            record.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    assert {record.id for record in records} == {
        "R01",
        "R02",
        "R03",
        "R07",
        "R21",
        "R22",
        "pllm.source.mpcache.arxiv-2501.06807v2",
    }


def test_slalom_full_text_lock_matches_the_quarantined_pdf() -> None:
    document = json.loads((SOURCES / "R07.json").read_text())
    paper = ROOT / document["local_path"]
    if paper.exists():
        assert hashlib.sha256(paper.read_bytes()).hexdigest() == document["paper_digest"]
    assert (
        document["paper_digest"]
        == "ffdccc42057482eada2ca836e93dafbc35832fb3670e0bce0524b3412f7ef536"
    )
    assert document["access"] == "primary_full_text_pdf"
    assert document.get("artifact_commit") is None


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(unexpected=True),
        lambda value: value.update(primary_url="http://example.test/paper"),
        lambda value: value.update(local_path="../paper.pdf"),
        lambda value: value.update(paper_digest="bad"),
        lambda value: value.update(authors=[]),
        lambda value: value.update(license_review="unchecked"),
    ],
)
def test_source_ingestion_rejects_unlocked_or_unsafe_records(change) -> None:
    document = source_document()
    change(document)
    with pytest.raises((TypeError, ValueError)):
        pllm.SourceRecord.from_dict(document)


def test_artifacts_are_quarantined_pinned_and_license_isolated() -> None:
    locked = artifact()
    assert locked.status == "quarantined"
    assert locked.evidence_digest is None
    verified = locked.verify(evidence_digest="2" * 64)
    assert verified.status == "reproduction_verified"
    assert locked.status == "quarantined"
    with pytest.raises(ValueError, match="quarantined"):
        verified.verify(evidence_digest="3" * 64)
    with pytest.raises(ValueError, match="external-process"):
        pllm.ArtifactLock(
            id="unsafe",
            source_record_id="R01",
            revision=source_document()["artifact_commit"],
            artifact_digest="1" * 64,
            license_review="external_reproduction_only",
            isolation="clean_room_fixture",
        )
    pending_license = pllm.ArtifactLock(
        id="pending-license",
        source_record_id="R01",
        revision=source_document()["artifact_commit"],
        artifact_digest="1" * 64,
        license_review="required_before_execution",
        isolation="external_process",
    )
    with pytest.raises(ValueError, match="license review"):
        pending_license.verify(evidence_digest="2" * 64)


def test_existing_upstream_lock_ingests_only_as_quarantined_oracle() -> None:
    path = SOURCES / "mpcache-55e401b17db3c47dd6920b9a92ad937c7e9bb452.json"
    document = json.loads(path.read_text())
    schema = json.loads((ROOT / "schemas/upstream-artifact-lock.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    lock = pllm.ArtifactLock.from_dict(document)
    assert lock.status == "quarantined"
    assert lock.artifact_digest is None
    assert lock.isolation == "external_process"
    with pytest.raises(ValueError, match="artifact and evidence"):
        lock.verify(evidence_digest="2" * 64)
    verified = lock.lock_artifact(artifact_digest="1" * 64).verify(
        evidence_digest="2" * 64
    )
    source = pllm.SourceRecord.from_dict(
        json.loads((SOURCES / "mpcache-arxiv-2501.06807v2.json").read_text())
    )
    registry = pllm.ResearchRegistry(sources=[source], artifacts=[verified])
    assert registry.artifact(lock.id).status == "reproduction_verified"


def test_registry_rejects_unresolved_or_mismatched_research_links() -> None:
    source = pllm.SourceRecord.from_dict(source_document())
    method = pllm.MethodRecord.from_dict(method_document())
    with pytest.raises(ValueError, match="unresolved"):
        pllm.ResearchRegistry(sources=[source], methods=[method])
    with pytest.raises(ValueError, match="revision"):
        pllm.ResearchRegistry(
            sources=[source],
            artifacts=[
                pllm.ArtifactLock(
                    id="dash-upstream",
                    source_record_id="R01",
                    revision="abcdef0",
                    artifact_digest="1" * 64,
                    license_review="external_reproduction_only",
                    isolation="external_process",
                )
            ],
        )


def test_promotion_gate_preserves_planned_blocked_and_promoted_states() -> None:
    source = pllm.SourceRecord.from_dict(source_document())
    planned = pllm.MethodRecord.from_dict(method_document())
    registry = pllm.ResearchRegistry(
        sources=[source],
        methods=[planned],
        artifacts=[artifact()],
    )
    decision = registry.promotion(planned.id)
    assert decision.eligible is False
    assert "artifact:dash-upstream:not_reproduced" in decision.blockers
    assert "method:native_implementation" in decision.blockers
    assert "gate:fidelity:unchecked" in decision.blockers
    assert "method:no_eligible_profile" in decision.blockers

    structural_document = method_document(promoted=True)
    structural_document["classification"] = "structural_pllm_adaptation"
    structural = pllm.MethodRecord.from_dict(structural_document)
    structural_decision = pllm.ResearchRegistry(
        sources=[source], methods=[structural], artifacts=[artifact(verified=True)]
    ).promotion(structural.id)
    assert structural_decision.eligible is False
    assert "method:structural_only" in structural_decision.blockers

    promoted = pllm.MethodRecord.from_dict(method_document(promoted=True))
    registry = pllm.ResearchRegistry(
        sources=[source],
        methods=[promoted],
        artifacts=[artifact(verified=True)],
    )
    assert registry.promotion(promoted.id) == pllm.PromotionDecision(promoted.id, True, ())
    assert registry.source("R01") is source
    assert registry.method(promoted.id) is promoted
    assert registry.artifact("dash-upstream").status == "reproduction_verified"


def test_required_before_vendoring_stays_external_without_blocking_clean_room_promotion() -> None:
    document = source_document()
    document["license_review"] = "required_before_vendoring"
    source = pllm.SourceRecord.from_dict(document)
    lock = pllm.ArtifactLock(
        id="dash-upstream",
        source_record_id="R01",
        revision=document["artifact_commit"],
        artifact_digest="1" * 64,
        license_review="required_before_vendoring",
        isolation="external_process",
    ).verify(evidence_digest="2" * 64)
    method = pllm.MethodRecord.from_dict(method_document(promoted=True))
    decision = pllm.ResearchRegistry(
        sources=[source], methods=[method], artifacts=[lock]
    ).promotion(method.id)
    assert decision.eligible is True
    assert decision.blockers == ()
    assert lock.isolation == "external_process"


def test_tracked_slalom_method_remains_blocked_and_unpromoted() -> None:
    source = pllm.SourceRecord.from_dict(
        json.loads((SOURCES / "R07.json").read_text())
    )
    path = ROOT / "docs/data/research/methods/R07-masked-linear.json"
    document = json.loads(path.read_text())
    schema = json.loads((ROOT / "schemas/method-record.schema.json").read_text())
    Draft202012Validator(schema).validate(document)
    method = pllm.MethodRecord.from_dict(document)
    decision = pllm.ResearchRegistry(sources=[source], methods=[method]).promotion(method.id)
    assert decision.eligible is False
    assert "method:native_implementation" in decision.blockers
    assert "gate:fidelity:blocked" in decision.blockers
    assert "method:no_eligible_profile" in decision.blockers


def test_method_records_are_canonical_and_lifecycle_fields_do_not_conflate() -> None:
    document = method_document(promoted=True)
    schema = json.loads((ROOT / "schemas/method-record.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    record = pllm.MethodRecord.from_dict(document)
    assert record.to_dict()["classification"] == "engineering_improvement"
    assert record.to_dict()["lifecycle_status"] == document["lifecycle_status"]
    assert len(record.digest) == 64
    forged = copy.deepcopy(document)
    forged["lifecycle_status"].pop("assurance")
    with pytest.raises(ValueError, match="fields"):
        pllm.MethodRecord.from_dict(forged)


def test_research_registry_has_no_execution_or_network_boundary() -> None:
    source = (ROOT / "python/pllm/research/__init__.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not imports & {"httpx", "requests", "subprocess", "urllib.request", "importlib"}
    assert "exec(" not in source
    assert "eval(" not in source

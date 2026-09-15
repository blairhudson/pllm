"""Pure, deterministic assessment of supplied research publication records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import re
from types import MappingProxyType
from typing import Any

from ._validation import ResearchMetadataError, canonical_bytes, decode_json, validate_json_value

REQUEST_VERSION = "pllm.publication_assessment_request.v1"
ASSESSMENT_VERSION = "pllm.publication_assessment.v1"
POLICY = {"id": "pllm.publication_policy", "version": "1"}
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_KINDS = {
    "scope": "documentation",
    "negative_result": "negative_result",
    "reproduction": "reproduction",
    "engineering_improvement": "engineering",
    "scientific_contribution": "scientific",
}
_RELATIONS = {"implements", "composes", "measured_by", "supports_claim", "supersedes", "retracts"}
_AXES = ("source", "method", "plan_lineage", "benchmark", "assurance", "claim")


def _freeze(value: object) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _normalize(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _object(value: object, required: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    optional = optional or set()
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or set(value) - required - optional
    ):
        raise ResearchMetadataError("publication assessment input has an invalid structure")
    return value


def _text(value: object) -> str:
    if type(value) is not str or not value:
        raise ResearchMetadataError("publication assessment input has an invalid string")
    return value


def _digest(value: object) -> str:
    text = _text(value)
    if _DIGEST.fullmatch(text) is None:
        raise ResearchMetadataError("publication assessment input has an invalid digest")
    return text


def _texts(value: object, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ResearchMetadataError("publication assessment input has an invalid string array")
    result = [_text(item) for item in value]
    if len(result) != len(set(result)):
        raise ResearchMetadataError("publication assessment input contains duplicate identities")
    return result


def _record_digest(value: object) -> str:
    return "sha256:" + sha256(canonical_bytes(value)).hexdigest()


def _parse_request(request: Mapping[str, Any] | bytes | str) -> dict[str, Any]:
    if isinstance(request, bytes):
        value = decode_json(request)
    elif type(request) is str:
        value = decode_json(request.encode("utf-8"))
    elif isinstance(request, Mapping):
        value = _normalize(request)
        validate_json_value(value)
    else:
        raise TypeError("publication assessment request must be a mapping, bytes, or JSON string")
    document = _object(value, {"schema_version", "claim", "plan_lineages", "evidence_bundles"})
    if document["schema_version"] != REQUEST_VERSION:
        raise ResearchMetadataError("publication assessment input has an unsupported version")
    return document


def _validate_claim(value: object) -> dict[str, Any]:
    claim = _object(
        value,
        {
            "schema_version",
            "id",
            "kind",
            "statement",
            "scope",
            "limitations",
            "source_record_ids",
            "method_record_ids",
            "plan_lock_digests",
            "publication_class",
        },
    )
    if claim["schema_version"] != "pllm.claim_record.v1":
        raise ResearchMetadataError("claim record has an unsupported version")
    kind = _text(claim["kind"])
    if kind not in _KINDS or claim["publication_class"] != _KINDS[kind]:
        raise ResearchMetadataError("claim kind and publication class do not match")
    _text(claim["id"])
    _text(claim["statement"])
    _text(claim["scope"])
    _texts(claim["limitations"])
    _texts(claim["source_record_ids"], nonempty=True)
    _texts(claim["method_record_ids"])
    for value in _texts(claim["plan_lock_digests"]):
        _digest(value)
    return claim


def _validate_lineage(value: object) -> dict[str, Any]:
    lineage = _object(
        value,
        {
            "schema_version",
            "id",
            "plan_lock_digest",
            "logical_plan_digest",
            "execution_plan_digest",
            "source_record_ids",
            "method_record_ids",
            "edges",
        },
    )
    if lineage["schema_version"] != "pllm.plan_lineage_record.v1":
        raise ResearchMetadataError("plan lineage record has an unsupported version")
    _text(lineage["id"])
    for field in ("plan_lock_digest", "logical_plan_digest", "execution_plan_digest"):
        _digest(lineage[field])
    source_ids = _texts(lineage["source_record_ids"], nonempty=True)
    method_ids = _texts(lineage["method_record_ids"], nonempty=True)
    if not isinstance(lineage["edges"], list):
        raise ResearchMetadataError("plan lineage record has invalid edges")
    seen_edges: set[tuple[str, str, str]] = set()
    for raw in lineage["edges"]:
        edge = _object(raw, {"relation", "from_id", "to_id"})
        if edge["relation"] not in _RELATIONS:
            raise ResearchMetadataError("plan lineage record has an invalid relation")
        from_id = _text(edge["from_id"])
        to_id = _text(edge["to_id"])
        identity = (edge["relation"], from_id, to_id)
        if identity in seen_edges:
            raise ResearchMetadataError("plan lineage record has a duplicate edge")
        seen_edges.add(identity)
    if not seen_edges:
        raise ResearchMetadataError("plan lineage record has no edges")
    return lineage


def _validate_measurement(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "id",
        "origin",
        "scope",
        "metric",
        "privacy_cohort",
        "numeric_cohort",
        "value",
        "unit",
        "unavailable_reason",
        "plan_lock_digest",
        "environment_digest",
    }
    measurement = _object(
        value, required, {"role", "phase", "source_record_ids", "notes", "evidence_paths"}
    )
    if measurement["schema_version"] != "pllm.measurement.v1":
        raise ResearchMetadataError("measurement has an unsupported version")
    for field in ("id", "origin", "scope", "metric", "privacy_cohort", "numeric_cohort", "unit"):
        if field == "unit" and measurement[field] == "":
            continue
        _text(measurement[field])
    if measurement["value"] is not None and type(measurement["value"]) not in (int, float):
        raise ResearchMetadataError("measurement has an invalid value")
    for field in ("plan_lock_digest", "environment_digest"):
        if measurement[field] is not None:
            _digest(measurement[field])
    return measurement


def _validate_assurance(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "id",
        "claim_id",
        "outcome",
        "scope",
        "origin",
        "implementation_refinement",
        "assumptions",
        "evidence_paths",
        "tool",
        "tool_version",
        "code_digest",
    }
    result = _object(value, required)
    if result["schema_version"] != "pllm.assurance_result.v1":
        raise ResearchMetadataError("assurance result has an unsupported version")
    for field in ("id", "claim_id", "outcome", "scope", "origin", "implementation_refinement"):
        _text(result[field])
    _texts(result["assumptions"])
    _texts(result["evidence_paths"])
    if result["code_digest"] is not None:
        _digest(result["code_digest"])
    return result


def _validate_bundle(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "id",
        "created_at",
        "plan_lock_digest",
        "claim_ids",
        "measurements",
        "assurance_results",
        "method_record_ids",
        "source_record_ids",
        "artifacts",
    }
    bundle = _object(value, required)
    if bundle["schema_version"] != "pllm.evidence_bundle.v1":
        raise ResearchMetadataError("evidence bundle has an unsupported version")
    _text(bundle["id"])
    try:
        datetime.fromisoformat(_text(bundle["created_at"]).replace("Z", "+00:00"))
    except ValueError:
        raise ResearchMetadataError("evidence bundle has an invalid creation time") from None
    if bundle["plan_lock_digest"] is not None:
        _digest(bundle["plan_lock_digest"])
    for field in ("claim_ids", "method_record_ids", "source_record_ids"):
        _texts(bundle[field])
    if (
        not isinstance(bundle["measurements"], list)
        or not isinstance(bundle["assurance_results"], list)
        or not isinstance(bundle["artifacts"], list)
    ):
        raise ResearchMetadataError("evidence bundle has invalid record arrays")
    for item in bundle["measurements"]:
        _validate_measurement(item)
    for item in bundle["assurance_results"]:
        _validate_assurance(item)
    for raw in bundle["artifacts"]:
        artifact = _object(raw, {"path", "digest", "media_type"})
        _text(artifact["path"])
        _digest(artifact["digest"])
        _text(artifact["media_type"])
    return bundle


@dataclass(frozen=True, slots=True)
class PublicationAssessment:
    """Immutable assessment; it never represents publication approval."""

    _canonical_bytes: bytes

    @property
    def data(self) -> Mapping[str, Any]:
        return _freeze(decode_json(self._canonical_bytes))

    @property
    def decision(self) -> str:
        return str(self.data["decision"])

    @property
    def digest(self) -> str:
        payload = b"pllm.publication_assessment.v1\0" + self._canonical_bytes
        return "sha256:" + sha256(payload).hexdigest()

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        return decode_json(self._canonical_bytes)


def _axis(status: str, inputs: list[str]) -> dict[str, Any]:
    return {"status": status, "supporting_inputs": sorted(set(inputs))}


def assess_publication(
    request: Mapping[str, Any] | bytes | str,
) -> PublicationAssessment:
    """Assess supplied immutable records without executing any research workflow."""
    from . import list_methods, list_sources

    document = _parse_request(request)
    claim = _validate_claim(document["claim"])
    if not isinstance(document["plan_lineages"], list) or not isinstance(
        document["evidence_bundles"], list
    ):
        raise ResearchMetadataError("publication assessment input has invalid record arrays")
    lineages = [_validate_lineage(item) for item in document["plan_lineages"]]
    bundles = [_validate_bundle(item) for item in document["evidence_bundles"]]
    claim_id = claim["id"]
    source_ids = set(claim["source_record_ids"])
    method_ids = set(claim["method_record_ids"])
    plan_digests = set(claim["plan_lock_digests"])
    publication_class = claim["publication_class"]
    reasons: set[str] = set()
    if len({item["id"] for item in lineages}) != len(lineages) or len(
        {item["plan_lock_digest"] for item in lineages}
    ) != len(lineages):
        raise ResearchMetadataError("publication assessment has duplicate plan lineages")
    if len({item["id"] for item in bundles}) != len(bundles):
        raise ResearchMetadataError("publication assessment has duplicate evidence bundles")
    lineage_identities = {
        claim_id,
        *source_ids,
        *method_ids,
        *plan_digests,
        *(item["id"] for item in lineages),
    }
    if any(
        edge[endpoint] not in lineage_identities
        for item in lineages
        for edge in item["edges"]
        for endpoint in ("from_id", "to_id")
    ):
        raise ResearchMetadataError("publication assessment has an unknown lineage edge identity")

    sources_by_id = {record.identifier: record.to_dict() for record in list_sources()}
    methods_by_id = {record.identifier: record.to_dict() for record in list_methods()}
    missing_sources = source_ids - set(sources_by_id)
    resolved_sources = [sources_by_id[item] for item in sorted(source_ids - missing_sources)]
    if missing_sources:
        reasons.add("SOURCE_RECORD_MISSING")
        source_status = "missing"
    elif any(
        item["access"] not in {"primary_full_text", "primary_full_text_html"}
        or item["paper_digest"] is None
        for item in resolved_sources
    ):
        reasons.add("SOURCE_NOT_ACQUIRED_OR_LOCKED")
        source_status = "blocked"
    else:
        source_status = "satisfied"

    citations = [
        {
            "source_record_id": item["id"],
            "source_record_digest": _record_digest(item),
            "title": item["title"],
            "authors_as_supplied": item["authors"],
            "year": item["year"],
            "primary_url": item["primary_url"],
            "verified_on": item["verified_on"],
            "paper_digest": item["paper_digest"],
            "local_path": item["local_path"],
            "license_review": item["license_review"],
            "artifact_commit": item.get("artifact_commit"),
        }
        for item in resolved_sources
    ]

    method_required = publication_class != "documentation"
    missing_methods = method_ids - set(methods_by_id)
    resolved_methods = [methods_by_id[item] for item in sorted(method_ids - missing_methods)]
    if not method_required:
        method_status = "not_applicable"
    elif not method_ids or missing_methods:
        reasons.add("METHOD_RECORD_MISSING")
        method_status = "missing"
    elif any(not set(item["source_record_ids"]) <= source_ids for item in resolved_methods):
        reasons.add("METHOD_SOURCE_MISMATCH")
        method_status = "blocked"
    else:
        lifecycle = [item["lifecycle_status"] for item in resolved_methods]
        required_lifecycle = (
            ("fidelity",)
            if publication_class == "negative_result"
            else ("fidelity", "assurance", "benchmark")
        )
        states = [status[field] for status in lifecycle for field in required_lifecycle]
        allowed = {"passed", "not_applicable"}
        if publication_class == "negative_result":
            allowed.add("failed")
        if (
            any(state in {"failed", "blocked"} for state in states)
            and publication_class != "negative_result"
        ):
            reasons.add("METHOD_LIFECYCLE_FAILED")
            method_status = "blocked"
        elif any(state not in allowed for state in states):
            reasons.add("METHOD_LIFECYCLE_UNCHECKED")
            method_status = "missing"
        else:
            method_status = "satisfied"

    lineage_required = method_required
    lineage_by_digest = {item["plan_lock_digest"]: item for item in lineages}
    matching_lineages = [
        lineage_by_digest[item] for item in sorted(plan_digests & set(lineage_by_digest))
    ]
    if not lineage_required:
        lineage_status = "not_applicable"
    elif not plan_digests or set(lineage_by_digest) != plan_digests:
        reasons.add("PLAN_LINEAGE_MISSING")
        lineage_status = "missing"
    elif any(
        set(item["source_record_ids"]) != source_ids or set(item["method_record_ids"]) != method_ids
        for item in matching_lineages
    ):
        reasons.add("PLAN_LINEAGE_MISMATCH")
        lineage_status = "blocked"
    else:
        lineage_status = "satisfied"

    matching_bundles = [
        item
        for item in bundles
        if claim_id in item["claim_ids"]
        and item["plan_lock_digest"] in plan_digests
        and set(item["source_record_ids"]) == source_ids
        and set(item["method_record_ids"]) == method_ids
    ]
    benchmark_required = publication_class != "documentation"
    measured = [
        item
        for bundle in matching_bundles
        for item in bundle["measurements"]
        if item["value"] is not None
        and item["plan_lock_digest"] == bundle["plan_lock_digest"]
        and item["environment_digest"] is not None
    ]
    if not benchmark_required:
        benchmark_status = "not_applicable"
    elif not measured:
        reasons.add("BENCHMARK_EVIDENCE_MISSING")
        benchmark_status = "missing"
    else:
        benchmark_status = "satisfied"

    assurance_required = publication_class in {"reproduction", "engineering", "scientific"}
    assurance = [
        item
        for bundle in matching_bundles
        for item in bundle["assurance_results"]
        if item["claim_id"] == claim_id
        and item["outcome"] in {"proved_in_model", "not_refuted"}
        and item["implementation_refinement"] != "not_established"
        and item["code_digest"] is not None
    ]
    if not assurance_required:
        assurance_status = "not_applicable"
    elif not assurance:
        reasons.add("ASSURANCE_EVIDENCE_MISSING")
        assurance_status = "missing"
    else:
        assurance_status = "satisfied"

    axes = {
        "source": _axis(source_status, [item["id"] for item in resolved_sources]),
        "method": _axis(method_status, [item["id"] for item in resolved_methods]),
        "plan_lineage": _axis(lineage_status, [item["id"] for item in matching_lineages]),
        "benchmark": _axis(benchmark_status, [item["id"] for item in measured]),
        "assurance": _axis(assurance_status, [item["id"] for item in assurance]),
        "claim": _axis("satisfied", [claim_id]),
    }
    ready = all(axes[name]["status"] in {"satisfied", "not_applicable"} for name in _AXES)
    if ready:
        reasons.add("HUMAN_REVIEW_REQUIRED")

    input_records: list[tuple[str, dict[str, Any]]] = [("claim", claim)]
    input_records += [("plan_lineage", item) for item in lineages]
    input_records += [("evidence_bundle", item) for item in bundles]
    input_records += [("source", item) for item in resolved_sources]
    input_records += [("method", item) for item in resolved_methods]
    inputs = sorted(
        (
            {
                "kind": kind,
                "id": item["id"],
                "schema_version": item["schema_version"],
                "digest": _record_digest(item),
            }
            for kind, item in input_records
        ),
        key=lambda item: (item["kind"], item["id"]),
    )
    policy = {**POLICY, "digest": _record_digest(POLICY)}
    output = {
        "schema_version": ASSESSMENT_VERSION,
        "policy": policy,
        "inputs": inputs,
        "claim": claim,
        "axes": axes,
        "decision": "ready_for_human_review" if ready else "blocked",
        "reason_codes": sorted(reasons),
        "citations": citations,
        "publication_metadata": {
            "claim_id": claim_id,
            "publication_class": publication_class,
            "plan_lock_digests": sorted(plan_digests),
        },
    }
    return PublicationAssessment(canonical_bytes(output))

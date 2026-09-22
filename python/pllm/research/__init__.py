"""Static research attribution, quarantine, and promotion records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:+-]{0,255}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_COMMIT = re.compile(r"^[A-Fa-f0-9]{7,64}$")
_SOURCE_DOMAIN = b"pllm.source_record.v1\0"
_METHOD_DOMAIN = b"pllm.method_record.v1\0"
_ARTIFACT_DOMAIN = b"pllm.upstream_artifact_lock.v1\0"
_LICENSES = {
    "approved",
    "external_reproduction_only",
    "required_before_vendoring",
    "not_applicable",
    "required_before_execution",
}
_LIFECYCLE_VALUES = {"unchecked", "passed", "failed", "blocked", "not_applicable"}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("research record keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _identity(value: object, path: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{path} must be a bounded identity")
    return value


def _text(value: object, path: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError(f"{path} must be a nonempty bounded string")
    return value


def _digest(value: object, path: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _array(value: object, path: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{path} must be an array")
    items = tuple(_text(item, path, maximum=512) for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{path} must not contain duplicates")
    return items


def _fields(value: object, required: set[str], optional: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    keys = set(value)
    if not required.issubset(keys) or keys - required - optional:
        raise ValueError(f"{path} fields do not match the schema")
    return value


@dataclass(frozen=True, slots=True)
class SourceRecord:
    _canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self._canonical_bytes) is not bytes:
            raise TypeError("source record bytes must be immutable bytes")
        try:
            document = self._validate(json.loads(self._canonical_bytes))
        except (json.JSONDecodeError, UnicodeError, TypeError) as exc:
            raise ValueError("source record must be canonical JSON") from exc
        if _canonical(document) != self._canonical_bytes:
            raise ValueError("source record bytes are not canonical")

    @staticmethod
    def _validate(document: object) -> dict[str, Any]:
        required = {
            "schema_version",
            "id",
            "title",
            "authors",
            "year",
            "primary_url",
            "access",
            "verified_on",
            "paper_digest",
            "local_path",
            "license_review",
        }
        value = dict(_fields(document, required, {"artifact_commit", "notes"}, "source"))
        if value["schema_version"] != "pllm.source_record.v1":
            raise ValueError("unsupported source record schema")
        _identity(value["id"], "source.id")
        _text(value["title"], "source.title")
        value["authors"] = list(_array(value["authors"], "source.authors", nonempty=True))
        if type(value["year"]) is not int or not 1900 <= value["year"] <= date.today().year + 1:
            raise ValueError("source.year is invalid")
        url = urlsplit(str(value["primary_url"]))
        if url.scheme != "https" or not url.netloc or url.username or url.password:
            raise ValueError("source.primary_url must be a credential-free HTTPS URL")
        if value["access"] not in {
            "primary_full_text",
            "primary_full_text_html",
            "primary_full_text_pdf",
            "primary_abstract",
            "primary_author_metadata",
            "primary_author_search_excerpt",
            "unavailable",
        }:
            raise ValueError("source.access is invalid")
        try:
            date.fromisoformat(str(value["verified_on"]))
        except ValueError as exc:
            raise ValueError("source.verified_on must be an ISO date") from exc
        _digest(value["paper_digest"], "source.paper_digest", nullable=True)
        local_path = value["local_path"]
        if type(local_path) is not str or re.fullmatch(r"papers/[^/]+\.pdf", local_path) is None:
            raise ValueError("source.local_path must name one quarantined PDF")
        commit = value.get("artifact_commit")
        if commit is not None and (type(commit) is not str or _COMMIT.fullmatch(commit) is None):
            raise ValueError("source.artifact_commit is invalid")
        if value["license_review"] not in _LICENSES:
            raise ValueError("source.license_review is invalid")
        if "notes" in value:
            _text(value["notes"], "source.notes", maximum=8192)
        return value

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> SourceRecord:
        if not isinstance(document, Mapping):
            raise TypeError("source record must be a mapping")
        return cls(_canonical(cls._validate(document)))

    @property
    def id(self) -> str:
        return str(self.to_dict()["id"])

    @property
    def digest(self) -> str:
        return hashlib.sha256(_SOURCE_DOMAIN + self._canonical_bytes).hexdigest()

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes)


@dataclass(frozen=True, slots=True)
class MethodRecord:
    _canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self._canonical_bytes) is not bytes:
            raise TypeError("method record bytes must be immutable bytes")
        try:
            document = self._validate(json.loads(self._canonical_bytes))
        except (json.JSONDecodeError, UnicodeError, TypeError) as exc:
            raise ValueError("method record must be canonical JSON") from exc
        if _canonical(document) != self._canonical_bytes:
            raise ValueError("method record bytes are not canonical")

    @staticmethod
    def _validate(document: object) -> dict[str, Any]:
        required = {
            "schema_version",
            "id",
            "version",
            "registry_aliases",
            "source_record_ids",
            "upstream_artifact_lock_ids",
            "classification",
            "semantic_scope",
            "input_representations",
            "output_representations",
            "online_roles",
            "preparation_phase",
            "native_owner",
            "implementation_status",
            "assurance_obligations",
            "lifecycle_status",
        }
        value = dict(_fields(document, required, {"eligible_components"}, "method"))
        if value["schema_version"] != "pllm.method_record.v1":
            raise ValueError("unsupported method record schema")
        _identity(value["id"], "method.id")
        _text(value["version"], "method.version", maximum=128)
        for name, nonempty in (
            ("registry_aliases", False),
            ("source_record_ids", True),
            ("upstream_artifact_lock_ids", False),
            ("input_representations", False),
            ("output_representations", False),
            ("online_roles", False),
            ("assurance_obligations", False),
            ("eligible_components", False),
        ):
            if name in value:
                value[name] = list(_array(value[name], f"method.{name}", nonempty=nonempty))
        if value["classification"] not in {
            "source_method",
            "reproduction",
            "engineering_improvement",
            "scientific_contribution",
            "structural_pllm_adaptation",
        }:
            raise ValueError("method.classification is invalid")
        _text(value["semantic_scope"], "method.semantic_scope")
        if value["preparation_phase"] not in {"none", "offline", "online"}:
            raise ValueError("method.preparation_phase is invalid")
        if value["native_owner"] != "Rust":
            raise ValueError("method.native_owner must be Rust")
        if value["implementation_status"] not in {
            "planned",
            "reference_operator",
            "native_operator",
            "region",
            "full_model",
            "structural_adaptation",
        }:
            raise ValueError("method.implementation_status is invalid")
        lifecycle = dict(
            _fields(
                value["lifecycle_status"],
                {"fidelity", "protected_execution", "assurance", "benchmark", "promotion"},
                set(),
                "method.lifecycle_status",
            )
        )
        if any(status not in _LIFECYCLE_VALUES for status in lifecycle.values()):
            raise ValueError("method lifecycle status is invalid")
        value["lifecycle_status"] = lifecycle
        return value

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> MethodRecord:
        if not isinstance(document, Mapping):
            raise TypeError("method record must be a mapping")
        return cls(_canonical(cls._validate(document)))

    @property
    def id(self) -> str:
        return str(self.to_dict()["id"])

    @property
    def digest(self) -> str:
        return hashlib.sha256(_METHOD_DOMAIN + self._canonical_bytes).hexdigest()

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes)


@dataclass(frozen=True, slots=True)
class ArtifactLock:
    id: str
    source_record_id: str
    revision: str
    artifact_digest: str | None
    license_review: str
    isolation: str
    repository_url: str | None = None
    status: str = "quarantined"
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        _identity(self.id, "artifact.id")
        _identity(self.source_record_id, "artifact.source_record_id")
        if type(self.revision) is not str or _COMMIT.fullmatch(self.revision) is None:
            raise ValueError("artifact.revision must be a pinned commit")
        _digest(self.artifact_digest, "artifact.artifact_digest", nullable=True)
        if self.license_review not in _LICENSES:
            raise ValueError("artifact.license_review is invalid")
        if self.isolation not in {"external_process", "clean_room_fixture"}:
            raise ValueError("artifact.isolation is invalid")
        if self.repository_url is not None:
            url = urlsplit(self.repository_url)
            if url.scheme != "https" or not url.netloc or url.username or url.password:
                raise ValueError("artifact.repository_url must be a credential-free HTTPS URL")
        if self.status not in {"quarantined", "reproduction_verified", "rejected"}:
            raise ValueError("artifact.status is invalid")
        _digest(self.evidence_digest, "artifact.evidence_digest", nullable=True)
        if self.status == "reproduction_verified" and (
            self.evidence_digest is None or self.artifact_digest is None
        ):
            raise ValueError("verified artifacts require artifact and evidence digests")
        if self.status == "reproduction_verified" and self.license_review == "required_before_execution":
            raise ValueError("artifact license review is required before execution")
        if self.license_review in {"external_reproduction_only", "required_before_vendoring", "required_before_execution"} and self.isolation != "external_process":
            raise ValueError("restricted artifacts must remain external-process isolated")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_ARTIFACT_DOMAIN + self.canonical_bytes()).hexdigest()

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_digest": self.artifact_digest,
            "evidence_digest": self.evidence_digest,
            "id": self.id,
            "isolation": self.isolation,
            "license_review": self.license_review,
            "repository_url": self.repository_url,
            "revision": self.revision,
            "source_record_id": self.source_record_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ArtifactLock:
        required = {
            "schema_version",
            "id",
            "source_record_ids",
            "repository_url",
            "commit",
            "submodules_lock",
            "dependency_lock",
            "build_recipe",
            "artifact_digest",
            "license_review",
            "acquisition_status",
            "execution_status",
            "execution_boundary",
        }
        value = _fields(document, required, {"notes"}, "artifact lock")
        if value["schema_version"] != "pllm.upstream_artifact_lock.v1":
            raise ValueError("unsupported artifact lock schema")
        source_ids = _array(value["source_record_ids"], "artifact.source_record_ids", nonempty=True)
        if len(source_ids) != 1:
            raise ValueError("artifact locks currently require exactly one source record")
        if value["execution_boundary"] != "oracle_only":
            raise ValueError("upstream artifacts must remain oracle-only")
        if value["acquisition_status"] not in {"not_acquired", "source_acquired"}:
            raise ValueError("artifact acquisition status is invalid")
        if value["execution_status"] not in {"not_executed", "built", "failed", "blocked"}:
            raise ValueError("artifact execution status is invalid")
        if value["execution_status"] == "built" and value["artifact_digest"] is None:
            raise ValueError("built artifacts require an artifact digest")
        submodules = value["submodules_lock"]
        if submodules is not None and (
            not isinstance(submodules, Mapping)
            or any(type(path) is not str or not path or type(commit) is not str or re.fullmatch(r"[a-f0-9]{40}", commit) is None for path, commit in submodules.items())
        ):
            raise ValueError("artifact submodule lock is invalid")
        for name in ("dependency_lock", "build_recipe"):
            item = value[name]
            if item is not None and (type(item) is not str or not item):
                raise ValueError(f"artifact {name} is invalid")
        if "notes" in value:
            _text(value["notes"], "artifact.notes", maximum=8192)
        status = "rejected" if value["execution_status"] == "failed" else "quarantined"
        return cls(
            id=_identity(value["id"], "artifact.id"),
            source_record_id=source_ids[0],
            revision=str(value["commit"]),
            artifact_digest=value["artifact_digest"],
            license_review=str(value["license_review"]),
            isolation="external_process",
            repository_url=str(value["repository_url"]),
            status=status,
        )

    def lock_artifact(self, *, artifact_digest: str) -> ArtifactLock:
        if self.status != "quarantined":
            raise ValueError("only quarantined artifacts can be locked")
        _digest(artifact_digest, "artifact_digest")
        return replace(self, artifact_digest=artifact_digest)

    def verify(self, *, evidence_digest: str) -> ArtifactLock:
        if self.status != "quarantined":
            raise ValueError("only quarantined artifacts can be verified")
        _digest(evidence_digest, "evidence_digest")
        return replace(self, status="reproduction_verified", evidence_digest=evidence_digest)

    def reject(self, *, evidence_digest: str) -> ArtifactLock:
        if self.status != "quarantined":
            raise ValueError("only quarantined artifacts can be rejected")
        _digest(evidence_digest, "evidence_digest")
        return replace(self, status="rejected", evidence_digest=evidence_digest)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    method_id: str
    eligible: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True, init=False)
class ResearchRegistry:
    _sources: Mapping[str, SourceRecord]
    _methods: Mapping[str, MethodRecord]
    _artifacts: Mapping[str, ArtifactLock]

    def __init__(
        self,
        *,
        sources: Iterable[SourceRecord] = (),
        methods: Iterable[MethodRecord] = (),
        artifacts: Iterable[ArtifactLock] = (),
    ) -> None:
        source_map = self._unique(sources, SourceRecord, "source")
        method_map = self._unique(methods, MethodRecord, "method")
        artifact_map = self._unique(artifacts, ArtifactLock, "artifact")
        for artifact in artifact_map.values():
            if artifact.source_record_id not in source_map:
                raise ValueError(f"artifact {artifact.id!r} references an unknown source")
            source = source_map[artifact.source_record_id].to_dict()
            source_commit = source.get("artifact_commit")
            if source_commit is not None and artifact.revision != source_commit:
                raise ValueError(f"artifact {artifact.id!r} revision does not match its source")
        for method in method_map.values():
            document = method.to_dict()
            missing_sources = set(document["source_record_ids"]) - set(source_map)
            missing_artifacts = set(document["upstream_artifact_lock_ids"]) - set(artifact_map)
            unrelated_artifacts = {
                artifact_id
                for artifact_id in document["upstream_artifact_lock_ids"]
                if artifact_id in artifact_map
                and artifact_map[artifact_id].source_record_id
                not in document["source_record_ids"]
            }
            if missing_sources or missing_artifacts or unrelated_artifacts:
                raise ValueError(f"method {method.id!r} has unresolved research references")
        object.__setattr__(self, "_sources", MappingProxyType(source_map))
        object.__setattr__(self, "_methods", MappingProxyType(method_map))
        object.__setattr__(self, "_artifacts", MappingProxyType(artifact_map))

    @staticmethod
    def _unique(values: Iterable[Any], expected: type, label: str) -> dict[str, Any]:
        records = {}
        for record in values:
            if not isinstance(record, expected):
                raise TypeError(f"{label} registry contains the wrong record type")
            if record.id in records and records[record.id] != record:
                raise ValueError(f"conflicting {label} id {record.id!r}")
            records[record.id] = record
        return dict(sorted(records.items()))

    def sources(self) -> tuple[SourceRecord, ...]:
        return tuple(self._sources.values())

    def methods(self) -> tuple[MethodRecord, ...]:
        return tuple(self._methods.values())

    def artifacts(self) -> tuple[ArtifactLock, ...]:
        return tuple(self._artifacts.values())

    def source(self, identity: str) -> SourceRecord:
        try:
            return self._sources[identity]
        except KeyError:
            raise KeyError("research source not found") from None

    def method(self, identity: str) -> MethodRecord:
        try:
            return self._methods[identity]
        except KeyError:
            raise KeyError("research method not found") from None

    def artifact(self, identity: str) -> ArtifactLock:
        try:
            return self._artifacts[identity]
        except KeyError:
            raise KeyError("research artifact not found") from None

    def promotion(self, method_id: str) -> PromotionDecision:
        method = self.method(method_id).to_dict()
        blockers = []
        for source_id in method["source_record_ids"]:
            source = self.source(source_id).to_dict()
            if source["paper_digest"] is None or source["access"] not in {
                "primary_full_text",
                "primary_full_text_html",
                "primary_full_text_pdf",
            }:
                blockers.append(f"source:{source_id}:full_text_unlocked")
        for artifact_id in method["upstream_artifact_lock_ids"]:
            artifact = self.artifact(artifact_id)
            if artifact.status != "reproduction_verified":
                blockers.append(f"artifact:{artifact_id}:not_reproduced")
            if artifact.license_review == "required_before_execution":
                blockers.append(f"artifact:{artifact_id}:license_review")
        if method["implementation_status"] not in {"native_operator", "region", "full_model"}:
            blockers.append("method:native_implementation")
        if method["classification"] == "structural_pllm_adaptation":
            blockers.append("method:structural_only")
        lifecycle = method["lifecycle_status"]
        required = {
            "fidelity": {"passed"},
            "protected_execution": {"passed", "not_applicable"},
            "assurance": {"passed", "not_applicable"},
            "benchmark": {"passed"},
            "promotion": {"passed"},
        }
        for gate, accepted in required.items():
            if lifecycle[gate] not in accepted:
                blockers.append(f"gate:{gate}:{lifecycle[gate]}")
        if not method.get("eligible_components"):
            blockers.append("method:no_eligible_profile")
        blockers = tuple(sorted(set(blockers)))
        return PromotionDecision(method_id, not blockers, blockers)


__all__ = [
    "ArtifactLock",
    "MethodRecord",
    "PromotionDecision",
    "ResearchRegistry",
    "SourceRecord",
]

"""Immutable, inert access to canonical PLLM research metadata."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ._validation import (
    CATALOG_VERSION,
    SCHEMA_FILES,
    ResearchMetadataError,
    ResearchMetadataIOError,
    read_json,
    validate_document,
    validate_payload,
)

__all__ = [
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
]

from ._agents import render_agents_guide
from ._publication import PublicationAssessment, assess_publication


def _freeze(value: object) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Immutable publication/source metadata."""

    payload: Mapping[str, Any]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(dict(self.payload)))
        object.__setattr__(self, "aliases", tuple(self.aliases))

    @property
    def identifier(self) -> str:
        return str(self.payload["id"])

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.payload)


@dataclass(frozen=True, slots=True)
class MethodRecord:
    """Immutable method-semantics metadata."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(dict(self.payload)))

    @property
    def identifier(self) -> str:
        return str(self.payload["id"])

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(self.payload["registry_aliases"])

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.payload)


@dataclass(frozen=True, slots=True)
class RecipeRecord:
    """Immutable reproduction-workflow metadata; never an executable recipe."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(dict(self.payload)))

    @property
    def identifier(self) -> str:
        return str(self.payload.get("id", self.payload["registry_alias"]))

    @property
    def aliases(self) -> tuple[str, ...]:
        alias = str(self.payload["registry_alias"])
        return () if alias == self.identifier else (alias,)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.payload)


def _repository_root() -> Path | None:
    root = Path(__file__).resolve().parents[3]
    if (root / "research/methods/sources").is_dir() and (root / "schemas").is_dir():
        return root
    return None


def _paths(base: Path) -> tuple[Path, ...]:
    try:
        paths = tuple(sorted(base.glob("*.json")))
    except OSError:
        raise ResearchMetadataIOError("research metadata is unavailable") from None
    if not base.is_dir() or not paths:
        raise ResearchMetadataIOError("research metadata is unavailable")
    return paths


def _checkout_payload(root: Path) -> dict[str, Any]:
    schemas = {
        kind: read_json(root / "schemas" / filename) for kind, filename in SCHEMA_FILES.items()
    }
    locations = {
        "sources": root / "research/methods/sources",
        "methods": root / "research/methods/records",
        "recipes": root / "research/recipes",
    }
    records = {
        kind: [read_json(path, record=True) for path in _paths(location)]
        for kind, location in locations.items()
    }
    upstream_ids: list[str] = []
    for path in _paths(root / "research/methods/source-locks"):
        lock = read_json(path, record=True)
        if isinstance(lock, dict) and "schema_version" not in lock:
            continue
        if (
            not isinstance(lock, dict)
            or lock.get("schema_version") != "pllm.upstream_artifact_lock.v1"
            or type(lock.get("id")) is not str
            or not lock["id"]
        ):
            raise ResearchMetadataError("research upstream relationship metadata is invalid")
        upstream_ids.append(lock["id"])
    payload = {
        "catalog_version": CATALOG_VERSION,
        "records": records,
        "relationships": {"upstream_artifact_lock_ids": sorted(upstream_ids)},
        "schemas": schemas,
    }
    return validate_payload(payload)


def _bundled_payload() -> dict[str, Any]:
    document = read_json(Path(__file__).with_name("_catalog.json"))
    return validate_document(document)


@lru_cache(maxsize=1)
def _records() -> tuple[
    tuple[SourceRecord, ...], tuple[MethodRecord, ...], tuple[RecipeRecord, ...]
]:
    root = _repository_root()
    payload = _checkout_payload(root) if root is not None else _bundled_payload()
    raw = payload["records"]

    source_aliases: dict[str, list[str]] = {item["id"]: [] for item in raw["sources"]}
    for recipe in raw["recipes"]:
        source_id = recipe.get("source_record_id")
        alias = recipe["registry_alias"]
        if source_id is not None and alias != source_id:
            source_aliases[source_id].append(alias)
    sources = tuple(
        sorted(
            (
                SourceRecord(item, tuple(sorted(source_aliases[item["id"]])))
                for item in raw["sources"]
            ),
            key=lambda record: record.identifier,
        )
    )
    methods = tuple(
        sorted(
            (MethodRecord(item) for item in raw["methods"]), key=lambda record: record.identifier
        )
    )
    recipes = tuple(
        sorted(
            (RecipeRecord(item) for item in raw["recipes"]), key=lambda record: record.identifier
        )
    )
    return sources, methods, recipes


def list_sources() -> tuple[SourceRecord, ...]:
    """Return all source records in stable canonical-identity order."""
    return _records()[0]


def list_methods() -> tuple[MethodRecord, ...]:
    """Return all method records in stable canonical-identity order."""
    return _records()[1]


def list_recipes() -> tuple[RecipeRecord, ...]:
    """Return all inert recipe records in stable canonical-identity order."""
    return _records()[2]


def _get(identity: str, records: tuple[Any, ...], kind: str) -> Any:
    if type(identity) is not str:
        raise TypeError("research identity must be a string")
    matches = [
        record for record in records if identity == record.identifier or identity in record.aliases
    ]
    if len(matches) != 1:
        raise KeyError(f"research {kind} not found")
    return matches[0]


def get_source(identity: str) -> SourceRecord:
    """Resolve one source by canonical identity or unique registry alias."""
    return _get(identity, list_sources(), "source")


def get_method(identity: str) -> MethodRecord:
    """Resolve one method by canonical identity or unique registry alias."""
    return _get(identity, list_methods(), "method")


def get_recipe(identity: str) -> RecipeRecord:
    """Resolve one inert recipe by canonical identity or unique registry alias."""
    return _get(identity, list_recipes(), "recipe")

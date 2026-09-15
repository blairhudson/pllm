"""Strict, dependency-free validation for read-only research catalog data."""

from __future__ import annotations

import json
import math
import re
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

CATALOG_VERSION = "pllm.research_catalog.v1"
SCHEMA_VERSIONS = {
    "sources": "pllm.source_record.v1",
    "methods": "pllm.method_record.v1",
    "recipes": "pllm.reproduction_recipe.v1",
}
SCHEMA_FILES = {
    "sources": "source-record.schema.json",
    "methods": "method-record.schema.json",
    "recipes": "reproduction-recipe.schema.json",
}
MAX_CATALOG_BYTES = 4_194_304
MAX_RECORD_BYTES = 1_048_576
MAX_RECORDS_PER_KIND = 10_000
MAX_CONTAINER_ITEMS = 10_000
MAX_DEPTH = 32
MAX_STRING_CHARS = 1_048_576


class ResearchMetadataError(ValueError):
    """Research metadata is malformed or internally inconsistent."""


class ResearchMetadataIOError(OSError):
    """Research metadata could not be read safely."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResearchMetadataError("research metadata contains a duplicate field")
        result[key] = value
    return result


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ResearchMetadataError("research metadata contains a non-finite number")
    return parsed


def _constant(_value: str) -> object:
    raise ResearchMetadataError("research metadata contains a non-finite number")


def decode_json(data: bytes) -> Any:
    if len(data) > MAX_CATALOG_BYTES:
        raise ResearchMetadataError("research metadata exceeds size limit")
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_float,
            parse_constant=_constant,
        )
    except ResearchMetadataError:
        raise
    except (UnicodeError, json.JSONDecodeError):
        raise ResearchMetadataError("research metadata is not valid JSON") from None
    validate_json_value(value)
    return value


def read_json(path: Path, *, record: bool = False) -> Any:
    try:
        data = path.read_bytes()
        if len(data) > (MAX_RECORD_BYTES if record else MAX_CATALOG_BYTES):
            raise ResearchMetadataError("research metadata exceeds size limit")
        return decode_json(data)
    except ResearchMetadataError:
        raise
    except OSError:
        raise ResearchMetadataIOError("research metadata is unavailable") from None


def validate_json_value(value: object, *, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise ResearchMetadataError("research metadata exceeds nesting limit")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ResearchMetadataError("research metadata contains a non-finite number")
        return
    if type(value) is str:
        if len(value) > MAX_STRING_CHARS:
            raise ResearchMetadataError("research metadata string exceeds size limit")
        return
    if isinstance(value, list):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ResearchMetadataError("research metadata container exceeds size limit")
        for item in value:
            validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ResearchMetadataError("research metadata container exceeds size limit")
        for key, item in value.items():
            if type(key) is not str:
                raise ResearchMetadataError("research metadata keys must be strings")
            validate_json_value(key, depth=depth + 1)
            validate_json_value(item, depth=depth + 1)
        return
    raise ResearchMetadataError("research metadata contains a non-JSON value")


def canonical_bytes(value: object) -> bytes:
    validate_json_value(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ResearchMetadataError("research metadata cannot be serialized") from None


def payload_digest(payload: object) -> str:
    return sha256(canonical_bytes(payload)).hexdigest()


def _matches_type(value: object, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": type(value) is str,
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "null": value is None,
    }.get(expected, False)


def _validate_schema_value(value: object, schema: object, *, depth: int = 0) -> None:
    if depth > MAX_DEPTH or not isinstance(schema, dict):
        raise ResearchMetadataError("research metadata schema is invalid")
    expected = schema.get("type")
    if isinstance(expected, str):
        accepted = _matches_type(value, expected)
    elif isinstance(expected, list) and expected:
        accepted = all(type(item) is str for item in expected) and any(
            _matches_type(value, item) for item in expected
        )
    else:
        accepted = True
    if not accepted:
        raise ResearchMetadataError("research metadata does not match its schema")
    if "const" in schema and value != schema["const"]:
        raise ResearchMetadataError("research metadata has an unexpected schema version")
    choices = schema.get("enum")
    if isinstance(choices, list) and value not in choices:
        raise ResearchMetadataError("research metadata does not match its schema")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ResearchMetadataError("research metadata schema is invalid")
        if any(type(field) is not str for field in required) or not set(required) <= set(value):
            raise ResearchMetadataError("research metadata is missing a required field")
        extra = set(value) - set(properties)
        additional = schema.get("additionalProperties", True)
        if extra and additional is False:
            raise ResearchMetadataError("research metadata has an unsupported field")
        for key in set(value) & set(properties):
            _validate_schema_value(value[key], properties[key], depth=depth + 1)
        if isinstance(additional, dict):
            for key in extra:
                _validate_schema_value(value[key], additional, depth=depth + 1)
    elif isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ResearchMetadataError("research metadata array is too short")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ResearchMetadataError("research metadata array is too long")
        if schema.get("uniqueItems") is True:
            encoded = [canonical_bytes(item) for item in value]
            if len(encoded) != len(set(encoded)):
                raise ResearchMetadataError("research metadata array contains duplicates")
        if "items" in schema:
            for item in value:
                _validate_schema_value(item, schema["items"], depth=depth + 1)
    elif type(value) is str:
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ResearchMetadataError("research metadata string is too short")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ResearchMetadataError("research metadata string is too long")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise ResearchMetadataError("research metadata string has an invalid format")
        if schema.get("format") == "date":
            try:
                date.fromisoformat(value)
            except ValueError:
                raise ResearchMetadataError("research metadata date is invalid") from None
        if schema.get("format") == "uri":
            parsed = urlsplit(value)
            if not parsed.scheme or (parsed.scheme in {"http", "https"} and not parsed.netloc):
                raise ResearchMetadataError("research metadata URI is invalid")
    elif type(value) in (int, float):
        numeric_value = cast(int | float, value)
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and numeric_value < minimum:
            raise ResearchMetadataError("research metadata number is below its minimum")
        if isinstance(maximum, (int, float)) and numeric_value > maximum:
            raise ResearchMetadataError("research metadata number exceeds its maximum")


def _validate_schema(schema: object, kind: str) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ResearchMetadataError("research metadata schema is invalid")
    expected_id = f"https://pllm.dev/schemas/{SCHEMA_FILES[kind]}"
    if (
        schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("$id") != expected_id
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        raise ResearchMetadataError("research metadata schema is invalid")
    return schema


def validate_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "catalog_version",
        "records",
        "relationships",
        "schemas",
    }:
        raise ResearchMetadataError("research catalog has an invalid structure")
    if payload.get("catalog_version") != CATALOG_VERSION:
        raise ResearchMetadataError("research catalog has an unsupported version")
    records = payload.get("records")
    schemas = payload.get("schemas")
    relationships = payload.get("relationships")
    if not isinstance(records, dict) or set(records) != set(SCHEMA_VERSIONS):
        raise ResearchMetadataError("research catalog records are invalid")
    if not isinstance(schemas, dict) or set(schemas) != set(SCHEMA_VERSIONS):
        raise ResearchMetadataError("research catalog schemas are invalid")
    if not isinstance(relationships, dict) or set(relationships) != {"upstream_artifact_lock_ids"}:
        raise ResearchMetadataError("research catalog relationships are invalid")

    validated_schemas = {kind: _validate_schema(schemas[kind], kind) for kind in SCHEMA_VERSIONS}
    for kind, expected_version in SCHEMA_VERSIONS.items():
        items = records[kind]
        if not isinstance(items, list) or len(items) > MAX_RECORDS_PER_KIND:
            raise ResearchMetadataError("research catalog record count is invalid")
        for item in items:
            if not isinstance(item, dict) or item.get("schema_version") != expected_version:
                raise ResearchMetadataError("research metadata has an unexpected schema version")
            if len(canonical_bytes(item)) > MAX_RECORD_BYTES:
                raise ResearchMetadataError("research metadata record exceeds size limit")
            _validate_schema_value(item, validated_schemas[kind])

    upstream_ids = relationships["upstream_artifact_lock_ids"]
    if (
        not isinstance(upstream_ids, list)
        or len(upstream_ids) > MAX_RECORDS_PER_KIND
        or any(type(item) is not str or not item for item in upstream_ids)
        or upstream_ids != sorted(set(upstream_ids))
    ):
        raise ResearchMetadataError("research catalog relationships are invalid")
    _validate_relationships(records, set(upstream_ids))
    return payload


def _identities(records: list[dict[str, Any]], kind: str) -> tuple[set[str], dict[str, str]]:
    identifiers: set[str] = set()
    aliases: dict[str, str] = {}
    for record in records:
        identifier = record.get("id")
        if kind == "recipes" and identifier is None:
            identifier = record.get("registry_alias")
        if type(identifier) is not str or not identifier or identifier in identifiers:
            raise ResearchMetadataError("research metadata contains a duplicate record identity")
        identifiers.add(identifier)
        raw_aliases = (
            record.get("registry_aliases", [])
            if kind == "methods"
            else [record.get("registry_alias")]
            if kind == "recipes"
            else []
        )
        for alias in raw_aliases:
            if alias == identifier:
                continue
            if type(alias) is not str or not alias or alias in aliases:
                raise ResearchMetadataError("research metadata contains a duplicate alias")
            aliases[alias] = identifier
    for alias, identifier in aliases.items():
        if alias in identifiers and alias != identifier:
            raise ResearchMetadataError("research metadata contains an ambiguous alias")
    return identifiers, aliases


def _validate_relationships(
    records: dict[str, list[dict[str, Any]]], upstream_ids: set[str]
) -> None:
    source_ids, source_aliases = _identities(records["sources"], "sources")
    method_ids, _ = _identities(records["methods"], "methods")
    recipe_ids, _ = _identities(records["recipes"], "recipes")
    recipe_aliases = {record["registry_alias"] for record in records["recipes"]}
    if len(recipe_aliases) != len(records["recipes"]):
        raise ResearchMetadataError("research metadata contains a duplicate alias")

    for method in records["methods"]:
        if not set(method["source_record_ids"]) <= source_ids:
            raise ResearchMetadataError("research method has a missing source relationship")
        if not set(method["upstream_artifact_lock_ids"]) <= upstream_ids:
            raise ResearchMetadataError("research method has a missing upstream relationship")
    for recipe in records["recipes"]:
        if not set(recipe["dependencies"]) <= recipe_aliases:
            raise ResearchMetadataError("research recipe has a missing dependency relationship")
        source_id = recipe.get("source_record_id")
        if source_id is not None and source_id not in source_ids:
            raise ResearchMetadataError("research recipe has a missing source relationship")
        if recipe["source_acquisition_status"] == "acquired":
            implied_source = source_id or recipe["registry_alias"]
            if implied_source not in source_ids:
                raise ResearchMetadataError("research recipe has a missing source relationship")
        method_id = recipe.get("method_record_id")
        if method_id is not None and method_id not in method_ids:
            raise ResearchMetadataError("research recipe has a missing method relationship")
        upstream_id = recipe.get("upstream_artifact_lock_id")
        if upstream_id is not None and upstream_id not in upstream_ids:
            raise ResearchMetadataError("research recipe has a missing upstream relationship")
        if source_id is not None:
            alias = recipe["registry_alias"]
            existing = source_aliases.get(alias)
            if existing is not None and existing != source_id:
                raise ResearchMetadataError("research metadata contains an ambiguous alias")
            if alias in source_ids and alias != source_id:
                raise ResearchMetadataError("research metadata contains an ambiguous alias")
            source_aliases[alias] = source_id

    if len(recipe_ids) != len(records["recipes"]):
        raise ResearchMetadataError("research metadata contains a duplicate record identity")


def validate_document(document: object) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != {"_generated", "payload"}:
        raise ResearchMetadataError("research catalog has an invalid generated header")
    header = document.get("_generated")
    payload = document.get("payload")
    if not isinstance(header, dict) or set(header) != {
        "catalog_version",
        "generator",
        "notice",
        "payload_sha256",
    }:
        raise ResearchMetadataError("research catalog has an invalid generated header")
    if (
        header.get("catalog_version") != CATALOG_VERSION
        or header.get("generator") != "scripts/generate_research_catalog.py"
        or header.get("notice") != "Generated file; do not edit."
        or header.get("payload_sha256") != payload_digest(payload)
    ):
        raise ResearchMetadataError("research catalog integrity check failed")
    return validate_payload(payload)

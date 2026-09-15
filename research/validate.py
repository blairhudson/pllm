#!/usr/bin/env python3
"""Dependency-free integrity checks for canonical design, schemas, and research records."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / "research" / "methods"
RECIPES = ROOT / "research" / "recipes"
ASSURANCE = ROOT / "research" / "assurance"
COMPONENTS = ROOT / "research" / "components"
SCHEMAS = ROOT / "schemas"
OUTCOMES = {
    "proved_in_model",
    "refuted_in_scope",
    "not_refuted",
    "inconclusive",
    "outside_contract",
    "unchecked",
}
REPRODUCTION_GATES = [
    "acquire",
    "specify",
    "reference",
    "fidelity",
    "native",
    "compose",
    "assure",
    "benchmark",
    "document",
]


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{path.relative_to(ROOT)}: invalid JSON: {exc}")
        return None


def check_json_and_schemas(errors: list[str]) -> None:
    roots = [SCHEMAS, METHODS, RECIPES, ASSURANCE, COMPONENTS]
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            load_json(path, errors)

    ids: dict[str, Path] = {}
    for path in sorted(SCHEMAS.glob("*.schema.json")):
        schema = load_json(path, errors)
        if not isinstance(schema, dict):
            continue
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append(f"{path.relative_to(ROOT)}: must use draft 2020-12")
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id:
            errors.append(f"{path.relative_to(ROOT)}: missing $id")
        elif schema_id in ids:
            errors.append(f"{path.relative_to(ROOT)}: duplicate $id also in {ids[schema_id]}")
        else:
            ids[schema_id] = path
        expected_id = f"https://pllm.dev/schemas/{path.name}"
        if schema_id != expected_id:
            errors.append(f"{path.relative_to(ROOT)}: $id must be {expected_id}")

        def check_strict_objects(value: object, location: str) -> None:
            if isinstance(value, dict):
                if value.get("type") == "object" and value.get("additionalProperties") is not False:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{location}: object must set additionalProperties false"
                    )
                for key, nested in value.items():
                    check_strict_objects(nested, f"{location}/{key}")
            elif isinstance(value, list):
                for index, nested in enumerate(value):
                    check_strict_objects(nested, f"{location}/{index}")

        if path.name in {
            "logical-plan.schema.json",
            "execution-plan.schema.json",
            "plan-lock.schema.json",
            "assurance-result.schema.json",
            "privacy-contract.schema.json",
            "locked-context.schema.json",
            "region-program.schema.json",
            "compile-request.schema.json",
        }:
            check_strict_objects(schema, "")

        def check_refs(value: object) -> None:
            if isinstance(value, dict):
                reference = value.get("$ref")
                if isinstance(reference, str) and not reference.startswith(("#", "http://", "https://")):
                    reference_path = reference.split("#", 1)[0]
                    if not (path.parent / reference_path).is_file():
                        errors.append(f"{path.relative_to(ROOT)}: missing schema reference {reference}")
                for nested in value.values():
                    check_refs(nested)
            elif isinstance(value, list):
                for nested in value:
                    check_refs(nested)

        check_refs(schema)

    assurance_schema = load_json(SCHEMAS / "assurance-result.schema.json", errors)
    if isinstance(assurance_schema, dict):
        actual = set(assurance_schema["properties"]["outcome"]["enum"])
        if actual != OUTCOMES:
            errors.append("schemas/assurance-result.schema.json: outcome vocabulary changed")


def schema_fixture_errors(value: object, schema: dict[str, object], path: str) -> list[str]:
    errors: list[str] = []
    external_schemas: dict[str, dict[str, object]] = {}

    def resolve(
        node: dict[str, object], owner: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        reference = node.get("$ref")
        if not isinstance(reference, str):
            return node, owner
        target_owner = owner
        fragment = reference
        if not reference.startswith("#"):
            filename, _, suffix = reference.partition("#")
            if filename not in external_schemas:
                external_path = SCHEMAS / filename
                try:
                    loaded = json.loads(external_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    errors.append(f"{path}: invalid external schema {filename}: {exc}")
                    return {}, owner
                if not isinstance(loaded, dict):
                    errors.append(f"{path}: external schema {filename} is not an object")
                    return {}, owner
                external_schemas[filename] = loaded
            target_owner = external_schemas[filename]
            fragment = f"#{suffix}"
        if fragment == "#" or fragment == "":
            return target_owner, target_owner
        if not fragment.startswith("#/"):
            errors.append(f"{path}: unsupported schema reference {reference}")
            return {}, owner
        target: object = target_owner
        for component in fragment[2:].split("/"):
            if not isinstance(target, dict) or component not in target:
                errors.append(f"{path}: unresolved local reference {reference}")
                return {}, owner
            target = target[component]
        if not isinstance(target, dict):
            errors.append(f"{path}: reference {reference} is not a schema object")
            return {}, owner
        return target, target_owner

    def matches_type(item: object, expected: str) -> bool:
        return {
            "object": isinstance(item, dict),
            "array": isinstance(item, list),
            "string": type(item) is str,
            "integer": type(item) is int,
            "number": type(item) in (int, float),
            "boolean": type(item) is bool,
            "null": item is None,
        }.get(expected, True)

    def check(
        item: object,
        raw_node: object,
        location: str,
        owner: dict[str, object],
    ) -> None:
        if not isinstance(raw_node, dict):
            errors.append(f"{path}:{location}: malformed schema node")
            return
        node, node_owner = resolve(raw_node, owner)
        if "const" in node and item != node["const"]:
            errors.append(f"{path}:{location}: expected constant {node['const']!r}")
        choices = node.get("enum")
        if isinstance(choices, list) and item not in choices:
            errors.append(f"{path}:{location}: value {item!r} is outside enum")
        expected_type = node.get("type")
        if isinstance(expected_type, str) and not matches_type(item, expected_type):
            errors.append(f"{path}:{location}: expected {expected_type}")
            return
        if isinstance(expected_type, list) and not any(
            isinstance(candidate, str) and matches_type(item, candidate)
            for candidate in expected_type
        ):
            errors.append(f"{path}:{location}: type is not one of {expected_type}")
            return
        if isinstance(item, dict):
            properties = node.get("properties", {})
            required = node.get("required", [])
            if not isinstance(properties, dict) or not isinstance(required, list):
                errors.append(f"{path}:{location}: malformed object schema")
                return
            missing = set(required) - set(item)
            extra = set(item) - set(properties)
            additional = node.get("additionalProperties")
            if missing:
                errors.append(f"{path}:{location}: missing fields {sorted(missing)}")
            if extra and additional is False:
                errors.append(f"{path}:{location}: unallowed fields {sorted(extra)}")
            for key in sorted(set(item) & set(properties)):
                check(item[key], properties[key], f"{location}.{key}", node_owner)
            if isinstance(additional, dict):
                for key in sorted(extra):
                    check(item[key], additional, f"{location}.{key}", node_owner)
        elif isinstance(item, list):
            minimum = node.get("minItems")
            if isinstance(minimum, int) and len(item) < minimum:
                errors.append(f"{path}:{location}: fewer than {minimum} items")
            if node.get("uniqueItems") is True:
                encoded = [json.dumps(entry, sort_keys=True) for entry in item]
                if len(encoded) != len(set(encoded)):
                    errors.append(f"{path}:{location}: items are not unique")
            if "items" in node:
                for index, nested in enumerate(item):
                    check(nested, node["items"], f"{location}[{index}]", node_owner)
        elif type(item) is str:
            minimum = node.get("minLength")
            if isinstance(minimum, int) and len(item) < minimum:
                errors.append(f"{path}:{location}: string is too short")
            pattern = node.get("pattern")
            if isinstance(pattern, str) and re.fullmatch(pattern, item) is None:
                errors.append(f"{path}:{location}: string does not match {pattern}")
        elif type(item) in (int, float):
            assert isinstance(item, (int, float))
            minimum = node.get("minimum")
            maximum = node.get("maximum")
            if isinstance(minimum, (int, float)) and item < minimum:
                errors.append(f"{path}:{location}: value is below minimum")
            if isinstance(maximum, (int, float)) and item > maximum:
                errors.append(f"{path}:{location}: value is above maximum")

    check(value, schema, "$", schema)
    return errors


def check_canonical_fixtures(errors: list[str]) -> None:
    pairs = {
        "logical-plan.valid.json": "logical-plan.schema.json",
        "execution-plan.valid.json": "execution-plan.schema.json",
        "plan-lock.valid.json": "plan-lock.schema.json",
        "assurance-result.valid.json": "assurance-result.schema.json",
        "privacy-contract.valid.json": "privacy-contract.schema.json",
        "locked-context.valid.json": "locked-context.schema.json",
        "region-program.valid.json": "region-program.schema.json",
        "compile-request.valid.json": "compile-request.schema.json",
        "reproduction-recipe.valid.json": "reproduction-recipe.schema.json",
    }
    fixture_dir = SCHEMAS / "fixtures"
    for fixture_name, schema_name in pairs.items():
        fixture_path = fixture_dir / fixture_name
        schema_path = SCHEMAS / schema_name
        fixture = load_json(fixture_path, errors)
        schema = load_json(schema_path, errors)
        if fixture is None or not isinstance(schema, dict):
            continue
        errors.extend(
            schema_fixture_errors(fixture, schema, str(fixture_path.relative_to(ROOT)))
        )

    recipe_schema = load_json(SCHEMAS / "reproduction-recipe.schema.json", errors)
    if isinstance(recipe_schema, dict):
        for path in sorted(fixture_dir.glob("reproduction-recipe.invalid-*.json")):
            fixture = load_json(path, errors)
            if fixture is not None and not schema_fixture_errors(
                fixture, recipe_schema, str(path.relative_to(ROOT))
            ):
                errors.append(f"{path.relative_to(ROOT)}: invalid fixture unexpectedly validates")


def experiment_errors(value: object, path: str = "experiment") -> list[str]:
    errors: list[str] = []

    def mapping(item: object, expected: set[str], location: str) -> dict[str, object] | None:
        if not isinstance(item, dict):
            errors.append(f"{location} must be an object")
            return None
        actual = set(item)
        if actual != expected:
            errors.append(f"{location} fields differ: {sorted(actual ^ expected)}")
        return item

    def text(item: object, location: str) -> None:
        if type(item) is not str or not item:
            errors.append(f"{location} must be a non-empty string")

    def positive_integer(item: object, location: str) -> None:
        if type(item) is not int or item < 1:
            errors.append(f"{location} must be a positive integer")

    def json_value(item: object, location: str) -> None:
        if item is None or type(item) in (bool, str, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                errors.append(f"{location} must be finite")
            return
        if isinstance(item, list):
            for index, nested in enumerate(item):
                json_value(nested, f"{location}[{index}]")
            return
        if isinstance(item, dict):
            for key, nested in item.items():
                if type(key) is not str or not key or "__" in key:
                    errors.append(f"{location} has invalid key {key!r}")
                json_value(nested, f"{location}.{key}")
            return
        errors.append(f"{location} is not JSON-safe")

    root = mapping(value, {"schema", "name", "pipeline", "deployment", "budget"}, path)
    if root is None:
        return errors
    if root.get("schema") != "pllm.experiment.v1":
        errors.append(f"{path}.schema must be pllm.experiment.v1")
    text(root.get("name"), f"{path}.name")

    pipeline = mapping(root.get("pipeline"), {"profile", "model", "components"}, "pipeline")
    if pipeline is not None:
        text(pipeline.get("profile"), "pipeline.profile")
        model = mapping(pipeline.get("model"), {"source"}, "pipeline.model")
        if model is not None:
            text(model.get("source"), "pipeline.model.source")
        components = pipeline.get("components")
        if not isinstance(components, dict):
            errors.append("pipeline.components must be an object")
        else:
            for name, raw_component in components.items():
                if type(name) is not str or not name or "__" in name:
                    errors.append(f"pipeline.components has invalid key {name!r}")
                component = mapping(
                    raw_component, {"component", "params"}, f"pipeline.components.{name}"
                )
                if component is None:
                    continue
                component_id = component.get("component")
                text(component_id, f"pipeline.components.{name}.component")
                params = component.get("params")
                if not isinstance(params, dict):
                    errors.append(f"pipeline.components.{name}.params must be an object")
                    continue
                if "component" in params or "params" in params:
                    errors.append(f"pipeline.components.{name}.params uses reserved key")
                json_value(params, f"pipeline.components.{name}.params")
                if component_id == "pllm/cpu":
                    if set(params) != {"threads"}:
                        errors.append(f"pipeline.components.{name}.params must contain threads only")
                    positive_integer(params.get("threads"), f"pipeline.components.{name}.params.threads")
                elif component_id in {"pllm/masked-linear", "pllm/model-aware-corrections"} and params:
                    errors.append(f"pipeline.components.{name}.params must be empty")

    deployment = mapping(root.get("deployment"), {"kind", "root"}, "deployment")
    if deployment is not None:
        if deployment.get("kind") != "local":
            errors.append("deployment.kind must be local")
        text(deployment.get("root"), "deployment.root")

    budget = mapping(
        root.get("budget"), {"requests", "max_input_tokens", "max_new_tokens"}, "budget"
    )
    if budget is not None:
        for field in ("requests", "max_input_tokens", "max_new_tokens"):
            positive_integer(budget.get(field), f"budget.{field}")
    return errors


def check_experiment_fixtures(errors: list[str]) -> None:
    fixture_dir = SCHEMAS / "fixtures"
    valid = load_json(fixture_dir / "experiment.valid.json", errors)
    if valid is not None:
        for error in experiment_errors(valid):
            errors.append(f"schemas/fixtures/experiment.valid.json: {error}")
    for path in sorted(fixture_dir.glob("experiment.invalid-*.json")):
        fixture = load_json(path, errors)
        if fixture is not None and not experiment_errors(fixture):
            errors.append(f"{path.relative_to(ROOT)}: invalid fixture unexpectedly validates")


def check_portfolio(errors: list[str]) -> None:
    source_schema = load_json(SCHEMAS / "source-record.schema.json", errors)
    recipe_schema = load_json(SCHEMAS / "reproduction-recipe.schema.json", errors)
    upstream_schema = load_json(SCHEMAS / "upstream-artifact-lock.schema.json", errors)
    method_schema = load_json(SCHEMAS / "method-record.schema.json", errors)

    source_records: dict[str, dict[str, object]] = {}
    for path in sorted((METHODS / "sources").glob("*.json")):
        source = load_json(path, errors)
        if not isinstance(source, dict):
            continue
        if isinstance(source_schema, dict):
            errors.extend(schema_fixture_errors(source, source_schema, str(path.relative_to(ROOT))))
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id:
            continue
        if source_id in source_records:
            errors.append(f"{path.relative_to(ROOT)}: duplicate source record ID {source_id}")
        else:
            source_records[source_id] = source

    upstream_records: dict[str, dict[str, object]] = {}
    for path in sorted((METHODS / "source-locks").glob("*.json")):
        upstream = load_json(path, errors)
        if not isinstance(upstream, dict) or upstream.get("schema_version") != (
            "pllm.upstream_artifact_lock.v1"
        ):
            continue
        if isinstance(upstream_schema, dict):
            errors.extend(
                schema_fixture_errors(upstream, upstream_schema, str(path.relative_to(ROOT)))
            )
        upstream_id = upstream.get("id")
        if not isinstance(upstream_id, str) or not upstream_id:
            continue
        if upstream_id in upstream_records:
            errors.append(f"{path.relative_to(ROOT)}: duplicate upstream lock ID {upstream_id}")
        else:
            upstream_records[upstream_id] = upstream
        unknown_sources = set(upstream.get("source_record_ids", [])) - set(source_records)
        if unknown_sources:
            errors.append(
                f"{path.relative_to(ROOT)}: unknown source records {sorted(unknown_sources)}"
            )

    method_records: dict[str, dict[str, object]] = {}
    for path in sorted((METHODS / "records").glob("*.json")):
        method = load_json(path, errors)
        if not isinstance(method, dict):
            continue
        if isinstance(method_schema, dict):
            errors.extend(schema_fixture_errors(method, method_schema, str(path.relative_to(ROOT))))
        method_id = method.get("id")
        if not isinstance(method_id, str) or not method_id:
            continue
        if method_id in method_records:
            errors.append(f"{path.relative_to(ROOT)}: duplicate method record ID {method_id}")
        else:
            method_records[method_id] = method
        unknown_sources = set(method.get("source_record_ids", [])) - set(source_records)
        if unknown_sources:
            errors.append(
                f"{path.relative_to(ROOT)}: unknown source records {sorted(unknown_sources)}"
            )
        unknown_upstream = set(method.get("upstream_artifact_lock_ids", [])) - set(
            upstream_records
        )
        if unknown_upstream:
            errors.append(
                f"{path.relative_to(ROOT)}: unknown upstream locks {sorted(unknown_upstream)}"
            )

    registry = load_json(METHODS / "registry.json", errors)
    if not isinstance(registry, dict) or not isinstance(registry.get("papers"), list):
        return
    papers = registry["papers"]
    if not papers:
        errors.append("research/methods/registry.json: expected at least one method")
    expected = [f"R{index:02d}" for index in range(1, len(papers) + 1)]
    actual = [paper.get("id") if isinstance(paper, dict) else None for paper in papers]
    if actual != expected:
        errors.append(f"research/methods/registry.json: expected ordered IDs {expected}, got {actual}")
    known = set(expected)

    for method_id, method in method_records.items():
        aliases = method.get("registry_aliases", [])
        if not isinstance(aliases, list):
            continue
        unknown_aliases = set(aliases) - known
        if unknown_aliases:
            errors.append(f"{method_id}: unknown registry aliases {sorted(unknown_aliases)}")

    for paper in papers:
        if not isinstance(paper, dict):
            errors.append("research/methods/registry.json: paper entry must be an object")
            continue
        paper_id = paper.get("id")
        if not isinstance(paper_id, str):
            errors.append("research/methods/registry.json: paper ID must be a string")
            continue
        card = METHODS / str(paper.get("card", ""))
        if not card.is_file():
            errors.append(f"{paper_id}: missing card {card.relative_to(ROOT)}")
        if any(dependency not in known for dependency in paper.get("depends_on", [])):
            errors.append(f"{paper_id}: unknown dependency")
        lock = paper.get("source_lock")
        distinct_ids = {
            "source_record_id": paper.get("source_record_id"),
            "upstream_artifact_lock_id": paper.get("upstream_artifact_lock_id"),
            "method_record_id": paper.get("method_record_id"),
        }
        if lock is not None:
            if not isinstance(lock, dict) or set(lock) != {
                "paper_sha256",
                "artifact_commit",
                "license_review",
            }:
                errors.append(f"{paper_id}: malformed source_lock")
        elif not all(isinstance(value, str) and value for value in distinct_ids.values()):
            errors.append(f"{paper_id}: missing source lock or distinct record identities")
        for field, records in (
            ("source_record_id", source_records),
            ("upstream_artifact_lock_id", upstream_records),
            ("method_record_id", method_records),
        ):
            record_id = paper.get(field)
            if record_id is not None and record_id not in records:
                errors.append(f"{paper_id}: unknown {field} {record_id}")
            if record_id == paper_id:
                errors.append(f"{paper_id}: registry alias cannot be a {field}")
        if paper.get("fulltext_gate") and paper.get("implementation_status") != (
            "reproduction_target_not_implemented_in_this_bundle"
        ):
            errors.append(f"{paper_id}: gated source cannot claim implementation")

        recipe_path = RECIPES / f"{paper_id}.json"
        recipe = load_json(recipe_path, errors)
        if not isinstance(recipe, dict):
            continue
        comparisons = {
            "registry_alias": paper_id,
            "primary_url": paper.get("primary_url"),
            "dependencies": paper.get("depends_on"),
            "target_modules": paper.get("modules"),
        }
        if lock is not None:
            comparisons["source_lock"] = lock
        for field in ("source_record_id", "upstream_artifact_lock_id", "method_record_id"):
            if paper.get(field) is not None:
                comparisons[field] = paper.get(field)
        for key, expected_value in comparisons.items():
            if recipe.get(key) != expected_value:
                errors.append(f"{recipe_path.relative_to(ROOT)}: {key} differs from registry")
        if isinstance(recipe_schema, dict):
            errors.extend(
                schema_fixture_errors(recipe, recipe_schema, str(recipe_path.relative_to(ROOT)))
            )
        acquisition = recipe.get("source_acquisition_status")
        workflow = recipe.get("workflow_status")
        if recipe.get("kind") != "planned_workflow" or workflow not in {
            "not_executed",
            "in_progress",
            "blocked",
            "failed",
            "completed",
        }:
            errors.append(f"{recipe_path.relative_to(ROOT)}: invalid workflow status")
        if recipe.get("gates") != REPRODUCTION_GATES:
            errors.append(f"{recipe_path.relative_to(ROOT)}: lifecycle gates differ from standard")
        if paper.get("source_acquisition_status") is not None and acquisition != paper.get(
            "source_acquisition_status"
        ):
            errors.append(
                f"{recipe_path.relative_to(ROOT)}: source acquisition differs from registry"
            )
        if paper.get("workflow_status") is not None and workflow != paper.get("workflow_status"):
            errors.append(f"{recipe_path.relative_to(ROOT)}: workflow status differs from registry")
        if acquisition == "acquired":
            source_id = paper.get("source_record_id", paper_id)
            source = source_records.get(source_id)
            if source is None:
                errors.append(f"{recipe_path.relative_to(ROOT)}: acquired source has no source record")
                continue
            expected_source = {
                "id": source_id,
                "primary_url": paper.get("primary_url"),
                "access": paper.get("source_access"),
            }
            if isinstance(lock, dict):
                if any(lock.get(field) is None for field in ("paper_sha256", "artifact_commit")):
                    errors.append(f"{recipe_path.relative_to(ROOT)}: acquired source is not locked")
                expected_source.update(
                    {
                        "paper_digest": lock.get("paper_sha256"),
                        "artifact_commit": lock.get("artifact_commit"),
                        "license_review": lock.get("license_review"),
                    }
                )
            for field, expected_value in expected_source.items():
                if source.get(field) != expected_value:
                    errors.append(f"{source_id}: {field} differs from registry")

    recipe_ids = {path.stem for path in RECIPES.glob("R[0-9][0-9].json")}
    if recipe_ids != known:
        errors.append(f"research/recipes: expected {sorted(known)}, got {sorted(recipe_ids)}")

    bibliography = (METHODS / "references.bib").read_text(encoding="utf-8")
    missing_citations = [alias for alias in expected if f"{{pllm_{alias.lower()}," not in bibliography]
    if missing_citations:
        errors.append(
            f"research/methods/references.bib: missing registry citations {missing_citations}"
        )


def check_assurance(errors: list[str]) -> None:
    manifest_path = ASSURANCE / "formal" / "manifest.json"
    manifest = load_json(manifest_path, errors)
    if isinstance(manifest, dict):
        checks = manifest.get("checks", [])
        if len(checks) != 10:
            errors.append("research/assurance/formal/manifest.json: expected ten checks")
        for check in checks:
            target = manifest_path.parent / str(check.get("path", ""))
            if not target.is_file() or target.parent != manifest_path.parent:
                errors.append(f"formal check {check.get('id')}: missing local SMT target")
            if check.get("expected") not in {"sat", "unsat"}:
                errors.append(f"formal check {check.get('id')}: invalid expectation")

    obligations = load_json(ASSURANCE / "obligations.json", errors)
    registry = load_json(METHODS / "registry.json", errors)
    if isinstance(obligations, dict) and isinstance(registry, dict):
        papers = registry.get("papers", [])
        items = obligations.get("obligations", [])
        if not isinstance(papers, list) or not isinstance(items, list):
            return
        expected: set[str] = set()
        for paper in papers:
            if not isinstance(paper, dict) or paper.get("assurance_status") == "not_applicable":
                continue
            alias = paper.get("id")
            if isinstance(alias, str):
                expected.add(alias)
        aliases: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            alias = item.get("registry_alias")
            if isinstance(alias, str):
                aliases.append(alias)
        actual = set(aliases)
        if len(aliases) != len(actual):
            errors.append("research/assurance/obligations.json: duplicate registry alias")
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected, key=str)
            errors.append(
                "research/assurance/obligations.json: registry coverage differs; "
                f"missing {missing}, extra {extra}"
            )


def check_components(errors: list[str]) -> None:
    schema = load_json(SCHEMAS / "component-descriptor.schema.json", errors)
    source_ids: set[str] = set()
    for path in (METHODS / "sources").glob("*.json"):
        source = load_json(path, errors)
        if isinstance(source, dict) and isinstance(source.get("id"), str):
            source_ids.add(source["id"])
    if not isinstance(schema, dict):
        return
    for path in sorted(COMPONENTS.glob("*.json")):
        component = load_json(path, errors)
        if not isinstance(component, dict):
            continue
        errors.extend(schema_fixture_errors(component, schema, str(path)))
        unknown = set(component.get("source_record_ids", [])) - source_ids
        if unknown:
            errors.append(f"{path.relative_to(ROOT)}: unknown source records {sorted(unknown)}")


def check_markdown_links(errors: list[str]) -> None:
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    roots = [ROOT / "design", METHODS, RECIPES, ASSURANCE, ROOT / "schemas"]
    for root in roots:
        for path in sorted(root.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            for raw_target in link_pattern.findall(text):
                target = raw_target.split("#", 1)[0].split("?", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    errors.append(f"{path.relative_to(ROOT)}: broken link {raw_target}")


def check_stale_references(errors: list[str]) -> None:
    roots = [ROOT / "design", SCHEMAS, METHODS, RECIPES, ASSURANCE]
    for root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.suffix not in {".md", ".json", ".py", ".smt2", ".bib"}:
                continue
            text = path.read_text(encoding="utf-8")
            for stale in ("PLLM_" + "DESIGN", "PLLM_" + "UPLIFT", "top20.json"):
                if stale in text:
                    errors.append(f"{path.relative_to(ROOT)}: stale reference {stale}")


def main() -> int:
    errors: list[str] = []
    check_json_and_schemas(errors)
    check_experiment_fixtures(errors)
    check_canonical_fixtures(errors)
    check_portfolio(errors)
    check_assurance(errors)
    check_components(errors)
    check_markdown_links(errors)
    check_stale_references(errors)
    if errors:
        print("integrity validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("integrity validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

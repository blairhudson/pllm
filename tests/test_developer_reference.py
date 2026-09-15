from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_developer_reference as reference  # noqa: E402


DOCS_ROOT = ROOT / "docs" / "content" / "docs"
SDK_DOC_BRANCHES = (
    "sdk",
    "build",
    "models",
    "operators",
    "numerics",
    "representations",
    "conversions",
    "search",
    "pipeline",
    "compiler",
    "runtime",
    "protocols",
    "preparation",
    "kernels",
    "measure",
    "benchmarks",
    "assurance",
    "operate",
    "deployment",
    "reference",
    "contribute",
    "agents",
)
SDK_DOC_EXCLUSIONS = (
    Path("build/research"),
    Path("measure/compare"),
    Path("measure/reproduce"),
    Path("reference/cli"),
    Path("reference/research"),
    Path("agents/research-map"),
    Path("agents/reproduction-checklist"),
)
PYTHON_FENCE = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)


def sdk_doc_pages() -> tuple[Path, ...]:
    pages = {
        page
        for branch in SDK_DOC_BRANCHES
        for page in (DOCS_ROOT / branch).rglob("*.mdx")
    }
    included = []
    for page in pages:
        stem = page.relative_to(DOCS_ROOT).with_suffix("")
        if any(stem == excluded or stem.is_relative_to(excluded) for excluded in SDK_DOC_EXCLUSIONS):
            continue
        included.append(page)
    return tuple(sorted(included))


def test_generated_developer_reference_is_fresh_and_deterministic() -> None:
    first = reference.render_outputs()
    second = reference.render_outputs()
    assert first == second
    assert all(path.read_text(encoding="utf-8") == content for path, content in first.items())
    assert all(reference.GENERATED_NOTICE in content for content in first.values())


def test_every_sdk_doc_page_has_syntax_valid_python_example() -> None:
    pages = sdk_doc_pages()
    assert pages
    missing = []
    invalid = []
    for page in pages:
        examples = PYTHON_FENCE.findall(page.read_text(encoding="utf-8"))
        if not examples:
            missing.append(str(page.relative_to(ROOT)))
            continue
        for index, example in enumerate(examples, start=1):
            try:
                ast.parse(example, filename=f"{page} example {index}")
            except SyntaxError as exc:
                invalid.append(f"{page.relative_to(ROOT)} example {index}: {exc}")
    assert not missing, "SDK documentation missing Python examples:\n" + "\n".join(missing)
    assert not invalid, "SDK documentation has invalid Python examples:\n" + "\n".join(invalid)


def test_complete_cli_help_has_exact_parser_parity() -> None:
    generated = reference.render_cli_help()
    for command, help_text in reference.cli_help_sections():
        assert f"$ {command} --help\n{help_text}" in generated
    assert "pllm run" not in generated
    assert "pllm serve" not in generated
    assert "pllm benchmark" not in generated


def test_api_inventory_uses_public_objects_once_and_keeps_alias_identity() -> None:
    exports = reference.public_exports()
    inventory = reference.api_inventory()
    assert sum(len(item["exports"]) for item in inventory) == len(exports)
    assert len({name for item in inventory for name in item["exports"]}) == len(exports)
    by_name = {f"{item['module']}.{item['name']}": item["value"] for item in exports}
    assert by_name["pllm.Experiment"] is by_name["pllm.config.Experiment"]
    assert by_name["pllm.ModelPlan"] is by_name["pllm.models.ModelPlan"]
    assert by_name["pllm.ComponentRef"] is by_name["pllm.components.ComponentRef"]


def test_component_and_research_catalogs_match_public_apis() -> None:
    from pllm.components import list_components
    from pllm.research import list_methods, list_recipes, list_sources

    assert reference.component_catalog() == tuple(item.to_dict() for item in list_components())
    assert reference.research_catalog() == {
        "sources": tuple(item.to_dict() for item in list_sources()),
        "methods": tuple(item.to_dict() for item in list_methods()),
        "recipes": tuple(item.to_dict() for item in list_recipes()),
    }

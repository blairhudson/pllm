from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

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
    Path("agents/research-map"),
    Path("agents/reproduction-checklist"),
)
PYTHON_FENCE = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)
PYTHON_EXAMPLE_HEADING = "## Python SDK example"
NO_PYTHON_API = "No public Python API"
API_LINK = re.compile(
    r"^API: .*\(/sdk/reference/python/pllm/#objects-and-signatures\)", re.MULTILINE
)


def sdk_doc_pages() -> tuple[Path, ...]:
    pages = {page for branch in SDK_DOC_BRANCHES for page in (DOCS_ROOT / branch).rglob("*.mdx")}
    included = []
    for page in pages:
        stem = page.relative_to(DOCS_ROOT).with_suffix("")
        if any(
            stem == excluded or stem.is_relative_to(excluded) for excluded in SDK_DOC_EXCLUSIONS
        ):
            continue
        included.append(page)
    return tuple(sorted(included))


def test_generated_developer_reference_is_fresh_and_deterministic() -> None:
    first = reference.render_outputs()
    second = reference.render_outputs()
    assert first == second
    # argparse help and Enum signatures differ across supported Python releases;
    # Python 3.13 is the canonical documentation generator used by Pages.
    if sys.version_info[:2] == (3, 13):
        assert all(path.read_text(encoding="utf-8") == content for path, content in first.items())
    assert all(content.strip() for content in first.values())
    assert all("do not edit" not in content.casefold() for content in first.values())


def test_cli_parser_help_is_split_exactly_across_generated_pages() -> None:
    sections = reference.cli_help_sections()
    parent_commands = {command.rsplit(" ", 1)[0] for command, _ in sections if " " in command}
    outputs = reference.render_cli_reference_outputs()
    pages = {path: content for path, content in outputs.items() if path.suffix == ".mdx"}

    for command, help_text in sections:
        words = command.split()[1:]
        if not words:
            source = reference.CLI_REFERENCE_ROOT / "index.mdx"
        elif command in parent_commands:
            source = reference.CLI_REFERENCE_ROOT.joinpath(*words, "index.mdx")
        else:
            source = reference.CLI_REFERENCE_ROOT.joinpath(*words[:-1], f"{words[-1]}.mdx")
        assert pages[source].count(help_text) == 1, command
        assert sum(content.count(help_text) for content in pages.values()) == 1, command

    generated_files = {path for path in reference.CLI_REFERENCE_ROOT.rglob("*") if path.is_file()}
    assert generated_files == set(outputs)


def test_cli_examples_cover_every_parser_leaf_and_parse() -> None:
    from pllm._cli.app import build_parser

    leaves = set(reference.cli_leaf_commands())
    assert set(reference.CLI_EXAMPLES) == leaves
    reference.validate_cli_examples()

    outputs = reference.render_cli_reference_outputs()
    for command, example in reference.CLI_EXAMPLES.items():
        arguments = shlex.split(example)
        assert arguments[0] == "pllm", command
        build_parser().parse_args(arguments[1:])

        words = command.split()[1:]
        source = reference.CLI_REFERENCE_ROOT.joinpath(*words[:-1], f"{words[-1]}.mdx")
        content = outputs[source]
        assert content.count("## Example") == 1, command
        assert f"```bash\n{example}\n```\n\n## Options\n\n```text\n" in content, command


def test_cli_example_validation_rejects_missing_and_extra_commands(monkeypatch) -> None:
    examples = dict(reference.CLI_EXAMPLES)
    examples.pop("pllm config show")
    examples["pllm imaginary"] = "pllm imaginary"
    monkeypatch.setattr(reference, "CLI_EXAMPLES", examples)

    with pytest.raises(ValueError, match=r"missing: pllm config show; extras: pllm imaginary"):
        reference.validate_cli_examples()


def test_cli_sidebar_metadata_preserves_command_hierarchy_and_order() -> None:
    outputs = reference.render_cli_reference_outputs()
    root = json.loads(outputs[reference.CLI_REFERENCE_ROOT / "meta.json"])
    assert root == {
        "title": "Command reference",
        "root": True,
        "pages": ["index", "gateway", "serve", "config", "components", "benchmark", "dev"],
    }
    assert json.loads(outputs[reference.CLI_REFERENCE_ROOT / "config/meta.json"])["pages"] == [
        "index",
        "show",
        "export",
    ]
    assert not (reference.CLI_REFERENCE_ROOT / "research").exists()


def test_sdk_pages_have_executable_examples_or_explicit_api_boundaries() -> None:
    pages = sdk_doc_pages()
    assert pages
    invalid_contract = []
    invalid = []
    runnable: list[tuple[Path, str]] = []
    for page in pages:
        content = page.read_text(encoding="utf-8")
        examples = PYTHON_FENCE.findall(content)
        has_example = PYTHON_EXAMPLE_HEADING in content
        has_boundary = NO_PYTHON_API in content
        relative = page.relative_to(ROOT)
        if has_example == has_boundary:
            invalid_contract.append(
                f"{relative}: choose one SDK example or explicit no-API boundary"
            )
            continue
        if has_boundary:
            if examples:
                invalid_contract.append(f"{relative}: no-API page contains a Python fence")
            if not re.search(r"\]\(/(?:sdk/reference/status|research)/", content):
                invalid_contract.append(
                    f"{relative}: no-API boundary has no status or research link"
                )
            continue
        if len(examples) != 1:
            invalid_contract.append(f"{relative}: expected exactly one Python example")
            continue
        if not re.search(r"(?:from|import)\s+pllm\b", examples[0]):
            invalid_contract.append(f"{relative}: example does not use public PLLM SDK")
        if not re.search(r"^assert\s", examples[0], re.MULTILINE):
            invalid_contract.append(f"{relative}: example has no checked result")
        if not API_LINK.search(content):
            invalid_contract.append(f"{relative}: example has no exact Python API link")
        for index, example in enumerate(examples, start=1):
            try:
                ast.parse(example, filename=f"{page} example {index}")
            except SyntaxError as exc:
                invalid.append(f"{relative} example {index}: {exc}")
            else:
                runnable.append((page, example))
    assert not invalid_contract, "SDK example contract failures:\n" + "\n".join(invalid_contract)
    assert not invalid, "SDK documentation has invalid Python examples:\n" + "\n".join(invalid)

    environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    for page, example in runnable:
        completed = subprocess.run(
            [sys.executable, "-c", example],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, (
            f"{page.relative_to(ROOT)} example failed:\n{completed.stdout}{completed.stderr}"
        )


def test_complete_cli_help_has_exact_parser_parity() -> None:
    generated = reference.render_cli_help()
    for command, help_text in reference.cli_help_sections():
        assert f"$ {command} --help\n{help_text}" in generated
    assert "pllm run" not in generated
    assert "$ pllm serve --help" in generated
    assert "$ pllm benchmark run --help" in generated
    assert "$ pllm research" not in generated
    assert "$ pllm benchmark search --help" not in generated
    assert "$ pllm benchmark compare --help" not in generated


def test_api_inventory_uses_public_objects_once_and_keeps_alias_identity() -> None:
    exports = reference.public_exports()
    inventory = reference.api_inventory()
    assert sum(len(item["exports"]) for item in inventory) == len(exports)
    assert len({name for item in inventory for name in item["exports"]}) == len(exports)
    by_name = {f"{item['module']}.{item['name']}": item["value"] for item in exports}
    assert by_name["pllm.Experiment"] is by_name["pllm.config.Experiment"]
    assert by_name["pllm.ModelPlan"] is by_name["pllm.models.ModelPlan"]
    assert by_name["pllm.ComponentRef"] is by_name["pllm.components.ComponentRef"]


def test_component_catalog_matches_public_api_and_research_catalog_is_absent() -> None:
    from pllm.components import list_components

    assert reference.component_catalog() == tuple(item.to_dict() for item in list_components())
    assert not hasattr(reference, "research_catalog")
    assert not (DOCS_ROOT / "reference/research.mdx").exists()

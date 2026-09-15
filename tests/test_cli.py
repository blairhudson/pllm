from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pllm._cli.app import build_parser
from pllm.research import list_methods, list_recipes, list_sources, render_agents_guide

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"


def run_cli(*arguments: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ, PYTHONPATH=str(PYTHON))
    return subprocess.run(
        [sys.executable, "-m", "pllm", *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_version_and_metadata_commands_keep_heavy_modules_unloaded() -> None:
    commands = [
        ["--help"],
        ["--version"],
        ["config", "show", str(ROOT / "examples/pllm.yaml"), "--format", "json"],
        ["components", "list", "--format", "json"],
        ["research", "recipes", "list", "--format", "json"],
        [
            "research",
            "assess",
            str(ROOT / "examples/publication-assessment.json"),
            "--format",
            "json",
        ],
    ]
    for command in commands:
        code = f"""
import sys
from pllm.cli import main
try:
    main({command!r})
except SystemExit as exc:
    assert exc.code == 0
banned = {{
    'numpy', 'fastapi', 'httpx', 'cryptography', 'pllm._native', 'pllm.provider',
    'pllm.runtime.dashboard', 'pllm.runtime.server'
}}
assert not banned.intersection(sys.modules), banned.intersection(sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=str(PYTHON)),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_every_parser_disables_abbreviation() -> None:
    pending = [build_parser()]
    while pending:
        parser = pending.pop()
        assert parser.allow_abbrev is False
        for action in parser._actions:
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                pending.extend(choices.values())

    result = run_cli("config", "show", "examples/pllm.yaml", "--trust-pyth")
    assert result.returncode == 2
    assert result.stdout == ""


def test_declarative_and_python_target_forms_have_canonical_parity(tmp_path: Path) -> None:
    json_path = tmp_path / "experiment.json"
    yaml_path = ROOT / "examples/pllm.yaml"
    shown = run_cli("config", "show", str(yaml_path), "--format", "json")
    expected = json.loads(shown.stdout)["data"]
    json_path.write_text(json.dumps(expected["configuration"]), encoding="utf-8")

    targets = [
        str(yaml_path),
        str(json_path),
        f"{ROOT / 'examples/composition.py'}:experiment",
        "examples.composition:experiment",
    ]
    digests = set()
    for target in targets:
        arguments = ["config", "show", target, "--format", "json"]
        if ":" in target:
            arguments.append("--trust-python")
        result = run_cli(*arguments)
        assert result.returncode == 0, result.stderr
        digests.add(json.loads(result.stdout)["data"]["configuration_digest"])
    assert digests == {expected["configuration_digest"]}


def test_explicit_factory_and_python_trust_policy(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text(
        "from examples.composition import experiment\n"
        "def build():\n"
        "    return experiment\n",
        encoding="utf-8",
    )
    reference = f"{target}:build"

    rejected = run_cli("config", "show", reference, "--factory", "--no-input")
    assert rejected.returncode == 3
    assert rejected.stdout == ""
    assert "PYTHON_TRUST_REQUIRED" in rejected.stderr

    accepted = run_cli(
        "config", "show", reference, "--factory", "--no-input", "--trust-python", "--format", "json"
    )
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stderr == ""
    assert json.loads(accepted.stdout)["data"]["target_kind"] == "python-factory"


def test_interactive_python_target_warns_and_confirms(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pllm.cli import main

    class Input(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", Input("yes\n"))
    main(["config", "show", "examples/composition.py:experiment"])
    captured = capsys.readouterr()
    assert '"schema": "pllm.experiment.v1"' in captured.out
    assert "warning[PYTHON_CODE_EXECUTION]" in captured.err
    assert "Continue? [y/N]" in captured.err


@pytest.mark.parametrize(
    "target,extra",
    [
        ("examples/composition.py:_private", ["--trust-python"]),
        ("examples/composition.py:experiment()", ["--trust-python"]),
        ("examples/composition.py:experiment.__class__", ["--trust-python"]),
        ("payload.pkl", []),
    ],
)
def test_target_resolver_rejects_private_attributes_expressions_and_pickle(
    target: str, extra: list[str]
) -> None:
    result = run_cli("config", "show", target, *extra)
    assert result.returncode == 3
    assert result.stdout == ""


def test_export_is_exclusive_and_dry_run_does_not_write(tmp_path: Path) -> None:
    output = tmp_path / "experiment.json"
    dry = run_cli(
        "--format", "json", "--dry-run", "config", "export", "examples/pllm.yaml", "--output", str(output)
    )
    assert dry.returncode == 0, dry.stderr
    assert not output.exists()
    assert json.loads(dry.stdout)["data"]["written"] is False

    first = run_cli("config", "export", "examples/pllm.yaml", "--output", str(output))
    assert first.returncode == 0, first.stderr
    assert output.read_bytes().endswith(b"\n")
    second = run_cli("config", "export", "examples/pllm.yaml", "--output", str(output))
    assert second.returncode == 4
    dry_existing = run_cli(
        "config", "export", "examples/pllm.yaml", "--output", str(output), "--dry-run"
    )
    assert dry_existing.returncode == 4
    forced = run_cli(
        "config", "export", "examples/pllm.yaml", "--output", str(output), "--force"
    )
    assert forced.returncode == 0, forced.stderr
    yaml_output = tmp_path / "experiment.yaml"
    yaml_export = run_cli(
        "config", "export", "examples/pllm.yaml", "--output", str(yaml_output)
    )
    assert yaml_export.returncode == 0, yaml_export.stderr
    assert run_cli("config", "show", str(yaml_output)).returncode == 0


def test_machine_envelopes_jsonl_and_stream_separation() -> None:
    shown = run_cli("--format", "json", "components", "show", "pllm/cpu")
    payload = json.loads(shown.stdout)
    assert shown.stderr == ""
    assert payload["schema_version"] == "pllm.cli.result.v1"
    assert payload["command"] == "components.show"

    listed = run_cli("research", "sources", "list", "--format", "jsonl")
    lines = [json.loads(line) for line in listed.stdout.splitlines()]
    assert listed.stderr == ""
    assert len(lines) == len(list_sources())
    assert all(line["schema_version"] == "pllm.cli.result.v1" for line in lines)
    assert all(set(line["data"]) == {"count", "dry_run", "item"} for line in lines)

    failed = run_cli("--format", "json", "components", "show", "missing")
    assert failed.returncode == 3
    assert failed.stdout == ""
    error = json.loads(failed.stderr)
    assert error["schema_version"] == "pllm.cli.error.v1"
    assert error["error"]["exit_code"] == 3


def test_research_records_are_immutable_and_do_not_execute_workflows() -> None:
    sources = list_sources()
    methods = list_methods()
    recipes = list_recipes()
    assert sources and methods and recipes
    with pytest.raises(TypeError):
        recipes[0].payload["workflow_status"] = "completed"
    assert all(record.payload["schema_version"] == "pllm.source_record.v1" for record in sources)
    assert all(record.payload["schema_version"] == "pllm.method_record.v1" for record in methods)
    assert all(
        record.payload["schema_version"] == "pllm.reproduction_recipe.v1" for record in recipes
    )


def test_research_assess_emits_a_pure_machine_readable_decision() -> None:
    result = run_cli(
        "research",
        "assess",
        "examples/publication-assessment.json",
        "--format",
        "json",
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["command"] == "research.assess"
    assert payload["data"]["assessment"]["decision"] == "ready_for_human_review"
    assert payload["data"]["assessment"]["claim"]["statement"].startswith("PLLM's MPCache")
    assert payload["data"]["digest"].startswith("sha256:")


def test_research_agents_help_and_generation(tmp_path: Path) -> None:
    help_result = run_cli("research", "agents", "--help")
    assert help_result.returncode == 0
    assert help_result.stderr == ""
    assert "--output PATH" in help_result.stdout
    assert "--force" in help_result.stdout

    output = tmp_path / "AGENTS.md"
    result = run_cli("research", "agents", "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == f"Wrote {output}\n"
    assert output.read_text(encoding="utf-8") == render_agents_guide()
    assert list(tmp_path.iterdir()) == [output]


def test_research_agents_dry_run_json_and_output_failures(tmp_path: Path) -> None:
    output = tmp_path / "AGENTS.md"
    dry = run_cli(
        "--dry-run", "research", "agents", "--output", str(output), "--format", "json"
    )
    assert dry.returncode == 0, dry.stderr
    assert dry.stderr == ""
    assert not output.exists()
    payload = json.loads(dry.stdout)
    assert payload["command"] == "research.agents"
    assert payload["data"]["output"] == str(output)
    assert payload["data"]["written"] is False
    assert payload["data"]["dry_run"] is True
    assert payload["data"]["digest"].startswith("sha256:")

    output.write_text("keep me", encoding="utf-8")
    existing = run_cli("research", "agents", "--output", str(output), "--format", "json")
    assert existing.returncode == 4
    assert existing.stdout == ""
    assert json.loads(existing.stderr)["error"]["code"] == "OUTPUT_EXISTS"
    assert output.read_text(encoding="utf-8") == "keep me"
    dry_existing = run_cli("--dry-run", "research", "agents", "--output", str(output))
    assert dry_existing.returncode == 4
    assert dry_existing.stdout == ""
    assert output.read_text(encoding="utf-8") == "keep me"

    missing_parent = run_cli(
        "research", "agents", "--output", str(tmp_path / "missing" / "AGENTS.md")
    )
    assert missing_parent.returncode == 4
    assert missing_parent.stdout == ""
    assert "OUTPUT_WRITE" in missing_parent.stderr

    forced = run_cli("research", "agents", "--output", str(output), "--force")
    assert forced.returncode == 0, forced.stderr
    assert output.read_text(encoding="utf-8") == render_agents_guide()


def test_bundled_research_snapshot_matches_checkout_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pllm.research as research

    source_records = [record.to_dict() for record in list_sources()]
    method_records = [record.to_dict() for record in list_methods()]
    recipe_records = [record.to_dict() for record in list_recipes()]
    research._records.cache_clear()
    monkeypatch.setattr(research, "_repository_root", lambda: None)
    assert [record.to_dict() for record in list_sources()] == source_records
    assert [record.to_dict() for record in list_methods()] == method_records
    assert [record.to_dict() for record in list_recipes()] == recipe_records
    research._records.cache_clear()


def test_validation_io_usage_and_interrupt_exit_classes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    usage = run_cli("components", "show")
    assert usage.returncode == 2
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema":"wrong","api_key":"do-not-print"}', encoding="utf-8")
    validation = run_cli("config", "show", str(invalid))
    assert validation.returncode == 3
    assert "do-not-print" not in validation.stderr
    missing = run_cli("config", "show", str(tmp_path / "missing.json"))
    assert missing.returncode == 4

    import pllm._cli.app as app

    monkeypatch.setattr(app, "_components", lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(SystemExit, match="130"):
        app.main(["components", "list"])


def test_public_configuration_rejects_nested_secret_fields(tmp_path: Path) -> None:
    configuration = json.loads((ROOT / "schemas/fixtures/experiment.valid.json").read_text())
    configuration["pipeline"]["components"]["unsafe"] = {
        "component": "example/custom",
        "params": {"service_api_key": "never-print-this"},
    }
    target = tmp_path / "secret.json"
    target.write_text(json.dumps(configuration), encoding="utf-8")

    result = run_cli("config", "show", str(target), "--format", "json")
    assert result.returncode == 3
    assert result.stdout == ""
    assert "CONFIGURATION_SECRET_FIELD" in result.stderr
    assert "never-print-this" not in result.stderr


def test_help_snapshot_matches_real_parser() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert result.stderr == ""
    snapshot = (ROOT / "docs/public/downloads/cli-help.txt").read_text(encoding="utf-8")
    root_help = snapshot.split("$ pllm --help\n", 1)[1].split("\n$ pllm config --help\n", 1)[0]
    assert result.stdout == root_help

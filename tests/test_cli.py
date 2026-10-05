from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pllm._cli.app import build_parser

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


def test_design_status_matches_parser_visible_command_families() -> None:
    parser = build_parser()
    choices = next(
        action.choices
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    )
    assert set(choices) == {
        "config", "components", "topology", "gateway", "serve", "benchmark", "dev", "network", "plan"
    }
    status = (ROOT / "docs/plans/cli.md").read_text(encoding="utf-8").split(
        "## Implementation status", 1
    )[1]
    for shipped in (
        "`config show TARGET`",
        "`config export TARGET --output PATH [--force]`",
        "`components list|show`",
        "`topology inspect TARGET`",
        "`gateway`",
        "`serve inference|preparation`",
        "`benchmark run`",
        "`dev dashboard`",
    ):
        row = next(line for line in status.splitlines() if line.startswith(f"| {shipped} |"))
        assert "**Shipped" in row
    research = next(
        line
        for line in status.splitlines()
        if line.startswith("| `research sources|methods|recipes list|show` |")
    )
    assert "**Unavailable**" in research


def test_help_version_and_metadata_commands_keep_heavy_modules_unloaded() -> None:
    commands = [
        ["--help"],
        ["--version"],
        ["config", "show", str(ROOT / "examples/pllm.yaml"), "--format", "json"],
        ["components", "list", "--format", "json"],
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
    'numpy', 'fastapi', 'httpx', 'cryptography', 'pllm._native', 'pllm.providers',
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


def test_topology_inspection_uses_exact_experiment_target_and_reports_local_separation() -> None:
    target = str(ROOT / "examples/pllm.yaml")
    shown = run_cli("topology", "inspect", target, "--format", "json")
    assert shown.returncode == 0, shown.stderr
    report = json.loads(shown.stdout)
    assert report["command"] == "topology.inspect"
    data = report["data"]
    assert data["topology"]["schema"] == "pllm.role_topology.v1"
    assert data["placement"]["separation_violations"] == [["inference", "preparation"]]
    assert {role["id"] for role in data["topology"]["roles"]} == {
        "client", "preparation", "inference"
    }
    assert len(data["topology_digest"]) == 64
    assert not data["python_executed"]

    rejected = run_cli("topology", "inspect", "examples/composition.py:experiment", "--no-input")
    assert rejected.returncode == 3
    assert "PYTHON_TRUST_REQUIRED" in rejected.stderr
    trusted = run_cli(
        "topology", "inspect", "examples/composition.py:experiment",
        "--trust-python", "--no-input", "--format", "json",
    )
    assert trusted.returncode == 0, trusted.stderr
    assert json.loads(trusted.stdout)["data"]["topology_digest"] == data["topology_digest"]

    explicit = run_cli(
        "topology", "inspect", "examples/prepared-topology.yaml", "--format", "json"
    )
    assert explicit.returncode == 0, explicit.stderr
    selected = json.loads(explicit.stdout)["data"]
    assert selected["topology_digest"] == data["topology_digest"]
    assert selected["composition_digest"] != data["composition_digest"]
    admitted = run_cli(
        "--dry-run", "benchmark", "run", "--experiment", "examples/prepared-topology.yaml",
        "--max-output-tokens", "1", "--repetitions", "1", "--format", "json",
    )
    assert admitted.returncode == 0, admitted.stderr
    assert json.loads(admitted.stdout)["data"]["configuration"]["experiments"][0][
        "pipeline_digest"
    ] == selected["composition_digest"]


def test_topology_inspection_checks_declared_placement_and_research_baselines() -> None:
    target = "examples/prepared-topology.yaml"
    inspected = run_cli(
        "topology", "inspect", target,
        "--role-deployment", "examples/prepared-role-deployment.json", "--format", "json",
    )
    assert inspected.returncode == 0, inspected.stderr
    report = json.loads(inspected.stdout)["data"]
    declaration = report["role_deployment"]
    assert declaration["digest"]
    assert declaration["assessment"]["declared_separation_violations"] == []
    assert declaration["assessment"]["operator_independence_verified"] is False
    assert declaration["assessment"]["runtime_admission_supported"] is False
    assert report["placement"]["separation_violations"] == [["inference", "preparation"]]
    assert report["executable_topology"] is True

    for kind, count in (("client-only", 1), ("two-online-offset", 3)):
        comparator = run_cli(
            "topology", "inspect", target, "--reference", kind, "--format", "json"
        )
        assert comparator.returncode == 0, comparator.stderr
        data = json.loads(comparator.stdout)["data"]
        assert data["executable_topology"] is False
        assert data["reference"] == kind
        assert len(data["topology"]["roles"]) == count
        assert data["composition_digest"] == report["composition_digest"]
        assert data["topology_digest"] != report["topology_digest"]
    mismatched = run_cli(
        "topology", "inspect", target, "--reference", "client-only",
        "--role-deployment", "examples/prepared-role-deployment.json", "--format", "json",
    )
    assert mismatched.returncode != 0
    assert "TOPOLOGY_PLACEMENT" in mismatched.stderr


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
    assert '"schema": "pllm.experiment.v2"' in captured.out
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

    failed = run_cli("--format", "json", "components", "show", "missing")
    assert failed.returncode == 3
    assert failed.stdout == ""
    error = json.loads(failed.stderr)
    assert error["schema_version"] == "pllm.cli.error.v1"
    assert error["error"]["exit_code"] == 3


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


def test_help_output_matches_real_parser() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == build_parser().format_help()

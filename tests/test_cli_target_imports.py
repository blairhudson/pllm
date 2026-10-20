"""Trusted target imports are scoped; console probes declare only, never load models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pllm._cli.errors import ResolutionError
from pllm._cli.targets import resolve_target


ROOT = Path(__file__).resolve().parents[1]
COMMON = '''from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import FuseFssSourceCpu
experiment = Experiment("bounded-cli-factory",
    FuseFssSourceCpu(Model.path("not-a-checkpoint", model_id="unit"),
        numeric_profile="fusefss.dense-rope-checked-ranges-ring50-q12.v1", capacity="decoder"),
    Deployment.local(root="unused-roles"),
    ExecutionBudget(requests=1, max_input_tokens=1, max_new_tokens=1))
'''

# Source-Python CLI tests must not depend on which native extension was last
# built in python/pllm. Exercise real Experiment.resolve through an explicit
# capability double; never stand in for source/numeric or worker admission.
CAPABILITY_DOUBLE = '''import json
from types import SimpleNamespace
from pllm import _native
def _capability_double(raw):
    record = json.loads(raw)
    return SimpleNamespace(canonical_composition=json.dumps(record["pipeline"]).encode(),
        composition_digest="a" * 64, configuration_digest="b" * 64)
_native.resolve_experiment = _capability_double
'''


@pytest.fixture
def package(tmp_path, monkeypatch):
    parent = tmp_path / "cli_fixture_package" / "papers"
    parent.mkdir(parents=True)
    (parent / "common.py").write_text(COMMON)
    monkeypatch.chdir(tmp_path)
    # Simulate a console script's path: neither '' nor the working directory.
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p and Path(p).resolve() != tmp_path])
    yield parent
    for name in list(sys.modules):
        if name == "cli_fixture_package" or name.startswith("cli_fixture_package."):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("module_target", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_console_style_import_context_covers_module_and_factory_then_restores(package, module_target, lazy):
    target = package / "recipe.py"
    target.write_text(("" if lazy else "from cli_fixture_package.papers.common import experiment\n")
        + "def make():\n"
        + ("    from cli_fixture_package.papers.common import experiment\n" if lazy else "")
        + "    import sys\n    sys.path.append('must-not-leak')\n    return experiment\n")
    original, entries = sys.path, sys.path[:]
    left = "cli_fixture_package.papers.recipe" if module_target else str(target)
    resolved = resolve_target(f"{left}:make", factory=True, no_input=True,
                              trust_python=True, output_format="json")
    assert resolved.configuration.name == "bounded-cli-factory"
    assert resolved.kind == "python-factory" and resolved.python_executed
    assert sys.path is original and sys.path == entries


@pytest.mark.parametrize("failure", ["module", "factory", "exit", "object", "type"])
def test_import_context_restored_on_every_target_failure(package, failure):
    target = package / "bad.py"
    body = {
        "module": "import sys\nsys.path = ['changed-by-target']\nraise RuntimeError('bad module')\n",
        "factory": "def make():\n    import sys\n    sys.path = ['changed-by-target']\n    raise RuntimeError('bad factory')\n",
        "exit": "def make():\n    raise SystemExit(7)\n",
        "object": "from cli_fixture_package.papers.common import experiment\n",
        "type": "def make():\n    return 1\n",
    }[failure]
    target.write_text(body)
    original, entries = sys.path, sys.path[:]
    with pytest.raises(ResolutionError):
        resolve_target(f"{target}:make", factory=True, no_input=True,
                       trust_python=True, output_format="json")
    assert sys.path is original and sys.path == entries


def test_untrusted_target_never_gets_import_context_or_executes(package):
    target, marker = package / "unsafe.py", package / "executed"
    target.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    original, entries = sys.path, sys.path[:]
    with pytest.raises(ResolutionError) as rejected:
        resolve_target(f"{target}:make", factory=True, no_input=True,
                       trust_python=False, output_format="json")
    assert rejected.value.code == "PYTHON_TRUST_REQUIRED"
    assert not marker.exists() and sys.path is original and sys.path == entries


@pytest.mark.parametrize("entrypoint,timeout,trusted", [
    ("console", 7200, True), ("module", 7200, True),
    ("console", 3600, True), ("console", 7200, False),
])
def test_generated_paper_factory_ordinary_cli_without_repository_on_pythonpath(tmp_path, entrypoint, timeout, trusted):
    # Same namespace imports and command shape as the Qwen factory. Source path
    # does not exist; --dry-run uses the explicit capability double above.
    paper = tmp_path / "examples" / "papers"
    paper.mkdir(parents=True)
    (paper / "common.py").write_text(CAPABILITY_DOUBLE + COMMON)
    (paper / "fusefss.py").write_text("from examples.papers.common import experiment\n"
        "def qwen_checked_smoke():\n"
        "    from examples.papers.common import experiment\n"
        "    return experiment\n")
    command = (["uv", "run", "--no-sync", "--project", str(ROOT), "pllm"] if entrypoint == "console"
               else [sys.executable, "-m", "pllm"])
    command += ["--dry-run", "benchmark", "run", "--experiment", "examples/papers/fusefss.py:qwen_checked_smoke",
                "--factory", "--backend", "native", "--prompt", "hello", "--max-output-tokens", "1",
                "--warmups", "0", "--repetitions", "1", "--timeout", str(timeout), "--format", "json"]
    if trusted: command.append("--trust-python")
    result = subprocess.run(command, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
                            capture_output=True, text=True, timeout=20)
    if trusted:
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)["data"]
        assert payload["dry_run"] is True
        assert payload["configuration"]["timeout_seconds"] == timeout
        assert payload["configuration"]["experiments"][0]["name"] == "bounded-cli-factory"
    else:
        assert result.returncode == 3 and "PYTHON_TRUST_REQUIRED" in result.stderr
    assert not (tmp_path / "not-a-checkpoint").exists() and not (tmp_path / "unused-roles").exists()

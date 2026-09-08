from __future__ import annotations
import ast
import importlib.util
import json
import re
import tomllib
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

def release_module():
    spec = importlib.util.spec_from_file_location("pllm_release_checks", ROOT / "scripts/release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_release_tag_matches_package_version():
    module = release_module()
    assert module.validate("v" + module.version()).startswith("v")

@pytest.mark.parametrize("tag", ["main", "v0.0.0", "v0.16.0;echo bad", ""])
def test_invalid_release_tag_is_rejected(tag):
    with pytest.raises(ValueError):
        release_module().validate(tag)

def test_publication_requires_real_dependency_locks(tmp_path, monkeypatch):
    module = release_module()
    (tmp_path / "python/pllm").mkdir(parents=True)
    (tmp_path / "python/pllm/_version.py").write_text('__version__ = "1.2.3"\n')
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="uv.lock"):
        module.validate("v1.2.3", require_locks=True)
    (tmp_path / "uv.lock").touch()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/package-lock.json").touch()
    (tmp_path / "Cargo.lock").touch()
    assert module.validate("v1.2.3", require_locks=True) == "v1.2.3"

def test_build_version_is_single_source():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["dynamic"] == ["version"]
    assert project["build-system"]["build-backend"] == "maturin"
    assert project["tool"]["maturin"]["module-name"] == "pllm._native"
    assert "extension-module" in project["tool"]["maturin"]["features"]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["version"] == release_module().version()

def test_all_workflows_parse_and_external_actions_are_pinned():
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        content = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        assert "on" in content and "jobs" in content
        assert "pull_request_target" not in content["on"]
        for action in re.findall(r"uses:\s*([^\s#]+)", path.read_text()):
            if not action.startswith("./"):
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action)

def test_release_jobs_separate_build_and_credentials():
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["jobs"]["pypi"]["permissions"] == {"id-token": "write"}
    assert workflow["jobs"]["pypi"]["environment"]["name"] == "pypi"
    assert "verify" in workflow["jobs"]["pypi"]["needs"]
    assert "PYPI_TOKEN" not in (ROOT / ".github/workflows/release.yml").read_text()

def test_docs_are_fumadocs_not_retired_mkdocs():
    project = json.loads((ROOT / "docs/package.json").read_text())
    assert "fumadocs-ui" in project["dependencies"]
    assert "fumadocs-mdx" in project["dependencies"]
    assert not (ROOT / "mkdocs.yml").exists()

def test_deploy_files_target_root_checkout():
    assert "/opt/pllm/runtime" not in (ROOT / "deploy/provider-start.sh").read_text()
    assert "COPY runtime/" not in (ROOT / "deploy/Dockerfile.runtime").read_text()

def test_documented_python_examples_parse():
    for path in (ROOT / "docs/content/docs").rglob("*.mdx"):
        for index, code in enumerate(re.findall(r"```python\n(.*?)```", path.read_text(), re.S)):
            ast.parse(code, filename=f"{path}:{index}")

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "pllm_docs_regen", ROOT / "scripts/docs_regen.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regenerate_runs_reference_then_publication_generation(monkeypatch) -> None:
    module = _module()
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, *, cwd, check: calls.append((command, cwd, check)),
    )
    module.run(check=False)
    assert calls == [
        ([sys.executable, str(module.DEVELOPER_REFERENCE)], module.ROOT, True),
        (["npm", "run", "generate"], module.DOCS, True),
    ]


def test_check_mode_materializes_ignored_references_then_runs_freshness_gates(
    monkeypatch,
) -> None:
    module = _module()
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, *, cwd, check: calls.append((command, cwd, check)),
    )
    module.run(check=True)
    assert calls == [
        ([sys.executable, str(module.DEVELOPER_REFERENCE)], module.ROOT, True),
        (
            [sys.executable, str(module.DEVELOPER_REFERENCE), "--check"],
            module.ROOT,
            True,
        ),
        (["npm", "test"], module.DOCS, True),
        (["npm", "run", "check:content"], module.DOCS, True),
    ]

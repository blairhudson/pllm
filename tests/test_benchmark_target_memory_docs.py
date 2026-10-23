"""Execute the documented selector against synthetic reports, never measurements."""

import importlib.util
from pathlib import Path
import re
import subprocess

import pytest

from pllm.runtime import client_process_usage


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/content/docs/sdk/evaluate/benchmark-targets.mdx"


@pytest.fixture
def reports(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("documentation comparison must not launch or sample processes")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(client_process_usage, "_birth_identity", forbidden)
    monkeypatch.setattr(client_process_usage, "supported", lambda: True)
    path = ROOT / "tests/test_client_endpoint_memory_comparison.py"
    spec = importlib.util.spec_from_file_location("_endpoint_doc_reports", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def example(run):
    section = DOC.read_text().split("### Guarded client-memory comparisons\n", 1)[1]
    section = section.split("### ", 1)[0]
    block = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert block is not None
    namespace = {"run": run}
    exec(compile(block.group(1), str(DOC), "exec"), namespace)
    return namespace["comparison"]


def test_documented_selector_preserves_endpoint_only_scope(reports):
    compared = example(reports())
    assert compared["status"] == "comparable_in_scope"
    assert compared["window"] == "client_process_birth_through_report_endpoint"
    assert compared["control_median"] == 1000
    assert compared["candidate_median"] == 10
    assert compared["reduction_status"] == "not_established"
    assert compared["whole_lifetime"] is False
    assert compared["physical_peak"] is False


def test_documented_selector_refuses_incomplete_owner_evidence(reports):
    def incomplete(row):
        scope = row["execution"]["details"]["report"]["client_owned_memory"]
        scope["lifetime_owner"]["membership_complete"] = False

    assert example(reports(change=incomplete))["status"] == "unranked"

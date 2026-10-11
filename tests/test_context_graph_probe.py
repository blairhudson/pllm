"""Probe provenance/order checks only; no workers, checkpoint imports or timings."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture(scope="module")
def probe():
    path = Path(__file__).resolve().parents[1] / "scripts/probe_context_graph.py"
    spec = importlib.util.spec_from_file_location("context_graph_probe_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preworker_freeze_includes_helpers_source_runtime_and_installed_package(probe):
    frozen = probe.freeze_sources()
    root = Path(__file__).resolve().parents[1]
    for relative in ("scripts/probe_context_graph.py", "scripts/decoder_probe_support.py",
                     "scripts/frontier_probe_support.py", "python/pllm/runtime/semantic_executor.py",
                     "python/pllm/runtime/model_binding.py", "python/pllm/runtime/prefill_cache.py",
                     "python/pllm/runtime/context_graph.py", "python/pllm/runtime/tiered_execution.py"):
        assert str((root / relative).resolve()) in frozen
    assert str(Path(sys.modules["pllm"].__file__).resolve()) in frozen
    probe.verify_sources(frozen)


def test_imported_sources_and_source_mutation_reject(probe, tmp_path, monkeypatch):
    frozen = probe.freeze_sources()
    fixture = tmp_path / "probe_fixture.py"
    fixture.write_text("value = 1\n")
    path = str(fixture.resolve())
    frozen[path] = probe.file_hash(fixture)
    name = "pllm.runtime.context_graph_probe_fixture"
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=path))
    assert probe.imported_sources(frozen)[name] == {"path": path, "sha256": frozen[path]}
    without_fixture = {p: h for p, h in frozen.items() if p != path}
    with pytest.raises(RuntimeError, match="absent from frozen source set"):
        probe.imported_sources(without_fixture)
    fixture.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="source changed during cohort"):
        probe.verify_sources(frozen)
    with pytest.raises(RuntimeError, match="absent from frozen source set"):
        probe.imported_sources(frozen)


def test_three_repeat_two_geometry_storage_order_balances(probe):
    orders = [probe.storage_order(geometry, repeat) for geometry in range(2) for repeat in range(3)]
    assert orders.count(("tiered", "graph")) == orders.count(("graph", "tiered")) == 3
    assert probe.storage_order(0, 0) != probe.storage_order(0, 1)

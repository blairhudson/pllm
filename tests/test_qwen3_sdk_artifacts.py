"""Replay promoted artifact controls without checkpoint or provider loading."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_factories_preserve_measured_pipelines_and_only_change_tokenizer(monkeypatch, tmp_path):
    from pllm.tokenization import IndexedTokenizer

    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv("PLLM_RESEARCH_ARTIFACTS", str(tmp_path))
    publisher = json.loads((ROOT / "docs/evidence/research-qwen3-sdk-artifacts-publisher.json").read_text())
    (tmp_path / "publisher.json").write_text(json.dumps(publisher))
    report = json.loads((ROOT / "docs/evidence/research-qwen3-sdk-artifacts.json").read_text())
    assert report["checks"]["passed"]
    factories = ["qwen3_sdk_control", "qwen3_sdk_indexed"]
    for name, candidate in zip(factories, report["candidates"], strict=True):
        experiment = importlib.import_module(f"benchmarks.research.{name}").experiment()
        if "tokenizer" in candidate["pipeline"]["components"]:
            selection = experiment.pipeline.components["tokenizer"]
            assert selection.params["path"] == str(tmp_path / "tokenizer")
            assert selection.params["digest"] == publisher["tokenizer_contract_sha256"]
            # Local paths are part of Pipeline identity; restore the recorded
            # public path only for this replay-digest check, without reading it.
            original = candidate["pipeline"]["components"]["tokenizer"]["params"]
            experiment = experiment.with_params(pipeline__tokenizer=IndexedTokenizer(**original))
        assert experiment.pipeline.to_spec() == candidate["pipeline"]
        assert experiment.configuration_digest() == candidate["configuration_digest"]
        assert candidate["report"]["checks"]["passed"]
        assert candidate["report"]["memory_guard"]["maximum_host_swap_growth_bytes"] == 0
    assert hashlib.sha256((ROOT / "benchmarks/research/qwen3_artifacts.py").read_bytes()).hexdigest() == publisher["configuration_source_sha256"]
    control, indexed = [item["pipeline"] for item in report["candidates"]]
    del indexed["components"]["tokenizer"]
    assert indexed == control
    publisher["revision"] = "f" * 40
    (tmp_path / "publisher.json").write_text(json.dumps(publisher))
    with pytest.raises(ValueError, match="another source"):
        importlib.import_module("benchmarks.research.qwen3_sdk_indexed").experiment()

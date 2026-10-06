"""Saved research factories preserve evidence identity and explicit artifact moves."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from pllm import BenchmarkResult, SearchCandidate
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse, PublicPrefixCapsule
from pllm.tokenization import IndexedTokenizer
from test_search import experiment, result_for


@pytest.fixture
def research(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("benchmarks.research.confirm")


def test_finalist_factory_replays_and_relocates_only_public_artifacts(monkeypatch, tmp_path, research):
    original = experiment().with_params(
        pipeline__quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
        pipeline__cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=8),
        pipeline__tokenizer=IndexedTokenizer("/recorded/index", digest="a" * 64),
        pipeline__public_prefix=PublicPrefixCapsule("/recorded/prefix.msgpack", digest="b" * 64, size_bytes=128),
    )
    candidate = SearchCandidate("replay", 0, original, {}, original.configuration_digest())
    namespace = {}
    exec(research.factory_source(candidate), namespace)
    monkeypatch.delenv("PLLM_RESEARCH_ARTIFACTS", raising=False)
    replay = namespace["experiment"]()
    assert replay.configuration_digest() == original.configuration_digest()
    assert replay.pipeline.profile == original.pipeline.profile
    monkeypatch.setenv("PLLM_RESEARCH_ARTIFACTS", str(tmp_path))
    relocated = namespace["experiment"]()
    assert relocated.configuration_digest() != original.configuration_digest()
    assert relocated.pipeline.profile == original.pipeline.profile
    assert relocated.pipeline.components["tokenizer"].params == {
        "path": str(tmp_path / "index"), "digest": "a" * 64,
    }
    assert relocated.pipeline.components["public_prefix"].params == {
        "path": str(tmp_path / "prefix.msgpack"), "digest": "b" * 64, "size_bytes": 128,
    }
    assert original.pipeline.components["tokenizer"].params["path"] == "/recorded/index"
    assert relocated.pipeline.model == original.pipeline.model


def finalist_document():
    evaluations = []
    for index, threads in enumerate((1, 2)):
        value = experiment().with_params(pipeline__kernels__threads=threads)
        candidate = SearchCandidate(f"trial-{index}", index, value, {}, value.configuration_digest())
        result = result_for(candidate).to_dict()
        metrics = {metric["id"]: metric for metric in result["metrics"]}
        metrics["latency"].update(id="request_tps", component="pllm/throughput",
            parameters={"basis": "tokens"}, unit="per_second", unit_detail="tokens_per_second",
            value=float(threads), origin="external_measured")
        metrics["communication"].update(id="covered_bytes", origin="external_measured")
        evaluations.append({"trial_id": candidate.trial_id, "parameters": {},
                            "experiment": value.to_spec(), "result": BenchmarkResult.from_dict(result).to_dict()})
    return {"schema_version": "pllm.search_benchmark.v1", "search": {"evaluations": evaluations}}


def test_finalists_select_distinct_sdk_objectives_with_exact_cohort_checks(research):
    document = finalist_document()
    selected = research.selected_candidates(document)
    assert len(selected) == 2
    assert selected[0]["candidate"].experiment.pipeline.components["kernels"].params["threads"] == 2
    assert selected[1]["candidate"].experiment.pipeline.components["kernels"].params["threads"] == 1
    document["search"]["evaluations"][1]["result"]["workload_digest"] = "f" * 64
    with pytest.raises(ValueError):
        research.selected_candidates(document)


def test_finalist_export_remains_replayable_after_artifact_directory_moves(monkeypatch, tmp_path, research):
    original = experiment().with_params(pipeline__tokenizer=IndexedTokenizer("/recorded/index", digest="a" * 64))
    candidate = SearchCandidate("original", 0, original, {}, original.configuration_digest())
    report = tmp_path / "search.json"
    report.write_text("{}")
    output = tmp_path / "confirmation.json"
    monkeypatch.setattr(research, "ROOT", tmp_path)
    monkeypatch.setattr(research, "selected_candidates", lambda _: ({"candidate": candidate, "selected_for": []},))
    monkeypatch.setattr(research.sys, "argv", ["confirm", str(output), "--search", f"test={report}", "--export-only"])
    manifests = []
    for directory in (tmp_path / "first", tmp_path / "moved"):
        monkeypatch.setenv("PLLM_RESEARCH_ARTIFACTS", str(directory))
        research.main()
        selected = json.loads(output.with_suffix(".selection.json").read_text())["finalists"][0]
        namespace = {}
        exec((tmp_path / selected["file"]).read_text(), namespace)
        assert namespace["experiment"]().configuration_digest() == selected["configuration_digest"]
        manifests.append(selected)
    assert manifests[0]["file_sha256"] == manifests[1]["file_sha256"]
    assert manifests[0]["configuration_digest"] != manifests[1]["configuration_digest"]

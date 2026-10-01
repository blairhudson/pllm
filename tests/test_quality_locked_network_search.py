"""Reference fidelity must be locked before comparing W4/W8 traffic."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.evidence import benchmark_network_candidates, benchmark_reference_quality
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.search import QualityLockedNetworkSearch, SearchError
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint


def test_quality_locked_search_rejects_invalid_thresholds() -> None:
    with pytest.raises(SearchError, match="finite fraction"):
        QualityLockedNetworkSearch(float("nan"), 0, 10)
    with pytest.raises(SearchError, match="finite and non-negative"):
        QualityLockedNetworkSearch(1, 0, float("inf"))
    with pytest.raises(SearchError, match="versioned"):
        QualityLockedNetworkSearch(1, 0, 10).select({}, [])
    with pytest.raises(ValueError, match="one to eight"):
        benchmark_network_candidates([], "a prompt", max_output_tokens=1)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("PLLM_RUN_QUALITY_NETWORK_GATE"), reason="explicit locked comparison")
def test_w4_w8_quality_and_online_body_comparison_over_one_source_lock(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "quality-network", num_hidden_layers=1, model_type="qwen2",
        tie_word_embeddings=True,
    )
    model_id = "quality-locked-decoder"
    source = Model.path(str(root), model_id=model_id)
    prompt = "A bounded input."
    candidates = []
    for bits in (4, 8):
        candidates.append(Experiment(
            name=f"quality-w{bits}a{bits}",
            pipeline=MaskedLinearCpu(
                source, quantization=SymmetricPerRow(
                    weight_bits=bits, activation_bits=bits,
                ),
            ),
            deployment=Deployment.local(root=str(tmp_path)),
            budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=1),
        ))
    quality = benchmark_reference_quality(candidates, [prompt], top_k=5)
    reports = list(benchmark_network_candidates(
        candidates, prompt, max_output_tokens=1,
        repetitions=1, timeout_seconds=120, inventory_policy="request-sized",
    ))
    assert all(result["checks"]["passed"] for _, result in reports)
    assert [result["configuration"]["source_lock_digest"] for _, result in reports] == [
        quality["model"]["source_lock_digest"]
    ] * len(reports), (
        quality["model"]["source_lock_digest"],
        [result["configuration"]["source_lock_digest"] for _, result in reports],
    )
    for experiment, result in reports:
        assert result.get("experiment", {}).get("configuration_digest") == experiment.configuration_digest(), result.get("experiment")
        assert result.get("experiment", {}).get("pipeline_digest") == experiment.pipeline.digest(), result.get("experiment")
        assert result.get("runs")
    search = QualityLockedNetworkSearch(0.0, 0.0, 1_000_000.0)
    selection = search.select(quality, reports)
    assert len(selection["qualified"]) == 2
    assert selection["winner_configuration_digest"] == selection["qualified"][0]["configuration_digest"]
    assert selection["source_lock_digest"] == quality["model"]["source_lock_digest"]
    assert reports[0][1]["runs"][0]["model_fingerprint"] != reports[1][1]["runs"][0]["model_fingerprint"]
    assert not selection["generation_quality_established"]

    reports[1][1]["configuration"]["source_lock_digest"] = "0" * 64
    with pytest.raises(SearchError, match="checkpoint-bound"):
        search.select(quality, reports)

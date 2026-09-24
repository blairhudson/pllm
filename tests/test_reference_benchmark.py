from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pllm
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.reference_benchmark import ReferenceBenchmarkError, run_reference_benchmark


def _experiment(name: str, bits: int) -> pllm.Experiment:
    return pllm.Experiment(
        name=name,
        pipeline=MaskedLinearCpu(
            pllm.Model.tiny(), quantization=SymmetricPerRow(weight_bits=bits, activation_bits=bits)
        ),
        deployment=pllm.Deployment.local(root="local://reference-benchmark-test"),
        budget=pllm.ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=1),
    )


def test_reference_benchmark_rejects_malformed_cohorts_before_import() -> None:
    experiment = _experiment("tiny-w4", 4)
    with pytest.raises(ReferenceBenchmarkError, match="1 to 32 prompts"):
        run_reference_benchmark([experiment], [])
    with pytest.raises(ReferenceBenchmarkError, match="prompt"):
        run_reference_benchmark([experiment], [""])
    with pytest.raises(ReferenceBenchmarkError, match="top_k"):
        run_reference_benchmark([experiment], ["Hello"], top_k=True)
    with pytest.raises(ReferenceBenchmarkError, match="request budget"):
        run_reference_benchmark([experiment], ["A", "B", "C"])


@pytest.mark.rust
@pytest.mark.slow
def test_reference_benchmark_compares_real_tiny_decoder_without_retaining_payloads() -> None:
    pytest.importorskip("transformers")
    prompts = ["Hello", "The capital of France is"]
    w4 = _experiment("tiny-w4", 4)
    w8 = _experiment("tiny-w8", 8)
    report = run_reference_benchmark([w4, w8], prompts, top_k=5)
    assert report["schema_version"] == "pllm.reference_quality_benchmark.v1"
    assert report["scope"] == "local-compiled-clear-kernel-prefill-reference"
    assert len(report["cohort"]["dataset_digest"]) == 64
    assert len(report["cohort"]["token_cohort_digest"]) == 64
    assert report["cohort"]["prompt_count"] == 2
    assert report["cohort"]["metric"]["component"] == "pllm/reference-agreement/v1"
    assert [item["weight_bits"] for item in report["candidates"]] == [4, 8]
    assert [item["configuration_digest"] for item in report["candidates"]] == [
        w4.configuration_digest(),
        w8.configuration_digest(),
    ]
    for item in report["candidates"]:
        assert item["sample_count"] == 2
        assert 0 <= item["top1_agreement"] <= 1
        assert 0 <= item["top_k_recall"] <= 1
        assert item["max_abs_logit_error"] >= 0
    serialized = json.dumps(report, allow_nan=False)
    assert all(prompt not in serialized for prompt in prompts)
    assert "token_ids" not in serialized and "logits" not in serialized


@pytest.mark.rust
@pytest.mark.slow
def test_quality_cli_compares_typed_experiments_without_emitting_prompts(tmp_path) -> None:
    pytest.importorskip("transformers")
    prompt = "Hello"
    cohort = tmp_path / "prompts.json"
    cohort.write_text(json.dumps([prompt]), encoding="utf-8")
    candidates = []
    for name, bits in (("tiny-w4", 4), ("tiny-w8", 8)):
        path = tmp_path / f"{name}.json"
        path.write_bytes(_experiment(name, bits).canonical_bytes())
        candidates.extend(["--experiment", str(path)])
    command = [
        sys.executable,
        "-m",
        "pllm",
        "benchmark",
        "quality",
        *candidates,
        "--prompts-file",
        str(cohort),
        "--format",
        "json",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=120)
    payload = json.loads(result.stdout)
    assert payload["command"] == "benchmark.quality"
    assert [row["weight_bits"] for row in payload["data"]["report"]["candidates"]] == [4, 8]
    assert prompt not in result.stdout


@pytest.mark.rust
@pytest.mark.slow
@pytest.mark.parametrize(
    ("env_name", "model_id", "model_key", "evidence_file"),
    [
        (
            "PLLM_REAL_QWEN_PATH",
            "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
            "qwen2",
            "qwen2.5-0.5b-reference-quality-2026-09-24.json",
        ),
        (
            "PLLM_REAL_QWEN3_PATH",
            "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
            "qwen3",
            "qwen3-0.6b-reference-quality-2026-09-24.json",
        ),
    ],
)
def test_pinned_reference_cohort_matches_retained_report(
    env_name: str, model_id: str, model_key: str, evidence_file: str
) -> None:
    if not os.getenv(env_name):
        pytest.skip(f"pinned checkpoint path is not configured: {env_name}")
    pytest.importorskip("transformers")
    source = pllm.Model.path(
        os.environ[env_name],
        model_id=model_id,
    )
    candidates = [
        pllm.Experiment(
            name=f"{model_key}-w{bits}a{bits}",
            pipeline=MaskedLinearCpu(
                source,
                quantization=SymmetricPerRow(weight_bits=bits, activation_bits=bits),
            ),
            deployment=pllm.Deployment.local(root=f"local://{model_key}-reference-quality"),
            budget=pllm.ExecutionBudget(requests=2, max_input_tokens=16, max_new_tokens=2),
        )
        for bits in (4, 8)
    ]
    root = Path(__file__).resolve().parents[1]
    prompts = json.loads((root / "examples/benchmarks/reference_prompts.json").read_text())
    evidence_text = (root / "docs/evidence" / evidence_file).read_text()
    assert all(prompt not in evidence_text for prompt in prompts)
    expected = json.loads(evidence_text)
    report = run_reference_benchmark(candidates, prompts)
    assert report["model"] == expected["model"]
    assert report["cohort"] == expected["cohort"]
    for actual, archived in zip(report["candidates"], expected["candidates"], strict=True):
        assert actual["name"] == archived["name"]
        assert actual["pipeline_digest"] == archived["pipeline_digest"]
        assert actual["configuration_digest"] == archived["configuration_digest"]
        assert actual["top1_agreement"] == archived["top1_agreement"]
        assert actual["top_k_recall"] == archived["top_k_recall"]
        assert abs(actual["max_abs_logit_error"] - archived["max_abs_logit_error"]) < 0.1

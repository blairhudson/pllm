from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    with pytest.raises(ReferenceBenchmarkError, match="reference_storage"):
        run_reference_benchmark([experiment], ["A"], reference_storage="unknown")
    with pytest.raises(ReferenceBenchmarkError, match="decode_steps"):
        run_reference_benchmark([experiment], ["A"], decode_steps=True)
    with pytest.raises(ReferenceBenchmarkError, match="output budget"):
        run_reference_benchmark([experiment], ["A"], decode_steps=1)


@pytest.mark.rust
@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_delegated_quality_uses_verified_worker_outputs_and_forced_decode(tmp_path, family):
    import numpy as np

    from pllm.runtime.delegated_source import admit_source
    from pllm.runtime.reference_benchmark import _delegated_candidate
    from test_delegated_decoder import clear_oracle
    from test_delegated_sdk import experiment_fixture

    experiment, config, arrays = experiment_fixture(tmp_path, family=family)
    admitted = admit_source(experiment)
    continuation = (7, 9)
    metadata, prefill, decode, ids, selected = _delegated_candidate(
        experiment, ("hello world",), admitted, 2, [continuation],
    )
    expected = clear_oracle(config, arrays, ids[0], 2, decode_tokens=continuation)
    assert [row.tobytes() for row in prefill + decode] == list(expected[0])
    assert selected == [continuation]
    assert metadata["numeric_profile"] == admitted.contract["numeric_profile"]
    assert metadata["execution"] == "two-worker-request-end-verified"
    assert metadata["weight_bits"] == 16 and metadata["activation_bits"] is None
    assert all(np.isfinite(row).all() for row in prefill + decode)


@pytest.mark.rust
@pytest.mark.parametrize("delegated_first", [True, False])
def test_quality_compares_delegated_and_prepared_on_one_token_trajectory(tmp_path, delegated_first):
    pytest.importorskip("transformers")
    from examples.papers.gkm import experiment as delegated_experiment
    from examples.papers.prepared import experiment as prepared_experiment
    from test_delegated_sdk import experiment_fixture

    fixture, _, _ = experiment_fixture(tmp_path)
    delegated = delegated_experiment(fixture.pipeline.model, fixture.budget)
    prepared = prepared_experiment(fixture.pipeline.model, fixture.budget)
    experiments = [delegated, prepared] if delegated_first else [prepared, delegated]
    report = run_reference_benchmark(experiments, ["hello world"], decode_steps=2,
                                     reference_storage="streamed")
    assert report["cohort"]["same_token_decode"]["teacher_configuration_digest"] == experiments[0].configuration_digest()
    assert report["cohort"]["input_token_counts"] == [2]
    assert [row["weight_bits"] for row in report["candidates"]] == ([16, 8] if delegated_first else [8, 16])
    for row in report["candidates"]:
        assert row["sample_count"] == 1 and row["same_token_decode"]["sample_count"] == 2
        assert 0 <= row["top1_agreement"] <= 1
        assert row["max_abs_logit_error"] >= 0
    assert report["scope"] == "local-compiled-decoder-prefill-and-same-token-decode-reference"
    encoded = json.dumps(report, allow_nan=False)
    assert "hello world" not in encoded and '"logits"' not in encoded and '"token_ids"' not in encoded


@pytest.mark.rust
def test_delegated_quality_failure_retires_workers_without_report(tmp_path, monkeypatch):
    from pllm.runtime import delegated_source, servers
    from pllm.runtime.reference_benchmark import _delegated_candidate
    from test_delegated_sdk import experiment_fixture

    experiment, _, _ = experiment_fixture(tmp_path)
    admitted = delegated_source.admit_source(experiment)
    original_load = delegated_source.AdmittedSource.load_transport
    original_build = servers.build_roles
    topologies = []

    class RejectFinish:
        def __init__(self, owner):
            self.owner = owner

        def __getattr__(self, name):
            return getattr(self.owner, name)

        def finish(self):
            self.owner.finish()
            raise ValueError("controlled final verification rejection")

    def load(self, *args, **kwargs):
        owner = original_load(self, *args, **kwargs)
        return RejectFinish(owner) if kwargs.get("worker") is None else owner

    def build(*args, **kwargs):
        topology = original_build(*args, **kwargs)
        topologies.append(topology)
        return topology

    monkeypatch.setattr(delegated_source.AdmittedSource, "load_transport", load)
    monkeypatch.setattr(servers, "build_roles", build)
    with pytest.raises(ValueError, match="final verification"):
        _delegated_candidate(experiment, ("hello world",), admitted, 1, None)
    assert len(topologies) == 1
    assert all(not status.running for status in topologies[0].statuses)


@pytest.mark.rust
def test_delegated_quality_cli_uses_ordinary_report(tmp_path):
    pytest.importorskip("transformers")
    from test_delegated_sdk import experiment_fixture

    experiment, _, _ = experiment_fixture(tmp_path)
    config = tmp_path / "experiment.json"
    config.write_bytes(experiment.canonical_bytes())
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["hello world"]))
    result = subprocess.run([
        sys.executable, "-m", "pllm", "benchmark", "quality",
        "--experiment", str(config), "--prompts-file", str(prompts),
        "--decode-steps", "2", "--reference-storage", "streamed", "--format", "json",
    ], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)["data"]["report"]
    assert report["schema_version"] == "pllm.reference_quality_benchmark.v1"
    assert report["candidates"][0]["same_token_decode"]["sample_count"] == 2
    assert "hello world" not in result.stdout


def test_quality_cli_reports_memory_rejection_without_success_file(tmp_path, monkeypatch, capsys):
    from pllm.cli import main
    from pllm.runtime import reference_benchmark
    from pllm.runtime.benchmark_memory import BenchmarkMemoryError

    config = tmp_path / "experiment.json"
    config.write_bytes(_experiment("memory-rejected", 8).canonical_bytes())
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["hello world"]))
    output = tmp_path / "quality.json"

    def reject(*args, **kwargs):
        raise BenchmarkMemoryError("benchmark memory preflight rejected: required exceeds available")

    monkeypatch.setattr(reference_benchmark, "run_reference_benchmark", reject)
    with pytest.raises(SystemExit) as error:
        main(["benchmark", "quality", "--experiment", str(config),
              "--prompts-file", str(prompts), "--output", str(output), "--format", "json"])
    assert error.value.code == 6
    captured = capsys.readouterr()
    assert captured.out == "" and not output.exists()
    diagnostic = json.loads(captured.err)
    assert diagnostic["error"]["code"] == "QUALITY_MEMORY"
    assert "required exceeds available" in diagnostic["error"]["message"]


@pytest.mark.rust
def test_streamed_quality_reference_matches_resident_metrics_and_cohort():
    pytest.importorskip("transformers")
    experiment = _experiment("streamed-tiny-w8", 8)
    resident = run_reference_benchmark([experiment], ["Hello"])
    streamed = run_reference_benchmark([experiment], ["Hello"], reference_storage="streamed")
    assert streamed["model"] == resident["model"]
    assert streamed["cohort"] == resident["cohort"]
    assert streamed["candidates"] == resident["candidates"]
    assert streamed["environment"]["reference_weight_storage"] == "module-streamed"
    loading = streamed["reference_loading"]
    assert 0 < loading["peak_active_weight_bytes"] <= loading["active_weight_bound_bytes"]


@pytest.mark.rust
def test_same_token_decode_uses_one_trajectory_and_matches_reference_storage():
    pytest.importorskip("transformers")
    experiments = [_experiment(f"quality-decode-w{bits}", bits).with_params(budget__max_new_tokens=4)
                   for bits in (8, 4)]
    resident = run_reference_benchmark(experiments, ["Hello"], decode_steps=3)
    streamed = run_reference_benchmark(experiments, ["Hello"], decode_steps=3,
                                       reference_storage="streamed")
    assert streamed["cohort"] == resident["cohort"]
    assert streamed["candidates"] == resident["candidates"]
    assert streamed["cohort"]["same_token_decode"]["teacher_forcing"] == "first-candidate-greedy"
    assert all(row["same_token_decode"]["sample_count"] == 3 for row in streamed["candidates"])
    assert "Hello" not in json.dumps(streamed)


@pytest.mark.rust
def test_quality_benchmark_preserves_prepared_residue_encoding():
    pytest.importorskip("transformers")
    from pllm.protocols import MaskedLinear

    plain = _experiment("quality-raw", 8)
    packed = plain.with_params(
        name="quality-residues",
        pipeline__linear=MaskedLinear(
            output_encoding="row_residues", request_encoding="stage_packed"),
    )
    report = run_reference_benchmark([plain, packed], ["Hello"], reference_storage="streamed")
    for metric in ("top1_agreement", "top_k_recall", "max_abs_logit_error"):
        assert report["candidates"][0][metric] == report["candidates"][1][metric]


def test_quality_preflights_checkpoint_and_aggregate_logit_memory(monkeypatch) -> None:
    experiment = _experiment("bounded-w8", 8)
    manifest = SimpleNamespace(
        metadata={
            "source_lock": {"files": [{"path": "model.safetensors", "size": 12 * 1024**3 + 1}]}
        },
        vocab_size=200064,
    )
    resolved = SimpleNamespace(path=Path("/unused"), manifest=manifest, source_lock_digest="0" * 64)
    monkeypatch.setattr("pllm.runtime.reference_benchmark.resolve_model", lambda _: resolved)
    with pytest.raises(ReferenceBenchmarkError, match="12 GiB"):
        run_reference_benchmark([experiment], ["A"])
    manifest.metadata["source_lock"]["files"][0]["size"] = 12 * 1024**3
    manifest.vocab_size = 200_000_000
    with pytest.raises(ReferenceBenchmarkError, match="512 MiB"):
        run_reference_benchmark([experiment], ["A"])


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
def test_quality_benchmark_uses_selected_metal_stage_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    import platform

    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or importlib.util.find_spec("mlx") is None
    ):
        pytest.skip("Apple Silicon and pllm.run[metal] are required")
    pytest.importorskip("transformers")
    pytest.importorskip("mlx.core")
    from pllm.kernels import AppleMetal
    from pllm.runtime.metal import MetalCompiledMatrix

    calls: list[int] = []
    original_clear = MetalCompiledMatrix.clear

    def counted_clear(self, inputs):
        calls.append(inputs.shape[0])
        return original_clear(self, inputs)

    monkeypatch.setattr(MetalCompiledMatrix, "clear", counted_clear)

    cpu = _experiment("quality-cpu", 8)
    metal = pllm.Experiment(
        name="quality-metal",
        pipeline=MaskedLinearCpu(
            cpu.pipeline.model,
            kernels=AppleMetal(min_rows=2),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        deployment=cpu.deployment,
        budget=cpu.budget,
    )
    report = run_reference_benchmark([cpu, metal], ["Hello"], top_k=5)
    assert calls and all(rows >= 2 for rows in calls)
    assert [row["kernel_backend"] for row in report["candidates"]] == [
        "pllm/cpu",
        "pllm/apple-metal-int8/v1",
    ]
    assert report["candidates"][0]["top1_agreement"] == report["candidates"][1]["top1_agreement"]
    assert (
        report["candidates"][0]["max_abs_logit_error"]
        == report["candidates"][1]["max_abs_logit_error"]
    )


@pytest.mark.rust
def test_quality_benchmark_preserves_selected_client_attention_numeric_path() -> None:
    pytest.importorskip("transformers")
    from pllm.preparation import PreparedInventory
    from pllm.protocols import ClientBundleTransport
    from pllm.roles import ClientLinearRoles

    plain = _experiment("quality-plain", 8)
    placed = pllm.Experiment(
        "quality-attention",
        MaskedLinearCpu(
            plain.pipeline.model,
            quantization=plain.pipeline.quantization,
            placement=ClientLinearRoles(["qkv_projection", "attention_output"]),
            inventory=PreparedInventory(),
            delivery=ClientBundleTransport(),
        ),
        plain.deployment,
        plain.budget,
    )
    report = run_reference_benchmark([plain, placed], ["Hello"], top_k=5)
    for metric in ("top1_agreement", "top_k_recall", "max_abs_logit_error"):
        assert report["candidates"][0][metric] == report["candidates"][1][metric]


@pytest.mark.rust
@pytest.mark.slow
@pytest.mark.parametrize("reference_storage,decode_steps", [("resident", 0), ("streamed", 3)])
def test_quality_cli_compares_typed_experiments_without_emitting_prompts(
    tmp_path, reference_storage, decode_steps,
) -> None:
    pytest.importorskip("transformers")
    prompt = "Hello"
    cohort = tmp_path / "prompts.json"
    cohort.write_text(json.dumps([prompt]), encoding="utf-8")
    candidates = []
    for name, bits in (("tiny-w4", 4), ("tiny-w8", 8)):
        path = tmp_path / f"{name}.json"
        path.write_bytes(_experiment(name, bits).with_params(budget__max_new_tokens=4).canonical_bytes())
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
        "--reference-storage",
        reference_storage,
        "--decode-steps",
        str(decode_steps),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=120)
    payload = json.loads(result.stdout)
    assert payload["command"] == "benchmark.quality"
    assert [row["weight_bits"] for row in payload["data"]["report"]["candidates"]] == [4, 8]
    if decode_steps:
        report = payload["data"]["report"]
        assert report["environment"]["reference_weight_storage"] == "module-streamed"
        assert all(row["same_token_decode"]["sample_count"] == 3 for row in report["candidates"])
    assert prompt not in result.stdout


@pytest.mark.rust
@pytest.mark.slow
@pytest.mark.parametrize(
    (
        "env_name",
        "model_id",
        "model_key",
        "evidence_file",
        "max_input",
        "max_new",
        "deployment_root",
    ),
    [
        (
            "PLLM_REAL_QWEN_PATH",
            "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
            "qwen2",
            "qwen2.5-0.5b-reference-quality-2026-09-24.json",
            16,
            2,
            "local://qwen2-reference-quality",
        ),
        (
            "PLLM_REAL_QWEN3_PATH",
            "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
            "qwen3",
            "qwen3-0.6b-reference-quality-2026-09-24.json",
            16,
            2,
            "local://qwen3-reference-quality",
        ),
        (
            "PLLM_REAL_PHI_PATH",
            "microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083",
            "phi4-mini",
            "phi4mini-reference-quality-2026-09-24.json",
            32,
            1,
            "local://phi-real-quality",
        ),
    ],
)
def test_pinned_reference_cohort_matches_retained_report(
    env_name: str,
    model_id: str,
    model_key: str,
    evidence_file: str,
    max_input: int,
    max_new: int,
    deployment_root: str,
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
            deployment=pllm.Deployment.local(root=deployment_root),
            budget=pllm.ExecutionBudget(
                requests=2,
                max_input_tokens=max_input,
                max_new_tokens=max_new,
            ),
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


@pytest.mark.rust
@pytest.mark.slow
@pytest.mark.parametrize(
    ("cohort_path", "evidence_file", "request_count"),
    [
        ("reference_prompts.json", "phi4mini-equalization-quality-2026-09-26.json", 2),
        ("phi4mini_holdout_prompts.json", "phi4mini-equalization-holdout-2026-09-26.json", 5),
    ],
)
def test_pinned_phi_public_equalization_reproduces_both_reference_cohorts(
    cohort_path: str,
    evidence_file: str,
    request_count: int,
) -> None:
    if not os.getenv("PLLM_REAL_PHI_PATH"):
        pytest.skip("pinned Phi checkpoint path is not configured")
    pytest.importorskip("transformers")
    from examples.benchmarks.phi4mini_equalized_quality import equalized
    from examples.benchmarks.phi4mini_reference_quality import w8a8

    root = Path(__file__).resolve().parents[1]
    prompts = json.loads((root / "examples/benchmarks" / cohort_path).read_text())
    evidence_text = (root / "docs/evidence" / evidence_file).read_text()
    assert all(prompt not in evidence_text for prompt in prompts)
    archived = json.loads(evidence_text)
    report = run_reference_benchmark(
        [
            w8a8.with_params(budget__requests=request_count),
            equalized.with_params(budget__requests=request_count),
        ],
        prompts,
    )
    assert report["model"] == archived["model"]
    assert report["cohort"] == archived["cohort"]
    for actual, expected in zip(report["candidates"], archived["candidates"], strict=True):
        assert actual["pipeline_digest"] == expected["pipeline_digest"]
        assert actual["configuration_digest"] == expected["configuration_digest"]
        assert actual["top1_agreement"] == expected["top1_agreement"]
        assert actual["top_k_recall"] == expected["top_k_recall"]
        assert abs(actual["max_abs_logit_error"] - expected["max_abs_logit_error"]) < 0.1
        if "public_equalization_profile_digest" in expected:
            assert (
                actual["public_equalization_profile_digest"]
                == expected["public_equalization_profile_digest"]
            )
            assert actual["public_calibration_digest"] == expected["public_calibration_digest"]
            assert actual["public_profile_bytes"] == expected["public_profile_bytes"]

"""Final-token quality uses the installed protected topology and shared metrics."""
import json
import subprocess
import sys

import pytest

from pllm import _native
from pllm.runtime.reference_benchmark import ReferenceBenchmarkError, _replicated_candidate, run_reference_benchmark
from pllm.runtime.replicated_source import admit_source
from test_replicated_sdk import experiment_fixture, native_execution


@pytest.mark.parametrize("family,pin", [
    ("qwen2", "e90df0711955251cb06f996518826bcf80d8cad5607187f8d99b975579473bd7"),
    ("qwen3", "268fdae236b2811a5adb33e351d779f5b46e54fe10a58c90342fab57649033af"),
    ("llama", "b1683e410cc9bd612cd37aeb0f19de419158c6b1cb8d185774000fef1a9d7df9"),
])
def test_bounded_program_identity_matches_prior_installed_wheel(tmp_path, family, pin):
    from safetensors.numpy import load_file

    experiment_fixture(tmp_path, family=family)
    root = tmp_path / "source"
    arrays = load_file(root / "model.safetensors")
    owner = _native.ReplicatedPublisher((root / "config.json").read_bytes(), 2, 2, b"s" * 32, b"c" * 32)
    try:
        for key, _ in owner.shapes:
            owner.append(key, 0, arrays[key].astype("<f4").tobytes())
        manifest = json.loads(owner.seal((3).to_bytes(4, "little") + (4).to_bytes(4, "little")))
        assert bytes(manifest["program"]).hex() == pin
    finally:
        owner.close()


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_wide_mlp_uses_native_chunked_dot_in_complete_decoder(tmp_path, family):
    experiment = experiment_fixture(tmp_path, family=family, intermediate_size=4864)
    source = admit_source(experiment)
    # Zero projections preserve the original embedding; wide shared activations
    # and the full down contraction still execute. Signed nonzero dot oracles
    # independently exercise the native linear compiler tests.
    assert native_execution(source, experiment, (3,), 2) == (2, 1)


@pytest.mark.parametrize("replicated_first", [True, False])
def test_quality_uses_common_prefixes_and_retains_unknown_logit_metrics(tmp_path, replicated_first):
    pytest.importorskip("transformers")
    from examples.papers.prepared import experiment as prepared_experiment

    replicated = experiment_fixture(tmp_path, active=True)
    prepared = prepared_experiment(replicated.pipeline.model, replicated.budget)
    experiments = [replicated, prepared] if replicated_first else [prepared, replicated]
    report = run_reference_benchmark(experiments, ["hello world"], decode_steps=1,
                                     reference_storage="streamed")
    assert report["cohort"]["same_token_decode"]["teacher_configuration_digest"] == experiments[0].configuration_digest()
    assert report["cohort"]["input_token_counts"] == [2]
    assert report["limitations"][0].startswith("local worker execution;")
    row = next(row for row in report["candidates"] if row["name"] == replicated.name)
    assert row["execution"] == "three-worker-final-token-full-prefix"
    assert row["diagnostic_requests"] == 2 and row["output_selections_per_request"] == 1
    for result in (row, row["same_token_decode"]):
        assert result["sample_count"] == 1 and 0 <= result["top1_agreement"] <= 1
        assert result["top_k_recall"] is None and result["max_abs_logit_error"] is None
    encoded = json.dumps(report, allow_nan=False)
    assert "hello world" not in encoded and '"token_ids"' not in encoded and '"logits"' not in encoded


def test_quality_budgets_reject_before_worker_launch(tmp_path, monkeypatch):
    from pllm.runtime import servers

    experiment = experiment_fixture(tmp_path)
    source = admit_source(experiment)

    def launch(*args, **kwargs):
        pytest.fail("invalid quality workload reached worker launch")

    monkeypatch.setattr(servers, "build_roles", launch)
    with pytest.raises(ReferenceBenchmarkError, match="request budget"):
        run_reference_benchmark([experiment.with_params(budget__requests=1)], ["hello"], decode_steps=1)
    with pytest.raises(ReferenceBenchmarkError, match="input budget"):
        _replicated_candidate(experiment, (" ".join(["hello"] * 8),), source, 1, None)
    with pytest.raises(ReferenceBenchmarkError, match="continuation"):
        _replicated_candidate(experiment, ("hello",), source, 1, [(999,)])


def test_large_public_geometry_rejects_before_value_decoding_or_workers(tmp_path, monkeypatch):
    from pllm.runtime import delegated_source, servers

    experiment = experiment_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("metadata rejection reached tensor values or worker launch")

    monkeypatch.setattr(delegated_source, "_source_float_frame", forbidden)
    monkeypatch.setattr(servers, "build_roles", forbidden)
    config_path = tmp_path / "source" / "config.json"
    config = json.loads(config_path.read_bytes())
    config.update(hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                  num_attention_heads=14, num_key_value_heads=2, vocab_size=151936)
    config.pop("head_dim", None)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source expansion budget"):
        run_reference_benchmark([experiment], ["hello"])


def test_quality_failure_retires_all_three_workers(tmp_path, monkeypatch):
    from pllm.runtime import replicated_source, servers

    experiment = experiment_fixture(tmp_path)
    source = admit_source(experiment)
    original_build, original_publisher = servers.build_roles, replicated_source.ReplicatedSource.publisher
    topologies = []

    class RejectFinish:
        def __init__(self, owner):
            self.owner = owner

        def __getattr__(self, name):
            return getattr(self.owner, name)

        def finish(self, *frames):
            self.owner.finish(*frames)
            raise ValueError("controlled final disclosure rejection")

    def publish(self, *args):
        owner, manifest = original_publisher(self, *args)
        return RejectFinish(owner), manifest

    def build(*args, **kwargs):
        topology = original_build(*args, **kwargs)
        topologies.append(topology)
        return topology

    monkeypatch.setattr(replicated_source.ReplicatedSource, "publisher", publish)
    monkeypatch.setattr(servers, "build_roles", build)
    with pytest.raises(ValueError, match="final disclosure rejection"):
        _replicated_candidate(experiment, ("hello",), source, 1, None)
    assert len(topologies) == 1 and len(topologies[0].statuses) == 3
    assert all(not status.running for status in topologies[0].statuses)


def test_quality_cli_keeps_missing_metrics_unknown(tmp_path):
    pytest.importorskip("transformers")
    experiment = experiment_fixture(tmp_path)
    config, prompts, output = (tmp_path / name for name in ("experiment.json", "prompts.json", "report.json"))
    config.write_bytes(experiment.canonical_bytes())
    prompts.write_text(json.dumps(["hello world"]))
    result = subprocess.run([sys.executable, "-m", "pllm", "benchmark", "quality",
        "--experiment", str(config), "--prompts-file", str(prompts), "--decode-steps", "1",
        "--reference-storage", "streamed", "--output", str(output)],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert "recall unknown" in result.stdout and "logit error unknown" in result.stdout
    report = json.loads(output.read_bytes())
    assert report["candidates"][0]["same_token_decode"]["top1_agreement"] == 1

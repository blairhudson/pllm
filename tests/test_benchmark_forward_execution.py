"""Real installed-worker forward integration; run only after source/review gates.

No mocked compiled owner, native contract, admission, execution, or subprocess.
This exercises report serialization and output-contract checks, NOT an independent
numeric oracle or a comparative-performance/precise-RSS assertion.
"""
import json
from pathlib import Path

import pytest

from pllm import Deployment, Experiment, ForwardBudget, Model
from pllm._benchmark_contract import digest
from pllm.kernels import Cpu
from pllm.metrics import BenchmarkCase, BenchmarkRun, BenchmarkSuite, BenchmarkTarget, Sampling, benchmark_suite
from pllm.numeric import SourceFloat32
from pllm.profiles import ClientOnlyCpu
from pllm.sources.generated_encoder import write_checkpoint


RUNTIME_MEMORY_BYTES = 128 << 20
# 704 MiB = 320 MiB overhead + 3 * 128 MiB runtime; shared partition unchanged.
BENCHMARK_MEMORY_BYTES = 704 << 20


@pytest.mark.integration
def test_actual_production_metadata_preflight_all_saved_requests(tmp_path):
    from pllm.runtime.benchmark_memory import benchmark_memory
    root = write_checkpoint(tmp_path / "checkpoint")
    experiment = Experiment("generated-forward-preflight",
        ClientOnlyCpu(Model.path(str(root)), kernels=Cpu(threads=2), numeric=SourceFloat32()),
        Deployment.local(root=str(tmp_path / "runtime")), ForwardBudget(requests=4, max_input_tokens=6))
    report = benchmark_memory(experiment, memory_budget_bytes=BENCHMARK_MEMORY_BYTES)
    assert report["memory_budget_bytes"] == BENCHMARK_MEMORY_BYTES
    assert report["runtime_memory_bytes"] == RUNTIME_MEMORY_BYTES
    assert report["metadata_only"] is True and report["host_reservation_admitted"] is True
    assert report["execution_admitted"] is report["numerical_finite_validated"] is report["performance_measured"] is False
    source = report["forward_source"]
    assert source["schema_version"] == "pllm.forward_source_description.v1"
    assert source["execution_admitted"] is source["numerical_finite_validated"] is False
    assert source["memory_bytes"] == report["runtime_memory_bytes"]
    assert source["memory_bytes"] == RUNTIME_MEMORY_BYTES
    assert source["requests"] == 4 and source["max_sequence"] == 6
    reservation = source["lifetime_reservation"]
    assert reservation["admission_bytes"] <= RUNTIME_MEMORY_BYTES
    assert reservation["requests"] == 4
    assert reservation["required_bytes"] >= reservation["max_call_bytes"] + 4 * (
        reservation["output_bytes_per_result"] + reservation["result_metadata_bytes_per_result"])
    assert reservation["max_call_bytes"] == source["quote"]["required_bytes"] + source["python_workspace_bytes"]
    assert report["estimate"]["lifetime_reservation"] == reservation
    assert report["estimate"]["required_bytes"] > reservation["admission_bytes"]
    assert report["estimate"]["required_bytes"] <= report["memory_budget_bytes"]
    estimate = report["estimate"]
    assert estimate["owners"]["production_lifetime_with_margin"] == reservation["admission_bytes"]
    assert estimate["required_bytes"] == sum(estimate["owners"].values())
    assert estimate["required_bytes"] <= BENCHMARK_MEMORY_BYTES
    assert source["metadata_bytes"] > 0 and source["header_bytes"] > 0
    assert source["snapshot_bytes"] >= source["source_tensor_bytes"] > 0
    assert source["tensor_count"] > 0
    expected = {"last_hidden_state": [1, 6, 8], "pooler_output": [1, 8], "logits": [1, 3]}
    assert set(source["outputs"]) == set(expected)
    for name, shape in expected.items():
        assert source["outputs"][name]["dtype"] == "float32"
        assert source["outputs"][name]["shape"] == shape
    for value in source["identities"].values():
        assert isinstance(value, str) and len(value) == 64
    assert source["execution_metadata"]["backend"] == "numpy_einsum_unoptimized_f32_v1"
    assert "metrics" not in report and "samples" not in report
    json.dumps(report, allow_nan=False)


@pytest.mark.integration
@pytest.mark.parametrize("preflight_only", [False, True])
def test_actual_cli_generated_example_factory_and_standard_flags(tmp_path, monkeypatch, capsys, preflight_only):
    from pllm._cli.app import main
    example = Path(__file__).resolve().parents[1] / "examples/benchmarks/generated_encoder.py"
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "target/generated-encoder/benchmark.json"
    arguments = ["benchmark", "run", "--experiment", f"{example}:candidate", "--factory", "--trust-python",
        "--input-file", "target/generated-encoder/inputs.json", "--output", str(path), "--no-input", "--format", "json"]
    if preflight_only:
        arguments.append("--preflight-only")
    assert main(arguments) is None
    data = json.loads(capsys.readouterr().out)["data"]
    document = json.loads(path.read_text())
    if preflight_only:
        assert data == document and document["preflight_only"] is True
        assert document["execution_admitted"] is document["numerical_finite_validated"] is False
        assert document["performance_measured"] is False
        assert document["memory"]["candidate"]["forward_source"]["requests"] == 1
        assert "results" not in document
    else:
        assert data["report"] == document
        assert BenchmarkRun.load(path).to_dict() == document
        assert document["complete"] is True
        assert document["results"][0]["status"] == "completed"
        metrics = document["results"][0]["metrics"]
        assert not any(any(word in metric["id"].lower() for word in ("ttft", "decode", "generated", "generation", "token"))
                       for metric in metrics)


@pytest.mark.integration
def test_generated_encoder_actual_suite_worker_finalized_report(tmp_path):
    root = write_checkpoint(tmp_path / "checkpoint")
    experiment = Experiment("generated-forward-integration",
        ClientOnlyCpu(Model.path(str(root), model_id="generated-encoder"),
                      kernels=Cpu(threads=2), numeric=SourceFloat32()),
        Deployment.local(root=str(tmp_path / "runtime")),
        ForwardBudget(requests=1, max_input_tokens=6))
    case = BenchmarkCase("candidate", BenchmarkTarget.forward(experiment))
    suite = BenchmarkSuite((case,), Sampling(processes=1, warmups=0, repetitions=1,
        threads=2, memory_bytes=BENCHMARK_MEMORY_BYTES), name="generated-forward-integration")
    private = {"schema_version": "pllm.forward_inputs.v1", "tokens": [[1, 2, 3]],
        "positions": [[0, 1, 2]], "segments": [[0, 0, 1]], "attention_mask": [[True, True, False]]}
    path = tmp_path / "benchmark.json"
    run = benchmark_suite(suite, inputs={"candidate": private}, output=path)
    document = run.to_dict()
    assert document["schema"] == "pllm.benchmark_run.v1" and document["complete"] is True
    assert BenchmarkRun.load(path).digest == run.digest
    assert json.loads(path.read_text()) == document
    assert len(document["results"]) == 1
    row = document["results"][0]
    assert row["schema_version"] == "pllm.benchmark_result.v2"
    assert row["status"] == "completed", row["execution"]["failure_code"]
    assert row["scope"] == "full_model" and row["warmups"] == 0 and row["repetitions"] == 1
    execution = row["execution"]
    assert execution["case"] == case.to_spec()
    assert execution["isolation"]["fresh_process"] is True
    assert execution["isolation"]["controller_included"] is False
    assert execution["failure_code"] is None
    assert execution["admission"]["admitted"] is True
    assert execution["admission"]["scope"] == "conservative_forward_envelope_reservation"
    details, comparison = execution["details"], execution["comparison"]
    identities = details["identities"]
    for name in ("source_digest", "checkpoint_digest", "source_lock_digest", "plan_digest", "schedule_digest",
                 "binding_digest", "numeric_digest", "composition_digest", "configuration_digest", "workload_digest"):
        assert isinstance(identities[name], str) and len(identities[name]) == 64
        int(identities[name], 16)
    assert row["plan_lock_digest"] == identities["plan_digest"] == details["plan_digest"]
    assert identities["schedule_digest"] == details["schedule_digest"]
    assert identities["binding_digest"] == details["binding_digest"]
    assert comparison["source_digest"] == identities["source_lock_digest"] == row["model"]["source_lock_digest"]
    assert identities["checkpoint_digest"] == row["model"]["checkpoint_digest"]
    assert comparison["numeric_digest"] == identities["numeric_digest"]
    assert comparison["workload_digest"] == identities["workload_digest"] == row["workload_digest"]
    assert all(isinstance(value, str) and len(value) == 64 for value in comparison.values())
    assert details["execution_metadata"] == {"backend": "numpy_einsum_unoptimized_f32_v1",
        "requested_compute_threads": 2, "effective_compute_threads": 1,
        "thread_policy": "serial_compute_within_requested_budget"}
    assert comparison["measurement_digest"] == digest({
        "scope": "complete_forward_excluding_compile_import", "validation": "outside_timing",
        "output_contract": "pllm.forward.encoder_outputs.v1", "execution_metadata": details["execution_metadata"]})
    assert details["output_contract"] == "pllm.forward.encoder_outputs.v1"
    assert details["sequence"] == 3 and details["visible_key_count"] == 2
    assert details["attempted_requests"] == details["completed_requests"] == 1
    assert details["stateless"] is True
    assert details["privacy_scope"] == "public_client_local_no_protocol_privacy"
    expected = {"last_hidden_state": [1, 3, 8], "pooler_output": [1, 8], "logits": [1, 3]}
    assert set(details["outputs"]) == set(expected)
    for name, shape in expected.items():
        assert details["outputs"][name]["shape"] == shape
        assert details["outputs"][name]["dtype"] == "F32"
        assert details["outputs"][name]["nbytes"] == {"last_hidden_state": 96, "pooler_output": 32, "logits": 12}[name]
        assert len(details["outputs"][name]["content_commitment"]) == 64
    assert execution["oracle"]["kind"] == "forward_output_contract_finite_checks_not_numeric_oracle"
    assert execution["oracle"]["outcome"] == "passed" and execution["oracle"]["checked_values"] == 35
    assert details["compile_import"]["wall_seconds"] >= 0
    assert details["compile_import"]["cpu_seconds"] >= 0
    assert details["resource_quote"]["native_quote_scope"] == "logical_payload_not_process_RSS"
    assert details["resource_quote"]["lifetime_reservation"]["requests"] == experiment.budget.requests
    # Inspect metrics AFTER worker finalization and controller serialization.
    metrics = {metric["id"]: metric for metric in row["metrics"]}
    assert {"target.latency", "target.cpu", "process.peak_rss", "process.cpu"} <= set(metrics)
    assert metrics["target.latency"]["measurement_scope"] == "complete_forward_excluding_compile_import"
    assert metrics["target.latency"]["sample_count"] == 1
    assert metrics["process.peak_rss"]["measurement_scope"] == "process_lifetime_high_water"
    for metric in metrics.values():
        assert metric["component"] != "pllm/throughput"
        assert metric["phase"] not in {"prefill", "decode"}
        assert not any(word in metric["id"].lower() for word in ("ttft", "decode", "generated", "generation", "token"))
    serialized = path.read_text()
    assert "little_endian_content" not in serialized and '"tokens":' not in serialized
    assert any("not an independent numeric oracle" in note for note in row["limitations"])

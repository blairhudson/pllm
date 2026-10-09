from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import pllm
from pllm._benchmark_worker import Measurement
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, benchmark_suite
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.benchmark_targets import execute_target


@pytest.fixture
def experiment(tmp_path):
    return pllm.Experiment("targets", MaskedLinearCpu(pllm.Model.tiny()),
        pllm.Deployment.local(root=str(tmp_path / "roles")), pllm.ExecutionBudget(2, 128, 2))


def measurement(tmp_path, *, inputs=None):
    return Measurement(dict(sampling=Sampling(processes=1).to_spec(), salt="ab" * 32,
                            inputs=inputs or {}, directory=str(tmp_path)))


@pytest.mark.parametrize("digits", [4, 8])
@pytest.mark.parametrize("bits", [16, 24, 32])
def test_digit_gemm_extremes_carries_and_partial_sum_bound(bits, digits):
    from pllm.runtime.benchmark_digit_gemm import DigitGEMM
    # Cross the worst-case block boundary, with -128 and maximal unsigned values.
    width = ((1 << 24) - 1) // (128 * (1 << (digits - 1))) + 1
    weights = np.stack([np.full(width, -128, np.int8), np.full(width, 127, np.int8),
                        np.resize(np.array([-128, 127, 0], np.int8), width)])
    row = np.resize(np.array([0, 1, (1 << bits) - 1, (1 << (bits - 1)), 127, 128], np.uint32), width)
    values = np.stack([row, np.full(width, (1 << bits) - 1, np.uint32)])
    expected = (values.astype(np.int64) @ weights.astype(np.int64).T) % (1 << bits)
    actual = DigitGEMM(weights, digit_bits=digits).modular(values, 1 << bits)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind,options", [
    ("geometry", {}), ("preparation", {}), ("preparation", {"chunk_rows": 3}),
    ("preparation", {"kernel": "digit4"}), ("preparation", {"kernel": "digit8", "chunk_rows": 3}),
    ("preparation", {"storage": "paged"}), ("stage", {}),
    ("artifacts", {"storage": "shared"}), ("artifacts", {"storage": "copied", "operation": "bind"}),
    ("artifacts", {"weights": "checkpoint"}),
    ("state", {}), ("state", {"storage": "sealed_local"}),
    ("state", {"storage": "sealed_local", "operation": "attention"}),
    ("state", {"storage": "sealed_local", "operation": "restart"}),
    ("state", {"storage": "sealed_stream"}),
    ("state", {"storage": "sealed_stream", "operation": "attention"}),
    ("state", {"storage": "sealed_stream", "operation": "restart"}),
    ("projection", {"construction": "layer_kv", "parameters": {"fixed_bytes": 1024, "live_layers": 1}}),
    ("projection", {"construction": "restart", "parameters": {"bootstrap_bytes": 1000, "publication_bytes": 100,
                    "control_bytes": 100, "resumed_bytes": 5, "resumptions": 200}}),
])
def test_targets_use_bounded_real_primitives(experiment, tmp_path, kind, options):
    m = measurement(tmp_path)
    execute_target(getattr(BenchmarkTarget, kind)(experiment, **options), m)
    assert m.admission["admitted"]
    assert m.oracle["outcome"] == "passed"
    assert m.oracle["checked_values"] > 0
    assert len(m.samples) == 1
    assert m.details["timings"]["cpu_seconds"][0] >= 0


def test_sealed_state_binding_corruption_rollback_and_failure(tmp_path):
    from pllm.runtime.benchmark_state_targets import SealedStateFixture
    layout = [{"layer": 0, "kind": "key", "shape": [1, 1, 2, 4]}]
    key = b"x" * 32
    store = SealedStateFixture(tmp_path, key, "binding", layout)
    value = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)
    old = store.commit([value], 0)
    with pytest.raises(ValueError, match="injected"):
        store.commit([value + 1], 1, fail_after=0)
    np.testing.assert_array_equal(next(store.read(old)), value)
    new = store.commit([value + 1], 1)
    with pytest.raises(ValueError):
        list(store.read({**new, "revision": 0}))
    with pytest.raises(ValueError):
        list(SealedStateFixture(tmp_path, key, "other-binding", layout).read(new))
    path = tmp_path / "1" / "0"
    sealed = bytearray(path.read_bytes())
    sealed[-1] ^= 1
    path.write_bytes(sealed)
    with pytest.raises(ValueError):
        list(store.read(new))
    assert not list(tmp_path.glob("pending-*"))


def test_native_wide_stage_selects_wrap32(experiment, tmp_path):
    from pllm.sources import _tiny_model_config
    config = {**_tiny_model_config(), "intermediate_size": 4096}
    m = measurement(tmp_path)
    execute_target(BenchmarkTarget.preparation(experiment, role="mlp_down", config=config), m)
    assert m.details["ring"]["wire_bits"] == 32
    assert m.oracle["outcome"] == "passed"


def test_pipeline_aliases_survive_target_serialization(experiment):
    from pllm.preparation import PreparedInventory
    from pllm.state import ClientPrefixReuse
    experiment = experiment.with_params(pipeline=experiment.pipeline.with_params(
        inventory=PreparedInventory(refill="on-demand"),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128)))
    target = BenchmarkTarget.from_spec(BenchmarkTarget.response(experiment).to_spec())
    alias = target.with_params(pipeline__inventory__stage_window=4)
    explicit = target.with_params(pipeline__components__inventory__stage_window=4)
    assert alias.digest == explicit.digest
    stripped = alias.with_params(pipeline__cache=None)
    assert "cache" not in stripped.to_spec()["parameters"]["model"]["experiment"]["pipeline"]["components"]
    with pytest.raises(ValueError, match="duplicate"):
        target.with_params(pipeline__cache=None, pipeline__components__cache=None)
    with pytest.raises(ValueError, match="overlapping"):
        target.with_params(pipeline__cache=None, pipeline__cache__max_bytes=2 << 20)


def test_isolated_suite_parity_negative_results_and_private_input_redaction(experiment, tmp_path):
    suite = BenchmarkSuite((
        BenchmarkCase("native", BenchmarkTarget.preparation(experiment)),
        BenchmarkCase("digit", BenchmarkTarget.preparation(experiment, kernel="digit4")),
        BenchmarkCase("unsupported", BenchmarkTarget.stage(experiment, execution="transport")),
    ), Sampling(processes=1))
    result = benchmark_suite(suite, output=tmp_path / "evidence.json")
    assert [r.to_dict()["status"] for r in result.results] == ["completed", "completed", "unsupported"]
    assert result.compare("native", "digit", "target.cpu")["status"] == "comparable"
    assert result.compare("native", "unsupported", "target.cpu")["status"] == "unranked"
    from pllm.metrics import BenchmarkRun
    assert BenchmarkRun.load(tmp_path / "evidence.json").digest == result.digest
    private = "never-archive-this-private-value"
    failed = benchmark_suite(BenchmarkSuite((suite.cases[0],), Sampling(processes=1)),
                             inputs={"native": {"private": private}})
    assert private not in json.dumps(failed.to_dict())
    assert failed.results[0].to_dict()["status"] == "unsupported"


@pytest.mark.integration
def test_response_target_uses_existing_roles_and_seven_scoped_metrics(experiment, tmp_path):
    from pllm.metrics import Scenario
    m = measurement(tmp_path, inputs={"prompt": "Hi"})
    m.sampling = Sampling(processes=1, memory_bytes=2 << 30)
    execute_target(BenchmarkTarget.response(experiment, scenario=Scenario.repeat(max_output_tokens=2)), m)
    assert m.oracle["outcome"] == "passed"
    from pllm.runtime.benchmark_response_target import SERVING_METRICS
    metrics = {v["id"]: v for v in m.metrics}
    assert set(SERVING_METRICS) <= set(metrics)
    assert metrics["network.all_link_stream_bytes"]["value"] is None
    assert metrics["network.covered_body_bytes"]["value"] > 0
    assert len(m.details["report"]["runs"]) == 2


@pytest.mark.integration
@pytest.mark.parametrize("owner", ["python", "native"])
@pytest.mark.parametrize("history", [False, True])
def test_response_target_research_stage_uses_real_roles_and_restores_adapter(experiment, tmp_path, owner, history):
    from pllm.metrics import Scenario
    from pllm.protocols import MaskedLinear
    from pllm.runtime.model_binding import CompiledRuntimeModel
    original = CompiledRuntimeModel.runtime
    experiment = experiment.with_params(pipeline__linear=MaskedLinear(
        request_encoding="stage_packed", output_encoding="row_residues", prefill_chunk_rows=4))
    first = [{"role": "user", "content": "Hi"}]
    second = first + [{"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Bye"}]
    scenario = Scenario.sequence(count=2, max_output_tokens=2) if history else Scenario.cold(max_output_tokens=2)
    target = BenchmarkTarget.response(experiment, scenario=scenario,
        client_stage={"owner": owner, "role": "mlp_gate_up", "layer": 0})
    assert BenchmarkTarget.from_spec(target.to_spec()).digest == target.digest
    m = measurement(tmp_path, inputs={"prompts": [first, second]} if history else {"prompt": first})
    m.sampling = Sampling(processes=1, memory_bytes=2 << 30)
    execute_target(target, m)
    assert m.oracle["outcome"] == "passed"
    records = m.details["client_stage_research"]["bindings"]
    assert records and sum(row["calls"] for row in records) == 2 * scenario.count
    assert sum(row["rows"] for row in records) > 2
    assert len(m.details["report"]["runs"]) == scenario.count
    assert CompiledRuntimeModel.runtime is original


@pytest.mark.parametrize("inputs", [
    {"prompt": [{"role": "user", "content": "private", "extra": "secret"}]},
    {"prompt": "x" * 16385},
    {"prompts": [[{"role": "user", "content": "private"}]]},
])
def test_response_contexts_reject_before_admission_without_disclosing_input(experiment, tmp_path, inputs):
    from pllm._benchmark_contract import TargetUnavailable
    from pllm.metrics import Scenario
    scenario = Scenario.sequence(count=2) if "prompts" in inputs else Scenario.cold()
    m = measurement(tmp_path, inputs=inputs)
    with pytest.raises(TargetUnavailable, match="^response_input_contract$"):
        execute_target(BenchmarkTarget.response(experiment, scenario=scenario), m)
    assert m.admission["admitted"] is None
    assert m.admission["required_bytes"] is None
    assert "topology_admission" not in m.details


def test_research_stage_selection_is_explicit_bounded_and_retires_on_error(experiment):
    from pllm.runtime.benchmark_client_stage import client_stage_execution
    from pllm.runtime.model_binding import CompiledRuntimeModel
    base = BenchmarkTarget.response(experiment)
    assert "client_stage" not in base.to_spec()["parameters"]
    selection = {"owner": "native", "role": "mlp_gate_up", "layer": 0}
    for change in ({"owner": "automatic"}, {"layer": -1}, {"layer": True}, {"callback": "anything"}):
        with pytest.raises(ValueError):
            BenchmarkTarget.response(experiment, client_stage={**selection, **change})
    original = CompiledRuntimeModel.runtime
    with pytest.raises(RuntimeError, match="injected"):
        with client_stage_execution(selection):
            raise RuntimeError("injected")
    assert CompiledRuntimeModel.runtime is original


@pytest.mark.integration
@pytest.mark.parametrize("rows", [0, 7])
def test_response_mlp_selector_measures_real_execution_and_restores_owners(experiment, tmp_path, rows):
    from pllm.metrics import Scenario
    from pllm.runtime.model_binding import CompiledRuntimeModel
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    original, forward = CompiledRuntimeModel.runtime, SemanticDecoderRuntime._forward
    target = BenchmarkTarget.response(experiment, scenario=Scenario.repeat(max_output_tokens=2),
                                      client_mlp={"tile_rows": rows})
    assert BenchmarkTarget.from_spec(target.to_spec()).digest == target.digest
    m = measurement(tmp_path, inputs={"prompt": "Hi there."})
    m.sampling = Sampling(processes=1, memory_bytes=2 << 30)
    execute_target(target, m)
    records = m.details["client_mlp_research"]["bindings"]
    assert sum(row["prefill_calls"] for row in records) == 2
    assert all(row["metadata_python_bytes"] > 0 for row in records)
    assert (sum(row["tiles"] for row in records) > 0) == bool(rows)
    assert CompiledRuntimeModel.runtime is original and SemanticDecoderRuntime._forward is forward
    assert m.oracle["outcome"] == "passed"


def test_research_mlp_selector_failure_cleanup_and_bounds(experiment):
    from pllm.runtime.benchmark_client_mlp import client_mlp_execution
    from pllm.runtime.model_binding import CompiledRuntimeModel
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    for bad in (True, -1, 257):
        with pytest.raises(ValueError):
            BenchmarkTarget.response(experiment, client_mlp={"tile_rows": bad})
    original, forward = CompiledRuntimeModel.runtime, SemanticDecoderRuntime._forward
    with pytest.raises(RuntimeError, match="cancel"):
        with client_mlp_execution({"tile_rows": 16}):
            raise RuntimeError("cancel")
    assert CompiledRuntimeModel.runtime is original and SemanticDecoderRuntime._forward is forward

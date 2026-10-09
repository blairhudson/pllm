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

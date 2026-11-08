"""Forward benchmark contracts; execution boundaries doubled, never numeric parity."""
from copy import deepcopy
import json

import pytest

from pllm._benchmark_contract import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling
from pllm._cli.app import build_parser
from pllm._cli.errors import ResolutionError
from pllm.runtime.benchmark_cli import run_suite_factory_cli


@pytest.fixture
def experiment(tmp_path):
    from pllm import Deployment, Experiment, ForwardBudget, Model
    from pllm.kernels import Cpu
    from pllm.numeric import SourceFloat32
    from pllm.profiles import ClientOnlyCpu
    return Experiment("forward-test", ClientOnlyCpu(Model.path(str(tmp_path / "checkpoint")),
        kernels=Cpu(threads=2), numeric=SourceFloat32()),
        Deployment.local(root=str(tmp_path / "runtime")), ForwardBudget(1, 6))


@pytest.fixture
def suite(experiment):
    return BenchmarkSuite((BenchmarkCase("candidate", BenchmarkTarget.forward(experiment)),),
                          Sampling(processes=1, threads=2))


def test_closed_forward_target_roundtrip_and_source_config(experiment):
    target = BenchmarkTarget.forward(experiment)
    assert target.kind == "forward"
    assert set(target.to_spec()["parameters"]) == {"model"}
    assert target.to_spec()["parameters"]["model"]["config"] is None
    assert BenchmarkTarget.from_spec(target.to_spec()).digest == target.digest
    for name, value in (("scenario", {}), ("config", {}), ("backend", "native")):
        spec = deepcopy(target.to_spec())
        spec["parameters"][name] = value
        with pytest.raises(ValueError):
            BenchmarkTarget.from_spec(spec)
    spec = deepcopy(target.to_spec())
    spec["parameters"]["model"]["config"] = {"hidden_size": 8}
    with pytest.raises(ValueError, match="overrides"):
        BenchmarkTarget.from_spec(spec)
    with pytest.raises(ValueError, match="forward"):
        BenchmarkTarget.geometry(experiment)


def test_decoder_target_does_not_become_forward():
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.profiles import ClientOnlyCpu
    decoder = Experiment("decoder", ClientOnlyCpu(Model.tiny()), Deployment.local(root="unused"),
                         ExecutionBudget(1, 6, 2))
    original = BenchmarkTarget.response(decoder)
    assert "numeric" not in original.to_spec()["parameters"]["model"]["experiment"]["pipeline"]["components"]
    assert BenchmarkTarget.from_spec(original.to_spec()).digest == original.digest
    with pytest.raises(ValueError, match="forward"):
        BenchmarkTarget.forward(decoder)


def args(*options):
    return build_parser().parse_args(["benchmark", "run", *map(str, options)])


def test_parser_input_file_is_explicit_without_changing_decoder_defaults(tmp_path):
    legacy = args()
    assert (legacy.warmups, legacy.repetitions, legacy.max_output_tokens, legacy.timeout) == (0, 1, 24, 900)
    assert not hasattr(legacy, "benchmark_explicit_options")
    selected = args("--input-file", tmp_path / "inputs.json")
    assert "input_file" in selected.benchmark_explicit_options


@pytest.mark.parametrize("options", [
    ("--prompt", "private"), ("--prompt-file", "private.txt"),
    ("--max-output-tokens", "24"), ("--temperature", "0"),
    ("--prefill-cache-mib", "0"), ("--prefill-cache-mode", "exact"),
    ("--backend", "native"), ("--wan",), ("--wan-estimate",),
    ("--inventory-policy", "prewarm"), ("--bundle-compression", "none"),
    ("--show-dashboard",), ("--tiny",), ("--model", "anything"),
])
def test_explicit_decoder_flags_reject_before_input_read(suite, options):
    with pytest.raises(ResolutionError) as error:
        run_suite_factory_cli(suite, args(*options), "json", True)
    assert error.value.code == "BENCHMARK_SUITE_OVERRIDE"


def test_default_decoder_flags_do_not_reject_forward_dry_run(suite, tmp_path, capsys):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"schema_version": "pllm.forward_inputs.v1", "tokens": [[1, 2, 3]],
        "segments": [[0, 0, 1]], "positions": [[0, 1, 2]], "attention_mask": [[True, True, False]]}))
    run_suite_factory_cli(suite, args("--input-file", path), "json", True)
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["suite"] == suite.to_spec()
    assert data["report_schema"] == "pllm.benchmark_run.v1"
    assert '"tokens":' not in json.dumps(data)


@pytest.mark.parametrize("batched", [False, True])
def test_cli_reuses_published_input_schema_without_flattening(suite, tmp_path, monkeypatch, batched):
    import numpy as np
    from pllm.runtime.benchmark_cli import _forward_suite_inputs
    tokens, mask = [1, 2, 3], [True, True, False]
    original = {"schema_version": "pllm.forward_inputs.v1", "tokens": [tokens] if batched else tokens,
                "attention_mask": [mask] if batched else mask}
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(original))
    for name in ("array", "asarray", "arange", "full"):
        monkeypatch.setattr(np, name, lambda *a, **kw: pytest.fail("array allocated in CLI input admission"))
    assert _forward_suite_inputs(suite, args("--input-file", path)) == {"candidate": original}


@pytest.mark.parametrize("payload", [
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[true]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[1]],"attention_mask":[[1]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[1]],"tokens":[[2]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[1]],"unknown":0}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[NaN]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[1,2]],"segments":[[0]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[[1]]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[1,2,3,4,5,6,7]]}',
    b'{"schema":"pllm.forward_inputs.v1","tokens":[[1]]}',
])
def test_cli_reuses_strict_production_input_admission(suite, tmp_path, payload):
    path = tmp_path / "inputs.json"
    path.write_bytes(payload)
    with pytest.raises(ResolutionError) as error:
        run_suite_factory_cli(suite, args("--input-file", path), "json", True)
    assert error.value.code == "BENCHMARK_FORWARD_INPUTS"


def test_input_reader_bounds_read_before_parsing(monkeypatch, tmp_path):
    from pllm.runtime import forward_inputs
    from io import BytesIO
    from pathlib import Path
    calls = []

    class Stream(BytesIO):
        def read(self, size=-1):
            calls.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *a, **kw: Stream(b" " * ((1 << 20) + 2)))
    with pytest.raises(ValueError):
        forward_inputs.read_forward_inputs(tmp_path / "large.json")
    assert calls == [(1 << 20) + 1]


def test_forward_workload_commitment_binds_every_private_field(tmp_path):
    from pllm._benchmark_worker import Measurement
    commitments = []
    for changed in ({}, {"tokens": [[2]]}, {"positions": [[1]]}, {"segments": [[1]]},
                    {"attention_mask": [[False]]}):
        inputs = {"schema_version": "pllm.forward_inputs.v1", "tokens": [[1]], "positions": [[0]],
                  "segments": [[0]], "attention_mask": [[True]], **changed}
        m = Measurement({"sampling": Sampling(processes=1).to_spec(), "salt": "ab" * 32,
                         "inputs": inputs, "directory": str(tmp_path)})
        # Failed targets retain ordinary private workload binding too.
        m.finalize_workload(type("Target", (), {"kind": "forward"})(), completed=False)
        assert m.details["identities"]["workload_digest"] == m.comparison["workload_digest"]
        commitments.append(m.comparison["workload_digest"])
    assert len(set(commitments)) == 5


def test_forward_memory_uses_all_request_production_lifetime_without_arrays(monkeypatch):
    from pllm.runtime.benchmark_memory import forward_owner_memory
    import numpy as np
    monkeypatch.setattr(np, "array", lambda *a, **kw: pytest.fail("array allocated before admission"))
    reservation = {"requests": 1, "max_sequence": 3, "output_bytes_per_result": 256,
        "result_metadata_bytes_per_result": 4096, "max_call_bytes": 64000,
        "import_peak_bytes": 50000, "required_bytes": 68352, "admission_bytes": 85440}
    small = forward_owner_memory(reservation)
    large = forward_owner_memory({**reservation, "max_sequence": 6,
        "max_call_bytes": 70000, "required_bytes": 74352, "admission_bytes": 92940})
    retained = forward_owner_memory({**reservation, "requests": 4,
        "required_bytes": 81408, "admission_bytes": 101760})
    assert small["lifetime_reservation"] == reservation
    assert large["required_bytes"] > small["required_bytes"]
    assert small["owners"]["production_lifetime_with_margin"] == 85440
    assert retained["owners"]["benchmark_result_references"] == 4 * small["owners"]["benchmark_result_references"]
    assert retained["required_bytes"] > small["required_bytes"]
    assert small["required_bytes"] > reservation["admission_bytes"]
    assert small["native_quote_scope"] == "logical_payload_not_process_RSS"
    metadata_dominated = forward_owner_memory({**reservation, "metadata_phase_peak_bytes": 800000,
        "required_bytes": 800000, "admission_bytes": 1000000})
    assert metadata_dominated["lifetime_reservation"]["metadata_phase_peak_bytes"] == 800000
    # Production already applied its margin ONCE, including metadata-phase peak.
    assert metadata_dominated["required_bytes"] - small["required_bytes"] == 1000000 - 85440


def test_unit_preflight_serializes_production_description_without_compiling(experiment, monkeypatch):
    from types import MappingProxyType
    from pllm.configuration import Experiment
    from pllm.runtime import benchmark_memory, forward_binding
    reservation = {"requests": 1, "max_sequence": 6, "output_bytes_per_result": 236,
        "result_metadata_bytes_per_result": 8192, "max_call_bytes": 64000,
        "import_peak_bytes": 50000, "required_bytes": 72428, "admission_bytes": 90535}
    description = MappingProxyType({"schema_version": "pllm.forward_source_description.v1",
        "execution_admitted": False, "numerical_finite_validated": False,
        "lifetime_reservation": MappingProxyType(reservation),
        "identities": MappingProxyType({"binding_digest": "a" * 64}),
        "outputs": MappingProxyType({"logits": MappingProxyType({"shape": (1, 3), "dtype": "float32"})})})
    calls = []
    def describe(value, *, memory_bytes):
        calls.append((value, memory_bytes))
        return description
    monkeypatch.setattr(forward_binding, "describe_forward_source", describe)
    monkeypatch.setattr(Experiment, "compile_forward",
        lambda *a, **kw: pytest.fail("metadata preflight compiled tensor values"))
    monkeypatch.setattr(benchmark_memory, "host_memory", lambda: benchmark_memory.HostMemory(
        total=16 << 30, available=12 << 30, swap_used=0, disk_free=100 << 30))
    report = benchmark_memory.benchmark_memory(experiment, memory_budget_bytes=512 << 20)
    assert calls == [(experiment, benchmark_memory.forward_runtime_memory_bytes(512 << 20))]
    assert report["admitted"] is True and report["execution_admitted"] is False
    assert report["numerical_finite_validated"] is report["performance_measured"] is False
    assert report["forward_source"]["identities"]["binding_digest"] == "a" * 64
    assert report["forward_source"]["outputs"]["logits"]["shape"] == [1, 3]
    assert report["estimate"]["lifetime_reservation"] == reservation
    assert "metrics" not in report and "samples" not in report
    json.dumps(report, allow_nan=False)


def test_worker_envelope_limit_includes_metadata_overhead(suite, monkeypatch, tmp_path):
    from pllm.metrics import benchmark_runner
    from pllm.runtime import benchmark_memory
    from types import SimpleNamespace
    monkeypatch.setattr(benchmark_memory, "host_memory", lambda: SimpleNamespace())
    monkeypatch.setattr(benchmark_runner.subprocess, "Popen",
                        lambda *a, **kw: pytest.fail("oversized envelope launched worker"))
    with pytest.raises(ValueError, match="16 MiB"):
        benchmark_runner._execute(suite.cases[0], suite.sampling,
            {"payload": "x" * ((16 << 20) - 32)}, bytes(32), tmp_path)


@pytest.fixture
def forward_worker(experiment, monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from pllm._benchmark_worker import Measurement
    from pllm._benchmark_contract import canonical
    from pllm.configuration import Experiment
    experiment = experiment.with_params(budget__requests=3)
    identities = {key: "a" * 64 for key in ("source_digest", "checkpoint_digest", "source_lock_digest",
        "plan_digest", "schedule_digest", "composition_digest", "numeric_digest",
        "configuration_digest", "binding_digest")}
    shapes = {"last_hidden_state": [1, "sequence", 8], "pooler_output": [1, 8], "logits": [1, 3]}
    plan = {"output_contract": "pllm.forward.encoder_outputs.v1", "outputs": {
        name: {"tensor": {"dtype": "float32", "shape": shape}} for name, shape in shapes.items()}}
    quote = {"sequence_length": 6, "input_bytes": 150, "retained_tensor_bytes": 1024,
             "output_bytes": 236, "workspace_bytes": 6144, "required_bytes": 11650}
    calls = {"compile": [], "forward": 0, "close": 0}
    control = {"failure": None}

    class Compiled:
        lifetime_reservation = {"requests": 3, "max_sequence": 6, "output_bytes_per_result": 236,
            "result_metadata_bytes_per_result": 8192, "max_call_bytes": 100000,
            "import_peak_bytes": 9 << 20, "required_bytes": 9 << 20,
            "admission_bytes": (9 << 20) * 125 // 100}
        execution_metadata = {"backend": "numpy_einsum_unoptimized_f32_v1", "requested_compute_threads": 2,
            "effective_compute_threads": 1, "thread_policy": "serial_compute_within_requested_budget"}
        contract = SimpleNamespace(plan_bytes=lambda: canonical(plan), quote=lambda n: canonical(quote),
            requirements_bytes=lambda: canonical({"schema_version": "pllm.forward_requirements.v1",
                "weight_bytes": 4096, "tensors": [{"name": "weight", "dtype": "float32", "shape": [32, 32]}]}))

        def __init__(self):
            self.identities = identities

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            calls["close"] += 1

        def forward(self, inputs):
            calls["forward"] += 1
            if control["failure"] == "execute":
                raise ValueError("private exception must not enter evidence")
            values = {name: np.frombuffer(bytes(4 * np.prod([3 if n == "sequence" else n for n in shape])),
                         dtype=np.float32).reshape([3 if n == "sequence" else n for n in shape])
                      for name, shape in shapes.items()}
            if control["failure"] == "dtype":
                values["logits"] = np.zeros((1, 3), dtype=np.float64)
            return SimpleNamespace(outputs=values, identities=identities)

    def compile(self, *, memory_bytes):
        calls["compile"].append(memory_bytes)
        return Compiled()

    monkeypatch.setattr(Experiment, "compile_forward", compile)
    m = Measurement({"sampling": Sampling(processes=1, warmups=1, repetitions=2).to_spec(),
        "salt": "ab" * 32, "inputs": {"schema_version": "pllm.forward_inputs.v1", "tokens": [[1, 2, 3]],
        "attention_mask": [[True, True, False]]}, "directory": str(tmp_path)})
    m.admit = lambda required: m.admission.update(admitted=True, required_bytes=required)
    return SimpleNamespace(experiment=experiment, m=m, calls=calls, control=control)


def test_unit_worker_one_compiled_owner_all_attempts_and_target_metadata(forward_worker):
    from pllm.runtime.benchmark_targets import execute_target
    state = forward_worker
    execute_target(BenchmarkTarget.forward(state.experiment), state.m)
    assert len(state.calls["compile"]) == 1
    assert state.calls["forward"] == 3 and state.calls["close"] == 1
    assert [sample["kind"] for sample in state.m.samples] == ["warmup", "measurement", "measurement"]
    details = state.m.details
    assert details["attempted_requests"] == details["completed_requests"] == 3
    assert details["visible_key_count"] == 2 and details["sequence"] == 3
    assert details["output_contract"] == "pllm.forward.encoder_outputs.v1"
    assert details["execution_metadata"] == {"backend": "numpy_einsum_unoptimized_f32_v1",
        "requested_compute_threads": 2, "effective_compute_threads": 1,
        "thread_policy": "serial_compute_within_requested_budget"}
    assert details["outputs"]["last_hidden_state"]["shape"] == [1, 3, 8]
    assert details["outputs"]["logits"]["dtype"] == "F32"
    assert details["compile_import"]["wall_seconds"] >= 0
    assert state.m.comparison["output_digest"]
    assert "not_numeric_oracle" in state.m.oracle["kind"]
    assert not any("token" in metric["id"] or "ttft" in metric["id"] or "decode" in metric["id"]
                   for metric in state.m.metrics)
    assert "little_endian_content" not in json.dumps(details)


@pytest.mark.parametrize("failure", ["execute", "dtype"])
def test_worker_closes_shared_context_on_execution_or_output_failure(forward_worker, failure):
    from pllm.runtime.benchmark_targets import execute_target
    from pllm._benchmark_contract import TargetUnavailable
    state = forward_worker
    state.control["failure"] = failure
    with pytest.raises(ValueError if failure == "execute" else TargetUnavailable):
        execute_target(BenchmarkTarget.forward(state.experiment), state.m)
    assert state.calls["close"] == 1 and len(state.calls["compile"]) == 1


def test_worker_rechecks_json_extents_before_compile(forward_worker):
    from pllm.runtime.benchmark_targets import execute_target
    state = forward_worker
    state.m.inputs["tokens"] = [[1] * 7]
    with pytest.raises(ValueError):
        execute_target(BenchmarkTarget.forward(state.experiment), state.m)
    assert not state.calls["compile"] and state.calls["forward"] == 0


def test_worker_refuses_budget_overrun_including_warmups(forward_worker):
    from pllm.runtime.benchmark_targets import execute_target
    from pllm._benchmark_contract import TargetUnavailable
    state = forward_worker
    experiment = state.experiment.with_params(budget__requests=2)
    with pytest.raises(TargetUnavailable) as error:
        execute_target(BenchmarkTarget.forward(experiment), state.m)
    assert error.value.code == "forward_request_budget"
    assert not state.calls["compile"]


def test_forward_ordinary_result_scope_and_schema(suite):
    from pllm.metrics.benchmark_runner import _result
    from pllm._benchmark_contract import digest
    row = _result(suite.cases[0], suite, 0, "a" * 64,
        {"digest": digest({}), "attributes": {}},
        {"status": "blocked", "failure_code": "memory_limit"}).to_dict()
    assert row["schema_version"] == "pllm.benchmark_result.v2"
    assert row["scope"] == "full_model"
    assert row["execution"]["case"] == suite.cases[0].to_spec()
    assert row["execution"]["admission"]["reason"] == "memory_limit"


def test_decoder_loopback_guard_precedes_timeout_and_dashboard(experiment, monkeypatch):
    from pllm.runtime import benchmark_cli
    monkeypatch.setattr(benchmark_cli, "validate_benchmark_timeout",
        lambda *a, **kw: pytest.fail("decoder timeout work preceded forward rejection"))
    monkeypatch.setattr(benchmark_cli.gc, "collect",
        lambda: pytest.fail("decoder dashboard work preceded forward rejection"))
    with pytest.raises(ValueError):
        benchmark_cli.run_loopback_benchmark(model="unused", model_id=None, tiny=False,
            prompt="unused", max_output_tokens=1, warmups=0, repetitions=1,
            timeout_seconds=300, experiment=experiment)


def test_decoder_memory_rejects_bypassed_fp32_pipeline_before_host(experiment, monkeypatch):
    from pllm.runtime import benchmark_memory
    monkeypatch.setattr(benchmark_memory, "host_memory",
        lambda: pytest.fail("host telemetry preceded FP32 pipeline rejection"))
    with pytest.raises(ValueError):
        benchmark_memory.benchmark_memory(experiment.pipeline)

"""Quality authority gates, not checkpoint or cryptographic conformance evidence.

Real Python adapter/decoder bodies; mock native acceptance and import boundaries.
The tiny Q11 fixture has independent literal full-head/teacher-forcing oracles.
No model/source loader, native extension, Torch, workers, or checkpoint is loaded.
Mock acceptance bytes are NOT native verification proof.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NUMERIC = "gkm.nearest-even.2048.i16.ring64.signed.f32-before-div22.le-finite-f32.sha256.secp256k1.v1"
LOCAL = "semantic.float32.dense-full-kv.client-bias-after-div22.v1"
TABLE = ((.5, 1.), (-1., .5), (1., -.5))
WEIGHTS = ((1, 0), (0, 1), (-1, 1))
PREFILL = np.array([.5, 1., .5], dtype="<f4")
TEACHER = np.array([1., -.5, -1.5], dtype="<f4")


@pytest.fixture
def lane(monkeypatch, tmp_path):
    import pllm
    import pllm.runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected model/native/local-kernel fallback")

    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    native = module("pllm._native")
    monkeypatch.setattr(pllm, "_native", native, raising=False)
    module("pllm.configuration", Experiment=type("Experiment", (), {}))
    module("pllm.metrics", ReferenceAgreement=forbidden, measure_reference_agreement=forbidden)
    reader = Mock()
    module("pllm.model_loader", resolve_model=forbidden, _source_lock=reader)
    module("pllm.profiles", resolve_runtime_composition=forbidden)
    module("pllm.protocols", FuseFssSourceDecoder=type("Fss", (), {}))
    module("pllm.runtime.model_binding", compile_runtime_model=forbidden)
    module("pllm.runtime.public_equalization", equalize_activation=forbidden)
    module("pllm.runtime.quantization", dequantize_matmul=forbidden, quantize_activation_per_row=forbidden)
    module("pllm.runtime.transformer_engine", MaskedTransformerEngine=forbidden)
    module("pllm.runtime.transformer_client", ClientBundle=forbidden, LayerCache=forbidden)

    class LocalSelectionOnly:
        def _declared_state_kinds(self, graph):
            return {}

        def _local(self, operation, values, kinds, pending):
            assert operation["operator"] == "greedy_token_selection"
            return np.argmax(values[operation["inputs"][0]])

    module("pllm.runtime.semantic_executor", SemanticLocalOperators=LocalSelectionOnly)

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT / "python/pllm/runtime" / filename)
        result = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, result)
        spec.loader.exec_module(result)
        return result

    # Aliases avoid poisoning cached real adapters in a combined pytest process.
    outcomes = load("pllm.runtime._gkm_authority_outcomes", "quality_outcomes.py")
    monkeypatch.setitem(sys.modules, "pllm.runtime.quality_outcomes", outcomes)
    monkeypatch.setattr(pllm.runtime, "quality_outcomes", outcomes, raising=False)
    decoder = load("pllm.runtime._gkm_authority_decoder", "delegated_decoder.py")
    adapter = load("pllm.runtime._gkm_authority_quality", "reference_benchmark.py")
    source_file = tmp_path / "tiny-numeric-fixture-not-checkpoint.f32"
    source_file.write_bytes(np.asarray(WEIGHTS, dtype="<f4").tobytes())

    def fingerprint(path, model):
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        return checksum, checksum, {"fixture_sha256": checksum}

    pins = fingerprint(source_file, None)
    reader.side_effect = fingerprint
    resolved = NS(path=source_file, model=object(), checkpoint_digest=pins[0],
        source_lock_digest=pins[1], manifest=NS(metadata={"source_lock": pins[2]}))
    experiment = NS(name="test-only", budget=NS(max_input_tokens=8),
        pipeline=NS(components={"kernels": NS(params={"threads": 1}, component="pllm/cpu")},
                    digest=lambda: "b" * 64), configuration_digest=lambda: "a" * 64)
    source = NS(contract={"vocabulary": 3, "numeric_profile": NUMERIC, "local_numeric_profile": LOCAL},
        tokenizer=lambda: NS(encode=lambda *args, **kwargs: (0,)),
        tokenizer_descriptor={"add_bos_token": False})
    guard = NS(start=Mock(), close=Mock(), check=Mock())
    module("pllm.runtime.benchmark_memory", MemoryWatchdog=lambda *args: guard,
           BenchmarkMemoryError=type("BenchmarkMemoryError", (ValueError,), {}),
           benchmark_memory=lambda *args, **kwargs: {}, require_admission=lambda report: None)
    module("pllm.runtime.client", ProtocolError=type("ProtocolError", (RuntimeError,), {}))
    topology = NS(worker_connections=lambda: {"worker_a": "mock", "worker_b": "mock"})
    module("pllm.runtime.servers", build_roles=lambda experiment: nullcontext(topology))
    execute = Mock()
    module("pllm.runtime.delegated_transport", execute_request=execute)
    yield NS(adapter=adapter, decoder=decoder, source=source, resolved=resolved,
             source_file=source_file, reader=reader, experiment=experiment, execute=execute,
             guard=guard, module=module, monkeypatch=monkeypatch, pins=pins)


def contract():
    operations = [
        {"id": "embedding", "operator": "token_lookup", "inputs": ["input.tokens"]},
        {"id": "head", "operator": "output_head", "inputs": ["embedding"]},
        {"id": "selected", "operator": "greedy_token_selection", "inputs": ["head"]},
    ]
    graph = {"operations": operations, "state_inputs": [], "state_outputs": []}
    steps = [{"executor": "remote", "operators": [op["operator"]],
              "operation_ids": [op["id"]], "input_ids": op["inputs"]} for op in operations]
    steps[-1]["executor"] = "client_local"
    return {"input_rows": 1, "decode_steps": 1, "vocabulary": 3, "width": 2,
        "plan": {"prefill": deepcopy(graph), "decode": deepcopy(graph)},
        "prefill": {"steps": deepcopy(steps)}, "decode": {"steps": deepcopy(steps)},
        "matrices": [{"rows": 3, "columns": 2,
                      "members": [{"offset": 0, "rows": 3, "bias": None}]}]}


class MockAcceptedNative:
    """Numeric/acceptance test double, not cryptographic verification authority."""

    def __init__(self, admitted, *, failure=False):
        self.admitted, self.failure = admitted, failure
        self.events = []

    def contract(self):
        return json.dumps(self.admitted).encode()

    def coefficients(self):
        return {}

    def lookup(self, token):
        self.events.append(("lookup", token))
        return np.asarray(TABLE[token], dtype="<f4").tobytes()

    def linear(self, stage, row):
        assert stage == 0
        # Exact small Q11 integers; cast signed accumulator to f32 BEFORE /2^22.
        encoded = [round(float(value) * 2048) for value in np.frombuffer(row, dtype="<f4")]
        integers = [sum((weight * 2048) * value for weight, value in zip(weights, encoded, strict=True))
                    for weights in WEIGHTS]
        self.events.append(("head", tuple(encoded)))
        return (np.asarray(integers, dtype="<f4") / np.float32(1 << 22)).tobytes()

    def finish(self):
        self.events.append(("finish",))
        if self.failure:
            raise ValueError("mock final acceptance refused")
        return b"test-double-only-native-acceptance"

    def abort(self):
        self.events.append(("abort",))


def test_real_decoder_returns_literal_full_head_only_after_final_acceptance(lane):
    admitted = contract()
    native = MockAcceptedNative(admitted)
    result = lane.decoder._execute_verified_window_inner(native, admitted, (0,), decode_tokens=(2,))
    assert result.executed == (0, 2) and result.selected == (1, 0)
    assert result.position == 2
    np.testing.assert_array_equal(np.frombuffer(result.logits[0], dtype="<f4"), PREFILL)
    np.testing.assert_array_equal(np.frombuffer(result.logits[1], dtype="<f4"), TEACHER)
    assert native.events[:5] == [("lookup", 0), ("head", (1024, 2048)),
                                ("lookup", 2), ("head", (2048, -1024)), ("finish",)]
    assert native.events[-1] == ("abort",)


def test_failed_final_acceptance_never_becomes_quality_values(lane):
    native = MockAcceptedNative(contract(), failure=True)

    def execute(*args, **kwargs):
        return lane.decoder._execute_verified_window_inner(native, native.admitted, (0,), decode_tokens=(2,))

    lane.execute.side_effect = execute
    with pytest.raises(ValueError, match="final acceptance refused"):
        lane.adapter._delegated_candidate(lane.experiment, ("fixture",), lane.source, 1, [(2,)])
    assert ("finish",) in native.events and native.events[-1] == ("abort",)
    lane.guard.close.assert_called_once()


@pytest.mark.parametrize("sharing", ["full", "seeded"])
def test_quality_adapter_consumes_actual_decoder_vectors_and_frozen_teacher(lane, sharing):
    natives = []
    # Native eligibility remains the existing SDK's job. Check adapter forwarding,
    # not counterfeit admission by this numeric/acceptance double.
    lane.experiment.pipeline.components["linear"] = NS(params={"sharing": sharing})

    def execute(experiment, source, ids, selections, connections, timeout, cancelled, *, decode_tokens):
        assert source is lane.source and ids == (0,) and selections == 2
        assert experiment.pipeline.components["linear"].params["sharing"] == sharing
        assert decode_tokens == (2,)
        native = MockAcceptedNative(contract())
        natives.append(native)
        return lane.decoder._execute_verified_window_inner(native, native.admitted, ids, decode_tokens=decode_tokens)

    lane.execute.side_effect = execute
    metadata, actual, decoded, ids, teacher = lane.adapter._delegated_candidate(
        lane.experiment, ("fixture",), lane.source, 1, [(2,)])
    assert ids == ((0,),) and teacher == [(2,)]
    np.testing.assert_array_equal(actual[0], PREFILL)
    np.testing.assert_array_equal(decoded[0], TEACHER)
    assert metadata["execution"] == "two-worker-request-end-verified"
    assert metadata["numeric_profile"] == NUMERIC and metadata["local_numeric_profile"] == LOCAL
    assert len(natives) == 1 and ("finish",) in natives[0].events


@pytest.mark.parametrize("vectors", [[], [PREFILL.tobytes()],
    [PREFILL[:2].tobytes()] * 2, [np.array([0, np.nan, 1], dtype="<f4").tobytes()] * 2])
def test_quality_adapter_rejects_missing_partial_or_nonfinite_full_head(lane, vectors):
    lane.execute.return_value = NS(logits=vectors, executed=(0, 2))
    with pytest.raises(lane.adapter.ReferenceBenchmarkError, match="malformed"):
        lane.adapter._delegated_candidate(lane.experiment, ("fixture",), lane.source, 1, [(2,)])


@pytest.mark.parametrize("storage", ["resident", "streamed"])
@pytest.mark.parametrize("when", ["before", "during"])
def test_changed_reference_checkpoint_cannot_be_scored_under_original_pins(lane, storage, when):
    def load(*args):
        lane.source_file.write_bytes(b"changed numeric source")
        return [[PREFILL, TEACHER]], None

    loader = Mock(side_effect=load)
    lane.monkeypatch.setattr(lane.adapter, "_load_reference_trajectories", loader)
    if when == "before":
        lane.source_file.write_bytes(b"changed before reference construction")
    from pllm.runtime.quality_outcomes import QualityIntegrityError
    with pytest.raises(QualityIntegrityError, match="source binding"):
        lane.adapter._reference_trajectories(lane.resolved, (), ((0,),), [(2,)], storage, 1, 25)
    assert loader.call_count == (0 if when == "before" else 1)


@pytest.mark.parametrize("error", [FileNotFoundError("fixture missing"), ValueError("bad source")])
def test_source_reader_failure_is_integrity_failure_not_numeric_acceptance(lane, error):
    from pllm.runtime.quality_outcomes import QualityIntegrityError
    lane.reader.side_effect = error
    with pytest.raises(QualityIntegrityError, match="source binding"):
        lane.adapter._check_reference_source(lane.resolved)


def test_record_mode_classifies_changed_reference_as_integrity_failure(lane):
    from pllm.runtime.quality_outcomes import CaptureFailure
    lane.source_file.write_bytes(b"different reference")
    with CaptureFailure(True) as phase:
        lane.adapter._reference_trajectories(lane.resolved, (), ((0,),), [(2,)], "streamed", 1, 25)
    assert phase.failed and phase.code == "source_integrity_failed"


@pytest.mark.parametrize("storage", ["resident", "streamed"])
def test_reference_adapter_preserves_fp32_same_source_and_teacher_coupling(lane, storage):
    calls = []

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def __getitem__(self, key):
            return Tensor(self.value[key])

        def float(self):
            return self

        def numpy(self):
            return self.value

    class Resident:
        def eval(self):
            return self

        def __call__(self, *, input_ids, use_cache, past_key_values=None):
            calls.append((tuple(np.asarray(input_ids).reshape(-1)), past_key_values))
            token = int(np.asarray(input_ids).reshape(-1)[-1])
            logits = np.asarray(WEIGHTS, dtype=np.float32) @ np.asarray(TABLE[token], dtype=np.float32)
            return NS(logits=Tensor(logits[None, None, :]), past_key_values="fixture-cache")

    def from_pretrained(path, **kwargs):
        assert path == lane.resolved.path
        assert kwargs == {"local_files_only": True, "trust_remote_code": False,
                          "dtype": "fixture-float32", "attn_implementation": "eager"}
        return Resident()

    lane.module("torch", float32="fixture-float32", tensor=np.asarray, inference_mode=nullcontext)
    lane.module("transformers", AutoModelForCausalLM=NS(from_pretrained=from_pretrained))

    def streamed(path, **kwargs):
        assert path == lane.resolved.path
        assert kwargs == {"max_input_tokens": 8, "max_decode_steps": 1,
                          "memory_safety_margin_percent": 25}

        def trajectory(ids, continuation):
            calls.append((ids, continuation))
            return [PREFILL.copy(), TEACHER.copy()]

        return nullcontext(NS(trajectory=trajectory, weight_bound=24, allocation_estimate_bytes=24,
            kv_bound_bytes=0, buffer_bytes=0, stats={}, _guard=NS(maximum_swap_growth_bytes=0)))

    lane.module("pllm.runtime.streamed_reference", StreamedFloat32Reference=streamed)
    configs = [(lane.experiment, None, lane.resolved, lane.source)]
    trajectories, loading = lane.adapter._reference_trajectories(
        lane.resolved, configs, ((0,),), [(2,)], storage, 1, 25)
    np.testing.assert_array_equal(trajectories[0][0], PREFILL)
    np.testing.assert_array_equal(trajectories[0][1], TEACHER)
    assert calls == ([((0,), (2,))] if storage == "streamed"
                     else [((0,), None), ((2,), "fixture-cache")])
    assert (loading is None) == (storage == "resident")
    assert lane.reader.call_count == 2

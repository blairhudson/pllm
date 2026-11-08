"""Genuine native-factory ownership, admission, source and lease regressions."""
import json
import math
import threading

import numpy as np
import pytest

from pllm import _native
from pllm.configuration import Deployment, Experiment, ForwardBudget, Model
from pllm.kernels import Cpu
from pllm.numeric import SourceFloat32
from pllm.profiles import ClientOnlyCpu
from pllm.runtime.forward_binding import CompiledForwardModel, ForwardBindingError
from pllm.runtime.forward_inputs import ForwardInputError, ForwardInputs
from pllm.sources.generated_encoder import tiny_checkpoint, write_checkpoint


# Positive/precondition setup: reuse genuine JSON tests' validated 128-MiB
# headroom, not a production cap. Explicit early refusals and boundaries stay put.
FUNCTIONAL_SETUP_BYTES = 128 << 20


def experiment(root, *, requests=4):
    return Experiment("forward-owner", ClientOnlyCpu(Model(str(root)), numeric=SourceFloat32()),
                      Deployment.local(root=str(root / "deployment")), ForwardBudget(requests, 8))


@pytest.fixture
def source(tmp_path):
    return write_checkpoint(tmp_path / "source")


def test_exact_native_owner_and_frozen_coefficients(source):
    with experiment(source).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        assert type(model) is CompiledForwardModel
        assert type(model.contract) is _native.ForwardContract
        assert model.contract.source_config_bytes() == (source / "config.json").read_bytes()
        requirements = json.loads(model.contract.requirements_bytes())["tensors"]
        key = requirements[0]["name"]
        value = model.local_tensor(key)
        original = value.shape
        with pytest.raises(ValueError):
            value.setflags(write=True)
        value.shape = (value.size,)
        assert model.local_tensor(key).shape == original
        assert model.local_tensor(key) is not model.local_tensor(key)


def test_invalid_preflight_unspent_success_spent_and_close_idempotent(source):
    model = experiment(source, requests=1).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES)
    with pytest.raises(ForwardInputError):
        model.forward(ForwardInputs([True]))
    result = model.forward(ForwardInputs([1, 2], positions=[7, 1], segments=[1, 0],
                                          attention_mask=[True, False]))
    with pytest.raises(ForwardBindingError, match="exhausted"):
        model.forward(ForwardInputs([1]))
    before = result.outputs["last_hidden_state"].tobytes()
    model.close()
    model.close()
    assert result.outputs["last_hidden_state"].tobytes() == before
    for array in result.outputs.values():
        assert array.dtype == np.float32
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.outputs["new"] = np.zeros(1)
    with pytest.raises(ForwardBindingError, match="closed"):
        model.forward(ForwardInputs([1]))


@pytest.mark.parametrize("damage", ["missing", "suffix", "f16", "shape", "nonfinite"])
def test_source_tensor_contracts_exact_not_suffix_or_cast(tmp_path, damage):
    config, tensors = tiny_checkpoint()
    key = "bert.embeddings.word_embeddings.weight"
    if damage == "missing":
        del tensors[key]
    elif damage == "suffix":
        tensors["wrapper." + key] = tensors.pop(key)
    elif damage == "f16":
        tensors[key] = tensors[key].astype(np.float16)
    elif damage == "shape":
        tensors[key] = tensors[key][:-1].copy()
    else:
        tensors[key][0, 0] = np.nan
    root = write_checkpoint(tmp_path / damage, config, tensors)
    error = {"missing": "required exact checkpoint key missing",
             "suffix": "required exact checkpoint key missing",
             "f16": "required exact F32 tensor differs",
             "shape": "required exact F32 tensor differs",
             "nonfinite": "checkpoint tensor is nonfinite"}[damage]
    with pytest.raises((ForwardBindingError, ValueError), match=error):
        experiment(root).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES)


def test_original_mutation_after_verified_boundary_cannot_change_snapshot_binding(source, monkeypatch):
    from pllm.runtime.bundle_storage import FilePayload
    original = FilePayload.read_at
    config_bytes = (source / "config.json").read_bytes()
    with experiment(source).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as baseline:
        expected = baseline.forward(ForwardInputs([1, 2])).outputs["last_hidden_state"].copy()
    changed = False
    def read(snapshot, offset, size):
        nonlocal changed
        result = original(snapshot, offset, size)
        if offset > 8 and not changed:
            changed = True
            path = source / "config.json"
            path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(FilePayload, "read_at", read)
    with experiment(source).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        assert changed and model.contract.source_config_bytes() == config_bytes
        np.testing.assert_array_equal(model.forward(ForwardInputs([1, 2])).outputs["last_hidden_state"], expected)


def test_concurrency_rejected_and_close_drains_actual_lease(source, monkeypatch):
    from pllm.runtime.forward_execution import _BoundForwardOperation
    model = experiment(source).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES)
    reservation = model.lifetime_reservation
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = _BoundForwardOperation.__call__
    def blocked(kernel, values):
        entered.set()
        assert release.wait(10)
        return original(kernel, values)
    monkeypatch.setattr(_BoundForwardOperation, "__call__", blocked)
    results, errors = [], []
    def run():
        try:
            results.append(model.forward(ForwardInputs([1, 2])))
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(10)
    with pytest.raises(ForwardBindingError, match="in-flight"):
        model.forward(ForwardInputs([1]))
    def close():
        model.close()
        closed.set()
    closer = threading.Thread(target=close)
    closer.start()
    assert not closed.wait(0.05)
    release.set()
    worker.join(10)
    closer.join(10)
    assert closed.is_set() and not worker.is_alive() and not closer.is_alive()
    assert not errors and len(results) == 1
    assert model.lifetime_reservation is reservation
    assert results[0].outputs["last_hidden_state"].shape == (1, 2, 8)


def test_failure_spends_attempt_but_binding_remains_usable(source, monkeypatch):
    from pllm.runtime.forward_execution import _BoundForwardOperation
    with experiment(source, requests=2).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        reservation = model.lifetime_reservation
        original = _BoundForwardOperation.__call__
        def fail(*args):
            raise ArithmeticError("injected kernel failure")
        with monkeypatch.context() as patch:
            patch.setattr(_BoundForwardOperation, "__call__", fail)
            with pytest.raises(ArithmeticError):
                model.forward(ForwardInputs([1]))
        assert _BoundForwardOperation.__call__ is original
        assert "pooler_output" in model.forward(ForwardInputs([1])).outputs
        assert model.lifetime_reservation is reservation
        with pytest.raises(ForwardBindingError, match="exhausted"):
            model.forward(ForwardInputs([1]))


@pytest.mark.parametrize("exception", [KeyboardInterrupt, BaseException])
def test_runtime_activation_baseexception_releases_lease_and_spends_attempt(source, monkeypatch, exception):
    from pllm.runtime import forward_execution
    spec = experiment(source, requests=2)
    inputs = ForwardInputs([[1, 2]], positions=[[7, 1]], segments=[[1, 0]],
                           attention_mask=[[True, False]])
    with spec.compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as baseline:
        expected = baseline.forward(inputs)
    model = spec.compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES)
    original_activation = forward_execution.source_float32_activation
    identities = dict(model.identities)
    source_config = model.contract.source_config_bytes()
    key = json.loads(model.contract.requirements_bytes())["tensors"][0]["name"]
    coefficient = model.local_tensor(key).tobytes()
    reservation = model.lifetime_reservation
    calls = []
    cancellation = exception("controlled runtime activation cancellation")
    def interrupted(array, kind):
        # This is the actual dispatch cache point used by bound forward ops.
        # Run the normal native-backed activation, then cancel the real request.
        original_activation(array, kind)
        calls.append(kind)
        with pytest.raises(ForwardBindingError, match="in-flight"):
            model.forward(inputs)
        raise cancellation
    try:
        with monkeypatch.context() as patch:
            patch.setattr(forward_execution, "source_float32_activation", interrupted)
            with pytest.raises(exception, match="controlled runtime activation cancellation") as caught:
                model.forward(inputs)
            assert caught.value is cancellation
        assert len(calls) == 1
        assert model.identities == identities
        assert model.contract.source_config_bytes() == source_config
        assert model.local_tensor(key).tobytes() == coefficient
        assert model.lifetime_reservation is reservation
        result = model.forward(inputs)  # No stuck lease, immutable binding reusable.
        for name, wanted in expected.outputs.items():
            np.testing.assert_allclose(result.outputs[name], wanted, atol=1e-5, rtol=1e-4)
        assert result.identities == identities
        with pytest.raises(ForwardBindingError, match="exhausted"):
            model.forward(inputs)  # The cancelled first request was consumed.
    finally:
        model.close()
    model.close()
    assert model.identities == identities
    for name, wanted in expected.outputs.items():
        output = result.outputs[name]
        np.testing.assert_allclose(output, wanted, atol=1e-5, rtol=1e-4)
        assert not output.flags.writeable
        with pytest.raises(ValueError):
            output.setflags(write=True)
    with pytest.raises(ForwardBindingError, match="closed"):
        model.forward(inputs)


def test_serial_projection_dispatch_and_immutable_execution_metadata(source, monkeypatch):
    spec = Experiment("serial-einsum", ClientOnlyCpu(Model(str(source)),
                      numeric=SourceFloat32(), kernels=Cpu(threads=7)),
                      Deployment.local(root=str(source / "deployment")), ForwardBudget(1, 8))
    expressions = []
    original = np.einsum
    def traced(expression, *operands, **kwargs):
        if expression == "...i,oi->...o":
            assert kwargs["dtype"] is np.float32 and kwargs["casting"] == "no"
            assert kwargs["optimize"] is False
            assert type(kwargs["out"]) is np.ndarray and kwargs["out"].dtype == np.float32
            expressions.append(expression)
        else:
            assert kwargs["optimize"] is False  # Shared attention path remains unoptimized.
        return original(expression, *operands, **kwargs)
    monkeypatch.setattr(np, "einsum", traced)
    with spec.compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        assert model.execution_metadata == {
            "backend": "numpy_einsum_unoptimized_f32_v1",
            "requested_compute_threads": 7,
            "effective_compute_threads": 1,
            "thread_policy": "serial_compute_within_requested_budget",
        }
        with pytest.raises(TypeError):
            model.execution_metadata["effective_compute_threads"] = 7
        assert "logits" in model.forward(ForwardInputs([1, 2])).outputs
        assert len(expressions) == 14  # Six projections/layer, pooler, classifier.


def test_production_description_never_reads_numerical_tensors(source, monkeypatch):
    from pllm.runtime.bundle_storage import FilePayload
    from pllm.runtime.forward_binding import describe_forward_source
    from pllm.runtime.safetensors_store import SafeTensorStore
    original = FilePayload.read_at
    def metadata_only(snapshot, offset, size):
        assert offset in (0, 8), "coefficient payload materialized during description"
        return original(snapshot, offset, size)
    def no_array(*args, **kwargs):
        raise AssertionError("FP32 array constructed during description")
    def no_store(*args, **kwargs):
        raise AssertionError("original source metadata reopened through SafeTensorStore")
    monkeypatch.setattr(SafeTensorStore, "__init__", no_store)
    with monkeypatch.context() as patch:
        patch.setattr(FilePayload, "read_at", metadata_only)
        patch.setattr(np, "frombuffer", no_array)
        description = describe_forward_source(experiment(source), memory_bytes=FUNCTIONAL_SETUP_BYTES)
    assert description["execution_admitted"] is False
    assert description["numerical_finite_validated"] is False
    assert description["outputs"]["last_hidden_state"]["shape"] == (1, 8, 8)
    with pytest.raises(TypeError):
        description["execution_admitted"] = True
    with experiment(source).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        assert description["identities"] == model.identities
        assert description["lifetime_reservation"] == model.lifetime_reservation


def test_description_makes_no_finite_weight_claim(tmp_path):
    from pllm.runtime.forward_binding import describe_forward_source
    config, tensors = tiny_checkpoint()
    tensors["classifier.bias"][0] = np.nan
    source = write_checkpoint(tmp_path / "nonfinite", config, tensors)
    assert describe_forward_source(experiment(source), memory_bytes=FUNCTIONAL_SETUP_BYTES)["numerical_finite_validated"] is False
    with pytest.raises(ForwardBindingError, match="nonfinite"):
        experiment(source).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES)


def test_lifetime_exact_cap_and_one_byte_short_before_payload_read(source, monkeypatch):
    from pllm.runtime.bundle_storage import FilePayload
    from pllm.runtime.forward_binding import describe_forward_source
    spec = experiment(source, requests=3)
    description = describe_forward_source(spec, memory_bytes=FUNCTIONAL_SETUP_BYTES)
    reservation = description["lifetime_reservation"]
    assert reservation["max_call_bytes"] == description["quote"]["required_bytes"] + description["python_workspace_bytes"]
    assert reservation["output_bytes_per_result"] == description["quote"]["output_bytes"]
    assert reservation["required_bytes"] == max(reservation["metadata_phase_peak_bytes"],
        reservation["native_compile_prerequisite_bytes"], reservation["import_peak_bytes"],
        reservation["max_call_bytes"] + 3 * (reservation["output_bytes_per_result"]
                                           + reservation["result_metadata_bytes_per_result"]))
    cap = reservation["admission_bytes"]
    original = FilePayload.read_at
    def no_coefficients(snapshot, offset, size):
        assert offset in (0, 8), "short lifetime admission reached numerical RAM read"
        return original(snapshot, offset, size)
    with monkeypatch.context() as patch:
        patch.setattr(FilePayload, "read_at", no_coefficients)
        with pytest.raises(ForwardBindingError, match="memory admission|reservation"):
            spec.compile_forward(memory_bytes=cap - 1)
    with spec.compile_forward(memory_bytes=cap) as model:
        assert model.lifetime_reservation == reservation
        with pytest.raises(TypeError):
            model.lifetime_reservation["required_bytes"] = 0
        results = [model.forward(ForwardInputs([1, 2])) for _ in range(3)]
    assert all(result.outputs["last_hidden_state"].shape == (1, 2, 8) for result in results)


def test_lifetime_request_product_checked_before_numerical_read(source, monkeypatch):
    from pllm.runtime.bundle_storage import FilePayload
    original = FilePayload.read_at
    def no_coefficients(snapshot, offset, size):
        assert offset in (0, 8)
        return original(snapshot, offset, size)
    monkeypatch.setattr(FilePayload, "read_at", no_coefficients)
    with pytest.raises(ForwardBindingError, match="lifetime"):
        experiment(source, requests=(1 << 64) - 1).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES)


@pytest.mark.parametrize("payload", [
    b'{"weight_map":{},"weight_map":{}}',
    b'{"weight_map":{},"metadata":{"invalid":NaN}}',
])
def test_first_resolver_index_read_is_strict_and_bounded(source, monkeypatch, payload):
    from pllm.runtime.hf_hub import HuggingFaceSourceError
    from pllm.runtime.bundle_storage import FilePayload
    (source / "model.safetensors.index.json").write_bytes(payload)
    def no_snapshot(*args, **kwargs):
        raise AssertionError("invalid resolver metadata reached checkpoint snapshot")
    monkeypatch.setattr(FilePayload, "__init__", no_snapshot)
    with pytest.raises(HuggingFaceSourceError):
        experiment(source).compile_forward(memory_bytes=64 << 20)


def test_oversized_first_index_read_refused_before_snapshot(source, monkeypatch):
    from pllm.runtime.hf_hub import HuggingFaceSourceError
    from pllm.runtime.bundle_storage import FilePayload
    with (source / "model.safetensors.index.json").open("wb") as stream:
        stream.truncate((16 << 20) + 1)
    def no_snapshot(*args, **kwargs):
        raise AssertionError("oversized resolver metadata reached checkpoint snapshot")
    monkeypatch.setattr(FilePayload, "__init__", no_snapshot)
    with pytest.raises(HuggingFaceSourceError):
        experiment(source).compile_forward(memory_bytes=64 << 20)


def test_source_policy_hash_bound_and_descriptor_change(tmp_path, monkeypatch):
    from pathlib import Path
    from pllm.runtime.source_io import SourceReadError, SourceReadPolicy
    oversized = tmp_path / "oversized.safetensors"
    with oversized.open("wb") as stream:
        stream.truncate((4 << 30) + 1)
    with pytest.raises(SourceReadError, match="envelope"):
        SourceReadPolicy().hash_file(oversized)
    path = tmp_path / "metadata.json"
    path.write_bytes(b'{"x":1}')
    original = Path.open
    class ChangingReader:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def fileno(self):
            return self.stream.fileno()
        def read(self, size):
            result = self.stream.read(size)
            with original(path, "ab") as changed:
                changed.write(b" ")
            return result
    def opened(candidate, mode="r", *args, **kwargs):
        stream = original(candidate, mode, *args, **kwargs)
        return ChangingReader(stream) if candidate == path and mode == "rb" else stream
    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(SourceReadError, match="changed"):
        SourceReadPolicy().read_json(path)


def test_transient_source_rewrite_detected_even_when_bytes_restored(source, monkeypatch):
    from pllm.runtime.source_io import SourceReadError, SourceReadPolicy
    original = SourceReadPolicy.chunks
    target = source / "model.safetensors"
    changed = False
    def chunks(policy, path):
        nonlocal changed
        for block in original(policy, path):
            yield block
            if path == target and not changed:
                changed = True
                with target.open("r+b") as stream:
                    first = stream.read(1)
                    stream.seek(0)
                    stream.write(bytes([first[0] ^ 1]))
                    stream.flush()
                    stream.seek(0)
                    stream.write(first)
                    stream.flush()
    monkeypatch.setattr(SourceReadPolicy, "chunks", chunks)
    with pytest.raises((SourceReadError, ValueError), match="changed"):
        experiment(source).compile_forward(memory_bytes=64 << 20)


def test_disk_admission_uses_actual_snapshot_filesystem_and_owned_reads(source, tmp_path, monkeypatch):
    from collections import namedtuple
    from pathlib import Path
    from pllm.runtime import forward_binding as binding
    destination = tmp_path / "snapshot-volume"
    destination.mkdir()
    seen = []
    original_usage = binding.shutil.disk_usage
    Usage = namedtuple("Usage", "total used free")
    def usage(path):
        if Path(path) == Path.home():
            return original_usage(path)  # Ordinary host telemetry is separate.
        assert path == destination.resolve()
        seen.append(path)
        return Usage(1 << 40, 0, 1 << 40)
    monkeypatch.setattr(binding.tempfile, "gettempdir", lambda: str(destination))
    monkeypatch.setattr(binding.shutil, "disk_usage", usage)
    with experiment(source).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        assert "pooler_output" in model.forward(ForwardInputs([1])).outputs
    assert seen and not tuple(destination.iterdir())


def test_fallback_snapshot_cleanup_on_disk_full(source, tmp_path, monkeypatch):
    from pllm.runtime import bundle_storage, forward_binding
    destination = tmp_path / "snapshot-volume"
    destination.mkdir()
    original = bundle_storage.tempfile.NamedTemporaryFile
    class FullDisk:
        def __init__(self, stream):
            self.stream, self.name = stream, stream.name
        def write(self, data):
            raise OSError("simulated disk full")
        def close(self):
            self.stream.close()
    def full(*args, **kwargs):
        assert kwargs["dir"] == destination.resolve()
        return FullDisk(original(*args, **kwargs))
    monkeypatch.setattr(forward_binding.tempfile, "gettempdir", lambda: str(destination))
    monkeypatch.setattr(bundle_storage.tempfile, "NamedTemporaryFile", full)
    with pytest.raises(OSError, match="disk full"):
        experiment(source).compile_forward(memory_bytes=64 << 20)
    assert not tuple(destination.iterdir())


@pytest.mark.parametrize("section", ["top", "text_config"])
def test_authenticated_config_quantization_never_reopens_store(tmp_path, monkeypatch, section):
    from pllm.runtime.safetensors_store import SafeTensorStore
    config, tensors = tiny_checkpoint()
    if section == "top":
        config["quantization_config"] = {"bits": 4}
    else:
        config["text_config"] = {"quantization_config": {"bits": 4}}
    source = write_checkpoint(tmp_path / section, config, tensors)
    def forbidden(*args, **kwargs):
        raise AssertionError("forward reopened original SafeTensorStore")
    monkeypatch.setattr(SafeTensorStore, "__init__", forbidden)
    with pytest.raises(ForwardBindingError, match="quantized"):
        experiment(source).compile_forward(memory_bytes=64 << 20)


def test_policy_preserves_ordinary_source_lock_digests(source):
    from pllm.model_loader import _source_lock
    from pllm.runtime.source_io import SourceReadPolicy
    model = Model(str(source))
    assert _source_lock(source, model, read_policy=SourceReadPolicy()) == _source_lock(source, model)


@pytest.mark.parametrize("raw", [b'{"x":0,"x":1}', b'{"x":NaN}', b'{"x":1e999}'])
def test_source_policy_json_duplicate_and_nonfinite_refusal(tmp_path, raw):
    from pllm.runtime.source_io import SourceReadError, SourceReadPolicy
    path = tmp_path / "metadata.json"
    path.write_bytes(raw)
    with pytest.raises(SourceReadError):
        SourceReadPolicy().read_json(path)


def test_policy_inventory_bounds_before_list_retention(tmp_path):
    from pllm.runtime.source_io import SourceReadError, SourceReadPolicy
    for name in ("a.json", "b.json", "c.json"):
        (tmp_path / name).write_bytes(b"{}")
    with pytest.raises(SourceReadError, match="inventory"):
        SourceReadPolicy(inventory_nodes=2).directory_files(tmp_path, patterns=("*.json",))
    policy = SourceReadPolicy()
    assert policy.max_file_bytes == 4 << 30
    assert policy.max_metadata_bytes == 16 << 20
    assert policy.max_files == 65536
    assert policy.max_name_bytes == policy.max_metadata_bytes


def _write_resolver_metadata(source, *, repo_id):
    import hashlib
    manifest = {"schema": 1, "repo_id": repo_id, "revision": None, "commit": None,
                "files": [{"path": name, "size": (source / name).stat().st_size,
                           "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
                          for name in ("config.json", "model.safetensors")]}
    raw = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    (source / ".pllm-model.json").write_bytes(raw)
    return raw


def test_large_retained_resolver_identities_and_complete_metadata_cap(source, monkeypatch):
    from pllm.runtime import forward_binding as binding
    from pllm.runtime.bundle_storage import FilePayload
    # SourceLock owns a newly decoded, large resolver string throughout preload;
    # final verification also owns a second resolver DOM while it hashes files.
    label = "r" * (2 << 20)
    _write_resolver_metadata(source, repo_id=label)
    observed = []
    original_admit = binding._MetadataAdmission.admit
    def admission(budget, scratch):
        original_admit(budget, scratch)
        observed.append(budget.peak_bytes)
    monkeypatch.setattr(binding._MetadataAdmission, "admit", admission)
    spec = experiment(source, requests=2)
    description = binding.describe_forward_source(spec, memory_bytes=256 << 20)
    reservation = description["lifetime_reservation"]
    assert description["metadata_bytes"] >= 4 * len(label)
    assert description["metadata_phase_peak_bytes"] == reservation["metadata_phase_peak_bytes"]
    assert max(observed) == reservation["metadata_phase_peak_bytes"]
    assert reservation["required_bytes"] == max(reservation["metadata_phase_peak_bytes"],
        reservation["native_compile_prerequisite_bytes"], reservation["import_peak_bytes"],
        reservation["max_call_bytes"] + 2 * (
            reservation["output_bytes_per_result"] + reservation["result_metadata_bytes_per_result"]))
    assert reservation["metadata_phase_peak_bytes"] >= reservation["native_compile_prerequisite_bytes"] > 0
    assert reservation["admission_bytes"] == (reservation["required_bytes"] * 125 + 99) // 100
    assert sum(item["nbytes"] for item in description["outputs"].values()) == description["quote"]["output_bytes"]
    for item in description["outputs"].values():
        assert item["nbytes"] == 4 * math.prod(item["shape"])
    original_read = FilePayload.read_at
    def metadata_only(snapshot, offset, size):
        assert offset in (0, 8), "incomplete admission reached coefficient RAM"
        return original_read(snapshot, offset, size)
    with monkeypatch.context() as patch:
        patch.setattr(FilePayload, "read_at", metadata_only)
        with pytest.raises(ForwardBindingError, match="memory admission|reservation"):
            spec.compile_forward(memory_bytes=reservation["admission_bytes"] - 1)
    observed.clear()
    with spec.compile_forward(memory_bytes=reservation["admission_bytes"]) as model:
        assert model.lifetime_reservation == reservation
        assert max(observed) <= reservation["required_bytes"]
        assert "pooler_output" in model.forward(ForwardInputs([1, 2])).outputs


def test_final_resolver_transient_expansion_restoration_precedes_all_coefficients(source, monkeypatch, tmp_path):
    from pathlib import Path
    from pllm.runtime import forward_binding as binding
    from pllm.runtime.bundle_storage import FilePayload
    from pllm.runtime.source_io import SourceReadPolicy
    original_bytes = _write_resolver_metadata(source, repo_id="original")
    expanded_bytes = _write_resolver_metadata(source, repo_id="temporary" * (1 << 18))
    path = source / ".pllm-model.json"
    path.write_bytes(original_bytes)
    destination = tmp_path / "snapshots"
    destination.mkdir()
    monkeypatch.setattr(binding.tempfile, "gettempdir", lambda: str(destination))
    original_read = SourceReadPolicy.read_bytes
    original_payload_read = FilePayload.read_at
    reads = 0
    def transient(policy, candidate, *, limit=None):
        nonlocal reads
        if Path(candidate) == path:
            reads += 1
            if reads == 2:
                path.write_bytes(expanded_bytes)
                try:
                    return original_read(policy, candidate, limit=limit)
                finally:
                    path.write_bytes(original_bytes)
        return original_read(policy, candidate, limit=limit)
    def no_coefficients(snapshot, offset, size):
        assert offset in (0, 8), "final original verification ran after coefficients"
        return original_payload_read(snapshot, offset, size)
    monkeypatch.setattr(SourceReadPolicy, "read_bytes", transient)
    monkeypatch.setattr(FilePayload, "read_at", no_coefficients)
    with pytest.raises(ForwardBindingError, match="changed"):
        experiment(source).compile_forward(memory_bytes=256 << 20)
    assert reads == 2 and path.read_bytes() == original_bytes
    assert not tuple(destination.iterdir())


@pytest.mark.parametrize("exception", [KeyboardInterrupt, BaseException])
@pytest.mark.parametrize("phase", ["header", "coefficient"])
def test_controlled_baseexception_closes_every_snapshot(source, monkeypatch, tmp_path, exception, phase):
    from pathlib import Path
    from pllm.runtime import forward_binding as binding
    from pllm.runtime.bundle_storage import FilePayload
    destination = tmp_path / "snapshots"
    destination.mkdir()
    monkeypatch.setattr(binding.tempfile, "gettempdir", lambda: str(destination))
    snapshots = []
    original_init, original_read = FilePayload.__init__, FilePayload.read_at
    def created(snapshot, *args, **kwargs):
        original_init(snapshot, *args, **kwargs)
        snapshots.append(snapshot)
    def interrupted(snapshot, offset, size):
        if phase == "header" and offset == 8 or phase == "coefficient" and offset > 8:
            raise exception("controlled forward cancellation")
        return original_read(snapshot, offset, size)
    monkeypatch.setattr(FilePayload, "__init__", created)
    monkeypatch.setattr(FilePayload, "read_at", interrupted)
    with pytest.raises(exception, match="controlled forward cancellation"):
        experiment(source).compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES)
    assert snapshots and not tuple(destination.iterdir())
    for snapshot in snapshots:
        assert not Path(snapshot.path).exists()
        with pytest.raises(ValueError):
            original_read(snapshot, 0, 1)


def test_target_disk_low_home_ample_refuses_before_snapshot_creation(source, monkeypatch, tmp_path):
    from collections import namedtuple
    from pathlib import Path
    from pllm.runtime import forward_binding as binding
    from pllm.runtime.bundle_storage import FilePayload
    destination = tmp_path / "full-snapshot-volume"
    destination.mkdir()
    Usage = namedtuple("Usage", "total used free")
    seen = []
    def disk(path):
        path = Path(path)
        if path == Path.home():
            return Usage(1 << 40, 0, 1 << 40)
        assert path == destination.resolve()
        seen.append(path)
        return Usage(1 << 40, 1 << 40, 0)
    def forbidden(*args, **kwargs):
        raise AssertionError("low target disk admitted a snapshot")
    monkeypatch.setattr(binding.tempfile, "gettempdir", lambda: str(destination))
    monkeypatch.setattr(binding.shutil, "disk_usage", disk)
    monkeypatch.setattr(FilePayload, "__init__", forbidden)
    with pytest.raises(ForwardBindingError, match="insufficient disk"):
        experiment(source).compile_forward(memory_bytes=64 << 20)
    assert seen and not tuple(destination.iterdir())

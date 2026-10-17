"""Bounded native source import: independent formats, identities and retirement."""

import json
import math

import numpy as np
import pytest

from pllm import _native
from pllm.runtime import delegated_source as source

import test_delegated_transport_binding as binding
from test_delegated_source import sdk_experiment, write_source_fixture

admitted = binding.admitted


def finish_zero(client, args, tensors):
    decoder = json.loads(client.decoder())
    publication, pin, starts = client.publication(), client.publication_digest(), client.starts()
    workers = []
    try:
        for role in range(2):
            workers.append(_native._GkmTransportWorker(*args, tensors, publication, pin, role, starts[role]))

        def accept(frames):
            return client.accept(*(worker.execute(frame) for worker, frame in zip(workers, frames, strict=True)))

        assert accept(client.lookup(0)) == tensors[decoder["table"]][:decoder["width"] * 4]
        for stage, matrix in enumerate(decoder["matrices"]):
            assert accept(client.linear(stage, bytes(matrix["columns"] * 4))) == bytes(matrix["rows"] * 4)
        for worker in workers:
            worker.finish()
        assert json.loads(client.finish())["schema_version"] == "pllm.gkm.sdk-acceptance.v1"
    finally:
        client.abort()
        for worker in workers:
            worker.abort()


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_streamed_safetensors_preserve_complete_original_bytes_without_eager_import(tmp_path, monkeypatch, dtype, wrapped):
    _, _, arrays, _ = write_source_fixture(tmp_path)
    pattern = np.array([-0.0, 0.125, -0.25, 1 / 2048], dtype="<f4")
    header, payload, expected = {}, bytearray(), {}
    for name, value in sorted(arrays.items()):
        values = np.resize(pattern, math.prod(value.shape)).reshape(value.shape)
        expected[name] = values.tobytes()
        if dtype == "BF16":
            raw = (values.view("<u4") >> 16).astype("<u2").tobytes()
        else:
            raw = values.astype("<f2" if dtype == "F16" else "<f4").tobytes()
        header[("wrapped." if wrapped else "") + name] = {
            "dtype": dtype, "shape": list(value.shape), "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (tmp_path / "model.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    experiment = sdk_experiment(tmp_path)
    admission = source.admit_source(experiment)
    original = _native._GkmTransportImport
    chunks = []

    class Import:
        def __init__(self, *args):
            self.inner = original(*args)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def push(self, name, offset, data):
            assert 0 < len(data) <= 64 << 10
            assert data == expected[name][offset:offset + len(data)]
            chunks.append((name, offset, len(data)))
            self.inner.push(name, offset, data)

    def forbidden(*args, **kwargs):
        raise AssertionError("eager float/source snapshot reached")

    monkeypatch.setattr(_native, "_GkmTransportImport", Import)
    monkeypatch.setattr(source.AdmittedSource, "load_values", forbidden)
    monkeypatch.setattr(source._AdmittedTensorStore, "get", forbidden)
    monkeypatch.setattr(_native, "_GkmTransportClient", forbidden)
    client = admission.load_transport(experiment, 1, 1)
    assert sum(size for _, _, size in chunks) == admission.source_float32_bytes
    args = (experiment.canonical_bytes(), admission.config_bytes, 1, 1,
            admission.source_lock_digest, admission.tokenizer_digest)
    finish_zero(client, args, expected)


@pytest.mark.parametrize("method,args,kwargs", [
    ("push", (), {}), ("push", ("x", 0, b"1234", "extra"), {}),
    ("push", ("x", 0, b"1234"), {"name": "x"}), ("push", (), {"unexpected": 0}),
    ("push", ("x", True, b"1234"), {}), ("push", ("x", "0", b"1234"), {}),
    ("push", ("x", 0, bytearray(4)), {}), ("push", (object(), 0, b"1234"), {}),
    ("client", (1,), {}), ("client", (), {"unexpected": 0}),
    ("worker", (), {}), ("worker", (b"", b"", False, b""), {}),
    ("abort", (1,), {}),
])
def test_import_signature_conversion_errors_retire_native_payload(admitted, method, args, kwargs):
    native_args, _, tensors = admitted
    owner = _native._GkmTransportImport(*native_args)
    with pytest.raises((TypeError, ValueError)):
        getattr(owner, method)(*args, **kwargs)
    name = min(tensors)
    with pytest.raises(ValueError, match="retired"):
        owner.push(name, 0, tensors[name])


def test_import_chunk_offsets_and_keyword_forms_cross_tensor_rows(admitted):
    args, _, tensors = admitted
    owner = _native._GkmTransportImport(*args)
    for name, data in sorted(tensors.items()):
        for offset in range(0, len(data), 28):
            owner.push(name, data=data[offset:offset + 28], offset=offset)
    finish_zero(owner.client(), args, tensors)
    with pytest.raises(ValueError, match="retired"):
        owner.client()


def test_exact_maximum_frames_split_larger_source_tensors(admitted):
    args, _, _ = admitted
    config = json.loads(args[1])
    config.update(hidden_size=128, intermediate_size=128, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=256)
    args = (args[0], binding._canonical(config), *args[2:])
    decoder = json.loads(_native.gkm_runtime_contract(*args))["decoder"]
    tensors = {name: np.full(shape, 0.125, dtype="<f4").tobytes()
               for name, shape in decoder["source_shapes"].items()}
    assert max(map(len, tensors.values())) > 64 << 10
    owner = _native._GkmTransportImport(*args)
    for name, data in sorted(tensors.items()):
        for offset in range(0, len(data), 64 << 10):
            owner.push(name, offset, data[offset:offset + (64 << 10)])
    finish_zero(owner.client(), args, tensors)


@pytest.mark.parametrize("failure", ["cancel", "read", "changed"])
def test_source_failure_retires_partial_import_before_publication(tmp_path, monkeypatch, failure):
    write_source_fixture(tmp_path)
    experiment = sdk_experiment(tmp_path)
    admission = source.admit_source(experiment)
    original, frame = _native._GkmTransportImport, source._source_float_frame
    owners, seen = [], []

    def imported(*args):
        owner = original(*args)
        owners.append(owner)
        return owner

    def changed(*args):
        data = frame(*args)
        if not seen:
            seen.append(True)
            if failure == "read":
                raise OSError("injected source failure")
            if failure == "changed":
                with (tmp_path / "model.safetensors").open("ab") as writer:
                    writer.write(b"changed")
        return data

    monkeypatch.setattr(_native, "_GkmTransportImport", imported)
    monkeypatch.setattr(source, "_source_float_frame", changed)
    with pytest.raises((OSError, ValueError)):
        admission.load_transport(experiment, 1, 1, cancelled=lambda: bool(seen) and failure == "cancel")
    assert len(owners) == 1 and seen
    with pytest.raises(ValueError, match="retired"):
        owners[0].client()

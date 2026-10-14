"""Research fixture checks; public test signing key grants no production trust."""
import copy

import numpy as np
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts import probe_cold_provider_artifacts as probe


@pytest.mark.parametrize("role", ["inference", "preparation"])
def test_exact_round_trip(role):
    from pllm.runtime.quantization import quantize_weight_per_row
    blob, binding = probe.publish(role)
    loaded = probe.import_artifact(blob, probe.public_key(), binding)
    files, tensors, _ = probe.generated_source()
    assert ("token_lookup" in loaded.matrices) == (role == "inference")
    assert ("lm_head" in loaded.matrices) == (role == "inference")
    for sid, matrix in loaded.matrices.items():
        q = quantize_weight_per_row(tensors[sid], bits=8)
        assert loaded.objects[sid + "/weight"] == q.values.tobytes()
        assert loaded.objects[sid + "/scales"] == q.scales.astype("<f4").tobytes()
        x = np.full((2, matrix.shape[1]), -127, dtype=np.int8)
        np.testing.assert_array_equal(loaded.execute(sid, x), x.astype(np.int64) @ q.values.astype(np.int64).T)
    if role == "inference":
        for name in ("config.json", "tokenizer.json", "norm.f32", "bias.f32"):
            assert loaded.objects[name] == files[name]
    loaded.close()
    assert not loaded.matrices and not loaded.objects
    with pytest.raises(ValueError, match="retired"):
        loaded.execute("layers.0.qkv", np.zeros((1, 4), np.int8))


@pytest.mark.parametrize("kind", ["payload", "manifest", "signature", "truncated", "trailing", "wrong_key"])
def test_untrusted_delivery(kind):
    blob, binding = probe.publish("inference")
    key = probe.public_key()
    if kind == "payload":
        blob = blob[:-1] + bytes([blob[-1] ^ 1])
    elif kind == "manifest":
        blob = blob.replace(b"inference", b"x nference", 1)
    elif kind == "signature":
        _, header, _, _ = probe.split(blob)
        index = len(probe.MAGIC) + 4 + len(header)
        blob = blob[:index] + bytes([blob[index] ^ 1]) + blob[index + 1:]
    elif kind == "truncated":
        blob = blob[:-3]
    elif kind == "trailing":
        blob += b"extra"
    else:
        key = Ed25519PrivateKey.from_private_bytes(b"z" * 32).public_key().public_bytes_raw()
    with pytest.raises((ValueError, InvalidSignature)):
        probe.import_artifact(blob, key, binding)


@pytest.mark.parametrize("field", ["role", "source_witness_digest", "model_plan", "composition", "compiler", "execution"])
def test_external_binding_mismatch(field):
    blob, binding = probe.publish("inference")
    expected = copy.deepcopy(binding)
    expected[field] = "substitution"
    with pytest.raises(ValueError, match="binding mismatch|unsupported expected role"):
        probe.import_artifact(blob, probe.public_key(), expected)


@pytest.mark.parametrize("kind", ["stage_order", "object_order", "source", "offset", "shape", "scale_digest", "retention"])
def test_signed_structural_substitution(kind):
    blob, binding = probe.publish("inference")
    manifest, _, _, payload = probe.split(blob)
    if kind == "stage_order":
        manifest["stages"].reverse()
    elif kind == "object_order":
        manifest["objects"].reverse()
    elif kind == "source":
        manifest["source_witness"]["revision"] = "another-source"
    elif kind == "offset":
        manifest["objects"][0]["offset"] = 1
    elif kind == "shape":
        manifest["objects"][0]["shape"] = [4, 7]
    elif kind == "scale_digest":
        manifest["stages"][0]["scales_digest"] = "0" * 64
    else:
        manifest["stages"][0]["retained"] = False
    with pytest.raises(ValueError):
        probe.import_artifact(probe.seal(manifest, payload), probe.public_key(), binding)


def test_source_content_not_self_assertion():
    files, _, witness = probe.generated_source()
    files["weights.bf16"] = b"\0" + files["weights.bf16"][1:]
    with pytest.raises(ValueError, match="source content"):
        probe.verify_source(files, witness)


def test_import_snapshot_survives_delivery_mutation():
    blob, binding = probe.publish("preparation")
    mutable = bytearray(blob)
    loaded = probe.import_artifact(mutable, probe.public_key(), binding)
    x = np.full((1, 4), 7, dtype=np.int8)
    before = loaded.execute("layers.0.qkv", x)
    mutable[:] = b"\0" * len(mutable)
    np.testing.assert_array_equal(before, loaded.execute("layers.0.qkv", x))
    with pytest.raises(ValueError):
        probe.import_artifact(mutable, probe.public_key(), binding)
    loaded.close()


def test_oversize_bytearray_refused_before_conversion():
    class Sentinel(bytearray):
        def __len__(self):
            return 1  # Must use real buffer byte count, not overridable length.

        def __bytes__(self):
            raise AssertionError("oversize input converted before rejection")

    with pytest.raises(ValueError, match="artifact framing"):
        probe.split(Sentinel(probe.MAX_BYTES + 1))


def test_oversize_typed_memoryview_uses_bytes_not_elements():
    # len(view) is below MAX_BYTES; nbytes exceeds cap. No conversion permitted.
    with memoryview(bytearray(probe.MAX_BYTES + 4)).cast("I") as view:
        assert len(view) < probe.MAX_BYTES < view.nbytes
        with pytest.raises(ValueError, match="artifact framing"):
            probe.split(view)


def test_unsupported_conversion_sentinel():
    class Sentinel:
        def __bytes__(self):
            raise AssertionError("unsupported input converted")

    with pytest.raises(ValueError, match="bytes-like type"):
        probe.split(Sentinel())


def test_signed_unsupported_expected_role():
    blob, binding = probe.publish("preparation")
    manifest, _, _, payload = probe.split(blob)
    binding["role"] = "unsupported-provider"
    manifest["binding"] = binding
    with pytest.raises(ValueError, match="unsupported expected role"):
        probe.import_artifact(probe.seal(manifest, payload), probe.public_key(), binding)


def test_close_releases_native_executor_owner():
    import weakref
    blob, binding = probe.publish("preparation")
    loaded = probe.import_artifact(blob, probe.public_key(), binding)
    owner = weakref.ref(loaded.kernel)
    loaded.close()
    loaded.close()  # Idempotent retirement.
    assert loaded.kernel is None and owner() is None


@pytest.mark.parametrize("failure", ["constructor", "non_native", "second_compile"])
def test_construction_failure_retires_owner(monkeypatch, failure):
    import weakref
    import pllm.runtime.native as native
    owners, calls = [], []
    instances = []

    class Kernel:
        def __init__(self, *, threads, build):
            assert threads == 1 and build is False
            owners.append(weakref.ref(self))
            self.native = failure != "non_native"
            self.count = 0
            if failure == "constructor":
                raise RuntimeError("constructor failure")

        def compile(self, weights):
            self.count += 1
            calls.append(self.count)
            if self.count == 2:
                raise RuntimeError("compile failure")
            return {"owner": self}

    class TrackedImport(probe.Imported):
        def __init__(self, *args):
            instances.append(self)
            super().__init__(*args)

    blob, binding = probe.publish("preparation")
    monkeypatch.setattr(native, "MaskedGEMM", Kernel)
    monkeypatch.setattr(probe, "Imported", TrackedImport)
    with pytest.raises((RuntimeError, ValueError)):
        probe.import_artifact(blob, probe.public_key(), binding)
    assert instances[0].closed and instances[0].kernel is None
    assert not instances[0].objects and not instances[0].matrices
    assert all(owner() is None for owner in owners)
    assert calls == ([1, 2] if failure == "second_compile" else [])


def test_full_bill_and_determinism():
    first, second = probe.run(), probe.run()
    assert first == second
    for sample in first["samples"]:
        assert sample["artifact_bytes"] == sample["manifest_bytes"] + sample["payload_bytes"] + sample["framing_and_signature_bytes"]
    assert first["both_role_delivery_bytes"] == sum(s["artifact_bytes"] for s in first["samples"])
    assert first["array_bound_bytes"] <= 128 << 20


def test_cold_import_does_not_read_or_quantize_source(monkeypatch):
    import pllm.runtime.quantization as quantization
    blob, binding = probe.publish("inference")

    def forbidden(*args, **kwargs):
        raise AssertionError("cold reader touched original source or quantizer")

    monkeypatch.setattr(probe, "generated_source", forbidden)
    monkeypatch.setattr(quantization, "quantize_weight_per_row", forbidden)
    loaded = probe.import_artifact(blob, probe.public_key(), binding)
    assert len(loaded.matrices) == 4
    loaded.close()


@pytest.mark.parametrize("name", ["layers.0.qkv/weight", "layers.0.qkv/scales", "tokenizer.json", "norm.f32", "bias.f32"])
def test_each_tensor_or_metadata_tamper(name):
    blob, binding = probe.publish("inference")
    manifest, _, _, payload = probe.split(blob)
    obj = next(o for o in manifest["objects"] if o["name"] == name)
    changed = bytearray(payload)
    changed[obj["offset"]] ^= 1
    with pytest.raises(ValueError, match="object digest"):
        probe.import_artifact(probe.seal(manifest, bytes(changed)), probe.public_key(), binding)


def test_signature_attests_publisher_not_quantization_derivation():
    # Deliberately dishonest *trusted* publisher can consistently replace q.
    # This passing import demonstrates the explicitly documented trust limit.
    blob, binding = probe.publish("preparation")
    manifest, _, _, payload = probe.split(blob)
    obj = manifest["objects"][0]
    changed = bytearray(payload)
    changed[obj["offset"]] ^= 1
    digest = probe.sha(bytes(changed[obj["offset"]:obj["offset"] + obj["bytes"]]))
    obj["sha256"] = digest
    stage = next(s for s in manifest["stages"] if s["id"] == "layers.0.qkv")
    stage["weight_digest"] = digest
    loaded = probe.import_artifact(probe.seal(manifest, bytes(changed)), probe.public_key(), binding)
    assert loaded.objects[obj["name"]] != payload[obj["offset"]:obj["offset"] + obj["bytes"]]
    loaded.close()

import hashlib

import numpy as np
import pytest

from pllm.runtime.benchmark_decoder_restart import bind, publish, restore, advance
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    source = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=2)
    compiled, contract, runtime, remote = bind(source, 32)
    history = [2, 4, 9, 3]
    logits = runtime.prepare_ids(history)[1]
    token = int(np.argmax(logits))
    logits = runtime.forward_ids([token])[-1]
    history.append(token)
    root, key = tmp_path / "checkpoint", b"r" * 32
    head, _ = publish(root, key, 0, compiled, contract, runtime, logits, history)
    return source, compiled, contract, runtime, remote, root, key, head, logits, history


def test_authenticated_restore_preserves_pending_logits_kv_and_failed_publication(checkpoint):
    source, compiled, contract, runtime, remote, root, key, head, logits, history = checkpoint
    with pytest.raises(ValueError, match="interruption"):
        publish(root, key, 1, compiled, contract, runtime, logits, history, fail=True)
    assert not list(root.glob("pending-*")) and not (root / "1").exists()
    saved = runtime.snapshot_for_response("original")
    runtime.restore_for_response(saved, "original")
    expected = advance(runtime, logits, history.copy())
    rebuilt, native, resumed, issued = bind(source, 32)
    previous, executed, pending = restore(root, key, head, rebuilt, native, resumed)
    assert resumed.position == len(history) == len(executed)
    assert pending == int(np.argmax(logits))
    np.testing.assert_array_equal(previous.view(np.uint32), logits.view(np.uint32))
    assert advance(resumed, previous, executed) == expected
    assert remote.attempts.isdisjoint(issued.attempts)


@pytest.mark.parametrize("failure", ["corruption", "wrong_key", "revision", "source", "pending"])
def test_untrusted_restart_cannot_substitute_state(checkpoint, failure):
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import msgpack
    from pllm.runtime.benchmark_decoder_restart import _aad
    source, compiled, contract, _, _, root, key, head, _, _ = checkpoint
    rebuilt, native, runtime, _ = bind(source, 32)
    path = root / "0"
    frame = path.read_bytes()
    if failure == "wrong_key":
        key = b"x" * 32
    elif failure == "revision":
        (root / "1").write_bytes(frame)
        head = {**head, "revision": 1}
    elif failure == "source":
        other = create_tiny_llama_checkpoint(source.parent / "other", seed=91, num_hidden_layers=2)
        rebuilt, native, runtime, _ = bind(other, 32)
    elif failure == "pending":
        raw = AESGCM(key).decrypt(frame[:12], frame[12:], _aad(compiled, contract, 0))
        doc = msgpack.unpackb(raw, raw=False)
        doc["pending"] = (doc["pending"] + 1) % int(runtime.cfg["vocab_size"])
        nonce = b"z" * 12
        frame = nonce + AESGCM(key).encrypt(nonce, msgpack.packb(doc, use_bin_type=True), _aad(compiled, contract, 0))
        path.write_bytes(frame)
        head = {**head, "digest": hashlib.sha256(frame).hexdigest()}
    else:
        path.write_bytes(frame[:-1] + bytes([frame[-1] ^ 1]))
    with pytest.raises((ValueError, InvalidTag)):
        restore(root, key, head, rebuilt, native, runtime)
    assert runtime.position == 0


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_fresh_process_decoder_restart_target(tmp_path, family):
    import pllm
    from pllm.metrics import BenchmarkTarget, Sampling
    from pllm._benchmark_worker import Measurement
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.benchmark_targets import execute_target
    from test_decoder_continuation import QuantizedRemote
    source = create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    compiled, _, runtime, remote = bind(source, 32)
    clear = compiled.runtime(QuantizedRemote(remote.engine, compiled._bundle.model_id))
    np.testing.assert_array_equal(runtime.prepare_ids([2, 3, 7])[1], clear.prepare_ids([2, 3, 7])[1])
    experiment = pllm.Experiment("restart", MaskedLinearCpu(pllm.Model.path(str(source)),
        quantization=SymmetricPerRow(causal_reduction="prefix_f32")),
        pllm.Deployment.local(root=str(tmp_path)), pllm.ExecutionBudget(1, 32, 4))
    m = Measurement(dict(sampling=Sampling(processes=1).to_spec(), salt="ac" * 32,
                         inputs={}, directory=str(tmp_path)))
    execute_target(BenchmarkTarget.state(experiment, tokens=8, operation="decoder_restart", storage="sealed_local"), m)
    assert m.oracle["outcome"] == "passed" and m.oracle["checked_values"] > 0
    assert m.details["restarted"][0]["fresh_attempts"] == 16
    assert m.details["publication"]["private_file_bytes"] > 0


def test_source_admission_precedes_loading_values(checkpoint, monkeypatch):
    from pllm._benchmark_contract import TargetUnavailable
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    monkeypatch.setattr(SafeTensorStore, "tensor_shape", lambda *_: (8_000_000,))
    def reject(*args):
        raise AssertionError("values loaded before source admission")
    monkeypatch.setattr(MaskedTransformerEngine, "load", reject)
    with pytest.raises(TargetUnavailable, match="restart_reference_source_limit"):
        bind(checkpoint[0], 32)

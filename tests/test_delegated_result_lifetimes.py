"""Result admission and exception ownership, without checkpoint value loading."""

from contextlib import contextmanager
import json
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from pllm.runtime import delegated_client, delegated_decoder, delegated_transport


@pytest.mark.parametrize("failure", ["finish", "phase", "cancel", "chain", "group", "cleanup"])
def test_failed_client_drains_tentative_frames_before_reservation_release(monkeypatch, failure):
    refs, releases, events = [], [], []
    contract = {"input_rows": 1, "decode_steps": 1 if failure == "cancel" else 0,
                "vocabulary": 2, "matrices": [{}]}

    def fail(message):
        private = np.ones(16, dtype=np.uint8)
        refs.append(weakref.ref(private))
        raise ValueError(message)

    class Phase:
        def __init__(self, *args):
            self.caches = []
            self.position = 1

        def phase(self, ids):
            vector = np.ones(8, dtype=np.float32)
            refs.append(weakref.ref(vector))
            if failure == "phase":
                fail("phase")
            return vector, 0

        def retire(self):
            events.append("retired")

    class Native:
        def contract(self):
            return b"bound"

        def decoder(self):
            return json.dumps(contract).encode()

        def publication(self):
            return b"publication"

        def publication_digest(self):
            return b"pin"

        def starts(self):
            return (b"a", b"b")

        def request_id(self):
            return bytes(32)

        def finish(self):
            if failure in {"chain", "group"}:
                try:
                    fail("inner")
                except ValueError as error:
                    if failure == "group":
                        raise ExceptionGroup("group", [error]) from None
                    raise RuntimeError("chain") from error
            fail("verification")

        def abort(self):
            events.append("aborted")

    class Peers:
        def __init__(self, *args, **kwargs):
            pass

        def admit(self, *args):
            pass

        def finish(self):
            return []

        def close(self):
            events.append("closed")
            if failure == "cleanup":
                fail("cleanup")

    @contextmanager
    def reservation(size):
        try:
            yield
        finally:
            releases.append(True)
            assert "aborted" in events and "closed" in events and "retired" in events
            assert refs and all(ref() is None for ref in refs), "unpriced tentative payload"

    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return failure == "cancel" and checks == 3

    source = SimpleNamespace(
        capacity="capsule", contract=contract, execution_payload_estimate_bytes=1024,
        runtime_contract=lambda *args: b"bound", load_transport=lambda *args, **kwargs: Native(),
    )
    monkeypatch.setattr(delegated_decoder, "_Request", Phase)
    monkeypatch.setattr(delegated_transport, "_Peers", Peers)
    monkeypatch.setattr(delegated_transport, "reserve_incremental_memory", reservation)
    with pytest.raises((ValueError, RuntimeError, ExceptionGroup)) as caught:
        delegated_transport.execute_request(
            None, source, (0,), contract["decode_steps"] + 1,
            {"worker_a": ("http://127.0.0.1:23001", "a" * 32),
             "worker_b": ("http://127.0.0.1:23002", "b" * 32)},
            5.0, cancelled,
        )
    # Holding the propagated error, including its chains/groups, cannot keep
    # original payloads alive after the reservation's finally block above.
    assert caught.value is not None and releases == [True]
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("rows,steps,expected", [(3, 1, 1313792),
                                                (150, 31, 23896064),
                                                (288, 31, 27287552)])
def test_qwen_result_bound_covers_all_native_admitted_logits_and_kv(rows, steps, expected):
    cfg = {"model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864,
           "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
           "vocab_size": 151936, "max_position_embeddings": 32768, "hidden_act": "silu",
           "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": True}
    contract = json.loads(delegated_decoder.decoder_contract(
        cfg, input_rows=rows, decode_steps=steps, capacity="decoder"))
    bound = contract["decoder_resources"]["accepted_result_bytes"]
    assert bound == expected + (16 << 10)
    count, position = steps + 1, rows + steps
    # Repeated immutable buffers exercise the complete logical result geometry
    # without allocating the complete official-checkpoint state or any weights.
    vector = bytes(4 * 151936)
    state = bytes(4 * position * 2 * 64)
    result = delegated_decoder.GkmDecoderResult(
        (vector,) * count, (((position, 2, 64), state, state),) * 24,
        (0,) * count, (0,) * position, position, b"accepted",
    )
    client = object.__new__(delegated_client.DelegatedClient)
    client._capacity = "decoder"
    source = SimpleNamespace(config=cfg, contract=contract)
    client._validate_result(result, source, (0,) * rows, count)
    client._capacity = "capsule"
    if expected > 16 << 20:
        with pytest.raises(delegated_client.ProtocolError, match="bounded owner"):
            client._validate_result(result, source, (0,) * rows, count)


@pytest.mark.parametrize("limit", [None, True, 0, -1, 2049, 1 << 40])
def test_decoder_missing_or_inconsistent_result_limit_rejects_before_loading(monkeypatch, limit):
    contract = {"vocabulary": 2, "decoder_resources": {"accepted_result_bytes": limit},
                "execution_payload_estimate_bytes": 2048}

    def forbidden(*args, **kwargs):
        pytest.fail("result admission followed reservation, source loading or network execution")

    source = SimpleNamespace(
        capacity="decoder", contract=contract,
        runtime_contract=lambda *args: b"bound", load_transport=forbidden,
    )
    monkeypatch.setattr(delegated_transport, "reserve_incremental_memory", forbidden)
    monkeypatch.setattr(delegated_transport, "_Peers", forbidden)
    with pytest.raises(ValueError, match="result has no admitted bound"):
        delegated_transport.execute_request(
            None, source, (0,), 1,
            {"worker_a": ("http://127.0.0.1:23001", "a" * 32),
             "worker_b": ("http://127.0.0.1:23002", "b" * 32)},
            5.0, lambda: False,
        )

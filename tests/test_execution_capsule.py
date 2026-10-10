import numpy as np
import pytest

from pllm.runtime.execution_capsule import capsule_runtime, publish_execution_capsule
from pllm.runtime.transformer_client import TransformerClientError
from test_batched_greedy_verification import fixture
from test_decoder_continuation import runtime_for


def test_native_steps_preserve_logits_kv_and_execution_basis(fixture):
    engine, compiled, _ = fixture
    original, _ = runtime_for(compiled, engine)
    _, remote = runtime_for(compiled, engine)
    payload, digest = publish_execution_capsule(compiled)
    candidate = capsule_runtime(compiled, remote, payload, digest)
    assert "operations" not in candidate._graphs["prefill"]
    assert "steps" not in candidate._schedule["decode"]
    _, expected, _ = original.prepare_ids([1, 3, 6, 9])
    _, actual, _ = candidate.prepare_ids([1, 3, 6, 9])
    for _ in range(4):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert candidate._snapshot_basis == original._snapshot_basis
        for left, right in zip(candidate.caches, original.caches, strict=True):
            for kind in ("key", "value"):
                np.testing.assert_array_equal(getattr(left, kind)[:left.length].view(np.uint32),
                                              getattr(right, kind)[:right.length].view(np.uint32))
        token = int(np.argmax(expected))
        actual = candidate.forward_ids([token])[-1]
        expected = original.forward_ids([token])[-1]


def test_capsule_substitution_bounds_failure_and_explicit_composition_reject(fixture):
    engine, compiled, contract = fixture
    _, remote = runtime_for(compiled, engine)
    payload, digest = publish_execution_capsule(compiled)
    for corrupted, commitment in ((payload[:-1] + b"x", digest), (payload, "0" * 64), (b"x" * ((8 << 20) + 1), digest)):
        with pytest.raises(ValueError, match="artifact"):
            capsule_runtime(compiled, remote, corrupted, commitment)
    candidate = capsule_runtime(compiled, remote, payload, digest)
    with pytest.raises(TransformerClientError, match="capsule"):
        candidate.install_continuation(contract)
    remote.inject = "nonfinite"
    with pytest.raises((ValueError, TransformerClientError)):
        candidate.prepare_ids([1, 2])
    assert candidate._snapshot_basis is None
    candidate.reset()
    assert all(cache.key is None for cache in candidate.caches)

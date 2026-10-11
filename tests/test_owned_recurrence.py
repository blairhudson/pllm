from __future__ import annotations

import copy

import numpy as np
import pytest

from pllm.state import BoundedGatedDeltaRecurrence, HybridStateError, OwnedGatedDeltaRecurrence


def inputs(geometry, tokens):
    kh, vh, kd, vd = geometry
    rng = np.random.default_rng(617)
    q = rng.normal(size=(1, tokens, kh, kd)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    v = rng.normal(size=(1, tokens, vh, vd)).astype(np.float32)
    g = -rng.random((1, tokens, vh), dtype=np.float32)
    b = rng.random(g.shape, dtype=np.float32)
    return q, k, v, g, b


@pytest.mark.parametrize("geometry", [(1, 1, 4, 7), (2, 6, 16, 4), (16, 32, 128, 128)])
@pytest.mark.parametrize("strided", [False, True])
def test_owned_prefill_decode_bits_independent_outputs_and_constant_owners(geometry, strided):
    reference = BoundedGatedDeltaRecurrence(*geometry)
    state = reference.initial_state()
    data = inputs(geometry, 7)
    if strided:
        data = tuple(x.transpose(0, 2, 3, 1).copy().transpose(0, 3, 1, 2)
                     if x.ndim == 4 else x for x in data)
    with OwnedGatedDeltaRecurrence(*geometry, maximum_tokens=7) as owner:
        address, scratch = id(owner._state), id(owner._workspace.outer)
        snapshots = []
        for begin, end in [(0, 3), (3, 4), (4, 5), (5, 6), (6, 7)]:
            mode = "prefill" if begin == 0 else "decode"
            rows = [a[:, begin:end] for a in data]
            expected, state = reference.evaluate(*rows, state, mode=mode)
            output = owner.evaluate(*rows, mode=mode)
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
            np.testing.assert_array_equal(owner.snapshot().view(np.uint32), state.view(np.uint32))
            assert id(owner._state) == address and id(owner._workspace.outer) == scratch
            snapshots.append((output, output.copy()))
        saved = owner.snapshot()
        saved.fill(0)
        assert np.any(owner.snapshot())
        with pytest.raises(HybridStateError, match="token bound"):
            owner.evaluate(*[a[:, :1] for a in data])
        for output, expected in snapshots:
            np.testing.assert_array_equal(output, expected)
        retained, workspace = owner._state, owner._workspace
    assert owner.state_bytes == owner.workspace_bytes == 0
    assert not np.any(retained) and not np.any(workspace.outer)
    with pytest.raises(HybridStateError, match="closed"):
        owner.snapshot()


def test_invalid_inputs_preserve_state_but_mid_transition_cancellation_burns(monkeypatch):
    import pllm.runtime.semantic_hybrid as hybrid
    owner = OwnedGatedDeltaRecurrence(2, 4, 8, 6)
    data = inputs((2, 4, 8, 6), 1)
    owner.evaluate(*data)
    saved = owner.snapshot()
    with pytest.raises(HybridStateError):
        owner.evaluate(*data[:-1], data[-1] + 2)
    np.testing.assert_array_equal(owner.snapshot(), saved)
    with pytest.raises(TypeError):
        copy.copy(owner)
    with pytest.raises(TypeError):
        copy.deepcopy(owner)
    retained = owner._state
    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(hybrid.np, "einsum", cancelled)
    with pytest.raises(KeyboardInterrupt):
        owner.evaluate(*data)
    assert not np.any(retained) and owner.state_bytes == 0


def test_pairing_uses_semantics_and_requires_exclusive_state():
    from pllm.runtime.recurrent_execution import recurrent_pairs
    a = {"id": "arbitrary-result", "operator": "gated_delta_rule", "layer": 2,
         "inputs": ["q", "k", "v", "g", "b", "state"], "attributes": {"same": True}}
    b = {**a, "id": "arbitrary-state", "operator": "gated_delta_state_update"}
    graph = {"operations": [b, a], "state_inputs": [{"id": "state", "kind": "recurrent"}],
             "state_outputs": [{"id": b["id"], "kind": "recurrent"}]}
    assert set(recurrent_pairs(graph)) == {a["id"], b["id"]}
    graph["operations"].append({"id": "extra", "operator": "reshape", "inputs": ["state"]})
    assert recurrent_pairs(graph) == {}
    graph["operations"].pop()
    b["attributes"] = {"same": False}
    assert recurrent_pairs(graph) == {}

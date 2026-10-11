"""Serial-only generated checkpoint gate (Torch imports; not lightweight gate)."""
import copy

import numpy as np

from pllm.runtime.context_graph import context_graph_runtime
from pllm.runtime.tiered_execution import tiered_runtime
from test_batched_greedy_verification import fixture
from test_decoder_continuation import runtime_for


def test_generated_decoder_shared_branches_logits_and_every_kv_word(fixture, tmp_path):
    engine, compiled, _ = fixture
    for factory in (tiered_runtime, context_graph_runtime):
        root = tmp_path / factory.__name__
        _, remote = runtime_for(compiled, engine)
        candidate = factory(compiled, remote, root, b"k" * 32, page_rows=2)
        candidate.prepare_ids([1, 4, 6])
        head = copy.deepcopy(candidate._tiered.head)
        pages = copy.deepcopy(candidate._tiered.records)
        candidate.reset()
        for first_token in (2, 3):
            serial, _ = runtime_for(compiled, engine)
            serial.prepare_ids([1, 4, 6])
            branch_head = head
            for index in range(18):
                _, remote = runtime_for(compiled, engine)
                candidate = factory(compiled, remote, root, b"k" * 32,
                                    head=branch_head, page_rows=2)
                token = first_token if index == 0 else int(np.argmax(expected))
                expected = serial.forward_ids([token])[-1]
                actual = candidate.forward_ids([token])[-1]
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                state = candidate._tiered
                assert all(new[:len(old)] == old for new, old in zip(state.records, pages))
                if factory is context_graph_runtime:
                    assert state.pending_token == int(np.argmax(expected))
                branch_head = copy.deepcopy(state.head)
                # Inspect complete active KV, not only logits or a horizon formula.
                state.begin(candidate, np.array([0], np.int64), False)
                for layer in range(candidate.layers):
                    state.load(candidate, layer)
                    for kind in ("key", "value"):
                        np.testing.assert_array_equal(
                            getattr(candidate.caches[layer], kind)[:candidate.position].view(np.uint32),
                            getattr(serial.caches[layer], kind)[:serial.position].view(np.uint32))
                    state._retire(candidate)
                candidate.reset()
            serial.reset()

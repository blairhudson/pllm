import numpy as np
import pytest
from tokenizers import Tokenizer, models, decoders

from pllm.runtime.checkpoint_drafter import CheckpointDrafter, target_suffix
from pllm.runtime.greedy_verification_reference import verify_batched_greedy
from test_batched_greedy_verification import fixture
from test_decoder_continuation import runtime_for


def test_token_boundary_requires_exact_prefix_even_across_bpe_merges():
    tokenizer = Tokenizer(models.BPE({"a": 0, "b": 1, "ab": 2}, [("a", "b")]))
    tokenizer.decoder = decoders.Fuse()
    assert target_suffix(tokenizer, [0], "b", 1) == []
    assert target_suffix(tokenizer, [2], "a", 1) == [0]
    assert target_suffix(tokenizer, [0, 1], "a", 1) == []


def test_actual_checkpoint_drafter_cannot_change_target_logits_or_kv(fixture):
    engine, compiled, contract = fixture
    reference, _ = runtime_for(compiled, engine)
    prompt = reference.encode_prompt("Hello")
    _, scores, _ = reference.prepare_ids(prompt)
    expected, golden = [], []
    for index in range(8):
        golden.append(scores.copy())
        expected.append(int(np.argmax(scores)))
        if index < 7:
            scores = reference.forward_ids([expected[-1]])[-1]
    target, _ = runtime_for(compiled, engine)
    small, _ = runtime_for(compiled, engine)
    drafter = CheckpointDrafter(small, contract, target.tokenizer, prompt, "Different prefix")
    checked = []
    try:
        result = verify_batched_greedy(target, contract, prompt, drafter, outputs=8, width=2, on_output=checked.append)
        assert result["tokens"] == expected
        np.testing.assert_array_equal(np.asarray(checked).view(np.uint32), np.asarray(golden).view(np.uint32))
        for actual, original in zip(target.caches, reference.caches, strict=True):
            for kind in ("key", "value"):
                np.testing.assert_array_equal(getattr(actual, kind)[:actual.length].view(np.uint32),
                    getattr(original, kind)[:original.length].view(np.uint32))
        assert drafter.counts["generated_rows"] > 0
        assert drafter.counts["prefill_rows"] > 0
    finally:
        drafter.close()
    assert all(cache.key is None for cache in small.caches)
    with pytest.raises(ValueError, match="closed"):
        drafter(prompt, 2)


def test_drafter_malformed_history_burns_state(fixture):
    engine, compiled, contract = fixture
    small, _ = runtime_for(compiled, engine)
    draft = CheckpointDrafter(small, contract, small.tokenizer, [1], "Hello")
    with pytest.raises(ValueError, match="history"):
        draft([2], 2)
    assert draft.closed and small.position == 0

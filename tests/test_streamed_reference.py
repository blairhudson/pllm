from __future__ import annotations

import numpy as np
import pytest

from pllm.runtime.streamed_reference import StreamedFloat32Reference, StreamedReferenceError


def _checkpoint(tmp_path, family="qwen3", tied=True):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config_cls = {"qwen3": transformers.Qwen3Config,
                  "qwen2": transformers.Qwen2Config,
                  "llama": transformers.LlamaConfig}[family]
    config = config_cls(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8,
                        tie_word_embeddings=tied, max_position_embeddings=64)
    torch.manual_seed(745)
    reference = transformers.AutoModelForCausalLM.from_config(
        config, attn_implementation="eager").eval()
    reference.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="5KB")
    return torch, reference


@pytest.mark.parametrize("family,tied", [("qwen3", True), ("qwen3", False),
                                         ("qwen2", True), ("llama", False)])
def test_streamed_reference_preserves_upstream_logits_and_retires_weights(tmp_path, family, tied):
    torch, reference = _checkpoint(tmp_path, family, tied)
    with StreamedFloat32Reference(tmp_path, max_input_tokens=16) as streamed:
        for ids in ([1, 3, 5], [2, 7, 9, 8], [1, 3, 5, 11]):
            with torch.inference_mode():
                expected = reference(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].numpy()
            np.testing.assert_array_equal(streamed.logits(ids), expected)
            assert all(parameter.is_meta for parameter in streamed.model.parameters())
            assert streamed._active_bytes == 0
        assert 0 < streamed.stats["peak_active_weight_bytes"] <= streamed.weight_bound
        assert streamed.weight_bound < sum(p.numel() * 4 for p in reference.parameters())
    assert not streamed._hooks
    with pytest.raises(StreamedReferenceError, match="closed"):
        streamed.logits([1])


def test_streamed_reference_preflights_values_and_retires_on_forward_failure(tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    with pytest.raises(StreamedReferenceError, match="bound"):
        StreamedFloat32Reference(tmp_path, max_weight_bytes=8)
    with StreamedFloat32Reference(tmp_path) as streamed:
        target = next(module for module, _, _ in streamed._owners)
        def fail(*args, **kwargs):
            assert any(not p.is_meta for p in target.parameters(recurse=False))
            raise RuntimeError("injected forward failure")
        monkeypatch.setattr(target, "forward", fail)
        with pytest.raises(RuntimeError, match="injected"):
            streamed.logits([1, 2])
        assert all(p.is_meta for p in streamed.model.parameters())
        assert streamed._active_bytes == 0
        assert not streamed._store._cache


def test_streamed_reference_rejects_changed_source_before_values(tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    with StreamedFloat32Reference(tmp_path) as streamed:
        path = next(iter(streamed._files))
        path.touch()
        monkeypatch.setattr(streamed, "_load", lambda *_: pytest.fail("loaded changed values"))
        with pytest.raises(StreamedReferenceError, match="changed"):
            streamed.logits([1, 2])


@pytest.mark.parametrize("family,tied", [("qwen3", True), ("qwen2", False), ("llama", True)])
def test_streamed_cached_decode_matches_upstream_and_bounds_state(tmp_path, family, tied):
    torch, reference = _checkpoint(tmp_path, family, tied)
    with StreamedFloat32Reference(tmp_path, max_input_tokens=4, max_decode_steps=3) as streamed:
        ids, tokens = [1, 3, 5], [11, 7, 9]
        with torch.inference_mode():
            output = reference(input_ids=torch.tensor([ids]), use_cache=True)
            expected = [output.logits[0, -1].numpy().copy()]
            for token in tokens:
                output = reference(input_ids=torch.tensor([[token]]),
                                   past_key_values=output.past_key_values, use_cache=True)
                expected.append(output.logits[0, -1].numpy().copy())
        actual = streamed.trajectory(ids, tokens)
        for row, oracle in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(row, oracle)
        assert streamed.kv_bound_bytes == 2 * 2 * 1 * 8 * 7 * 4
        assert streamed._active_bytes == 0
        assert all(parameter.is_meta for parameter in streamed.model.parameters())
        with pytest.raises(StreamedReferenceError, match="bounds"):
            streamed.trajectory(ids, tokens + [1])

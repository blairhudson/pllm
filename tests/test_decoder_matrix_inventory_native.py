"""Internal binding contract; requires a native build containing the inventory API."""

import hashlib
import json

import pytest

from pllm import _native


pytestmark = pytest.mark.skipif(
    not hasattr(_native, "decoder_matrix_inventory"),
    reason="native extension must be rebuilt with decoder_matrix_inventory",
)


def test_internal_inventory_is_canonical_immutable_metadata_without_composition():
    config = {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 32, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True,
    }
    plan = _native.lower_model(json.dumps(config).encode(), 1, 3, 2)
    encoded, digest = _native.decoder_matrix_inventory(plan)
    assert isinstance(encoded, bytes)
    inventory = json.loads(encoded)
    assert encoded == json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    assert digest == hashlib.sha256(b"pllm.decoder_matrix_inventory.v1\0" + encoded).hexdigest()
    for mode in ("prefill", "decode"):
        groups = inventory[mode]["groups"]
        assert len(groups) == 10
        assert groups[1]["logical_shape"] == [16, 8]
        assert [member["output_offset"] for member in groups[1]["members"]] == [0, 8, 12]
        assert groups[0]["artifact_kind"] == "token_table"
        assert groups[-1]["artifact_kind"] == "matrix"
        assert groups[0]["members"][0]["weight_id"] == groups[-1]["members"][0]["weight_id"]
    assert not {"composition_digest", "complete", "protected_execution"} & inventory.keys()
    assert _native.decoder_matrix_inventory(plan) == (encoded, digest)


def test_internal_inventory_rejects_oversized_and_ambiguous_metadata():
    with pytest.raises(ValueError, match="byte limit"):
        _native.decoder_matrix_inventory(b" " * ((8 << 20) + 1))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _native.decoder_matrix_inventory(b'{"weight":"one","weight":"two"}')
    with pytest.raises(TypeError):
        _native.decoder_matrix_inventory(bytearray(b"{}"))

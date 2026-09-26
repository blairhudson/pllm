from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pllm.modeling import lower_model
from pllm.runtime.semantic_numeric import SemanticNumericError, float32_rotary_text_mrope


def _declared_rotary() -> tuple[dict, dict]:
    source = Path(__file__).resolve().parents[1] / "crates/pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    plan = lower_model(config, batch=1, max_input_tokens=5, max_new_tokens=1)
    rotary = next(row for row in plan.prefill["operations"] if row["operator"] == "rotary_embedding")
    return rotary, config


@pytest.mark.parametrize("positions", [[0], [1, 2, 3, 4, 5], [2048, 2049]])
def test_text_only_interleaved_partial_rotary_matches_qwen35_torch(positions: list[int]) -> None:
    torch = pytest.importorskip("torch")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb

    rotary, config = _declared_rotary()
    assert rotary["attributes"] == {
        "theta": 10_000_000, "rope_type": "default", "rotary_dimensions": 64,
        "partial_rotary_factor": "0.25", "mrope_interleaved": True,
        "mrope_section": (11, 11, 10), "position_policy": "text_replicated_axes",
    }
    rng = np.random.default_rng(0x35500 + positions[0])
    value = rng.normal(size=(1, 2, len(positions), 256)).astype(np.float32)
    indices = np.asarray(positions, dtype=np.int64)
    expected_embedding = Qwen3_5TextRotaryEmbedding(Qwen3_5TextConfig(**config["text_config"]))
    q = torch.from_numpy(value)
    cosine, sine = expected_embedding(q, torch.from_numpy(indices[None, :]))
    expected, _ = apply_rotary_pos_emb(q, q, cosine, sine)
    actual = float32_rotary_text_mrope(value, indices, rotary["attributes"])
    np.testing.assert_allclose(actual, expected.numpy(), rtol=4e-5, atol=3e-5)
    np.testing.assert_array_equal(actual[..., 64:], value[..., 64:])


def test_text_only_rotary_rejects_multiaxis_positions_and_forged_descriptors() -> None:
    rotary, _ = _declared_rotary()
    value = np.ones((1, 2, 3, 256), dtype=np.float32)
    positions = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(SemanticNumericError, match="descriptor"):
        float32_rotary_text_mrope(value, np.stack([positions] * 3), rotary["attributes"])
    for attrs in (
        {**rotary["attributes"], "mrope_section": [11, 11, 11]},
        {**rotary["attributes"], "position_policy": "three_independent_axes"},
        {**rotary["attributes"], "partial_rotary_factor": "1"},
    ):
        with pytest.raises(SemanticNumericError, match="descriptor"):
            float32_rotary_text_mrope(value, positions, attrs)
    with pytest.raises(SemanticNumericError, match="position domain"):
        float32_rotary_text_mrope(value, np.array([0, 1, 1 << 18], dtype=np.int64), rotary["attributes"])

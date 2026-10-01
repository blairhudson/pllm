"""SDK cost probes execute real bounded arithmetic without whole-model claims."""

from __future__ import annotations

import pytest

from pllm import Model, lower_model
from pllm.metrics import EncryptedLinearCostProbe, ResidentMlpCostProbe
from pllm.profiles import MaskedLinearCpu


def test_share_resident_sdk_option_uses_compiler_schedule_and_resource_gate() -> None:
    plan = lower_model({
        "model_type": "qwen2", "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "vocab_size": 256, "max_position_embeddings": 256,
        "tie_word_embeddings": True, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    }, batch=1, max_input_tokens=2, max_new_tokens=2)
    composition = MaskedLinearCpu(Model("org/bounded-reference"))
    option = ResidentMlpCostProbe(
        fixed_scale_bits=8,
        maximum_material_bytes_per_party=256 << 20,
        maximum_online_all_link_body_bytes=95_805_056,
    )
    result = option.run(plan, composition, response_new_tokens=2)
    assert result["plan_digest"] == plan.digest
    assert result["composition_digest"] == plan.runtime_schedule(composition).composition_digest
    assert result["prefill_elements"] == 1024
    assert result["online_within_budget"]
    assert result["executable"] is False
    with pytest.raises(ValueError, match="fixed scale"):
        ResidentMlpCostProbe(11, 1 << 20, 1 << 20)
    with pytest.raises(ValueError, match="budgets"):
        ResidentMlpCostProbe(8, -1, 1 << 20)


@pytest.mark.he
@pytest.mark.parametrize("rows", (1, 8, 32))
def test_encrypted_linear_sdk_probe_measures_both_directions_with_exact_parity(rows: int) -> None:
    pytest.importorskip("tenseal")
    result = EncryptedLinearCostProbe(32, 64, rows).run()
    assert result["schema"] == "pllm.encrypted_linear_cost_probe.v1"
    assert result["exact_modular_parity"] is True
    assert result["whole_decoder_executable"] is False
    assert result["ciphertext_bodies"] > 0
    assert result["public_context_body_bytes"] > 1 << 20
    assert result["online_all_link_body_bytes"] == (
        result["client_to_provider_body_bytes"]
        + result["provider_to_client_body_bytes"]
    )
    # Exact packed-ring input/output width (3 bytes) is a lower bound on
    # plaintext residue traffic, not a matched prepared-protocol measurement.
    assert result["online_all_link_body_bytes"] > rows * (32 + 64) * 3
    assert not any("payload" in key or "input_values" in key for key in result)


def test_encrypted_probe_rejects_unbounded_or_unchecked_modulus() -> None:
    with pytest.raises(ValueError, match="output_width"):
        EncryptedLinearCostProbe(32, 1024)
    with pytest.raises(ValueError, match="modulus"):
        EncryptedLinearCostProbe(32, 64, modulus=257)

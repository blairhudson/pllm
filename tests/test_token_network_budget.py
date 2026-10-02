import pytest

from pllm.assurance import PublicPolynomialShiftRegression
from pllm.metrics import TokenNetworkBudgetProbe


@pytest.mark.rust
@pytest.mark.parametrize("bits", (8, 16, 24, 32, 64))
def test_exposed_shifted_coefficients_recover_exact_up_and_gate_residue(bits):
    modulus = 1 << bits
    g, u, a, b = modulus - 1, 193, 79, modulus - 3
    linear = 256 % modulus
    witness = PublicPolynomialShiftRegression(bits).evaluate(
        masked_gate=(g + a) % modulus,
        masked_up=(u + b) % modulus,
        coefficient_gu=(linear - 2 * a) % modulus,
        coefficient_g2=(-b) % modulus,
        linear_coefficient=linear,
    )
    assert witness.up_value == u
    assert witness.gate_residue == g % (modulus // 2)
    assert witness.gate_residue_bits == bits - 1


@pytest.mark.rust
def test_inconsistent_polynomial_claim_is_not_a_privacy_pass():
    with pytest.raises(ValueError, match="inconsistent"):
        PublicPolynomialShiftRegression().evaluate(
            masked_gate=18, masked_up=12, coefficient_gu=233, coefficient_g2=0
        )
    with pytest.raises(ValueError, match="residue"):
        PublicPolynomialShiftRegression().evaluate(
            masked_gate=True, masked_up=12, coefficient_gu=234, coefficient_g2=0
        )


@pytest.mark.rust
def test_per_token_budget_counts_prefill_and_feedback_instead_of_dropping_them():
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from test_shared_resources import CONFIG

    plan = lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=8)
    pipeline = MaskedLinearCpu(
        Model.hf("Qwen/Qwen2.5-0.5B-Instruct"),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    result = TokenNetworkBudgetProbe().project(
        plan,
        pipeline,
        response_new_tokens=8,
        baseline_online_body_bytes=1000000,
        baseline_total_body_bytes=2000000,
    )
    assert result["executed_rows"] == 46
    assert result["response_online_budget_bytes"] == 10000
    assert result["amortized_online_budget_bytes_per_generated_token"] == 1250
    assert result["material_budget_if_online_budget_spent_bytes"] == 10000
    assert result["full_width_client_feedback_scenario"]["optimistic_body_bytes"] > 0
    assert not result["complete_cost_admitted"]
    with pytest.raises(ValueError, match="baseline"):
        TokenNetworkBudgetProbe().project(
            plan,
            pipeline,
            response_new_tokens=8,
            baseline_online_body_bytes=200,
            baseline_total_body_bytes=100,
        )


@pytest.mark.parametrize(
    "args",
    (
        {"reduction_factor": True},
        {"reduction_factor": 1},
        {"ring_bits": 12},
        {"reduction_factor": 1001},
    ),
)
def test_budget_rejects_unbounded_or_silent_precision_options(args):
    with pytest.raises(ValueError):
        TokenNetworkBudgetProbe(**args)

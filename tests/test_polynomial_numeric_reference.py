"""The float oracle checks hypotheses; the Rust tests check protocol arithmetic."""

import pytest

torch = pytest.importorskip("torch")

from pllm.runtime.polynomial_numeric_reference import (
    evaluate_channel_polynomial,
    fit_channel_polynomial,
)


@pytest.mark.parametrize("degree", (1, 2, 4))
def test_public_calibration_recovers_independent_polynomial(degree):
    g = torch.linspace(-2, 2, 32, dtype=torch.float64).reshape(-1, 1).repeat(1, 3)
    g[:, 1] *= 3
    u = torch.cos(g) + 2
    target = sum((power + 1) * g.pow(power) for power in range(degree + 1))
    profile = fit_channel_polynomial(g, u, degree=degree, target=target * u)
    held = torch.tensor([[-0.3, 0.7, 0.1], [0.2, -1.1, 1.5]], dtype=torch.float64)
    expected = sum((power + 1) * held.pow(power) for power in range(degree + 1))
    torch.testing.assert_close(
        evaluate_channel_polynomial(held, profile), expected, rtol=1e-9, atol=1e-9
    )


def test_public_calibration_rejects_nonfinite_and_unbounded_inputs():
    g = torch.ones(10, 3)
    g[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_channel_polynomial(g, torch.ones_like(g), degree=2)
    with pytest.raises(ValueError, match="bounded"):
        fit_channel_polynomial(torch.ones(257, 3), torch.ones(257, 3), degree=2)
    with pytest.raises(ValueError, match="degree"):
        fit_channel_polynomial(torch.ones(8, 3), torch.ones(8, 3), degree=True)

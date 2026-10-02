import pytest

from pllm.deployment import LinkConditions, NetworkError
from pllm.deployment.network import strict_load


@pytest.mark.parametrize("values", [{"latency_ms": float("nan")}, {"latency_ms": True},
    {"latency_ms": -1}, {"latency_ms": 5001}, {"bytes_per_second": 0},
    {"loss_fraction": .3}, {"seed": False}, {"seed": 0}])
def test_invalid_link_controls_fail_before_resources(values):
    with pytest.raises(NetworkError):
        LinkConditions(**values)


def test_link_controls_roundtrip_and_canonical_digest():
    shape = LinkConditions(latency_ms=10, bytes_per_second=12500000, loss_fraction=.001)
    assert LinkConditions.from_spec(shape.to_spec()) == shape
    assert LinkConditions.from_spec(strict_load(shape.canonical_bytes())).digest == shape.digest

"""A public mask subspace must not be mistaken for a private ring mask."""

from itertools import product

import pytest

from pllm.assurance import PublicSubspaceMaskRegression
from pllm.components import ComponentRef, planned_component
from pllm.configuration import ConfigurationError


@pytest.mark.parametrize("ring_bits", [16, 24, 32])
def test_concrete_projection_recovers_input_parity_for_every_mask_choice(ring_bits: int) -> None:
    controls = PublicSubspaceMaskRegression(ring_bits=ring_bits)
    public = ((1, 1, 0), (0, 1, 1))
    private = (19, 27, 38)
    modulus = 1 << ring_bits
    for first in (0, 1):
        for second in (0, 1):
            masked = tuple(
                (private[index] + first * public[0][index] + second * public[1][index]) % modulus
                for index in range(3)
            )
            witness = controls.evaluate(public, masked)
            assert witness is not None
            assert witness.indices == (0, 1, 2)
            assert witness.leaked_parity == (sum(private) & 1)
            assert witness.public_rank_mod2 == 2
            assert witness.mask_count == 2


def test_full_mod2_rank_yields_no_witness_but_not_a_privacy_claim() -> None:
    control = PublicSubspaceMaskRegression()
    assert control.evaluate(((1, 0), (0, 1)), (11, 13)) is None
    assert control.evaluate((), (13,)) is not None
    assert control.evaluate(((2, 0), (0, 2)), (11, 13)).indices == (0,)


def test_bounded_binary_bases_match_an_independent_nullspace_oracle() -> None:
    control = PublicSubspaceMaskRegression()
    for width in range(1, 5):
        for count in range(min(width, 3) + 1):
            for basis in product(range(1 << width), repeat=count):
                vectors = tuple(tuple((row >> bit) & 1 for bit in range(width)) for row in basis)
                witness = control.evaluate(vectors, (1,) * width)
                candidates = [
                    probe for probe in range(1, 1 << width)
                    if all((probe & row).bit_count() % 2 == 0 for row in basis)
                ]
                assert (witness is None) == (not candidates)
                if witness is not None:
                    projection = sum(1 << index for index in witness.indices)
                    assert projection in candidates


@pytest.mark.parametrize(
    ("ring_bits", "vectors", "masked"),
    [
        (8, (), (1,)),
        (True, (), (1,)),
        (16, [(0,)], ()),
        (16, [[1, 0]], [1]),
        (16, [[-1]], [1]),
        (16, [[1]], [1 << 16]),
        (16, [[True]], [1]),
        (16, [[1]], [False]),
        (16, [[0] * 4097], [0] * 4097),
        (16, [[0]] * 4097, [0]),
    ],
)
def test_invalid_or_unbounded_claims_fail_closed(ring_bits: int, vectors: object, masked: object) -> None:
    with pytest.raises(ValueError):
        PublicSubspaceMaskRegression(ring_bits).evaluate(vectors, masked)


def test_assurance_control_is_implemented_but_not_a_pipeline_component() -> None:
    plan = planned_component("carnival")
    assert plan.status == "implemented"
    assert PublicSubspaceMaskRegression.paper_url.endswith("/research/papers/carnival/")
    assert PublicSubspaceMaskRegression.attack_url.endswith("/research/papers/maverick/")
    with pytest.raises(ConfigurationError, match="retired planned identity"):
        ComponentRef(plan.identity)

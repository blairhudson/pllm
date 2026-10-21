"""Negative algebraic controls for the pinned CPU TR/LRS mask width.

Specification evidence only: no upstream code or PLLM executor is called.
EzPC aa0f2f860266fb79901405ef5e9965b8c9fe2e88:
  api.cpp:4042,4176: rout = random_ge(1)
  include/llama/group_element.h:61-67: random_ge(bw) reduces modulo 2**bw
  api.cpp:4075-4076,4209-4210: reconstruct multi-bit arithmetic outputs
This tests the exposed-output contract y = floor(x/2**shift) + rout, not an
end-to-end exploit or correctness of the source's complete truncation protocol.
"""

from collections import Counter

import pytest


def observation(x: int, shift: int, output_bits: int, output_mask: int) -> int:
    return ((x // (1 << shift)) + output_mask) % (1 << output_bits)


@pytest.mark.parametrize("input_bits,shift", [(5, 1), (8, 1), (12, 6), (50, 12), (64, 16)])
@pytest.mark.parametrize("reduced", [False, True])
def test_one_bit_output_mask_distinguishes_private_truncation_results(
    input_bits: int, shift: int, reduced: bool
) -> None:
    output_bits = input_bits - shift if reduced else input_bits
    left, right = 0, 2 << shift
    # Both operands are positive signed inputs; these cases also fit the
    # public quarter-ring promise whenever GapARS is the intended caller.
    assert right < (1 << (input_bits - 1))
    if not reduced:
        assert right < (1 << (input_bits - 2))
    views = [
        {observation(x, shift, output_bits, mask) for mask in (0, 1)}
        for x in (left, right)
    ]
    assert views == [{0, 1}, {2, 3}]
    assert views[0].isdisjoint(views[1])


@pytest.mark.parametrize("output_bits", range(2, 9))
def test_full_output_ring_masks_remove_this_specific_observation_witness(
    output_bits: int,
) -> None:
    modulus = 1 << output_bits
    # Exhaustive distributions for this boundary only, not a security proof
    # for DPFs, correlated keys, composed schedules or another participant view.
    expected = Counter({value: 1 for value in range(modulus)})
    for clear_result in (0, 1, 2, modulus // 2, modulus - 1):
        assert Counter(
            observation(clear_result, 0, output_bits, mask) for mask in range(modulus)
        ) == expected

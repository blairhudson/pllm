"""Independent negative control for the inspected SIGMA CPU terminal algebra.

This does not import or execute upstream code. Source locations and the limited
scope of the finding are in paper/reproduction-contracts/sigma-dpf-terminal.md.
No PLLM source-profile backend is implemented or qualified by this test.
"""

import pytest


@pytest.mark.parametrize("suffix", range(128))
@pytest.mark.parametrize("first_control", [0, 1])
def test_cpu_terminal_algebra_reveals_secret_suffix_predicate(
    suffix: int, first_control: int
) -> None:
    seed_mask = ((1 << 128) - 1) ^ 1
    # Include zero, extremal and nontrivial independently selected seed pairs.
    for raw_left, raw_right in [
        (0, 0),
        (seed_mask, 0),
        (2, seed_mask),
        (0xA35F167CA45E92D78B1345E6289ADC10, 0x7E29183A654BDCF0),
    ]:
        left = raw_left & seed_mask
        right = raw_right & seed_mask
        second_control = first_control ^ 1
        # Source maps suffix 0..63 to the high word, 64..127 to the low.
        one_hot = 1 << (127 - suffix)
        terminal = left ^ right ^ first_control ^ second_control ^ one_hot

        # The observation uses one party's common terminal correction only:
        # neither root, the other key, controls nor any evaluation is needed.
        observed_predicate = 1 ^ (terminal & 1)
        assert observed_predicate == int(suffix == 127)


@pytest.mark.parametrize("bits", [8, 12, 37, 48, 50, 51, 63])
def test_terminal_predicate_distinguishes_points_at_every_tree_width(bits: int) -> None:
    prefix = ((1 << (bits - 7)) - 1) << 7
    observations = []
    for suffix in [0, 126, 127]:
        point = prefix | suffix
        # Upper tree decisions do not change the target-path control XOR of 1.
        terminal = (1 << (127 - (point % 128))) ^ 1
        observations.append(1 ^ (terminal & 1))
    assert observations == [0, 0, 1]

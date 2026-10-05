"""Exact old operator bits and bounded ownership, including cache misses."""
import numpy as np
import pytest

from pllm.runtime.rotary_coefficients import (
    MAX_ROTARY_COEFFICIENT_BYTES,
    PhaseRotaryCoefficients,
    rotary_coefficient_bound,
)
from pllm.runtime.semantic_executor import SemanticDecoderRuntime


@pytest.mark.parametrize("width", [2, 6, 8, 32, 128, 256])
def test_reused_rotary_matches_original_float32_expression(width):
    rng = np.random.default_rng(width)
    positions = np.asarray([0, 1, 7, 4095, 65535], np.int64)
    cache = PhaseRotaryCoefficients(positions)
    for indices, theta in ((positions, 10000.0), (positions, 1000000.0),
                           (positions + 1, 10000.0), (positions, 10000.0)):
        value = rng.standard_normal((1, 3, len(indices), width + 4), dtype=np.float32)
        frequency = 1.0 / (theta ** (np.arange(0, width, 2, dtype=np.float32) / width))
        angles = indices.astype(np.float32)[:, None] * frequency[None, :]
        cosine = np.concatenate([np.cos(angles), np.cos(angles)], -1)[None, None]
        sine = np.concatenate([np.sin(angles), np.sin(angles)], -1)[None, None]
        current = value[..., :width]
        rotated = np.concatenate([-current[..., width // 2:], current[..., :width // 2]], -1)
        expected = value.copy()
        expected[..., :width] = current * cosine + rotated * sine
        actual = SemanticDecoderRuntime._rotary(value, indices,
            {"rotary_dimensions": width, "theta": theta}, _coefficients=cache)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert cache.retained_bytes == 2 * 8 * len(positions) * width
    assert all(not table.flags.writeable for table in cache.get(positions, width, 10000.0))
    cache.clear()
    assert cache.retained_bytes == 0 and cache.positions is None


def test_rotary_tables_obey_aggregate_byte_and_entry_bounds():
    positions = np.arange(64, dtype=np.int64)
    cache = PhaseRotaryCoefficients(positions)
    for theta in range(1, 100):
        cache.get(positions, 128, float(theta))
    assert cache.retained_bytes == MAX_ROTARY_COEFFICIENT_BYTES
    positions = np.arange(2048, dtype=np.int64)
    cache = PhaseRotaryCoefficients(positions)
    cache.get(positions, 128, 10000.0)
    assert cache.retained_bytes == 0
    # A shorter actual input can fit even when the declared maximum cannot.
    assert rotary_coefficient_bound({"query_sequence": 2048, "operations": [
        {"operator": "rotary_embedding", "attributes": {
            "theta": 10000, "rotary_dimensions": 128}},
    ]}) == MAX_ROTARY_COEFFICIENT_BYTES


def test_nondefault_rotary_never_prices_the_default_table():
    assert rotary_coefficient_bound({"query_sequence": 128, "operations": [
        {"operator": "rotary_embedding", "attributes": {"output_dtype": "bfloat16"}},
        {"operator": "rotary_embedding", "attributes": {"frequency_scaling": {}}},
        {"operator": "rotary_embedding", "attributes": {"mrope_section": [1, 1, 1]}},
    ]}) == 0

from __future__ import annotations

import os

import numpy as np
import pytest

from pllm.runtime.packed_bfv import PackedBFVLinearFactory


@pytest.mark.he
def test_packed_bfv_matches_clear() -> None:
    factory = PackedBFVLinearFactory(
        tenseal_path=os.environ.get("PLLM_PYDEPS"), threads=1
    )
    rng = np.random.default_rng(1401)
    masks = rng.integers(0, 65537, size=(4, 16), dtype=np.int64)
    weight = rng.integers(-7, 8, size=(12, 16), dtype=np.int64)
    result = factory.evaluate(masks, weight)
    expected = masks @ (weight % 65537).T % 65537
    np.testing.assert_array_equal(result.clear, expected)
    assert result.streams == 4
    assert result.noise_budget_bits > 0


def test_capacity_is_shape_bounded() -> None:
    factory = object.__new__(PackedBFVLinearFactory)
    factory.row_size = 2048
    assert factory.capacity(258, 36) == 12
    assert factory.capacity(1536, 12288) == 0

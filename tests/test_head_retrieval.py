import dataclasses
import json

import numpy as np
import pytest

from pllm.metrics import PrivateHeadRetrievalProbe
from pllm import _native


@pytest.mark.parametrize("rank", [1, 8, 16, 32])
def test_public_fit_residual_bounds_cover_every_omitted_original_row(rank):
    probe = PrivateHeadRetrievalProbe(rank=rank, candidate_counts=(1, 8, 64))
    result = probe.run()
    assert result["all_checked_bounds_valid"] and result["private_page_exact"]
    assert not result["whole_decoder_executable"]
    complete = result["candidates"][-1]
    assert complete["winner_agreement"] == complete["certified_winners"] == 4
    for case in result["candidates"]:
        assert case["certified_winners"] <= case["winner_agreement"]
        assert (
            case["fixed_budget_head_body_bytes_per_token"]
            == case["candidate_count"] * result["private_page_body_bytes"]
        )
    json.dumps(result, allow_nan=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        probe.rank = 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": True},
        {"rank": 257},
        {"candidate_counts": (8, 1)},
        {"candidate_counts": (0,)},
        {"page_records": 65},
    ],
)
def test_reject_invalid_public_policy(kwargs):
    with pytest.raises(ValueError):
        PrivateHeadRetrievalProbe(**kwargs)


def test_reject_shapes_and_nonfinite_scales_before_public_fit():
    weights = np.zeros((64, 32), np.int8)
    with pytest.raises(ValueError, match="scales"):
        PrivateHeadRetrievalProbe().run(
            weights=weights,
            inputs=weights[:1],
            scales=np.full(64, np.nan, np.float32),
            input_scales=np.ones(1, np.float32),
        )
    with pytest.raises(ValueError, match="shape"):
        PrivateHeadRetrievalProbe(rank=64).run(
            weights=weights,
            inputs=weights[:1],
            scales=np.ones(64, np.float32),
            input_scales=np.ones(1, np.float32),
        )


def test_native_ranking_matches_independent_compressed_integer_oracle():
    rng = np.random.default_rng(54)
    weights = rng.integers(-50, 51, (31, 4), dtype=np.int8)
    projection = rng.integers(-1, 2, (4, 8)).astype("<f8")
    profiles = np.tile(np.asarray([0.125, 1000, 1000], dtype="<f8"), (31, 1))
    index = _native.HeadRetrievalIndex(
        31, 8, 4, weights.tobytes(), projection.tobytes(), profiles.tobytes()
    )
    for _ in range(8):
        query = rng.integers(-31, 32, 8, dtype=np.int8)
        coordinates = projection @ (query.astype(np.float64) * 0.25)
        scale = np.max(np.abs(coordinates)) / 127 or 1
        quantized = np.rint(coordinates / scale).clip(-127, 127).astype(np.int64)
        scores = (weights.astype(np.int64) @ quantized) * scale * 0.125
        expected = np.lexsort((np.arange(31), -scores))
        for count in (1, 8, 31):
            result, _ = index.candidates(query.tobytes(), 0.25, count)
            np.testing.assert_array_equal(np.frombuffer(result, "<u4"), expected[:count])


def test_public_calibration_excludes_evaluation_inputs_and_keeps_bounds_valid():
    rng = np.random.default_rng(195)
    weights = rng.integers(-127, 128, (64, 32), dtype=np.int8)
    calibration = rng.normal(size=(64, 32)).astype(np.float32)
    kwargs = dict(
        weights=weights,
        scales=np.full(64, 0.01, np.float32),
        input_scales=np.ones(2, np.float32),
        public_calibration=calibration,
    )
    probe = PrivateHeadRetrievalProbe(rank=8, candidate_counts=(1, 8))
    first = probe.run(inputs=np.ones((2, 32), np.int8), **kwargs)
    second = probe.run(inputs=rng.integers(-127, 128, (2, 32), dtype=np.int8), **kwargs)
    assert first["index_digest"] == second["index_digest"]
    assert first["all_checked_bounds_valid"] and second["all_checked_bounds_valid"]
    assert first["public_calibration_digest"] == second["public_calibration_digest"]

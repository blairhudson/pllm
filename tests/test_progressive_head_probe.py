import numpy as np
import pytest

from pllm import _native
from pllm.metrics import ProgressiveHeadProbe


def test_signed_intervals_match_independent_full_head_and_price_pir():
    rng = np.random.default_rng(813)
    weights = rng.integers(-128, 128, (37, 49), dtype=np.int8)
    queries = rng.integers(-128, 128, (7, 49), dtype=np.int8)
    scales = rng.uniform(0.001, 0.1, 37).astype(np.float32)
    report = ProgressiveHeadProbe().run(weights, scales, queries, np.ones(7, np.float32))
    assert report["exact_greedy_agreements"] == 7
    assert report["private_lookup_one_record_control"]["exact_byte_parity"]
    assert not report["public_capacity_certificate"]
    assert not report["executable"]
    assert report["oracle_retained_native_weight_bytes"] == weights.nbytes


def test_residual_records_reconstruct_signed_weights_and_have_zero_padding():
    weights = np.arange(-128, 127, dtype=np.int8).reshape(15, 17)
    head = _native._ProgressiveHead(weights.tobytes(), [1.0] * 15, 17)
    for bits in range(1, 8):
        shift = 8 - bits
        size = (17 * shift + 7) // 8
        table = head.residual_table(bits)
        for row in range(15):
            record = int.from_bytes(table[row * size:(row + 1) * size], "little")
            assert record >> (17 * shift) == 0
            residual = np.array([(record >> (i * shift)) & ((1 << shift) - 1) for i in range(17)])
            assert np.array_equal((weights[row].astype(np.int16) >> shift) * (1 << shift) + residual, weights[row])
    with pytest.raises(ValueError, match="finite"):
        head.query(bytes(17), float("nan"), 5)


def test_tie_break_and_zero_query_never_require_private_fallback():
    head = _native._ProgressiveHead(bytes([7, 8]), [8.0, 7.0], 1)
    assert head.query(bytes([1]), 1.0, 4)[0] == 0
    winner, rounds = head.query(bytes([0]), 1.0, 4)
    assert winner == 0 and rounds == [(4, 2, 1)]
    with pytest.raises(ValueError):
        ProgressiveHeadProbe(prefix_bits=True)

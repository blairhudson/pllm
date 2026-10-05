"""SHAKE wire compatibility, one-use lease burns and bounded mask residency."""
import dataclasses
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pllm.runtime.preparation_protocol import (
    PreparationRequest, expand_output_mask, expand_preparation_mask, seeded_ring_profile,
)
from pllm.runtime.transformer_client import (
    PreparedInventory, PreparedStageRows, TransformerClientError,
)

pytestmark = pytest.mark.rust


def request(bits=24, *, rows=71, width=47, output=29):
    profile = seeded_ring_profile({16: 123, 24: 40000, 32: 1 << 24}[bits])
    return PreparationRequest("a1" * 16, "inventory", "model", "body", "stage", "weight",
        rows, width, output, 8, 8, profile.signed_output_bound, profile.ring,
        profile.modulus, profile.wire_bits, bytes(range(32)))


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_disjoint_out_of_order_leases_match_hashlib_and_burn(bits):
    req = request(bits)
    r, s = expand_preparation_mask(req), expand_output_mask(req)
    stage = PreparedStageRows(req)
    assert stage.input_mask is stage.output_mask is None
    assert stage.retained_mask_bytes < 2048
    inventory = PreparedInventory("inventory", req.rows, {req.stage_id: stage})
    first, second = inventory.reserve(10), inventory.reserve(61)
    for lease, counts in ((second, [54, 1, 6]), (first, [3, 7])):
        begin = lease.start
        for count in counts:
            got_r, got_s, attempts = lease.take(req.stage_id, count)
            np.testing.assert_array_equal(got_r, r[begin:begin + count])
            np.testing.assert_array_equal(got_s, s[begin:begin + count])
            assert len(set(attempts)) == count
            begin += count
        lease.close()
        with pytest.raises(TransformerClientError, match="closed"):
            lease.take(req.stage_id, 1)
    assert inventory.status()["stage_rows_claimed"] == req.rows
    with pytest.raises(ValueError, match="consumed"):
        stage.take_masks(0, 1)


def test_unused_reservation_is_burned_without_expansion():
    req = request(rows=10)
    stage = PreparedStageRows(req)
    inventory = PreparedInventory("inventory", 10, {req.stage_id: stage})
    first, second = inventory.reserve(4), inventory.reserve(6)
    saved, _, _ = first.take(req.stage_id, 1)
    expected = saved.copy()
    first.close()
    # Cancelling the first lease cannot prevent the disjoint second lease.
    got, _, _ = second.take(req.stage_id, 6)
    np.testing.assert_array_equal(got, expand_preparation_mask(req)[4:])
    second.close()
    assert inventory.status()["stage_rows_burned"] == 3
    with pytest.raises(ValueError, match="consumed"):
        stage.take_masks(2, 1)
    inventory.cancel()
    np.testing.assert_array_equal(saved, expected)


def test_parallel_disjoint_claims_and_failure_burn():
    req = request(rows=24)
    stage = PreparedStageRows(req)
    inventory = PreparedInventory("inventory", 24, {req.stage_id: stage})
    leases = [inventory.reserve(3) for _ in range(8)]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda lease: lease.take(req.stage_id, 3)[0], reversed(leases)))
    expected = expand_preparation_mask(req)
    for lease, result in zip(reversed(leases), results, strict=True):
        np.testing.assert_array_equal(result, expected[lease.start:lease.start + 3])
        lease.close()
    failing = PreparedInventory("inventory", 24, {req.stage_id: PreparedStageRows(req)}).reserve(3)
    with pytest.raises(TransformerClientError, match="invalid"):
        failing.take("unknown", 1)
    with pytest.raises(TransformerClientError, match="closed"):
        failing.take(req.stage_id, 1)


def test_residency_depends_on_row_ledger_not_matrix_width():
    small = PreparedStageRows(request(width=1, output=1))
    wide = PreparedStageRows(request(width=2560, output=19456))
    assert small.retained_mask_bytes == wide.retained_mask_bytes < 2048
    wide.cancel()
    with pytest.raises(ValueError, match="closed"):
        wide.take_masks(0, 1)
    with pytest.raises(Exception, match="allocation"):
        PreparedStageRows(request(width=1 << 26))


def test_domain_separation_and_explicit_python_oracle(monkeypatch):
    from pllm.runtime import _native_support
    req = request()
    r, s = PreparedStageRows(req).take_masks(0, 1)
    np.testing.assert_array_equal(r, expand_preparation_mask(req)[:1])
    np.testing.assert_array_equal(s, expand_output_mask(req)[:1])
    for other in (dataclasses.replace(req, stage_id="other"),
                  dataclasses.replace(req, rows=72), dataclasses.replace(req, seed=b"b" * 32)):
        assert not np.array_equal(r, PreparedStageRows(other).take_masks(0, 1)[0])
    monkeypatch.setattr(_native_support, "extension", lambda: None)
    oracle = PreparedStageRows(req)
    np.testing.assert_array_equal(oracle.input_mask, expand_preparation_mask(req))
    assert oracle.retained_mask_bytes == (req.in_features + req.out_features) * req.rows * 4

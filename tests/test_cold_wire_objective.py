"""Bounded public synthetic wire-objective regression and independent oracle tests."""
import json

import numpy as np
import pytest

from scripts.probe_cold_wire_objective import (
    ROOT, bin_size, check_installed, framed_sizes, geometry, l1_cap,
    pack_integer, row_width, run, train_budget, unpack_integer,
)


@pytest.mark.parametrize("bound", [7, 127])
@pytest.mark.parametrize("width", [1, 8, 16, 20, 24])
def test_strict_signed_threshold(width, bound):
    cap = l1_cap(width, bound)
    assert row_width(cap, bound) <= width
    assert row_width(cap + 1, bound) > width
    # Actual installed native oracle at both sides; independent distributed row.
    from pllm import _native
    for l1 in (cap, cap + 1):
        full, remainder = divmod(l1, 127)
        q = bytes([127]) * full + bytes([remainder])
        assert _native.offset_row_bits(q, len(q), bound) == bytes([row_width(l1, bound)])


def test_zero_and_power_of_two_boundary():
    from pllm import _native
    assert _native.offset_row_bits(bytes([128]), 1, 127) == bytes([row_width(128)])
    assert row_width(0) == 1
    assert row_width(32767, 1) == 16
    assert row_width(32768, 1) == 17
    assert l1_cap(16, 127) == 258
    with pytest.raises(ValueError):
        row_width(1 << 31, 1)
    with pytest.raises(ValueError):
        row_width(1, 128)


def test_codec_padding_and_centered_signed_extrema():
    widths = bytes([1, 7, 16, 23, 32])
    values = np.array([[0, -63, 32767, -(1 << 22), (1 << 31) - 1],
                       [-1, 63, -32768, (1 << 22) - 1, -(1 << 31)]], dtype=np.int64)
    packed = pack_integer(values, widths)
    assert len(packed) == (2 * sum(widths) + 7) // 8
    assert np.array_equal(unpack_integer(packed, widths, 2), values)
    bad = packed[:-1] + bytes([packed[-1] | 128])
    with pytest.raises(ValueError):
        unpack_integer(bad, widths, 2)


@pytest.mark.parametrize("n", [0, 255, 256, 65535, 65536])
def test_msgpack_bin_header_cliffs(n):
    import msgpack
    assert bin_size(n) == len(msgpack.packb(bytes(n), use_bin_type=True))


def test_global_max_blocker_and_subbyte_rounding():
    a = framed_sizes(32, bytes([19, 18]), 16, 131072)
    b = framed_sizes(32, bytes([19, 17]), 16, 131072)
    assert a["input_tensor_bytes"] == b["input_tensor_bytes"]
    assert a["output_tensor_bytes"] - b["output_tensor_bytes"] == 2
    assert len(pack_integer([[0, 0]], bytes([1, 1]))) == 1
    assert len(pack_integer([[0, 0]], bytes([2, 1]))) == 1


def test_training_enforces_budget_and_improves_train_loss():
    x = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int64)
    q = np.array([[8, 2]], dtype=np.int64)
    target = x @ q.T
    with pytest.raises(ValueError):
        train_budget(q, x, target, -1)
    fitted, edits = train_budget(q, x, target, 7)
    assert np.abs(fitted).sum() <= 7
    assert edits > 0
    assert np.sum((x @ fitted.T - target) ** 2) <= np.sum((x @ np.array([[5, 2]]).T - target) ** 2)


def test_installed_serializers_and_mask_reconstruction():
    q = np.array([[127, -12, 7], [-127, 8, -1]], dtype=np.int64)
    assert len(check_installed(q, np.array([[127, -127, 127]]))) == 3


def test_geometry_not_real_measurement_and_all_directions():
    g = geometry()
    assert g["body_parameters"] == 357826560
    for record in g["records"]:
        assert record["full_request_bytes"] is None
        assert record["correction_tensor_bytes"] == record["response_tensor_bytes"]
        assert record["covered_hypothetical_total"] == (
            sum(record["delivery"].values()) + record["preparation_two_directed_bodies"]
            + record["online_two_directed_bodies"])


def test_deterministic_evidence_and_honest_tradeoff():
    d = run()
    saved = json.loads((ROOT / "docs/evidence/cold-wire-objective-synthetic.json").read_text())
    assert d == saved
    c = {v["name"]: v for v in d["candidates"]}
    base, mild, severe = (c[k] for k in ("unchanged_W8A8", "integer_trained_width_minus_1", "integer_trained_width_minus_2"))
    assert mild["stage_16_rows"]["covered_stage_bodies"] < base["stage_16_rows"]["covered_stage_bodies"]
    assert severe["heldout"]["relative_mse"] > mild["heldout"]["relative_mse"] > 0
    assert severe["heldout"]["argmax_agreement"] < 1
    assert d["real_quality"] is d["rss"] is d["latency"] is d["full_transport"] is None
    assert d["extra_array_design_upper_bound_bytes"] < 128 * 1024 * 1024
    assert d["scalar_max_normalized_requantization_same_integers"]
    assert d["exact_divisible_weight_reparameterization"]["integer_product_times_public_scale_exact"]

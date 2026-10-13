"""Standalone numeric codec checks; no native modules or model downloads."""
import importlib.util
from pathlib import Path
import struct
import sys

import pytest

spec = importlib.util.spec_from_file_location("cyclic_mask_codec_probe",
    Path(__file__).resolve().parents[1] / "scripts/probe_cyclic_mask_codec.py")
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
G = probe.Geometry


def test_exhaustive_geometry_and_independent_oracle():
    counts = probe.exhaustive()
    assert counts["bounded_recoveries"] > 300000
    assert counts["independent_oracle_color_checks"] > 300000
    assert counts["parent_grid_recoveries"] == 23096


def test_large_prime_boundaries_and_real_serializer():
    counts = probe.sampled_large()
    assert counts["bounded_recoveries"] > 1000
    assert counts["serialized_frames"] > 100


@pytest.mark.parametrize("q,lo,hi", [(2, 0, 1), (16, -7, 7), (17, -8, 8),
                                    (17, 100, 116), (17, -100, -84), (19, 7, 7)])
def test_even_odd_maximal_asymmetric_shifted_singleton(q, lo, hi):
    g = G(q, lo, hi)
    for h in range(q):
        xs = list(range(lo, hi + 1))
        layout = [g] * len(xs)
        frame = probe.serialize([(h + x) % q for x in xs], layout)
        assert probe.deserialize(frame, [h] * len(xs), layout) == xs
        assert len(frame) == probe.layout_stats(layout)["frame_bytes"]


def test_same_color_cyclic_spacing():
    for q in range(2, 65):
        for m in range(1, q + 1):
            g = G(q, 0, m - 1)
            for c in range(g.alphabet):
                ys = [y for y in range(q) if g.encode(y) == c]
                if len(ys) > 1:
                    assert min((ys[(i + 1) % len(ys)] - y) % q for i, y in enumerate(ys)) >= m


def test_outside_domain_is_not_certified():
    g = G.symmetric(17, 1)
    assert g.encode(4) == g.encode(0) == 0
    assert g.decode(g.encode(4), 0) == 0  # Wrong but accepted, original x was 4.
    with pytest.raises(ValueError, match="no unique"):
        g.decode(g.encode(3), 0)  # Some outside-domain values do reject.
    # Rejecting aliased ranges is essential: x=0 and x=17 give identical y.
    with pytest.raises(ValueError, match="aliases"):
        G(17, 0, 17)


def test_chosen_color_rejection_depends_on_secret_mask():
    g = G.symmetric(17, 1)
    with pytest.raises(ValueError, match="no unique"):
        g.decode(3, 0)
    assert g.decode(3, 3) == 0
    # The same canonical public frame can therefore trigger secret-dependent errors.
    frame = probe.serialize([3], [g])
    assert probe.parse_colors(frame, [g]) == [3]
    with pytest.raises(ValueError):
        probe.deserialize(frame, [0], [g])
    assert probe.deserialize(frame, [3], [g]) == [0]


def test_malformed_frame_padding_invalid_color_and_layout():
    g = G.symmetric(17, 1)  # K=4, two bits, one byte with six padding bits.
    layout = [g]
    frame = probe.serialize([0], layout)
    for bad in (frame[:-1], frame + b"\0", b"BAD!" + frame[4:],
                frame[:-1] + bytes([frame[-1] | 128]),
                probe.HEADER.pack(probe.MAGIC, 2, 1) + frame[12:]):
        with pytest.raises(ValueError):
            probe.parse_colors(bad, layout)
    g = G.symmetric(31, 2)  # K=6; colors 6/7 invalid in three-bit representation.
    bad = probe.HEADER.pack(probe.MAGIC, 1, 1) + b"\x07"
    with pytest.raises(ValueError, match="invalid packed color"):
        probe.parse_colors(bad, [g])
    for bad in (b"", b"x", struct.pack("<Qqq", 17, 0, 17)):
        with pytest.raises(ValueError):
            probe.unpack_layout(bad)


def test_heterogeneous_public_widths_and_zero_width():
    layout = [G.symmetric(17, 0), G.symmetric(31, 2), G(16, -7, 7), G(19, -3, 5)]
    masks = [16, 30, 0, 18]
    xs = [0, -2, 7, 5]
    frame = probe.serialize([(h + x) % g.q for h, x, g in zip(masks, xs, layout)], layout)
    assert probe.deserialize(frame, masks, layout) == xs
    assert probe.unpack_layout(probe.pack_layout(layout)) == tuple(layout)
    assert len(probe.pack_layout(layout)) == probe.layout_stats(layout)["public_layout_bytes"]
    zero = [G.symmetric(17, 0)] * 4
    frame = probe.serialize([0, 1, 15, 16], zero)
    assert len(frame) == 12
    assert probe.deserialize(frame, [0, 1, 15, 16], zero) == [0] * 4


def test_uint64_limits_resource_admission_and_noncanonical_inputs():
    g = G(probe.MAX_Q, -(1 << 63), (1 << 63) - 2)
    assert g.n == 1 and g.bits == 64
    frame = probe.serialize([probe.MAX_Q - 1], [g])
    x = g.decode(g.encode(probe.MAX_Q - 1), probe.MAX_Q - 1)
    assert probe.deserialize(frame, [probe.MAX_Q - 1], [g]) == [x]
    for args in ((1, 0, 0), (1 << 64, 0, 0), (17, 2, 1), (17, True, 1),
                 (17, -(1 << 63) - 1, 0), (17, 0, 1 << 63)):
        with pytest.raises(ValueError):
            G(*args)
    for layout in ([], [G.symmetric(17, 0)] * (probe.MAX_VALUES + 1), [None]):
        with pytest.raises(ValueError):
            probe.layout_stats(layout)
    g = G.symmetric(17, 1)
    for y in (-1, 17, True):
        with pytest.raises(ValueError):
            g.encode(y)
    for c, h in ((-1, 0), (g.alphabet, 0), (0, 17), (0, True)):
        with pytest.raises(ValueError):
            g.decode(c, h)
    with pytest.raises(ValueError):
        probe.deserialize(probe.serialize([0], [g]), [], [g])


def test_public_row_prices_do_not_claim_extra_gain():
    for width in (896, 4864, 16384):
        p = probe.price(4294967291, width * 128 * 128, 896)
        assert p["cyclic_bits_per_value"] == p["row_residue_bits_per_value"]
        assert p["payload_bytes"] == p["existing_row_residue_payload_bytes"]
        assert p["frame_bytes"] == p["payload_bytes"] + 12


def test_small_field_can_require_more_bits_than_ring_residue():
    p = probe.price(17, 3, 8)
    assert p["alphabet"] == 9
    assert p["cyclic_bits_per_value"] == 4
    assert p["row_residue_bits_per_value"] == 3
    assert p["payload_bytes"] == 4
    assert p["existing_row_residue_payload_bytes"] == 3

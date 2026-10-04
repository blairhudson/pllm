"""Bounds, commitment and small-ring view invariants for public row layouts."""
import base64
from collections import Counter
import itertools
import struct
import random
import zlib

import pytest
from pllm import _native
from pllm.runtime.residue_codec import encode_layout, decode_layout


def test_inline_public_layout_is_bounded_and_committed():
    widths = bytes(random.Random(73).choices([16, 17, 22, 24], k=8192))
    record = encode_layout(widths)
    assert decode_layout(record, len(widths)) == widths
    assert len(record["zlib_base64"]) > 512
    assert decode_layout({**record, "zlib_base64": record["zlib_base64"].encode()}, len(widths)) == widths
    for replacement in ({"count": len(widths) - 1}, {"sha256": "0" * 64}, {"zlib_base64": record["zlib_base64"] + "?"}):
        with pytest.raises(ValueError, match="layout"):
            decode_layout({**record, **replacement}, len(widths))
    for packed in (zlib.compress(widths + b"x"), zlib.compress(widths) + b"x", zlib.compress(widths)[:-1]):
        with pytest.raises(ValueError, match="layout"):
            decode_layout({**record, "zlib_base64": base64.b64encode(packed).decode()}, len(widths))
    for values in (b"", b"\0", b"\x21"):
        with pytest.raises(ValueError, match="layout"):
            encode_layout(values)


def test_reduced_corrections_do_not_distinguish_small_ring_inputs():
    # Public W=[[1,1],[1,0]], |x_i|<=1: q=8, output q_j=8,4.
    # Exhaust both independent uniform r and s. This checks the full inference
    # view (x-r, reduced correction), not merely a marginal mask histogram.
    distributions = []
    for x in ((0, 0), (1, -1)):
        view = Counter()
        for r0, r1, s0, s1 in itertools.product(range(8), repeat=4):
            correction = _native.prepared_pack_correction(
                struct.pack("<II", r0 + r1, r0), struct.pack("<II", s0, s1), bytes([3, 2]), 1)
            masked = ((x[0] - r0) % 8, (x[1] - r1) % 8)
            view[(masked, correction)] += 1
        distributions.append(view)
    assert distributions[0] == distributions[1]
    assert len(distributions[0]) == 8 * 8 * 8 * 4
    assert set(distributions[0].values()) == {2}

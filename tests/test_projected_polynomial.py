"""Rust protocol checked against independent unbounded Python integer algebra."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pllm import _native


def fixture(seed=12):
    rng = np.random.default_rng(seed)
    return tuple(rng.integers(-5, 6, shape, dtype=np.int8) for shape in ((9, 4), (9, 4), (3, 9)))


@pytest.mark.rust
@pytest.mark.parametrize("bits", (24, 32, 64))
@pytest.mark.parametrize("mode", ("dense", "derived", "contracted", "seeded"))
def test_native_modular_oracle_with_wrap(bits, mode):
    weights = fixture()
    rng = np.random.default_rng(674)
    clear = rng.integers(0, 1 << 63, (4, 4), dtype=np.uint64) & np.uint64((1 << bits) - 1)
    clear[0] = [0, 1, (1 << bits) - 1, 1 << (bits - 1)]
    output, metadata = _native.projected_polynomial_probe(
        mode,
        bits,
        4,
        4,
        9,
        3,
        *(w.tobytes() for w in weights),
        clear.astype("<u8").tobytes(),
        b"b" * 32,
    )
    x = clear.astype(object)
    gate = x @ weights[0].astype(object).T
    up = x @ weights[1].astype(object).T
    expected = (((gate * gate + 256 * gate) * up) @ weights[2].astype(object).T) % (1 << bits)
    actual = np.frombuffer(output, dtype="<u8").reshape(4, 3)
    np.testing.assert_array_equal(actual, expected.astype(np.uint64))
    measured = json.loads(metadata)
    assert measured["backend"] == "rust-pllm-garble/pllm-core"
    assert measured["opening_bytes"][0] == measured["opening_bytes"][1]
    if mode == "seeded":
        assert measured["material_bytes"][0] < measured["expanded_array_storage_bytes_per_party"][0]


@pytest.mark.rust
@pytest.mark.parametrize("bad", ("shape", "ring", "mode", "bytes", "residue", "binding"))
def test_native_boundary_rejects_invalid_requests(bad):
    weights = fixture()
    args = [
        "seeded",
        24,
        1,
        4,
        9,
        3,
        *(w.tobytes() for w in weights),
        np.zeros((1, 4), dtype="<u8").tobytes(),
        b"b" * 32,
    ]
    if bad == "shape":
        args[2] = 17
    elif bad == "ring":
        args[1] = 8
    elif bad == "mode":
        args[0] = "unknown"
    elif bad == "bytes":
        args[9] += b"x"
    elif bad == "residue":
        args[9] = np.full((1, 4), (1 << 24), dtype="<u8").tobytes()
    else:
        args[10] = b"bad"
    with pytest.raises(ValueError):
        _native.projected_polynomial_probe(*args)

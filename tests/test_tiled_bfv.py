from __future__ import annotations

import importlib.util
import threading

import msgpack
import numpy as np
import pytest

from pllm.runtime.tiled_bfv import (
    TiledBFVClient,
    TiledBFVError,
    TiledBFVServer,
    import_tenseal,
)


pytestmark = [
    pytest.mark.he,
    pytest.mark.skipif(
        importlib.util.find_spec("tenseal") is None,
        reason="TenSEAL test wheel unavailable",
    ),
]
MODULUS = 65537


@pytest.fixture(scope="module")
def dense_client() -> TiledBFVClient:
    return TiledBFVClient(11, 9, plain_modulus=MODULUS, threads=1)


def test_random_w4_and_w8_exact_parity_and_grouping(
    dense_client: TiledBFVClient,
) -> None:
    rng = np.random.default_rng(6107)
    masks = rng.integers(0, MODULUS, size=(7, 11), dtype=np.int64)
    payloads = dense_client.encrypt_many(masks)
    group_sizes = dense_client.group_sizes(len(masks))

    assert dense_client.capacity == 4
    assert group_sizes == [4, 3]
    assert len(payloads) == 2

    for low, high in [(-8, 8), (-128, 128)]:
        weight = rng.integers(low, high, size=(9, 11), dtype=np.int16)
        server = TiledBFVServer(dense_client.public_context, weight, threads=1)
        responses = server.evaluate_many(payloads)
        clear = dense_client.decrypt_many(responses, group_sizes)
        expected = (masks % MODULUS @ (weight.astype(np.int64) % MODULUS).T) % MODULUS
        np.testing.assert_array_equal(clear, expected)
        assert len(responses) == len(group_sizes)


def test_largest_configured_w8_modulus_is_exact() -> None:
    modulus = 1_073_692_673
    rng = np.random.default_rng(6207)
    weight = rng.integers(-127, 128, size=(5, 7), dtype=np.int16).astype(np.int8)
    masks = rng.integers(0, modulus, size=(4, 7), dtype=np.int64)
    client = TiledBFVClient(7, 5, plain_modulus=modulus)
    server = TiledBFVServer(client.public_context, weight)

    actual = client.decrypt_many(server.evaluate_many(client.encrypt_many(masks)), [4])
    expected = (masks @ weight.astype(np.int64).T) % modulus
    assert np.array_equal(actual, expected)


def test_public_server_context_has_no_secret_key(
    dense_client: TiledBFVClient,
) -> None:
    weight = np.ones((9, 11), dtype=np.int8)
    server = TiledBFVServer(dense_client.public_context, weight, threads=1)
    envelope = msgpack.unpackb(dense_client.public_context, raw=False)

    assert not server.has_secret_key
    assert "secret_key" not in envelope
    assert "public_key" not in envelope
    assert not hasattr(server, "_secret_key")
    seal = import_tenseal()
    with pytest.raises(ValueError, match="secret key is not valid"):
        seal.Decryptor(server.context, seal.SecretKey())


def test_server_honors_cooperative_cancellation(dense_client: TiledBFVClient) -> None:
    event = threading.Event()
    event.set()
    payloads = dense_client.encrypt_many(np.ones((1, 11), dtype=np.int64))
    server = TiledBFVServer(
        dense_client.public_context, np.ones((9, 11), dtype=np.int8), threads=1
    )

    with pytest.raises(TiledBFVError, match="cancelled"):
        server.evaluate_many(payloads, cancel_event=event)


def test_both_matrix_dimensions_are_tiled_and_match_modular_cleartext() -> None:
    width = 2051
    client = TiledBFVClient(width, width, plain_modulus=MODULUS, threads=1)
    assert client.input_tile_width < width
    assert client.output_tile_width < width
    assert client.segment_width == 2048
    assert client.input_tiles > 1
    assert client.output_tiles > 1

    rng = np.random.default_rng(8172)
    masks = rng.integers(0, MODULUS, size=(1, width), dtype=np.int64)
    weight = rng.integers(-8, 8, size=(width, width), dtype=np.int16).astype(np.int8)

    server = TiledBFVServer(client.public_context, weight, threads=1)
    responses = server.evaluate_many(client.encrypt_many(masks))
    clear = client.decrypt_many(responses, [1])
    expected = (masks @ (weight.astype(np.int64) % MODULUS).T) % MODULUS
    np.testing.assert_array_equal(clear, expected)


def test_context_is_reused_across_stage_shapes(dense_client: TiledBFVClient) -> None:
    reshaped = dense_client.for_shape(7, 5)
    assert reshaped.public_context is dense_client.public_context

    masks = np.arange(14, dtype=np.int64).reshape(2, 7)
    weight = np.arange(35, dtype=np.int16).reshape(5, 7) % 15 - 7
    responses = TiledBFVServer(reshaped.public_context, weight, threads=1).evaluate_many(
        reshaped.encrypt_many(masks)
    )
    actual = reshaped.decrypt_many(responses, reshaped.group_sizes(len(masks)))
    expected = (masks @ weight.astype(np.int64).T) % MODULUS
    np.testing.assert_array_equal(actual, expected)


def test_malformed_and_mismatched_dimensions_are_rejected(
    dense_client: TiledBFVClient,
) -> None:
    server = TiledBFVServer(dense_client.public_context, np.ones((9, 11), dtype=np.int8), threads=1)
    payload = dense_client.encrypt_many(np.ones((1, 11), dtype=np.int64))[0]
    envelope = msgpack.unpackb(payload, raw=False)
    envelope["input_width"] += 1
    malformed = msgpack.packb(envelope, use_bin_type=True)

    with pytest.raises(TiledBFVError, match="dimensions"):
        server.evaluate_many([malformed])
    other_server = TiledBFVServer(
        dense_client.public_context, np.ones((9, 12), dtype=np.int8), threads=1
    )
    with pytest.raises(TiledBFVError, match="dimensions"):
        other_server.evaluate_many([payload])
    with pytest.raises(TiledBFVError, match="signed W8"):
        TiledBFVServer(
            dense_client.public_context, np.full((9, 11), 128, dtype=np.int16), threads=1
        )

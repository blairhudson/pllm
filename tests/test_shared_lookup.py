from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import msgpack

from pllm.runtime.shared_lookup import (
    DenseReferenceLookupEncoder,
    PrivateLookupKey,
    SharedEmbeddingParty,
)
from pllm.runtime.shared_mpc import SharedMPCError, reconstruct


def test_dense_reference_lookup_hides_tokens_from_each_embedding_party() -> None:
    commitment = b"c" * 32
    channel = b"h" * 32
    table = np.array(
        [[1, 2, 3], [-2, 4, 1], [7, -1, 5], [3, 0, -4]],
        dtype=np.int64,
    )
    expected_table = table.copy()
    parties = (
        SharedEmbeddingParty(
            session_id="session",
            commitment_digest=commitment,
            channel_id=channel,
            party=0,
            table=table,
            scale=256,
            consumed_query_ids=set(),
        ),
        SharedEmbeddingParty(
            session_id="session",
            commitment_digest=commitment,
            channel_id=channel,
            party=1,
            table=table,
            scale=256,
            consumed_query_ids=set(),
        ),
    )
    keys = DenseReferenceLookupEncoder.generate(
        "session",
        "prompt-0",
        np.array([2, 0], dtype=np.int64),
        vocabulary_size=4,
        commitment_digest=commitment,
        channel_id=channel,
        table_fingerprint=parties[0].table_fingerprint,
    )
    table[:] = 999
    assert not np.array_equal(keys[0].shares, keys[1].shares)
    with pytest.raises(SharedMPCError, match="channel"):
        parties[0].lookup(replace(keys[0], channel_id=b"x" * 32))
    outputs = parties[0].lookup(keys[0]), parties[1].lookup(keys[1])
    assert np.array_equal(reconstruct(*outputs).view(np.int64), expected_table[[2, 0]])
    with pytest.raises(SharedMPCError, match="consumed"):
        parties[0].lookup(keys[0])


def test_private_lookup_key_round_trip_and_bounds() -> None:
    commitment = b"c" * 32
    channel = b"h" * 32
    table = np.eye(4, dtype=np.int64)
    party = SharedEmbeddingParty(
        session_id="session",
        commitment_digest=commitment,
        channel_id=channel,
        party=0,
        table=table,
        scale=1,
        consumed_query_ids=set(),
    )
    key = DenseReferenceLookupEncoder.generate(
        "session",
        "query",
        np.array([1], dtype=np.int64),
        vocabulary_size=4,
        commitment_digest=commitment,
        channel_id=channel,
        table_fingerprint=party.table_fingerprint,
    )[0]
    packed = key.pack()
    restored = PrivateLookupKey.unpack(packed, maximum_bytes=len(packed))
    assert restored.session_id == key.session_id
    assert restored.query_id == key.query_id
    assert np.array_equal(restored.shares, key.shares)
    with pytest.raises(SharedMPCError, match="exceeds"):
        PrivateLookupKey.unpack(packed, maximum_bytes=len(packed) - 1)
    envelope = msgpack.unpackb(packed, raw=False)
    envelope[b"v"] = 2
    with pytest.raises(SharedMPCError, match="invalid private lookup key"):
        PrivateLookupKey.unpack(
            msgpack.packb(envelope, use_bin_type=True), maximum_bytes=len(packed)
        )


def test_private_lookup_rejects_wrong_table_binding() -> None:
    commitment = b"c" * 32
    channel = b"h" * 32
    first = SharedEmbeddingParty(
        session_id="session",
        commitment_digest=commitment,
        channel_id=channel,
        party=0,
        table=np.eye(2, dtype=np.int64),
        scale=1,
        consumed_query_ids=set(),
    )
    second = SharedEmbeddingParty(
        session_id="session",
        commitment_digest=commitment,
        channel_id=channel,
        party=0,
        table=np.ones((2, 2), dtype=np.int64),
        scale=1,
        consumed_query_ids=set(),
    )
    key = DenseReferenceLookupEncoder.generate(
        "session",
        "query",
        np.array([0]),
        vocabulary_size=2,
        commitment_digest=commitment,
        channel_id=channel,
        table_fingerprint=first.table_fingerprint,
    )[0]
    with pytest.raises(SharedMPCError, match="another table"):
        second.lookup(key)

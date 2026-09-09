from __future__ import annotations

import numpy as np
import pytest
import msgpack

from pllm.runtime.shared_mpc import PartyRuntime, ReferenceDealer, SharedMPCError
from pllm.runtime.shared_output import (
    ReferenceJointSelector,
    SharedOutputHeadParty,
    TokenShare,
    reconstruct_token,
)


def test_shared_output_head_returns_only_token_shares() -> None:
    commitment = b"c" * 32
    channel = b"h" * 32
    dealer = ReferenceDealer("session")
    hidden = dealer.split(np.array([[3, -2, 1]], dtype=np.int64), "hidden", scale=256)
    weight = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1], [-1, 1, 0]], dtype=np.int8)
    logits = (
        SharedOutputHeadParty(PartyRuntime("session", 0), weight).project(
            hidden[0], "step-0", signed_bound=5
        ),
        SharedOutputHeadParty(PartyRuntime("session", 1), weight).project(
            hidden[1], "step-0", signed_bound=5
        ),
    )
    token_shares = ReferenceJointSelector.select_pair(
        logits[0],
        logits[1],
        generation_id="step-0",
        commitment_digest=commitment,
        channel_id=channel,
    )
    consumed: set[str] = set()
    with pytest.raises(SharedMPCError, match="unexpected generation"):
        reconstruct_token(
            *token_shares,
            vocabulary_size=4,
            expected_commitment_digest=commitment,
            expected_channel_id=channel,
            expected_generation_id="step-1",
            consumed_generation_ids=consumed,
        )
    assert (
        reconstruct_token(
            *token_shares,
            vocabulary_size=4,
            expected_commitment_digest=commitment,
            expected_channel_id=channel,
            expected_generation_id="step-0",
            consumed_generation_ids=consumed,
        )
        == 0
    )
    with pytest.raises(SharedMPCError, match="already consumed"):
        reconstruct_token(
            *token_shares,
            vocabulary_size=4,
            expected_commitment_digest=commitment,
            expected_channel_id=channel,
            expected_generation_id="step-0",
            consumed_generation_ids=consumed,
        )
    assert TokenShare.unpack(token_shares[0].pack()) == token_shares[0]
    assert TokenShare.unpack(token_shares[1].pack()) == token_shares[1]
    envelope = msgpack.unpackb(token_shares[0].pack(), raw=False)
    envelope[b"w"] = 2
    with pytest.raises(SharedMPCError, match="malformed"):
        TokenShare.unpack(msgpack.packb(envelope, use_bin_type=True))


def test_token_reconstruction_rejects_mismatched_or_out_of_range_shares() -> None:
    commitment = b"c" * 32
    channel = b"h" * 32
    first = TokenShare("session", "step", commitment, channel, 0, 10)
    with pytest.raises(SharedMPCError, match="belong together"):
        reconstruct_token(
            first,
            TokenShare("other", "step", commitment, channel, 1, 1),
            vocabulary_size=4,
            expected_commitment_digest=commitment,
            expected_channel_id=channel,
            expected_generation_id="step",
            consumed_generation_ids=set(),
        )
    with pytest.raises(SharedMPCError, match="outside vocabulary"):
        reconstruct_token(
            first,
            TokenShare("session", "step", commitment, channel, 1, 10),
            vocabulary_size=4,
            expected_commitment_digest=commitment,
            expected_channel_id=channel,
            expected_generation_id="step",
            consumed_generation_ids=set(),
        )

"""Faithful ring-2^64 truncation through party-local shares and one-use material."""

import asyncio
import hashlib

import numpy as np
import pytest

from pllm.runtime.shared_mpc import (
    ExactSessionSlot,
    ExactTruncationMaskShare,
    PartyRuntime,
    PreprocessingBurnLedger,
    ReferenceDealer,
    SharedMPCError,
    SharedTensor,
    reconstruct,
)
from pllm.runtime.shared_party import LocalPreprocessingInventory, SharedComputeParty
from pllm.runtime.shared_session import (
    SharedPeerConnection,
    SharedSessionCommitment,
    SharedSessionState,
)


def _execute(values: list[int], bits: int, mask: np.ndarray | None = None):
    dealer = ReferenceDealer(f"exact-{bits}", opened_element_budget=len(values))
    clear = np.asarray(values, dtype=np.int64)
    shares = dealer.split(clear, "input", scale=1 << bits)
    masks = dealer.exact_truncation_masks(clear.shape, "mask", bits=bits)
    if mask is not None:
        # Force wrap and low-bit carry cases, including on opposite parties.
        random_value = np.array([17] * len(values), dtype=np.uint64)
        random_shift = np.array([7] * len(values), dtype=np.uint64)
        random_bits = np.full((len(values), bits), 13, dtype=np.uint64)
        random_sign = np.array([11] * len(values), dtype=np.uint64)
        masks = (
            ExactTruncationMaskShare(
                dealer.session_id, "mask", 0, bits, len(values), random_value,
                random_shift, random_bits, random_sign,
            ),
            ExactTruncationMaskShare(
                dealer.session_id, "mask", 1, bits, len(values),
                mask - random_value, (mask >> np.uint64(bits)) - random_shift,
                np.stack([
                    ((mask >> np.uint64(bit)) & np.uint64(1)) - random_bits[:, bit]
                    for bit in range(bits)
                ], axis=-1),
                (mask >> np.uint64(63)) - random_sign,
            ),
        )
    ledger = PreprocessingBurnLedger(capacity=2 * (bits + 2))
    parties = [PartyRuntime(dealer.session_id, idx, burn_ledger=ledger) for idx in (0, 1)]
    states, frames = zip(*[
        party.begin_exact_truncate(share, mask_share, "shift", signed_bound=(1 << 62) - 1)
        for party, share, mask_share in zip(parties, shares, masks, strict=True)
    ], strict=True)
    opened, carry0 = parties[0].exact_carry_start(states[0], frames[0], frames[1])
    opened1, carry1 = parties[1].exact_carry_start(states[1], frames[1], frames[0])
    assert np.array_equal(opened, opened1)
    carry = [carry0, carry1]
    for bit in range(1, bits):
        triples = dealer.multiplication_triples(clear.shape, f"carry:{bit}")
        results = [party.begin_multiply(
            carry[index],
            SharedTensor(
                dealer.session_id, f"maskbit:{bit}", index,
                np.ascontiguousarray(masks[index].low_bit_shares[..., bit]), 1,
            ),
            triples[index],
            f"shift:carry:{bit}",
        ) for index, party in enumerate(parties)]
        products = [party.finish_multiply(
            results[index][0], results[index][1], results[1 - index][1], f"product:{bit}"
        ) for index, party in enumerate(parties)]
        carry = [party.exact_carry_bit(states[index], opened, carry[index], products[index], bit)
                 for index, party in enumerate(parties)]
    result = reconstruct(*[
        party.finish_exact_truncate(states[index], opened, carry[index], "output")
        for index, party in enumerate(parties)
    ])
    assert np.array_equal(result.view(np.int64), clear // (1 << bits))
    assert parties[0].stats.multiplication_rounds == bits - 1
    assert parties[1].stats.multiplication_rounds == bits - 1
    with pytest.raises(SharedMPCError, match="already consumed"):
        parties[0].begin_exact_truncate(shares[0], masks[0], "replay", signed_bound=1)


@pytest.mark.parametrize("bits", [1, 4, 10, 62])
def test_exact_truncation_signed_edges_with_uniform_64_bit_mask(bits: int):
    _execute([-(1 << 62) + 1, -129, -17, -1, 0, 1, 15, 16, 129, (1 << 62) - 1], bits)


@pytest.mark.parametrize("bits", [1, 4, 10])
def test_exact_truncation_forced_wrap_and_carry(bits: int):
    masks = np.asarray([
        0, (1 << 64) - 1, 1 << 63, (1 << 63) - 1, (1 << bits) - 1, 1 << 62,
    ], dtype=np.uint64)
    _execute([-1, 1, -17, 17, 5, -5], bits, masks)


def test_exact_truncation_rejects_unsupported_bounds_and_budget_before_consumption():
    dealer = ReferenceDealer("preflight", opened_element_budget=1)
    tensor = dealer.split([-1, 1], "input", scale=16)[0]
    with pytest.raises(SharedMPCError, match="budget"):
        dealer.exact_truncation_masks((2,), "invalid", bits=4)
    dealer = ReferenceDealer("preflight", opened_element_budget=2)
    mask = dealer.exact_truncation_masks((2,), "valid", bits=4)[0]
    party = PartyRuntime("preflight", 0, burn_ledger=PreprocessingBurnLedger(capacity=8))
    with pytest.raises(SharedMPCError, match="strict signed gap"):
        party.begin_exact_truncate(tensor, mask, "shift", signed_bound=1 << 62)
    party.begin_exact_truncate(tensor, mask, "shift", signed_bound=1)


def test_exact_inventory_preflights_comparison_material_and_burns_on_cancel():
    dealer = ReferenceDealer("preflight-exact", opened_element_budget=4)
    mask = dealer.exact_truncation_masks((2,), "shift", bits=4)[0]
    inventory = LocalPreprocessingInventory("preflight-exact", 0, capacity=4)
    inventory.add_exact_truncation("shift", mask)
    with pytest.raises(SharedMPCError, match="comparison material is incomplete"):
        inventory.preflight_exact_truncation("preflight-exact", "shift", (2,), 0, 4)
    assert inventory.remaining == 1
    inventory.cancel()
    assert inventory.remaining == 0
    with pytest.raises(SharedMPCError, match="cancelled"):
        inventory.add_exact_truncation("shift", mask)


def test_exact_session_preflights_aggregate_budgets_and_binds_issue_order():
    dealer = ReferenceDealer("planned-exact", opened_element_budget=4)
    graph_digest = hashlib.sha256(b"model-neutral-reference-plan").hexdigest()
    first = ExactSessionSlot("norm", (2,), 4, "beaver")
    second = ExactSessionSlot("activation", (2,), 10, "fss")
    with pytest.raises(SharedMPCError, match="aggregate opening budget"):
        dealer.admit_exact_session(graph_digest, (first, second, ExactSessionSlot("extra", (1,), 4, "beaver")))
    with pytest.raises(SharedMPCError, match="aggregate FSS key budget"):
        dealer.admit_exact_session(graph_digest, (first, second), max_key_bytes_per_party=100)
    admission = dealer.admit_exact_session(graph_digest, (first, second))
    assert admission.total_opened_elements == 4
    assert admission.per_party_key_bytes == 2 * 187
    assert len(admission.schedule_digest) == 64
    with pytest.raises(SharedMPCError, match="admitted slot and order"):
        dealer.fss_truncation_masks((2,), "activation", bits=10)
    with pytest.raises(SharedMPCError, match="admitted slot and order"):
        dealer.exact_truncation_masks((2,), "norm", bits=5)
    dealer.exact_truncation_masks((2,), "norm", bits=4)
    dealer.fss_truncation_masks((2,), "activation", bits=10)
    with pytest.raises(SharedMPCError, match="admitted slot and order"):
        dealer.exact_truncation_masks((1,), "extra", bits=4)


def test_exact_unplanned_issuer_enforces_cumulative_budget() -> None:
    dealer = ReferenceDealer("unplanned-exact", opened_element_budget=2)
    dealer.exact_truncation_masks((1,), "first", bits=4)
    dealer.fss_truncation_masks((1,), "second", bits=8)
    with pytest.raises(SharedMPCError, match="aggregate opening budget"):
        dealer.exact_truncation_masks((1,), "third", bits=4)


def test_exact_truncation_over_two_authenticated_party_channels():
    class Socket:
        def __init__(self):
            self.incoming: asyncio.Queue[bytes] = asyncio.Queue()
            self.peer: Socket | None = None

        async def send(self, payload: bytes) -> None:
            assert self.peer is not None
            await self.peer.incoming.put(payload)

        async def recv(self, maximum_bytes: int) -> bytes:
            del maximum_bytes
            return await self.incoming.get()

        async def close(self) -> None:
            pass

    async def run() -> None:
        session = "exact-network"
        dealer = ReferenceDealer(session, opened_element_budget=3)
        graph_digest = hashlib.sha256(b"graph").hexdigest()
        admission = dealer.admit_exact_session(
            graph_digest, (ExactSessionSlot("shift", (3,), 4, "beaver"),),
        )
        inputs = dealer.split([-17, 0, 17], "input", scale=16)
        masks = dealer.exact_truncation_masks((3,), "shift", bits=4)
        inventories = [LocalPreprocessingInventory(session, i, capacity=4) for i in (0, 1)]
        for i in (0, 1):
            inventories[i].add_exact_truncation("shift", masks[i])
        for bit in range(1, 4):
            triples = dealer.multiplication_triples((3,), f"shift.carry.{bit}")
            for i in (0, 1):
                inventories[i].add_triple(f"shift.carry.{bit}", triples[i])
        for inventory in inventories:
            inventory.bind_exact_admission(admission)
        sockets = Socket(), Socket()
        sockets[0].peer, sockets[1].peer = sockets[1], sockets[0]
        digests = tuple(hashlib.sha256(f"party-{i}".encode()).hexdigest() for i in (0, 1))
        commitment = SharedSessionCommitment(
            session, "model", hashlib.sha256(b"model").hexdigest(),
            graph_digest, hashlib.sha256(b"scales").hexdigest(),
            digests, 4,
        )
        parties = [SharedComputeParty(
            PartyRuntime(session, i, burn_ledger=PreprocessingBurnLedger(capacity=12)),
            SharedPeerConnection(
                SharedSessionState(commitment, party=i), sockets[i],
                max_payload_bytes=4096, authenticated_peer_fingerprint=digests[1 - i],
                timeout_seconds=1,
            ), inventories[i], max_opening_elements=3, truncation="exact",
        ) for i in (0, 1)]
        results = await asyncio.gather(*[
            parties[i].truncate_exact(inputs[i], "shift", bits=4, signed_bound=17)
            for i in (0, 1)
        ])
        np.testing.assert_array_equal(reconstruct(*results).view(np.int64), [-2, 0, 1])
        assert inventories[0].remaining == inventories[1].remaining == 0
        assert parties[0].runtime.stats.multiplication_rounds == 3

    asyncio.run(run())

"""SIGMA-style bounded FSS carry against independent signed-shift oracle."""

import asyncio
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from pllm.runtime.shared_mpc import (
    ExactSessionSlot,
    PartyRuntime,
    PreprocessingBurnLedger,
    ReferenceDealer,
    SharedMPCError,
    reconstruct,
)
from pllm.runtime.shared_party import LocalPreprocessingInventory, SharedComputeParty
from pllm.runtime.shared_session import (
    SharedPeerConnection,
    SharedSessionCommitment,
    SharedSessionState,
)


class _Socket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[bytes] = asyncio.Queue()
        self.peer: _Socket | None = None

    async def send(self, payload: bytes) -> None:
        assert self.peer is not None
        await self.peer.incoming.put(payload)

    async def recv(self, maximum_bytes: int) -> bytes:
        del maximum_bytes
        return await self.incoming.get()

    async def close(self) -> None:
        return None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.mark.parametrize("bits", [1, 4, 10])
def test_point_shared_carry_recovers_exact_signed_truncation(bits: int) -> None:
    values = np.asarray([-129, -17, -1, 0, 1, 17, 129], dtype=np.int64)
    dealer = ReferenceDealer(f"fss-{bits}", opened_element_budget=values.size)
    inputs = dealer.split(values, "input", scale=1 << bits)
    masks = dealer.fss_truncation_masks(values.shape, "shift", bits=bits)
    assert masks[0].key_bytes == masks[1].key_bytes == values.size * (17 + 17 * bits)
    parties = [
        PartyRuntime(dealer.session_id, party, burn_ledger=PreprocessingBurnLedger(capacity=16))
        for party in (0, 1)
    ]
    states, frames = zip(*[
        party.begin_exact_truncate(value, mask, "shift", signed_bound=1 << 20)
        for party, value, mask in zip(parties, inputs, masks, strict=True)
    ], strict=True)
    public = [parties[party]._open_exact(states[party], frames[party], frames[1 - party])
              for party in (0, 1)]
    assert np.array_equal(public[0], public[1])
    masked_bits = []
    for mask in masks:
        result = [
            key.less_than_share(int(opened & np.uint64((1 << bits) - 1))) ^ int(boolean)
            for key, opened, boolean in zip(
                mask.comparison_keys,
                public[0].flat,
                mask.xor_mask_share.flat,
                strict=True,
            )
        ]
        masked_bits.append(np.asarray(result, dtype=np.uint8).reshape(values.shape))
    opened_bit = np.bitwise_xor(*masked_bits).astype(np.uint64)
    outputs = [
        parties[party].finish_exact_truncate(
            states[party], public[party],
            parties[party].fss_carry_share(states[party], public[party], opened_bit),
            "out",
        )
        for party in (0, 1)
    ]
    np.testing.assert_array_equal(reconstruct(*outputs).view(np.int64), values // (1 << bits))
    assert all(party.stats.multiplication_rounds == 0 for party in parties)
    with pytest.raises(SharedMPCError, match="already consumed"):
        parties[0].begin_exact_truncate(inputs[0], masks[0], "again", signed_bound=1 << 20)


def test_fss_truncation_refuses_keys_larger_than_bounded_width() -> None:
    dealer = ReferenceDealer("fss-reject", opened_element_budget=1)
    with pytest.raises(SharedMPCError, match="bit width"):
        dealer.fss_truncation_masks((1,), "invalid", bits=11)


def test_fss_truncation_over_authenticated_separate_party_channels() -> None:
    async def run() -> None:
        session = "fss-channel"
        values = np.array([-17, -1, 0, 17], dtype=np.int64)
        dealer = ReferenceDealer(session, opened_element_budget=4)
        admission = dealer.admit_exact_session(
            _digest("fss-graph"), (ExactSessionSlot("stage", values.shape, 8, "fss"),),
        )
        inputs = dealer.split(values, "input", scale=256)
        masks = dealer.fss_truncation_masks(values.shape, "stage", bits=8)
        inventories = [LocalPreprocessingInventory(session, party, capacity=1)
                       for party in (0, 1)]
        for party in (0, 1):
            inventories[party].add_fss_truncation("stage", masks[party])
            assert all(key.expanded_bytes == 32 for key in masks[party].comparison_keys)
            inventories[party].bind_exact_admission(admission)
        sockets = _Socket(), _Socket()
        sockets[0].peer, sockets[1].peer = sockets[1], sockets[0]
        commitment = SharedSessionCommitment(
            session_id=session, model_id="model", model_fingerprint=_digest("model"),
            graph_fingerprint=_digest("fss-graph"), scale_fingerprint=_digest("scales"),
            party_fingerprints=(_digest("p0"), _digest("p1")), maximum_operations=2,
        )
        parties = [SharedComputeParty(
            PartyRuntime(session, party, minimum_truncation_security_bits=40,
                         burn_ledger=PreprocessingBurnLedger(capacity=8)),
            SharedPeerConnection(
                SharedSessionState(commitment, party=party), sockets[party],
                max_payload_bytes=4096,
                authenticated_peer_fingerprint=_digest(f"p{1 - party}"),
                timeout_seconds=1,
            ),
            inventories[party], max_opening_elements=4, truncation="fss",
        ) for party in (0, 1)]
        outputs = await asyncio.gather(*[
            parties[i].truncate(inputs[i], "stage", bits=8, signed_bound=1 << 20)
            for i in (0, 1)
        ])
        np.testing.assert_array_equal(reconstruct(*outputs).view(np.int64), values // 256)
        assert inventories[0].remaining == inventories[1].remaining == 0
        assert all(party.runtime.stats.multiplication_rounds == 0 for party in parties)
        assert all(key.expanded_bytes == 0 for mask in masks for key in mask.comparison_keys)

    asyncio.run(run())


def test_fss_inventory_rejects_key_budget_before_expansion() -> None:
    dealer = ReferenceDealer("fss-cap", opened_element_budget=1)
    mask = dealer.fss_truncation_masks((1,), "stage", bits=10)[0]
    inventory = LocalPreprocessingInventory("fss-cap", 0, capacity=1, max_fss_key_bytes=100)
    with pytest.raises(SharedMPCError, match="key budget"):
        inventory.add_fss_truncation("stage", mask)
    assert inventory.remaining == 0
    assert all(key.expanded_bytes == 0 for key in mask.comparison_keys)


@pytest.mark.parametrize("bits", [16, 24])
def test_hybrid_truncation_keeps_wide_carry_secret_shared(bits: int) -> None:
    async def run() -> None:
        session = f"hybrid-{bits}"
        values = np.array([-129, -17, -1, 0, 1, 17, 129], dtype=np.int64)
        dealer = ReferenceDealer(session, opened_element_budget=values.size)
        graph = _digest(f"hybrid-graph-{bits}")
        admission = dealer.admit_exact_session(
            graph, (ExactSessionSlot("shift", values.shape, bits, "hybrid", 8),),
        )
        assert admission.per_party_key_bytes == values.size * (17 + 17 * 8)
        assert admission.per_party_material_body_bytes == values.size * (
            33 + 8 * (bits - 8) + 24 * (bits - 8) + 17 + 17 * 8
        )
        masks = dealer.fss_truncation_masks(values.shape, "shift", bits=bits, low_bits=8)
        shares = dealer.split(values, "input", scale=1 << bits)
        inventories = [LocalPreprocessingInventory(session, party, capacity=bits - 7)
                       for party in (0, 1)]
        for party in (0, 1):
            inventories[party].add_fss_truncation("shift", masks[party])
        for bit in range(8, bits):
            op = f"shift.carry.{bit}"
            triples = dealer.multiplication_triples(values.shape, op)
            for party in (0, 1):
                inventories[party].add_triple(op, triples[party])
        for inventory in inventories:
            inventory.bind_exact_admission(admission)
        sockets = _Socket(), _Socket()
        sockets[0].peer, sockets[1].peer = sockets[1], sockets[0]
        commitment = SharedSessionCommitment(
            session, "model", _digest("model"), graph, _digest("scales"),
            (_digest("p0"), _digest("p1")), 2 + bits - 8,
        )
        parties = [SharedComputeParty(
            PartyRuntime(session, party, burn_ledger=PreprocessingBurnLedger(capacity=bits)),
            SharedPeerConnection(
                SharedSessionState(commitment, party=party), sockets[party],
                max_payload_bytes=4096, authenticated_peer_fingerprint=_digest(f"p{1 - party}"),
                timeout_seconds=2,
            ),
            inventories[party], max_opening_elements=values.size,
            truncation="fss_lowbits_exact_highbits",
        ) for party in (0, 1)]
        output = await asyncio.gather(*[
            party.truncate(shares[index], "shift", bits=bits, signed_bound=1 << 20)
            for index, party in enumerate(parties)
        ])
        np.testing.assert_array_equal(reconstruct(*output).view(np.int64), values // (1 << bits))
        assert all(party.runtime.stats.multiplication_rounds == bits - 8 for party in parties)
        assert all(party.peer.stats.sent_frames == bits - 8 + 2 for party in parties)
        body_floor = 2 * (
            8 * values.size + (values.size + 7) // 8
            + 16 * values.size * (bits - 8)
        )
        assert sum(party.peer.stats.uploaded_bytes for party in parties) > body_floor
        assert all(inventory.remaining == 0 for inventory in inventories)
        assert all(key.expanded_bytes == 0 for mask in masks for key in mask.comparison_keys)

    asyncio.run(run())


def test_hybrid_material_preflight_rejects_missing_high_carry_and_oversize_body() -> None:
    dealer = ReferenceDealer("hybrid-budget", opened_element_budget=128)
    slots = (ExactSessionSlot("shift", (128,), 24, "hybrid", 8),)
    with pytest.raises(SharedMPCError, match="material body budget"):
        dealer.admit_exact_session(_digest("graph"), slots, max_material_body_bytes_per_party=1024)
    dealer.admit_exact_session(_digest("graph"), slots)
    masks = dealer.fss_truncation_masks((128,), "shift", bits=24, low_bits=8)
    inventory = LocalPreprocessingInventory("hybrid-budget", 0, capacity=17)
    inventory.add_fss_truncation("shift", masks[0])
    with pytest.raises(SharedMPCError, match="high-bit carry material"):
        inventory.bind_exact_admission(dealer._exact_admission)
    assert inventory.remaining == 1
    inventory.cancel()
    assert all(key.expanded_bytes == 0 for key in masks[0].comparison_keys)


def test_hybrid_admission_rejects_forged_comparison_width() -> None:
    dealer = ReferenceDealer("hybrid-forgery", opened_element_budget=1)
    admitted = dealer.admit_exact_session(
        _digest("hybrid-forgery-graph"), (ExactSessionSlot("shift", (1,), 16, "hybrid", 8),),
    )
    mask = dealer.fss_truncation_masks((1,), "shift", bits=16, low_bits=8)[0]
    inventory = LocalPreprocessingInventory("hybrid-forgery", 0, capacity=9)
    inventory.add_fss_truncation("shift", mask)
    for bit in range(8, 16):
        name = f"shift.carry.{bit}"
        inventory.add_triple(name, dealer.multiplication_triples((1,), name)[0])
    forged = replace(admitted, slots=(ExactSessionSlot("shift", (1,), 16, "hybrid", 7),))
    with pytest.raises(SharedMPCError, match="comparison width"):
        inventory.bind_exact_admission(forged)
    assert inventory.exact_admission is None
    inventory.bind_exact_admission(admitted)
    inventory.cancel()

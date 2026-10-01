from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import pytest

from pllm.runtime.scalar_fss_reference import AffineScalarSpec, ScalarFssError, ScalarGateIssuer
from pllm.runtime.shared_mpc import (
    PartyRuntime,
    PreprocessingBurnLedger,
    ReferenceDealer,
    SharedTensor,
    ValueOpeningFrame,
    reconstruct,
)
from pllm.runtime.shared_party import LocalPreprocessingInventory, SharedComputeParty
from pllm.runtime.shared_session import SharedPeerConnection, SharedSessionCommitment, SharedSessionState


class _Socket:
    def __init__(self) -> None:
        self.peer: _Socket | None = None
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def send(self, payload: bytes) -> None:
        assert self.peer is not None
        await self.peer.queue.put(payload)

    async def recv(self, maximum_bytes: int) -> bytes:
        del maximum_bytes
        return await self.queue.get()

    async def close(self) -> None:
        return None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


_RELU = AffineScalarSpec(
    bits=8,
    boundaries=(0, 128, 256),
    coefficients=((1, 0), (0, 0)),
    predicates=(128,),
)


@pytest.mark.parametrize("input_value", [-128, -17, -1, 0, 1, 42, 127])
def test_compiled_scalar_relu_over_separate_authenticated_parties(input_value: int) -> None:
    async def run() -> None:
        session = f"scalar-gate-{input_value}"
        issuer = ScalarGateIssuer(session)
        keys = issuer.issue(_RELU, "relu")
        assert keys[0].spec.public_shape == keys[1].spec.public_shape == (8, 2, 3, 2)
        assert keys[0].offline_bytes == keys[1].offline_bytes == 4096 + 2 * 153 + 1
        dealer = ReferenceDealer(session, opened_element_budget=1)
        input_shares = dealer.split(np.array([input_value], dtype=np.int64), "input")
        triples = dealer.multiplication_triples((1,), "relu.mul")
        inventories = [LocalPreprocessingInventory(session, party, capacity=1) for party in (0, 1)]
        for party in (0, 1):
            inventories[party].add_triple("relu.mul", triples[party])

        sockets = _Socket(), _Socket()
        sockets[0].peer, sockets[1].peer = sockets[1], sockets[0]
        commitment = SharedSessionCommitment(
            session_id=session, model_id="tiny-public-scalar", model_fingerprint=_digest("body"),
            graph_fingerprint=_digest(repr(_RELU)), scale_fingerprint=_digest("ring2^64"),
            party_fingerprints=(_digest("p0"), _digest("p1")), maximum_operations=2,
        )
        workers = [
            SharedComputeParty(
                PartyRuntime(session, party, burn_ledger=PreprocessingBurnLedger(capacity=4)),
                SharedPeerConnection(
                    SharedSessionState(commitment, party=party),
                    sockets[party],
                    max_payload_bytes=4096,
                    authenticated_peer_fingerprint=_digest(f"p{1 - party}"),
                    timeout_seconds=1,
                ),
                inventories[party],
                max_opening_elements=4,
            ) for party in (0, 1)
        ]

        async def evaluate(party: int) -> tuple[SharedTensor, int]:
            local = input_shares[party]
            assert local.values.shape == (1,)
            local_low = np.uint64((int(local.values[0]) + int(keys[party].mask_share)) % 256)
            frame = ValueOpeningFrame(session, "relu.open", party, np.array([local_low], dtype=np.uint64))
            peer = ValueOpeningFrame.unpack(await workers[party].peer.exchange(
                "scalar.open", "relu.open", frame.pack(),
            ))
            assert peer.party == 1 - party
            public_masked = (int(local_low) + int(peer.value[0])) % 256
            (a, b), (nonnegative,) = keys[party].evaluate(public_masked)
            coefficient = SharedTensor(session, "relu.coeff", party, np.array([a], dtype=np.uint64))
            product = await workers[party].multiply(coefficient, local, "relu.mul")
            output = SharedTensor(session, "relu.out", party, product.values + b)
            return output, nonnegative

        result0, result1 = await asyncio.gather(evaluate(0), evaluate(1))
        actual = reconstruct(result0[0], result1[0]).view(np.int64)
        assert actual.tolist() == [max(input_value, 0)]
        assert result0[1] ^ result1[1] == int(input_value >= 0)
        assert inventories[0].remaining == inventories[1].remaining == 0
        with pytest.raises(ScalarFssError, match="consumed"):
            keys[0].evaluate(0)
        with pytest.raises(ScalarFssError, match="already issued"):
            issuer.issue(_RELU, "relu")

    asyncio.run(run())


def test_scalar_spec_rejects_incomplete_or_mutating_public_shapes() -> None:
    with pytest.raises(ScalarFssError, match="full partition"):
        AffineScalarSpec(8, (0, 64, 63, 256), ((0, 0), (0, 0), (0, 0)))
    with pytest.raises(ScalarFssError, match="two uint64"):
        AffineScalarSpec(8, (0, 256), ((1, 2, 3),))
    spec = AffineScalarSpec(8, (0, 128, 256), ((1, 0), (0, 0)), (128,))
    issuer = ScalarGateIssuer("bad-scalar")
    first, second = issuer.issue(spec, "bad")
    assert first.spec.public_shape == second.spec.public_shape
    with pytest.raises(ScalarFssError, match="outside the gate domain"):
        first.evaluate(256)
    with pytest.raises(ScalarFssError, match="consumed"):
        first.evaluate(0)
    second.cancel()
    with pytest.raises(ScalarFssError, match="consumed"):
        second.evaluate(0)

from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import pytest

from pllm.runtime.shared_attention import NetworkSharedAttention, SharedKVCache
from pllm.runtime.shared_mlp import SharedMLPWeights
from pllm.runtime.shared_mpc import (
    PartyRuntime,
    PublicQuantizedMatrix,
    ReferenceDealer,
    SharedMPCError,
    reconstruct,
)
from pllm.runtime.shared_norm import NetworkSharedRMSNorm, RMSNormApproximation
from pllm.runtime.shared_party import (
    LocalPreprocessingInventory,
    NetworkSharedMLP,
    SharedComputeParty,
)
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


def test_network_shared_mlp_uses_only_party_local_preprocessing() -> None:
    async def run() -> None:
        session = "network-mlp"
        scale = 1 << 8
        weights = SharedMLPWeights(
            gate=np.array([[1, -1, 1], [0, 1, -1]], dtype=np.int8),
            up=np.array([[1, 1, 0], [-1, 0, 1]], dtype=np.int8),
            down=np.array([[1, -1], [1, 1], [-1, 1]], dtype=np.int8),
        )
        clear = np.array([[64, -32, 64]], dtype=np.int64)
        dealer = ReferenceDealer(session, session_security_bits=16, opened_element_budget=4)
        hidden = dealer.split(clear, "hidden", scale=scale)
        inventories = (
            LocalPreprocessingInventory(session, 0, capacity=4),
            LocalPreprocessingInventory(session, 1, capacity=4),
        )
        for operation, values in (
            ("network_mlp.gate_square", dealer.multiplication_triples((1, 2), "square")),
            (
                "network_mlp.activation_truncate",
                dealer.truncation_masks((1, 2), "activation", bits=10),
            ),
            ("network_mlp.gate_product", dealer.multiplication_triples((1, 2), "product")),
            (
                "network_mlp.product_truncate",
                dealer.truncation_masks((1, 2), "product-truncate", bits=8),
            ),
        ):
            for party in (0, 1):
                value = values[party]
                if hasattr(value, "triple_id"):
                    inventories[party].add_triple(operation, value)
                else:
                    inventories[party].add_truncation(operation, value)

        sockets = _Socket(), _Socket()
        sockets[0].peer, sockets[1].peer = sockets[1], sockets[0]
        commitment = SharedSessionCommitment(
            session_id=session,
            model_id="model",
            model_fingerprint=_digest("model"),
            graph_fingerprint=_digest("graph"),
            scale_fingerprint=_digest("scales"),
            party_fingerprints=(_digest("party-0"), _digest("party-1")),
            maximum_operations=4,
        )
        models = []
        for party in (0, 1):
            runtime = PartyRuntime(session, party, minimum_truncation_security_bits=16)
            peer = SharedPeerConnection(
                SharedSessionState(commitment, party=party),
                sockets[party],
                max_payload_bytes=4096,
                authenticated_peer_fingerprint=_digest(f"party-{1 - party}"),
                timeout_seconds=1,
            )
            compute = SharedComputeParty(
                runtime,
                peer,
                inventories[party],
                max_opening_elements=16,
            )
            models.append(NetworkSharedMLP(compute, weights, prefix="network_mlp"))

        output = await asyncio.gather(
            models[0].run(hidden[0], hidden_bound=64),
            models[1].run(hidden[1], hidden_bound=64),
        )
        actual = reconstruct(output[0], output[1]).view(np.int64)
        assert output[0].scale == output[1].scale == scale
        np.testing.assert_allclose(actual, np.array([[13, 13, -13]]), atol=1)
        assert inventories[0].remaining == inventories[1].remaining == 0

    asyncio.run(run())


def test_preprocessing_inventory_burns_before_network_use() -> None:
    dealer = ReferenceDealer("burn")
    inventory = LocalPreprocessingInventory("burn", 0, capacity=1)
    triple = dealer.multiplication_triples((1,), "triple")[0]
    inventory.add_triple("multiply", triple)
    assert inventory.take_multiplication("burn", "multiply", (1,), 0) is triple
    with pytest.raises(SharedMPCError, match="consumed"):
        inventory.take_multiplication("burn", "multiply", (1,), 0)


def test_quantized_public_linear_applies_per_output_scales_without_wrap() -> None:
    async def run() -> None:
        session = "quantized-linear"
        dealer = ReferenceDealer(session, session_security_bits=16, opened_element_budget=1)
        inputs = dealer.split(np.array([[100, -20]], dtype=np.int64), "input", scale=256)
        matrix = PublicQuantizedMatrix(
            np.array([[2, -1]], dtype=np.int8),
            np.array([32], dtype=np.int64),
            16,
        )
        masks = dealer.truncation_masks((1, 1), "quantized", bits=4)
        inventories = (
            LocalPreprocessingInventory(session, 0, capacity=1),
            LocalPreprocessingInventory(session, 1, capacity=1),
        )
        for party in (0, 1):
            inventories[party].add_truncation("linear.truncate", masks[party])
        sockets = _Socket(), _Socket()
        sockets[0].peer, sockets[1].peer = sockets[1], sockets[0]
        commitment = SharedSessionCommitment(
            session,
            "model",
            _digest("model"),
            _digest("graph"),
            _digest("scales"),
            (_digest("party-0"), _digest("party-1")),
            1,
        )
        parties = []
        for party in (0, 1):
            parties.append(
                SharedComputeParty(
                    PartyRuntime(session, party, minimum_truncation_security_bits=16),
                    SharedPeerConnection(
                        SharedSessionState(commitment, party=party),
                        sockets[party],
                        max_payload_bytes=4096,
                        authenticated_peer_fingerprint=_digest(f"party-{1 - party}"),
                        timeout_seconds=1,
                    ),
                    inventories[party],
                    max_opening_elements=1,
                )
            )
        outputs = await asyncio.gather(
            parties[0].quantized_linear(inputs[0], matrix, "linear", signed_bound=100),
            parties[1].quantized_linear(inputs[1], matrix, "linear", signed_bound=100),
        )
        actual = reconstruct(outputs[0][0], outputs[1][0]).view(np.int64)
        np.testing.assert_allclose(actual, np.array([[440]]), atol=1)
        assert outputs[0][0].scale == outputs[1][0].scale == 256
        assert outputs[0][1] == outputs[1][1] == 601

    asyncio.run(run())


def test_network_rms_norm_keeps_values_and_statistic_secret_shared() -> None:
    async def run() -> None:
        session = "network-norm"
        scale = 1 << 8
        clear = np.array([[128, -128, 256, -256]], dtype=np.int64)
        dealer = ReferenceDealer(session, session_security_bits=16, opened_element_budget=14)
        hidden = dealer.split(clear, "hidden", scale=scale)
        approximation = RMSNormApproximation(
            weight=np.full(4, scale, dtype=np.int64),
            intercept=384,
            slope=-128,
            epsilon=128,
            scale=scale,
            minimum_statistic=128,
            maximum_statistic=512,
            maximum_sampled_relative_error=0.35,
        )
        operations = (
            ("norm.square", dealer.multiplication_triples((1, 4), "norm-square")),
            (
                "norm.square_truncate",
                dealer.truncation_masks((1, 4), "norm-square-t", bits=8),
            ),
            (
                "norm.mean_truncate",
                dealer.truncation_masks((1, 1), "norm-mean-t", bits=10),
            ),
            (
                "norm.inverse_truncate",
                dealer.truncation_masks((1, 1), "norm-inv-t", bits=8),
            ),
            ("norm.normalize", dealer.multiplication_triples((1, 4), "norm-normalize")),
            (
                "norm.normalize_truncate",
                dealer.truncation_masks((1, 4), "norm-normalize-t", bits=8),
            ),
            (
                "norm.weight_truncate",
                dealer.truncation_masks((1, 4), "norm-weight-t", bits=8),
            ),
        )
        inventories = (
            LocalPreprocessingInventory(session, 0, capacity=len(operations)),
            LocalPreprocessingInventory(session, 1, capacity=len(operations)),
        )
        for operation, values in operations:
            for party in (0, 1):
                value = values[party]
                if hasattr(value, "triple_id"):
                    inventories[party].add_triple(operation, value)
                else:
                    inventories[party].add_truncation(operation, value)

        sockets = _Socket(), _Socket()
        sockets[0].peer, sockets[1].peer = sockets[1], sockets[0]
        commitment = SharedSessionCommitment(
            session_id=session,
            model_id="model",
            model_fingerprint=_digest("model"),
            graph_fingerprint=_digest("norm-graph"),
            scale_fingerprint=_digest("scales"),
            party_fingerprints=(_digest("party-0"), _digest("party-1")),
            maximum_operations=len(operations),
        )
        models = []
        for party in (0, 1):
            runtime = PartyRuntime(session, party, minimum_truncation_security_bits=16)
            peer = SharedPeerConnection(
                SharedSessionState(commitment, party=party),
                sockets[party],
                max_payload_bytes=4096,
                authenticated_peer_fingerprint=_digest(f"party-{1 - party}"),
                timeout_seconds=1,
            )
            compute = SharedComputeParty(
                runtime,
                peer,
                inventories[party],
                max_opening_elements=16,
            )
            models.append(NetworkSharedRMSNorm(compute, approximation, prefix="norm"))

        output = await asyncio.gather(
            models[0].run(hidden[0], signed_bound=256),
            models[1].run(hidden[1], signed_bound=256),
        )
        actual = reconstruct(output[0], output[1]).view(np.int64)
        assert output[0].scale == output[1].scale == scale
        assert np.array_equal(actual, np.array([[120, -120, 240, -240]], dtype=np.int64))
        assert inventories[0].remaining == inventories[1].remaining == 0

    asyncio.run(run())


def test_malformed_inner_opening_aborts_session_after_preprocessing_burn() -> None:
    async def run() -> None:
        session = "malformed-opening"
        dealer = ReferenceDealer(session)
        values = dealer.split(np.array([2], dtype=np.int64), "value", scale=1)
        triples = dealer.multiplication_triples((1,), "bad")
        inventories = (
            LocalPreprocessingInventory(session, 0, capacity=1),
            LocalPreprocessingInventory(session, 1, capacity=1),
        )
        for party in (0, 1):
            inventories[party].add_triple("bad", triples[party])
        sockets = _Socket(), _Socket()
        sockets[0].peer, sockets[1].peer = sockets[1], sockets[0]
        commitment = SharedSessionCommitment(
            session,
            "model",
            _digest("model"),
            _digest("graph"),
            _digest("scales"),
            (_digest("party-0"), _digest("party-1")),
            1,
        )
        peers = tuple(
            SharedPeerConnection(
                SharedSessionState(commitment, party=party),
                sockets[party],
                max_payload_bytes=4096,
                authenticated_peer_fingerprint=_digest(f"party-{1 - party}"),
                timeout_seconds=1,
            )
            for party in (0, 1)
        )
        compute = SharedComputeParty(
            PartyRuntime(session, 0), peers[0], inventories[0], max_opening_elements=1
        )
        task = asyncio.create_task(compute.multiply(values[0], values[0], "bad"))
        await peers[1].send("multiply.open", "bad", b"not-an-opening")
        await peers[1].receive(kind="multiply.open", operation_id="bad")
        with pytest.raises(SharedMPCError, match="opening frame"):
            await task
        with pytest.raises(SharedMPCError, match="failed"):
            peers[0].state.outbound("later", "later", b"later")
        with pytest.raises(SharedMPCError, match="consumed"):
            inventories[0].take_multiplication(session, "bad", (1,), 0)

    asyncio.run(run())


def test_network_causal_attention_uses_resident_shared_kv() -> None:
    async def run() -> None:
        session = "network-attention"
        scale = 1 << 8
        dealer = ReferenceDealer(session, session_security_bits=16, opened_element_budget=80)
        query = dealer.split(
            np.array([[[[128, 0], [128, 0]], [[128, 0], [128, 0]]]], dtype=np.int64),
            "query",
            scale=scale,
        )
        key_parts = (
            dealer.split(np.array([[[[128, 0]]]], dtype=np.int64), "key-0", scale=scale),
            dealer.split(np.array([[[[-128, 0]]]], dtype=np.int64), "key-1", scale=scale),
        )
        value_parts = (
            dealer.split(np.array([[[[256, 512]]]], dtype=np.int64), "value-0", scale=scale),
            dealer.split(np.array([[[[768, 1024]]]], dtype=np.int64), "value-1", scale=scale),
        )
        caches = (SharedKVCache(session, 0, 4), SharedKVCache(session, 1, 4))
        for party in (0, 1):
            caches[party].append(key_parts[0][party], value_parts[0][party])
            caches[party].append(key_parts[1][party], value_parts[1][party])

        operation_values = [
            ("attention.qk.0", dealer.multiplication_triples((1, 2, 2, 2, 2), "attn-qk")),
            (
                "attention.score_truncate",
                dealer.truncation_masks((1, 2, 2, 2), "attn-score-t", bits=8),
            ),
            (
                "attention.scale_truncate",
                dealer.truncation_masks((1, 2, 2, 2), "attn-scale-t", bits=8),
            ),
            ("attention.square", dealer.multiplication_triples((1, 2, 2, 2), "attn-square")),
            (
                "attention.square_truncate",
                dealer.truncation_masks((1, 2, 2, 2), "attn-square-t", bits=8),
            ),
            (
                "attention.half_truncate",
                dealer.truncation_masks((1, 2, 2, 2), "attn-half-t", bits=8),
            ),
            (
                "attention.probability",
                dealer.multiplication_triples((1, 2, 2, 2), "attn-probability"),
            ),
            (
                "attention.probability_truncate",
                dealer.truncation_masks((1, 2, 2, 2), "attn-probability-t", bits=8),
            ),
            (
                "attention.value.0",
                dealer.multiplication_triples((1, 2, 2, 2, 2), "attn-value"),
            ),
            (
                "attention.value_truncate",
                dealer.truncation_masks((1, 2, 2, 2), "attn-value-t", bits=16),
            ),
        ]
        for iteration in range(4):
            shape = (1, 2, 2, 1)
            operation_values.extend(
                [
                    (
                        f"attention.inverse_refine_{iteration}.product",
                        dealer.multiplication_triples(shape, f"attn-refine-{iteration}-p"),
                    ),
                    (
                        f"attention.inverse_refine_{iteration}.truncate",
                        dealer.truncation_masks(shape, f"attn-refine-{iteration}-pt", bits=8),
                    ),
                    (
                        f"attention.inverse_refine_{iteration}.update",
                        dealer.multiplication_triples(shape, f"attn-refine-{iteration}-u"),
                    ),
                    (
                        f"attention.inverse_refine_{iteration}.update_truncate",
                        dealer.truncation_masks(shape, f"attn-refine-{iteration}-ut", bits=16),
                    ),
                ]
            )
        inventories = (
            LocalPreprocessingInventory(session, 0, capacity=len(operation_values)),
            LocalPreprocessingInventory(session, 1, capacity=len(operation_values)),
        )
        for operation, values in operation_values:
            for party in (0, 1):
                if hasattr(values[party], "triple_id"):
                    inventories[party].add_triple(operation, values[party])
                else:
                    inventories[party].add_truncation(operation, values[party])

        sockets = _Socket(), _Socket()
        sockets[0].peer, sockets[1].peer = sockets[1], sockets[0]
        commitment = SharedSessionCommitment(
            session_id=session,
            model_id="model",
            model_fingerprint=_digest("model"),
            graph_fingerprint=_digest("attention-graph"),
            scale_fingerprint=_digest("scales"),
            party_fingerprints=(_digest("party-0"), _digest("party-1")),
            maximum_operations=len(operation_values),
        )
        models = []
        for party in (0, 1):
            compute = SharedComputeParty(
                PartyRuntime(session, party, minimum_truncation_security_bits=16),
                SharedPeerConnection(
                    SharedSessionState(commitment, party=party),
                    sockets[party],
                    max_payload_bytes=4096,
                    authenticated_peer_fingerprint=_digest(f"party-{1 - party}"),
                    timeout_seconds=1,
                ),
                inventories[party],
                max_opening_elements=32,
            )
            models.append(
                NetworkSharedAttention(
                    compute, scale=scale, prefix="attention", probability_scale=1 << 16
                )
            )
        outputs = await asyncio.gather(
            models[0].run(
                query[0],
                *caches[0].tensors(),
                query_bound=128,
                key_bound=128,
                value_bound=1024,
            ),
            models[1].run(
                query[1],
                *caches[1].tensors(),
                query_bound=128,
                key_bound=128,
                value_bound=1024,
            ),
        )
        actual = reconstruct(outputs[0], outputs[1]).view(np.int64)
        np.testing.assert_allclose(
            actual,
            np.array(
                [[[[256, 512], [468, 724]], [[256, 512], [468, 724]]]],
                dtype=np.int64,
            ),
            atol=8,
        )
        assert inventories[0].remaining == inventories[1].remaining == 0

    asyncio.run(run())


def test_kv_cache_rejects_value_from_other_party() -> None:
    cache = SharedKVCache("session", 0, 4)
    key = ReferenceDealer("session").split(np.zeros((1, 1, 1, 2), dtype=np.int64), "key")[0]
    value = ReferenceDealer("session").split(np.zeros((1, 1, 1, 2), dtype=np.int64), "value")[1]
    with pytest.raises(SharedMPCError, match="another party"):
        cache.append(key, value)


def test_rms_norm_mean_bound_remains_precise_at_large_width() -> None:
    approximation = RMSNormApproximation(
        weight=np.full(4096, 256, dtype=np.int64),
        intercept=384,
        slope=-128,
        epsilon=128,
        scale=256,
        minimum_statistic=128,
        maximum_statistic=512,
        maximum_sampled_relative_error=0.35,
    )
    norm = NetworkSharedRMSNorm(None, approximation, prefix="wide")  # type: ignore[arg-type]
    assert norm.bounds(256).statistic - approximation.epsilon <= 258


def test_attention_initial_reciprocal_retains_long_context_precision() -> None:
    model = NetworkSharedAttention(None, scale=256, prefix="attention")  # type: ignore[arg-type]
    assert int(model._initial_inverse(past=4095, query_tokens=1).item()) == 4096
    assert int(model._initial_inverse(past=32767, query_tokens=1).item()) == 512

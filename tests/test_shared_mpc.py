from __future__ import annotations

import msgpack
import numpy as np
import pytest

from pllm.runtime.shared_mlp import (
    SharedMLPParty,
    SharedMLPWeights,
    StaticSharedMLPParty,
    clear_mlp_reference,
)
from pllm.runtime.shared_mpc import (
    OpeningFrame,
    PartyRuntime,
    PreprocessingBurnLedger,
    ReferenceDealer,
    PublicQuantizedMatrix,
    SharedMPCError,
    SharedTensor,
    SQLitePreprocessingBurnLedger,
    decode_fixed,
    encode_fixed,
    reconstruct,
)
from pllm.runtime.native import MaskedGEMM


def _multiply(
    first: PartyRuntime,
    second: PartyRuntime,
    left,
    right,
    dealer: ReferenceDealer,
    operation_id: str,
):
    triples = dealer.multiplication_triples(left[0].shape, operation_id)
    state0, frame0 = first.begin_multiply(left[0], right[0], triples[0], operation_id)
    state1, frame1 = second.begin_multiply(left[1], right[1], triples[1], operation_id)
    result0 = first.finish_multiply(
        state0, frame0, OpeningFrame.unpack(frame1.pack()), operation_id
    )
    result1 = second.finish_multiply(
        state1, frame1, OpeningFrame.unpack(frame0.pack()), operation_id
    )
    return result0, result1


def test_party_local_beaver_multiplication_and_single_use() -> None:
    dealer = ReferenceDealer("session")
    ledger = PreprocessingBurnLedger(capacity=4)
    first = PartyRuntime("session", 0, burn_ledger=ledger)
    second = PartyRuntime("session", 1)
    left = dealer.split(np.array([[3, -4, 8]], dtype=np.int64), "left")
    right = dealer.split(np.array([[-2, 5, 7]], dtype=np.int64), "right")
    triples = dealer.multiplication_triples(left[0].shape, "multiply")
    state0, frame0 = first.begin_multiply(left[0], right[0], triples[0], "multiply")
    state1, frame1 = second.begin_multiply(left[1], right[1], triples[1], "multiply")
    result0 = first.finish_multiply(state0, frame0, OpeningFrame.unpack(frame1.pack()), "result")
    result1 = second.finish_multiply(state1, frame1, OpeningFrame.unpack(frame0.pack()), "result")

    expected = np.array([[-6, -20, 56]], dtype=np.int64).view(np.uint64)
    assert np.array_equal(reconstruct(result0, result1), expected)
    assert first.stats.multiplication_rounds == 1
    assert second.stats.multiplication_rounds == 1
    with pytest.raises(SharedMPCError, match="consumed"):
        first.begin_multiply(left[0], right[0], triples[0], "replay")
    with pytest.raises(SharedMPCError, match="consumed"):
        PartyRuntime("session", 0, burn_ledger=ledger).begin_multiply(
            left[0], right[0], triples[0], "recreated-runtime"
        )


def test_sqlite_burn_ledger_survives_runtime_restart(tmp_path) -> None:
    dealer = ReferenceDealer("durable-reuse")
    left = dealer.split(np.array([4]), "left")
    right = dealer.split(np.array([5]), "right")
    triples = dealer.multiplication_triples((1,), "durable-triple")
    path = tmp_path / "burns.sqlite3"
    ledger = SQLitePreprocessingBurnLedger(path)
    PartyRuntime("durable-reuse", 0, burn_ledger=ledger).begin_multiply(
        left[0], right[0], triples[0], "first"
    )
    ledger.close()
    restarted = SQLitePreprocessingBurnLedger(path)
    with pytest.raises(SharedMPCError, match="consumed"):
        PartyRuntime("durable-reuse", 0, burn_ledger=restarted).begin_multiply(
            left[0], right[0], triples[0], "second"
        )
    restarted.close()


def test_opening_frame_is_bound_and_bounded() -> None:
    dealer = ReferenceDealer("session-a")
    runtime = PartyRuntime("session-a", 0)
    left = dealer.split(np.ones((2, 3), dtype=np.int64), "left")[0]
    right = dealer.split(np.ones((2, 3), dtype=np.int64), "right")[0]
    triple = dealer.multiplication_triples(left.shape, "triple")[0]
    _, frame = runtime.begin_multiply(left, right, triple, "operation")
    packed = frame.pack()
    assert b"session-a" in packed
    decoded = OpeningFrame.unpack(packed, max_elements=6)
    assert decoded.operation_id == "operation"

    value = msgpack.unpackb(packed, raw=False)
    value["h"] = [10_000, 10_000]
    with pytest.raises(SharedMPCError, match="too large"):
        OpeningFrame.unpack(msgpack.packb(value, use_bin_type=True), max_elements=6)


def test_shared_mlp_matches_independent_ring_reference() -> None:
    session = "mlp-session"
    scale = 256
    dealer = ReferenceDealer(session)
    runtimes = PartyRuntime(session, 0), PartyRuntime(session, 1)
    weights = SharedMLPWeights(
        gate=np.array([[1, -1, 1], [0, 1, -1]], dtype=np.int8),
        up=np.array([[1, 1, 0], [-1, 0, 1]], dtype=np.int8),
        down=np.array([[1, -1], [1, 1], [-1, 1]], dtype=np.int8),
    )
    clear_hidden = encode_fixed(np.array([[0.5, -0.25, 0.75]]), scale)
    hidden = dealer.split(clear_hidden.view(np.int64), "hidden", scale=scale)
    parties = SharedMLPParty(runtimes[0], weights), SharedMLPParty(runtimes[1], weights)
    projections = (
        parties[0].project(hidden[0], signed_bound=192),
        parties[1].project(hidden[1], signed_bound=192),
    )

    square_triples = dealer.multiplication_triples(projections[0].gate.shape, "gate-square")
    square0 = parties[0].begin_gate_square(projections[0], square_triples[0])
    square1 = parties[1].begin_gate_square(projections[1], square_triples[1])
    product_triples = dealer.multiplication_triples(projections[0].gate.shape, "gate-product")
    product0 = parties[0].finish_gate_square(
        square0, OpeningFrame.unpack(square1.opening.pack()), product_triples[0]
    )
    product1 = parties[1].finish_gate_square(
        square1, OpeningFrame.unpack(square0.opening.pack()), product_triples[1]
    )
    output0 = parties[0].finish(product0, OpeningFrame.unpack(product1.opening.pack()))
    output1 = parties[1].finish(product1, OpeningFrame.unpack(product0.opening.pack()))

    actual = reconstruct(output0, output1)
    expected, output_scale = clear_mlp_reference(clear_hidden, weights, scale=scale)
    assert output0.scale == output_scale
    assert np.array_equal(actual, expected)
    assert np.allclose(
        decode_fixed(actual, output_scale),
        np.array([[0.390625, 0.265625, -0.390625]]),
    )
    assert runtimes[0].stats.multiplication_rounds == 2
    assert runtimes[1].stats.multiplication_rounds == 2


def test_static_scale_mlp_truncates_back_to_input_scale() -> None:
    scale = 1 << 8
    clear_hidden = encode_fixed(np.array([[0.25, -0.125, 0.25]]), scale)
    weights = SharedMLPWeights(
        gate=np.array([[1, -1, 1], [0, 1, -1]], dtype=np.int8),
        up=np.array([[1, 1, 0], [-1, 0, 1]], dtype=np.int8),
        down=np.array([[1, -1], [1, 1], [-1, 1]], dtype=np.int8),
    )
    dealer = ReferenceDealer("static-mlp", session_security_bits=16, opened_element_budget=4)
    hidden = dealer.split(clear_hidden.view(np.int64), "hidden", scale=scale)
    runtimes = (
        PartyRuntime("static-mlp", 0, minimum_truncation_security_bits=16),
        PartyRuntime("static-mlp", 1, minimum_truncation_security_bits=16),
    )
    parties = StaticSharedMLPParty(runtimes[0], weights), StaticSharedMLPParty(runtimes[1], weights)
    projections = [
        party.project(hidden[index], signed_bound=64) for index, party in enumerate(parties)
    ]

    square_triples = dealer.multiplication_triples((1, 2), "static-square")
    square = [
        party.begin_gate_square(projections[index][0], square_triples[index])
        for index, party in enumerate(parties)
    ]
    activation_masks = dealer.truncation_masks((1, 2), "static-activation", bits=10)
    activation = [
        party.begin_activation_truncation(
            square[index],
            square[1 - index].opening,
            activation_masks[index],
            gate_bound=projections[index][1],
        )
        for index, party in enumerate(parties)
    ]

    product_triples = dealer.multiplication_triples((1, 2), "static-product")
    product = [
        party.begin_gate_product(
            activation[index],
            activation[1 - index].opening,
            product_triples[index],
            up_bound=projections[index][2],
        )
        for index, party in enumerate(parties)
    ]
    product_masks = dealer.truncation_masks((1, 2), "static-product-truncate", bits=8)
    truncated = [
        party.begin_product_truncation(
            product[index], product[1 - index].opening, product_masks[index]
        )
        for index, party in enumerate(parties)
    ]
    output = [
        party.finish(truncated[index], truncated[1 - index].opening)
        for index, party in enumerate(parties)
    ]
    actual = reconstruct(output[0], output[1])
    assert output[0].scale == output[1].scale == scale
    np.testing.assert_allclose(
        decode_fixed(actual, scale),
        np.array([[0.05078125, 0.05078125, -0.05078125]]),
        atol=2 / scale,
    )


def test_public_linear_can_use_cached_wrap64_kernel(monkeypatch) -> None:
    monkeypatch.delenv("PLLM_REQUIRE_RUST", raising=False)
    monkeypatch.setenv("PLLM_KERNEL_BACKEND", "python")
    runtime = PartyRuntime("session", 0, matrix_executor=MaskedGEMM(threads=1))
    values = np.array([[[0, 2**64 - 1, 7], [2**63, 11, 13]]], dtype=np.uint64)
    shared = SharedTensor("session", "input", 0, values, scale=1)
    weights = np.array([[127, -128, 1], [-1, 0, 7]], dtype=np.int8)
    expected = np.asarray(
        values.astype(object) @ weights.astype(object).T % (1 << 64), dtype=np.uint64
    )
    np.testing.assert_array_equal(
        runtime.linear_public(shared, weights, "first", signed_bound=6).values, expected
    )
    np.testing.assert_array_equal(
        runtime.linear_public(shared, weights, "second", signed_bound=6).values, expected
    )
    with pytest.raises(SharedMPCError, match="signed ring range"):
        runtime.linear_public(shared, weights, "overflow", signed_bound=1 << 62)
    assert len(runtime._compiled_matrices) == 1


def test_public_quantized_matrix_rejects_lossy_or_oversized_inputs() -> None:
    with pytest.raises(SharedMPCError, match="fit signed int8"):
        PublicQuantizedMatrix(np.array([[2**64 - 1]], dtype=np.uint64), np.array([1]), 1)
    with pytest.raises(SharedMPCError, match="must be integers"):
        PublicQuantizedMatrix(np.array([[1.5]]), np.array([1]), 1)


def test_each_initial_share_hides_clear_value(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_random = np.arange(8, dtype=np.uint64).reshape(2, 4)
    monkeypatch.setattr(
        "pllm.runtime.shared_mpc._random_ring",
        lambda shape: fixed_random.copy(),
    )
    dealer = ReferenceDealer("privacy")
    first_a, second_a = dealer.split(np.zeros((2, 4), dtype=np.int64), "a")
    first_b, second_b = dealer.split(np.full((2, 4), 99, dtype=np.int64), "b")
    assert np.array_equal(first_a.values, first_b.values)
    assert not np.array_equal(second_a.values, second_b.values)
    assert not np.array_equal(first_a.values, reconstruct(first_a, second_a))


def test_one_party_multiplication_transcript_is_independent_of_clear_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_random = np.arange(1, 5, dtype=np.uint64).reshape(1, 4)
    monkeypatch.setattr("pllm.runtime.shared_mpc._random_ring", lambda shape: fixed_random.copy())

    def transcript(left_clear: int, right_clear: int, material_label: str) -> bytes:
        dealer = ReferenceDealer("transcript")
        left = dealer.split(np.full((1, 4), left_clear, dtype=np.int64), "left")[0]
        right = dealer.split(np.full((1, 4), right_clear, dtype=np.int64), "right")[0]
        triple = dealer.multiplication_triples((1, 4), material_label)[0]
        _, opening = PartyRuntime("transcript", 0).begin_multiply(left, right, triple, "multiply")
        return opening.pack()

    assert transcript(1, 2, "multiply-a") == transcript(999, -777, "multiply-b")


def test_truncation_accounts_for_whole_session_security_and_budget() -> None:
    session = "truncate-security"
    dealer = ReferenceDealer(session, session_security_bits=16, opened_element_budget=2)
    masks = dealer.truncation_masks((1, 2), "mask", bits=8)
    assert masks[0].session_security_bits == 16
    assert masks[0].per_element_security_bits == 17
    runtime = PartyRuntime(session, 0, minimum_truncation_security_bits=16)
    values = dealer.split(np.ones((1, 2), dtype=np.int64), "input", scale=256)
    runtime.begin_truncate(values[0], masks[0], "truncate", signed_bound=1)
    more = ReferenceDealer(
        session, session_security_bits=16, opened_element_budget=2
    ).truncation_masks((1, 1), "more", bits=8)[0]
    recreated = PartyRuntime(session, 0, minimum_truncation_security_bits=16)
    with pytest.raises(SharedMPCError, match="budget exhausted"):
        recreated.begin_truncate(
            dealer.split(np.ones((1, 1), dtype=np.int64), "more", scale=256)[0],
            more,
            "more",
            signed_bound=1,
        )

    strict = PartyRuntime(session, 0)
    with pytest.raises(SharedMPCError, match="security target"):
        strict.begin_truncate(values[0], masks[0], "strict", signed_bound=1)


def test_ring64_truncation_rejects_unachievable_session_security() -> None:
    dealer = ReferenceDealer(
        "qwen-sized", session_security_bits=40, opened_element_budget=10_000_000
    )
    with pytest.raises(SharedMPCError, match="bit range"):
        dealer.truncation_masks((1,), "mask", bits=8)


def test_probabilistic_truncation_matches_fixed_point_with_one_ulp_error() -> None:
    session = "truncate"
    dealer = ReferenceDealer(session, session_security_bits=16, opened_element_budget=257)
    first = PartyRuntime(session, 0, minimum_truncation_security_bits=16)
    second = PartyRuntime(session, 1, minimum_truncation_security_bits=16)
    clear = np.arange(-128, 129, dtype=np.int64).reshape(1, -1)
    shares = dealer.split(clear, "input", scale=256)
    masks = dealer.truncation_masks(clear.shape, "truncate-8", bits=8)
    state0, frame0 = first.begin_truncate(shares[0], masks[0], "truncate-8", signed_bound=128)
    state1, frame1 = second.begin_truncate(shares[1], masks[1], "truncate-8", signed_bound=128)
    result0 = first.finish_truncate(state0, frame0, frame1, "result")
    result1 = second.finish_truncate(state1, frame1, frame0, "result")

    actual = reconstruct(result0, result1).view(np.int64)
    expected = np.floor_divide(clear, 256)
    assert np.all((actual == expected) | (actual == expected + 1))
    assert result0.scale == 1
    with pytest.raises(SharedMPCError, match="consumed"):
        first.begin_truncate(shares[0], masks[0], "truncate-replay", signed_bound=128)


def test_truncation_is_exact_for_values_at_output_scale() -> None:
    session = "truncate-exact"
    dealer = ReferenceDealer(session, session_security_bits=16, opened_element_budget=65)
    parties = (
        PartyRuntime(session, 0, minimum_truncation_security_bits=16),
        PartyRuntime(session, 1, minimum_truncation_security_bits=16),
    )
    clear = (np.arange(-32, 33, dtype=np.int64) * 256).reshape(1, -1)
    shares = dealer.split(clear, "input", scale=65536)
    masks = dealer.truncation_masks(clear.shape, "truncate-8", bits=8)
    states = [
        parties[index].begin_truncate(shares[index], masks[index], "truncate-8", signed_bound=8192)
        for index in range(2)
    ]
    outputs = [
        parties[index].finish_truncate(
            states[index][0],
            states[index][1],
            states[1 - index][1],
            "result",
        )
        for index in range(2)
    ]
    assert np.array_equal(reconstruct(*outputs).view(np.int64), clear // 256)
    assert outputs[0].scale == 256

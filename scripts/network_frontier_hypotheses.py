"""Five bounded counterexample/cost gates for new whole-decoder hypotheses.

These are synthetic public fixtures, not a cryptographic implementation or
Qwen execution. No secret material, private text or live protocol is used.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np


def correction_side_information() -> dict:
    """Existing uniform output masks make corrections useless to an input codec."""
    q, views = 4, []
    for x in itertools.product(range(q), repeat=2):
        counts = np.zeros((q, q, q), dtype=np.int64)
        for r0, r1, s in itertools.product(range(q), repeat=3):
            correction = (r0 + 3 * r1 - s) % q
            counts[correction, (x[0] - r0) % q, (x[1] - r1) % q] += 1
        assert np.all(counts == 1)
        views.append(counts)
    assert all(np.array_equal(views[0], view) for view in views)
    return {"modulus": q, "private_inputs_checked": len(views), "mask_choices_per_input": q**3,
            "masked_input_bits": 4, "conditional_entropy_given_correction_bits": math.log2(q**2),
            "scope": "exhaustive two-coordinate uniform-mask toy; no worst-case lossless saving"}


def reusable_label_witness() -> dict:
    """Public XOR epoch refresh can be undone before comparing wire labels."""
    first = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
    second = [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
    label = lambda wire, bit: int.from_bytes(hashlib.sha256(f"test-only/{wire}/{bit}".encode()).digest()[:16], "big")
    delta = int.from_bytes(hashlib.sha256(b"public-test-epoch").digest()[:16], "big")
    observed = [(label(i, a) == (label(i, b) ^ delta) ^ delta)
                for i, (a, b) in enumerate(zip(first, second, strict=True))]
    expected = [a == b for a, b in zip(first, second, strict=True)]
    assert observed == expected and any(observed) and not all(observed)
    return {"wire_labels": len(first), "exact_private_equality_relations_learned": len(first),
            "label_bytes": 16, "public_epoch_refresh_prevents_linkage": False,
            "scope": "reuse/public-XOR-refresh counterexample, not an attack on reviewed reusable encodings"}


def composed_polynomial_budget() -> dict:
    """Dense shifted coefficients for x -> x^2 repeated L times over a large field.

    F(x)=x^(2^L); F(X-a) has 2^L+1 nonzero coefficients for a != 0 when
    p > 2^L. Each coefficient is represented by two opaque 64-bit shares.
    This prices this representation, not every possible function-sharing scheme.
    """
    p = 2**61 - 1
    records = []
    for layers in (4, 8, 16, 24):
        degree = 2**layers
        assert degree < p
        records.append({"layers": layers, "degree": degree,
                        "dense_shifted_coefficient_bytes_two_parties": (degree + 1) * 16})
    # Independently evaluate the first composition and its shifted expansion.
    degree, mask = 16, 17
    coefficients = [math.comb(degree, k) * pow(-mask, degree - k, p) % p for k in range(degree + 1)]
    for x in range(-16, 17):
        lifted = x + mask
        assert sum(c * pow(lifted, k, p) for k, c in enumerate(coefficients)) % p == pow(x, degree, p)
    return {"field_modulus": p, "records": records, "one_scalar_evaluation_only": True,
            "scope": "translated dense-polynomial representation; excludes attention, rounding and all other channels"}


def prime_rank(values: np.ndarray) -> int:
    """Exact rank over GF(257), an independent finite-table factorization gate."""
    a = np.asarray(values, dtype=np.int64).copy() % 257
    if a.ndim != 2 or not a.size or max(a.shape) > 128:
        raise ValueError("bounded nonempty finite table required")
    rank = 0
    for column in range(a.shape[1]):
        pivots = np.flatnonzero(a[rank:, column])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        a[[rank, pivot]] = a[[pivot, rank]]
        a[rank] = a[rank] * pow(int(a[rank, column]), -1, 257) % 257
        for row in range(rank + 1, len(a)):
            a[row] = (a[row] - a[row, column] * a[rank]) % 257
        rank += 1
        if rank == len(a):
            break
    return rank


def nonlinear_tensor_factorization() -> dict:
    """Factor the exact *nonlinear function table*, not a linear weight matrix."""
    x = np.array(list(itertools.product((-1, 1), repeat=8)), dtype=np.int64)
    gate = x @ np.array([3, -2, 7, 4, -6, 5, 1, -3]) / 8
    up = x @ np.array([-4, 6, 2, -1, 7, -5, 3, 4]) / 8
    table = np.rint(128 * gate / (1 + np.exp(-gate)) * up).astype(np.int64) % 257
    ranks = [1, *(prime_rank(table.reshape(2**cut, -1)) for cut in range(1, 8)), 1]
    elements = sum(a * 2 * b for a, b in zip(ranks[:-1], ranks[1:], strict=True))
    return {"input_bits": 8, "field_modulus": 257, "unfolding_ranks": ranks,
            "table_sha256": hashlib.sha256(table.astype("<u2").tobytes()).hexdigest(),
            "flat_public_table_bytes": table.size * 2, "minimal_dense_tensor_train_bytes": elements * 2,
            "scope": "exact ranks of a stored synthetic gated table; not a Qwen lower bound or private evaluator"}


def observational_state_quotient() -> dict:
    """Equal next tokens are insufficient to merge persistent decoder states."""
    def trace(state, steps):
        hidden, velocity = state
        result = []
        for _ in range(steps):
            result.append(int(hidden >= 0))
            hidden += velocity
        return tuple(result)

    a, b = (1, 1), (1, -2)
    assert trace(a, 1) == trace(b, 1) and trace(a, 2) != trace(b, 2)
    states = list(itertools.product(range(-3, 4), range(-2, 3)))
    return {"enumerated_states": len(states), "same_next_token_merger_is_sound": False,
            "finite_horizon_equivalence_classes": {str(n): len({trace(s, n) for s in states}) for n in (1, 2, 4, 8)},
            "scope": "public affine-state counterexample; universal future/input bisimulation remains unimplemented"}


def report() -> dict:
    return {"schema": "pllm.network_frontier_gates.v1", "complete": True,
            "scope": "five new hypotheses with bounded synthetic vetoes; no executable decoder",
            "all_link_100x_target_bytes": 178_970_558 // 100,
            "target_source": "docs/evidence/prepared-stage-attribution-qwen25-2026-09-29.json",
            "side_information_codec": correction_side_information(),
            "epoch_reusable_encoding": reusable_label_witness(),
            "whole_span_polynomial_sharing": composed_polynomial_budget(),
            "secret_index_tensor_network": nonlinear_tensor_factorization(),
            "private_observational_automaton": observational_state_quotient(),
            "whole_response_compute": None, "whole_client_peak_rss_bytes": None,
            "full_wire_bytes": None, "admitted_pipeline": False}

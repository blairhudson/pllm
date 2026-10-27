"""Unqualified source-only oracle definitions. UNRUN; no backend evidence.

Six independent oracle families below. Physical-layout helpers are PURE PYTHON
CONTRACT ORACLE DEFINITIONS, NOT actual Rust sparse-schedule qualification.
Python has no current interface observing that program or its materializer.
Rust tests also remain UNRUN. Future integration needs separate approval and an
actual plan/materializer interface, not a Rust source-text parser, copied receipt
or new binding here. No SEAL/TenSEAL import, keys or parameter adoption here.
Expected values never use production scale/planner helpers. Logical Python
workspace alone says nothing about evaluator pools, zero-copy, native output
atomicity, precision, security or ciphertext provenance.
"""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import hashlib
import math
from pathlib import Path

import pytest

from pllm.runtime import ckks_reference as ckks


# Algebra fixtures ONLY. Not CKKS parameters: these do not support N=8192.
# Small integers 17, 97, 113 are prime by trial division through their square
# roots; no claim these equal any installed context's coefficient moduli.
IDS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12))
MODULI = ((17, 97, 113), (17, 97), (17,))


def _context(delta=8.0, identity=b"Unqualified algebra fixture"):
    return ckks.ScaleContext(identity, delta, tuple(
        ckks.ScaleLevel(parms_id, 2 - index, moduli)
        for index, (parms_id, moduli) in enumerate(zip(IDS, MODULI))))


def _states(delta=8.0):
    context = _context(delta)
    initial = context.initial_state(IDS[0], delta)
    # Expected metadata deliberately computed without production scale helpers.
    first_float = (delta * delta) / float(113)
    second_float = (first_float * delta) / float(97)
    product = initial.multiply_delta_rescale(IDS[1], first_float)
    # Generic four-child metadata publisher control, NOT four physical masks
    # per displacement. The physical contract's 14 masks are counted separately.
    branches = ckks.publish_product_scale_branches(
        product, ((IDS[2], second_float),) * 4)
    return initial, product, branches


# Oracle 1: exact rational arithmetic, separate from ordered binary64.
@pytest.mark.parametrize("delta", [8.0, 1.1, float(2**40)])
def test_exact_fraction_alpha_beta_and_bias(delta):
    initial, product, branches = _states(delta)
    numerator, denominator = delta.as_integer_ratio()
    exact_delta = Fraction(numerator, denominator)
    assert initial.exact_scale == exact_delta
    assert product.exact_scale == exact_delta**2 / 113
    assert product.expression == ckks.ScaleExpression(2, (113,))
    for child in (branches.support, branches.wrap, branches.remainder,
                  branches.a0_full_support):
        assert child.exact_scale == exact_delta**3 / (113 * 97)
        assert child.expression.delta_power == 3
        assert child.expression.dropped_primes == (113, 97)
        assert child.parms_id == IDS[2]
        branches.support.require_add_compatible(child)
    bias = branches.bias_encoding_state()
    assert bias.parms_id == IDS[2]
    assert bias.actual_scale == ((delta * delta) / float(113) * delta) / float(97)
    assert bias.actual_scale != delta  # No encode-at-Delta then retag.
    assert branches.a0_full_support is not branches.support
    assert branches.a0_full_support is not branches.wrap
    assert branches.a0_full_support is not branches.remainder


# Oracle 2: public SEAL operation order, NOT float(exact rational expression).
@pytest.mark.parametrize("delta", [8.0, 1.1, float(2**40)])
def test_ordered_float_is_distinct_from_exact_oracle(delta):
    _, product, branches = _states(delta)
    multiplied = delta * delta
    alpha = multiplied / float(113)
    masked = alpha * delta
    beta = masked / float(97)
    assert product.actual_scale.hex() == alpha.hex()
    assert branches.support.actual_scale.hex() == beta.hex()
    # 8**2/113 and 8**3/(113*97) are non-dyadic rationals. Binary64 cannot
    # represent them exactly even when backend's metadata equals its predictor.
    if delta == 8.0:
        assert Fraction.from_float(alpha) != Fraction(64, 113)
        assert Fraction.from_float(beta) != Fraction(512, 10961)


def _dense_physical_contract_definition(x, weights, bias, tokens):
    """Independent Y=XW+b, explicit B=X.T, A=W.T, C=AB=Y.T before bias.

    Two 4096-slot output frames, virtual H=4. No sparse displacement, input
    packing, mask or rotation helper participates in this dense control.
    """
    din, dout = len(weights), len(bias)
    linear_y = [[sum(x[t][f] * weights[f][o] for f in range(din))
                 for o in range(dout)] for t in range(tokens)]
    y = [[linear_y[t][o] + bias[o] for o in range(dout)] for t in range(tokens)]
    b_matrix = [[x[t][f] for t in range(tokens)] for f in range(din)]
    a_matrix = [[weights[f][o] for f in range(din)] for o in range(dout)]
    c_matrix = [[sum(a_matrix[o][f] * b_matrix[f][t] for f in range(din))
                 for t in range(tokens)] for o in range(dout)]
    # For T=0, B has Din empty columns-as-rows, A remains fully defined, and C
    # has Dout empty rows. This is mathematical emptiness, NOT native admission.
    assert len(b_matrix) == din and len(a_matrix) == dout and len(c_matrix) == dout
    for f in range(din):
        for t in range(tokens):
            assert b_matrix[f][t] == x[t][f]
    for o in range(dout):
        for f in range(din):
            assert a_matrix[o][f] == weights[f][o]
        for t in range(tokens):
            assert c_matrix[o][t] == linear_y[t][o]  # AB=(XW).T exactly.
            assert c_matrix[o][t] + bias[o] == y[t][o]
    frames = [[0] * 4096 for _ in range(2)]
    for t in range(tokens):
        for o in range(dout):
            q = (o % 2 - t) % 2
            frames[q][4 * t + o // 2] = y[t][o]
    return frames


def _contract_rotate_definition(values, step):
    """R_r(v)[s]=v[(s+r)%4096], pure contract definition, not backend call."""
    residue = step % 4096
    return values[residue:] + values[:residue]


def _contract_branch_definition(original_product, displacement):
    """Contract oracle: F, M_a and F-M_a; preserve ORIGINAL P for both masks."""
    support = [int(slot < 8) for slot in range(4096)]
    if displacement == 0:
        a0 = [f * p for f, p in zip(support, original_product)]
        return a0, None, None
    mask = [f * int(slot % 4 >= 4 - displacement)
            for slot, f in enumerate(support)]
    wrap = [m * p for m, p in zip(mask, original_product)]
    remainder = [(f - m) * p for f, m, p in zip(support, mask, original_product)]
    left = _contract_rotate_definition(wrap, 4 - displacement)
    right = _contract_rotate_definition(remainder, -displacement)
    aligned = [w + r for w, r in zip(left, right)]
    return aligned, wrap, remainder


def _sparse_physical_contract_definition(x, weights, bias, tokens):
    """Unqualified PURE PYTHON CONTRACT oracle, not actual Rust schedule.

    I_j[4t+k]=X[t,2k+(t+j)%2]. V_qaj uses i=(k+a)%4, then
    W[2k+(t+j)%2,2i+(t+q)%2], with all token/feature padding zero.
    Original P_qa=I_0*V_qa0+I_1*V_qa1 survives both independent masks.
    Bias is packed ONLY in post-alignment output coordinates.
    """
    din, dout = len(weights), len(bias)
    inputs = [[0] * 4096 for _ in range(2)]
    for j in range(2):
        for t in range(tokens):
            for k in range(din // 2):
                inputs[j][4 * t + k] = x[t][2 * k + (t + j) % 2]
    frames = [[0] * 4096 for _ in range(2)]
    packed_weights, products, aligned_terms = {}, {}, {}
    for q in range(2):
        for a in range(4):
            for j in range(2):
                v = [0] * 4096
                for t in range(tokens):
                    for k in range(din // 2):
                        i = (k + a) % 4
                        if i < dout // 2:
                            v[4 * t + k] = weights[2 * k + (t + j) % 2][
                                2 * i + (t + q) % 2]
                packed_weights[q, a, j] = v
            original = [i0 * v0 + i1 * v1 for i0, v0, i1, v1 in zip(
                inputs[0], packed_weights[q, a, 0], inputs[1], packed_weights[q, a, 1])]
            products[q, a] = original
            aligned, _, _ = _contract_branch_definition(original, a)
            aligned_terms[q, a] = aligned
            frames[q] = [acc + term for acc, term in zip(frames[q], aligned)]
        for t in range(tokens):
            for i in range(dout // 2):
                frames[q][4 * t + i] += bias[2 * i + (t + q) % 2]
    return frames, inputs, packed_weights, products, aligned_terms


# Oracle 3: physical two-frame CONTRACT, both rectangular directions, ALL lanes.
@pytest.mark.parametrize("din,dout", [(4, 8), (8, 4)])
@pytest.mark.parametrize("tokens", [0, 1, 2])
def test_physical_contract_dense_vs_sparse_two_frames(din, dout, tokens):
    # Signed, nonsymmetric values distinguish transpose, token and frame mistakes.
    x = [[(-1 if (t + f) % 2 else 1) * (11 * t + 3 * f + 2)
          for f in range(din)] for t in range(tokens)]
    weights = [[(-1 if (f + o) % 2 else 1) * (13 * f - 7 * o + 5)
                for o in range(dout)] for f in range(din)]
    bias = [(-1 if o % 2 else 1) * (o + 2) for o in range(dout)]
    expected = _dense_physical_contract_definition(x, weights, bias, tokens)
    actual, inputs, packed, _, _ = _sparse_physical_contract_definition(
        x, weights, bias, tokens)
    assert actual == expected  # Both complete 4096-slot frames, no active-only cut.
    assert len(actual) == 2 and all(len(frame) == 4096 for frame in actual)
    for q in range(2):
        for slot in range(4096):
            t, i = divmod(slot, 4)
            if t >= tokens or i >= dout // 2:
                assert actual[q][slot] == 0
            if t >= tokens or i >= din // 2:
                assert inputs[q][slot] == 0
            for a in range(4):
                for j in range(2):
                    if t >= tokens or i >= din // 2 or (i + a) % 4 >= dout // 2:
                        assert packed[q, a, j][slot] == 0

    zero_x = [[0] * din for _ in range(tokens)]
    zero_weights = [[0] * dout for _ in range(din)]
    bias_only, *_ = _sparse_physical_contract_definition(zero_x, zero_weights, bias, tokens)
    assert bias_only == _dense_physical_contract_definition(zero_x, zero_weights, bias, tokens)
    for o in range(dout):
        basis_bias = [0] * dout
        basis_bias[o] = 1
        bias_result, *_ = _sparse_physical_contract_definition(
            zero_x, zero_weights, basis_bias, tokens)
        bias_route = [[0] * 4096 for _ in range(2)]
        for t in range(tokens):
            bias_route[(o % 2 - t) % 2][4 * t + o // 2] = 1
        assert bias_result == bias_route

    # Exhaustive Cartesian one-hot input AND one-hot weight routes, including
    # mismatched feature pairs that must yield zero. Expected addresses come
    # directly from (t,f,o), not from sparse masks/rotations/packing helpers.
    for t in range(tokens):
        for f in range(din):
            basis_x = [[0] * din for _ in range(tokens)]
            basis_x[t][f] = 1
            for g in range(din):
                for o in range(dout):
                    basis_w = [[0] * dout for _ in range(din)]
                    basis_w[g][o] = 1
                    result, basis_inputs, basis_v, basis_p, basis_a = (
                        _sparse_physical_contract_definition(
                            basis_x, basis_w, [0] * dout, tokens))
                    route = [[0] * 4096 for _ in range(2)]
                    q = (o % 2 - t) % 2
                    target = 4 * t + o // 2
                    if f == g:
                        route[q][target] = 1
                        j = (f % 2 - t) % 2
                        k = f // 2
                        a = (o // 2 - k) % 4
                        origin = 4 * t + k
                        assert basis_inputs[j][origin] == 1
                        assert basis_v[q, a, j][origin] == 1
                        assert basis_p[q, a][origin] == 1
                        assert basis_a[q, a][target] == 1
                    assert result == route
                    assert result == _dense_physical_contract_definition(
                        basis_x, basis_w, [0] * dout, tokens)


# Oracle 4: signed rotations, residues/elements and ORIGINAL-P mask branches.
def test_signed_rotation_manifest_and_all_slot_wrap_definition():
    expected = ((-3, 4093, 6675), (-2, 4094, 3641), (-1, 4095, 10923),
                (1, 1, 3), (2, 2, 9), (3, 3, 27))
    assert ckks.SIGNED_ROTATION_MANIFEST == expected
    slots = list(range(4096))
    for step, residue, element in expected:
        assert residue == step % 4096
        # Independent modular equations verify literal elements, including inverse
        # powers. No production generator helper supplies expected Galois values.
        assert 0 < element < 16384 and element % 2 == 1
        if step < 0:
            assert element * 3**(-step) % 16384 == 1
        else:
            assert element == 3**step % 16384
        rotated = slots[residue:] + slots[:residue]
        for target, value in enumerate(rotated):
            assert value == (target + step) % 4096
        assert rotated[0] == step % 4096
        assert rotated[-1] == (4095 + step) % 4096


def test_physical_contract_support_and_branch_origins():
    # Every tail origin is adversarially nonzero, including origins just outside
    # F that an unbounded complement or unmasked a0 would incorrectly admit.
    product = [(-1 if slot % 2 else 1) * (slot + 1) for slot in range(4096)]
    original = tuple(product)
    f_mask = [int(slot < 8) for slot in range(4096)]
    a0, wrap, remainder = _contract_branch_definition(product, 0)
    assert wrap is None and remainder is None
    assert a0 == [product[slot] if slot < 8 else 0 for slot in range(4096)]
    assert a0 != product  # a0=P is WRONG, including at displacement zero.
    assert all(value == 0 for value in a0[8:])
    for a in (1, 2, 3):
        m_mask = [int(slot < 8 and slot % 4 >= 4 - a) for slot in range(4096)]
        r_mask = [f_mask[slot] - m_mask[slot] for slot in range(4096)]
        assert set(m_mask) == set(r_mask) == {0, 1}
        assert sum(m_mask) == 2 * a and sum(r_mask) == 8 - 2 * a
        aligned, masked_wrap, masked_remainder = _contract_branch_definition(product, a)
        assert masked_wrap == [m_mask[slot] * product[slot] for slot in range(4096)]
        assert masked_remainder == [r_mask[slot] * product[slot] for slot in range(4096)]
        assert any(masked_wrap) and any(masked_remainder)
        assert all(value == 0 for value in masked_wrap[8:] + masked_remainder[8:])
        expected = [0] * 4096
        # Independent SCATTER route, unlike the helper's two GATHER rotations.
        for origin in range(8):
            block, lane = divmod(origin, 4)
            destination = 4 * block + (lane + a) % 4
            expected[destination] = product[origin]
        assert aligned == expected
        assert all(value == 0 for value in aligned[8:])

        # Bad global 1-M keeps ALL off-support origins. Not F-M.
        bad_global_remainder = [(1 - m_mask[s]) * product[s] for s in range(4096)]
        assert bad_global_remainder != masked_remainder
        assert all(bad_global_remainder[s] != 0 for s in range(8, 4096))
        bad_global_aligned = [w + r for w, r in zip(
            _contract_rotate_definition(masked_wrap, 4 - a),
            _contract_rotate_definition(bad_global_remainder, -a))]
        assert bad_global_aligned != expected
        # Negative chained references: mutating P to either masked branch before
        # deriving the other kills a branch that should be nonzero.
        chained_remainder = [r_mask[s] * masked_wrap[s] for s in range(4096)]
        chained_wrap = [m_mask[s] * masked_remainder[s] for s in range(4096)]
        assert not any(chained_remainder) and not any(chained_wrap)
        assert chained_remainder != masked_remainder
        assert chained_wrap != masked_wrap
        wrong = [w + r for w, r in zip(
            _contract_rotate_definition(masked_wrap, 4 - a),
            _contract_rotate_definition(chained_remainder, -a))]
        assert wrong != expected
        assert tuple(product) == original

    # Exhaustive displacement routes, derived as block-local destination scatter.
    # Each assertion spans all 4096 slots, so wrong signs/global wrap cannot hide.
    for a in range(4):
        for origin in range(8):
            impulse = [0] * 4096
            impulse[origin] = 1
            actual, _, _ = _contract_branch_definition(impulse, a)
            expected = [0] * 4096
            block, lane = divmod(origin, 4)
            expected[4 * block + (lane + a) % 4] = 1
            assert actual == expected
        for origin in range(8, 4096):
            # Off-F origins are excluded BEFORE either rotation, not admitted
            # merely because their rotated destination might fall within F.
            assert f_mask[origin] == 0


def test_physical_contract_scale_order_per_frame_displacement():
    """Declarative scale CONTRACT trace, not observed native masks/ciphertexts.

    Runtime states validate this metadata only. They prove neither actual mask
    encoding nor ciphertext observations. Four-child publisher is a generic
    metadata API: NOT four masks per displacement in this physical contract.
    """
    from collections import Counter

    delta = 8.0
    context = _context(delta)
    initial = context.initial_state(IDS[0], delta)
    alpha = (delta * delta) / 113.0
    beta = (alpha * delta) / 97.0
    trace = []
    for q in range(2):
        accumulator = None
        for a in range(4):
            weight_metadata = []
            for j in range(2):
                weight_metadata.append((IDS[0], delta * delta))
                trace.append(("weight_product", q, a, j))
            # Same level/scale BEFORE summing the two weight products, followed
            # by ONE partial rescale to alpha, not one rescale per product.
            assert weight_metadata[0] == weight_metadata[1] == (IDS[0], 64.0)
            trace.append(("weight_sum", q, a))
            original_product = initial.multiply_delta_rescale(IDS[1], alpha)
            trace.append(("partial_rescale", q, a))
            assert original_product.expression.dropped_primes == (113,)
            assert original_product.exact_scale == Fraction(64, 113)
            saved = (original_product.level, original_product.expression,
                     original_product.actual_scale)
            if a == 0:
                # Separate F*P branch for a0, never the unmasked P itself.
                term = original_product.multiply_delta_rescale(IDS[2], beta)
                trace.append(("mask_product_rescale", q, a, "full_support"))
            else:
                wrap = original_product.multiply_delta_rescale(IDS[2], beta)
                trace.append(("mask_product_rescale", q, a, "wrap"))
                remainder = original_product.multiply_delta_rescale(IDS[2], beta)
                trace.append(("mask_product_rescale", q, a, "F-M"))
                # Compatibility check MUST precede alignment/branch addition.
                before_alignment = tuple(trace)
                with pytest.raises(ValueError):
                    wrap.require_add_compatible(original_product)
                assert tuple(trace) == before_alignment
                wrap.require_add_compatible(remainder)
                trace.append(("branch_compatibility", q, a))
                trace.append(("rotation", q, a, 4 - a))
                trace.append(("rotation", q, a, -a))
                trace.append(("branch_add", q, a))
                term = wrap  # Metadata unchanged by rotation/add, not execution.
                assert wrap is not remainder
            assert (original_product.level, original_product.expression,
                    original_product.actual_scale) == saved
            assert term.level == 2 and term.parms_id == IDS[2]
            assert term.expression.dropped_primes == (113, 97)
            assert term.actual_scale == beta
            assert term.exact_scale == Fraction(512, 113 * 97)
            with pytest.raises(ValueError):
                original_product.require_add_compatible(term)
            if accumulator is None:
                accumulator = term
            else:
                accumulator.require_add_compatible(term)
                trace.append(("accumulation_compatibility", q, a))
                trace.append(("accumulate", q, a))
        # Encode bias at resulting beta AFTER all four aligned terms accumulate.
        bias = ckks.ScaleState(context, 2, ckks.ScaleExpression(3, (113, 97)), beta)
        accumulator.require_add_compatible(bias)
        trace.append(("bias_encode_beta", q))
        trace.append(("bias_add", q))
        assert bias.actual_scale != delta and bias.parms_id == IDS[2]

    counts = Counter(event[0] for event in trace)
    assert counts["weight_product"] == 16
    assert counts["partial_rescale"] == 8
    assert counts["mask_product_rescale"] == 14
    assert counts["rotation"] == 12
    assert counts["weight_product"] + counts["mask_product_rescale"] == 30
    assert counts["partial_rescale"] + counts["mask_product_rescale"] == 22
    for q in range(2):
        for a in (1, 2, 3):
            gate = trace.index(("branch_compatibility", q, a))
            assert gate < trace.index(("rotation", q, a, 4 - a))
            assert gate < trace.index(("rotation", q, a, -a))
            assert gate < trace.index(("branch_add", q, a))
            assert trace.index(("accumulation_compatibility", q, a)) < trace.index(
                ("accumulate", q, a))
        assert trace.index(("accumulate", q, 3)) < trace.index(("bias_encode_beta", q))
        assert trace.index(("bias_encode_beta", q)) < trace.index(("bias_add", q))


def _rounding_witness_context(identity=b"Unqualified synthetic rounding witness"):
    # NONCERTIFIED odd modulus, not an asserted prime or qualified CKKS encoding
    # parameter. Delta=1 here is a rounding witness, NOT a scale-one mask workaround.
    p = 2**54 + 3
    moduli = ((17, 97, p), (17, 97), (17,))
    return ckks.ScaleContext(identity, 1.0, tuple(
        ckks.ScaleLevel(parms_id, 2 - index, limbs)
        for index, (parms_id, limbs) in enumerate(zip(IDS, moduli))))


def test_exact_rational_vs_ordered_float_rounding_witness():
    """Exact, predetermined binary64 witness; no search or parameter selection."""
    p = 2**54 + 3
    q = 2**54 + 4
    x = Fraction(1, 2**54)
    u = Fraction(1, 2**107)
    rational = Fraction(1, p)
    rounded_modulus_reciprocal = Fraction(1, q)
    assert x - 3 * u / 2 < rational < x - u
    assert x - 2 * u < rounded_modulus_reciprocal < x - 3 * u / 2
    # Round-to-nearest/ties-to-even binary64 rule; p itself is closer to Q,
    # not a midpoint tie (distance 1 to Q, distance 3 to 2**54).
    assert q - p == 1 and p - 2**54 == 3
    assert float(p).hex() == "0x1.0000000000001p+54"
    exact_rounded = float.fromhex("0x1.fffffffffffffp-55")  # x-u
    ordered = float.fromhex("0x1.ffffffffffffep-55")  # x-2u
    assert Fraction.from_float(exact_rounded) == x - u
    assert Fraction.from_float(ordered) == x - 2 * u
    assert float(rational).hex() == "0x1.fffffffffffffp-55"
    assert (1.0 / float(p)).hex() == "0x1.ffffffffffffep-55"
    assert exact_rounded != ordered
    context = _rounding_witness_context()
    initial = context.initial_state(IDS[0], 1.0)
    alpha = initial.multiply_delta_rescale(IDS[1], ordered)
    assert alpha.exact_scale == rational
    assert alpha.actual_scale.hex() == "0x1.ffffffffffffep-55"
    with pytest.raises(ValueError):
        initial.multiply_delta_rescale(IDS[1], exact_rounded)


def test_adjacent_scale_refusal():
    context = _rounding_witness_context()
    initial = context.initial_state(IDS[0], 1.0)
    identical_initial = _rounding_witness_context().initial_state(IDS[0], 1.0)
    initial.require_add_compatible(identical_initial)
    for direction in (-math.inf, math.inf):
        with pytest.raises(ValueError):
            context.initial_state(IDS[0], math.nextafter(1.0, direction))
    ordered_alpha = float.fromhex("0x1.ffffffffffffep-55")
    ordered_beta = ordered_alpha / 97.0
    product = initial.multiply_delta_rescale(IDS[1], ordered_alpha)
    for parent, next_id, expected in ((initial, IDS[1], ordered_alpha),
                                      (product, IDS[2], ordered_beta)):
        accepted = parent.multiply_delta_rescale(next_id, expected)
        # Independently constructed, identical state/context records are allowed.
        independent = ckks.ScaleState(_rounding_witness_context(), accepted.level,
                                      ckks.ScaleExpression(accepted.expression.delta_power,
                                                           accepted.expression.dropped_primes),
                                      expected)
        accepted.require_add_compatible(independent)
        foreign = ckks.ScaleState(_rounding_witness_context(b"foreign"), accepted.level,
                                  independent.expression, expected)
        with pytest.raises(ValueError):
            accepted.require_add_compatible(foreign)
        saved = (parent.level, parent.expression, parent.actual_scale)
        for direction in (-math.inf, math.inf):
            published = None
            with pytest.raises(ValueError):
                published = parent.multiply_delta_rescale(next_id, math.nextafter(expected, direction))
            assert published is None
            assert (parent.level, parent.expression, parent.actual_scale) == saved


# Oracle 5: refusal, immutable parents and no partial metadata publication.
@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -float("inf"),
                                   0.0, -1.0, True, 8, "8"])
def test_bad_scales_rejected(scale):
    with pytest.raises(ValueError):
        _context(scale)
    context = _context()
    with pytest.raises(ValueError):
        context.initial_state(IDS[0], scale)


@pytest.mark.parametrize("changes", [
    {"parms_id": (0, 0, 0, 0)}, {"parms_id": (True, 2, 3, 4)},
    {"parms_id": (2**64, 2, 3, 4)}, {"parms_id": [1, 2, 3, 4]},
    {"chain_index": True}, {"chain_index": -1},
    {"coefficient_moduli": ()}, {"coefficient_moduli": (17, 17)},
    {"coefficient_moduli": (17, 98)}, {"coefficient_moduli": (17, True)},
    {"coefficient_moduli": (17, 2**64 + 1)},
])
def test_malformed_level_metadata(changes):
    with pytest.raises(ValueError):
        replace(_context().levels[0], **changes)


def test_malformed_chains_and_symbolic_states():
    context = _context()
    for levels in (context.levels[:2], context.levels[::-1],
                   (context.levels[0], context.levels[2]),
                   (context.levels[0],) * 3,
                   (context.levels[0], replace(context.levels[1], coefficient_moduli=(17, 113)),
                    context.levels[2])):
        with pytest.raises(ValueError):
            replace(context, levels=levels)
    for power, divisors in ((0, ()), (True, ()), (3, (113,)),
                            (2, (float("nan"),)), (2, [113])):
        with pytest.raises(ValueError):
            ckks.ScaleExpression(power, divisors)
    initial, product, branches = _states()
    for changes in ({"level": True}, {"level": -1}, {"level": 2},
                    {"expression": ckks.ScaleExpression(2, (97,))},
                    {"actual_scale": product.actual_scale * (1.0 + 2**-40)}):
        with pytest.raises(ValueError):
            replace(product, **changes)
    with pytest.raises(ValueError):
        initial.require_add_compatible(product)
    foreign = replace(context, identity=b"other context")
    other_initial = foreign.initial_state(IDS[0], 8.0)
    other = other_initial.multiply_delta_rescale(IDS[1], (8.0 * 8.0) / 113.0)
    with pytest.raises(ValueError):
        product.require_add_compatible(other)
    # Same numeric metadata cannot erase symbolic Delta/context distinctions.
    with pytest.raises(ValueError):
        product.require_add_compatible(None)
    with pytest.raises(FrozenInstanceError):
        branches.support.actual_scale = 8.0


def test_failed_branch_publication_is_atomic_and_cannot_chain_masks():
    initial, product, branches = _states()
    saved = (initial, product, branches)
    beta = (((8.0 * 8.0) / 113.0) * 8.0) / 97.0
    good = (IDS[2], beta)
    for bad in ((IDS[1], beta), (IDS[0], beta), (IDS[2], 8.0),
                (IDS[2], float("nan")), (IDS[2], float("inf")),
                (IDS[2], beta * (1 + 2**-40)), ([9, 10, 11, 12], beta)):
        for position in range(4):
            observations = tuple(bad if index == position else good for index in range(4))
            published = None
            with pytest.raises(ValueError):
                published = ckks.publish_product_scale_branches(product, observations)
            assert published is None
            assert (initial, product, branches) == saved
    with pytest.raises(ValueError):
        ckks.publish_product_scale_branches(branches.support, (good,) * 4)
    with pytest.raises(ValueError):
        ckks.ProductScaleBranches(product, branches.support, branches.wrap,
                                  branches.remainder, product)
    for observations in ((), (good,) * 3, [good] * 4, (None,) * 4):
        with pytest.raises(ValueError):
            ckks.publish_product_scale_branches(product, observations)
    for delta in (1e200, 1e-200):
        context = _context(delta)
        parent = context.initial_state(IDS[0], delta)
        with pytest.raises(ValueError):
            parent.multiply_delta_rescale(IDS[1], delta)
        assert parent.actual_scale == delta


# Oracle 6: legacy implementation bytes unchanged, plus its normalization policy.
def test_legacy_body_unchanged_source_control():
    source = Path(ckks.__file__).read_text()
    legacy = "HEADER_BYTES = 81" + source.split("HEADER_BYTES = 81", 1)[1].split(
        "# Unqualified, source-only metadata contract.", 1)[0].rstrip() + "\n"
    # SHA256 of original HEAD body from HEADER_BYTES through diagonal_linear.
    assert hashlib.sha256(legacy.encode()).hexdigest() == (
        "8a86749e2ad2dd3165bfdf316a9f3034e8a011b2eafbe0b1fbb8963f14e6442b")
    assert ckks.SCALE_STATE_QUALIFICATION == "Unqualified"


def test_legacy_normalization_stays_separate_from_typed_states():
    from collections import Counter
    from types import SimpleNamespace

    result = SimpleNamespace(scale=float(2**40) * (1 + 2**-20))
    evaluator = SimpleNamespace(rescale_to_next=lambda source, destination: None)
    arithmetic = object.__new__(ckks.Arithmetic)
    arithmetic.seal = SimpleNamespace(Ciphertext=lambda: result)
    arithmetic.evaluator = evaluator
    arithmetic.scale = float(2**40)
    arithmetic.minimum_remaining_depth = 2
    arithmetic.level = lambda ciphertext: 1
    arithmetic.counts = Counter()
    assert arithmetic._rescale(object()) is result
    assert result.scale == float(2**40)
    assert arithmetic.counts["scale_normalizations"] == 1
    assert arithmetic.minimum_remaining_depth == 1
    # Legacy setup still refuses negative steps before touching any backend.
    with pytest.raises(ValueError, match="session geometry"):
        ckks.setup(None, None, None, depth=2, rotation_steps=(-1,),
                   relinearization=False, live_ciphertexts=1, queries=1,
                   binding=bytes(32))

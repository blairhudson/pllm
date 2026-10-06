"""Independent, bounded algebra oracles for ten non-selectable hypotheses.

All randomness is deterministic PUBLIC TEST DATA. These functions do not issue
keys, implement secure transports, or authorize a decoder numeric substitution.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def solve_field(matrix, target, prime=257):
    """Return one solution of A x = b, or None; bounded exact prime-field oracle."""
    a = np.asarray(matrix, dtype=np.int64)
    b = np.asarray(target, dtype=np.int64)
    if (
        prime not in (17, 61, 127, 257, 65537)
        or a.ndim != 2
        or not a.size
        or max(a.shape) > 512
        or b.shape != (a.shape[0],)
    ):
        raise ValueError("bounded prime-field system required")
    work = np.column_stack((a % prime, b % prime))
    pivots = []
    for column in range(a.shape[1]):
        choices = np.flatnonzero(work[len(pivots) :, column])
        if not len(choices):
            continue
        row = len(pivots)
        pivot = row + int(choices[0])
        work[[row, pivot]] = work[[pivot, row]]
        work[row] = work[row] * pow(int(work[row, column]), -1, prime) % prime
        factors = work[:, column].copy()
        factors[row] = 0
        work = (work - factors[:, None] * work[row]) % prime
        pivots.append(column)
        if len(pivots) == len(work):
            break
    if np.any(np.all(work[:, :-1] == 0, axis=1) & (work[:, -1] != 0)):
        return None
    result = np.zeros(a.shape[1], np.int64)
    result[pivots] = work[: len(pivots), -1]
    assert np.array_equal(a @ result % prime, b % prime)
    return result


def sparse_syndrome_decode(syndrome, width=32, prime=257):
    """Decode ONLY the promised at-most-one-nonzero domain, not arbitrary words."""
    if not 1 <= width < prime or prime != 257:
        raise ValueError("invalid sparse domain")
    total, moment = (int(x) % prime for x in syndrome)
    result = np.zeros(width, np.int64)
    if total == 0:
        if moment:
            raise ValueError("syndrome outside one-sparse domain")
        return result
    position = moment * pow(total, -1, prime) % prime
    if position >= width:
        raise ValueError("syndrome outside one-sparse domain")
    result[position] = total
    return result


def coset_oracle():
    """H(y+s)=H(z) works, but cannot certify its own sparse-domain promise."""
    p, n = 257, 32
    h = np.vstack((np.ones(n, np.int64), np.arange(n)))
    rng = np.random.default_rng(4901)
    checked = 0
    for position in range(n):
        for value in range(1, p):
            z = np.zeros(n, np.int64)
            z[position] = value
            mask = rng.integers(0, p, n, dtype=np.int64)
            encoded = h @ ((z - mask) % p) % p
            decoded = sparse_syndrome_decode((encoded + h @ mask) % p)
            assert np.array_equal(z, decoded)
            checked += 1
    outside = np.zeros(n, np.int64)
    outside[[0, 2]] = 1
    alias = sparse_syndrome_decode(h @ outside % p)
    assert not np.array_equal(alias, outside)
    assert np.array_equal(h @ alias % p, h @ outside % p)
    return {
        "width": n,
        "field": p,
        "nonzero_domain_words_checked": checked,
        "domain_symbols": n,
        "syndrome_symbols": 2,
        "component_symbol_reduction": n / 2,
        "two_sparse_aliases_one_sparse": True,
        "domain_membership_certified_by_syndrome": False,
        "scope": "output-side mask side information; promise-domain algebra, not a secure decoder",
    }


def span_oracle():
    """Previously acquired integer results span new inputs without new weights."""
    rng, p = np.random.default_rng(4902), 257
    basis = rng.integers(0, p, (3, 16), dtype=np.int64)
    weights = rng.integers(0, p, (7, 16), dtype=np.int64)
    coefficients = rng.integers(0, p, (32, 3), dtype=np.int64)
    inputs = coefficients @ basis % p
    saved_outputs = basis @ weights.T % p
    for x in inputs:
        recovered = solve_field(basis.T, x)
        assert recovered is not None
        assert np.array_equal(recovered @ saved_outputs % p, x @ weights.T % p)
    return {
        "rows": 32,
        "basis_rows": 3,
        "input_width": 16,
        "output_width": 7,
        "exact_field_outputs": True,
        "client_basis_and_outputs_bytes": (3 * 16 + 3 * 7) * 2,
        "scope": "public fixed-span toy; full-width domain certificate and private miss handling absent",
    }


def silu32(values):
    """Independent copy of the declared stable float32 arithmetic oracle."""
    x = np.asarray(values, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = x[pos] / (np.float32(1) + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = x[~pos] * e / (np.float32(1) + e)
    return out


def tropical_silu(values):
    x = np.asarray(values, dtype=np.float32)
    return np.where(x >= 0, x, np.copysign(np.float32(0), x))


def rational_silu(values, terms=24):
    """Euler tanh continued fraction via continuants; PUBLIC fixed [-32,32] bound.

    One final division, not a division at each level. Float64 fitting/evaluation
    is a distinct numeric candidate, not float32 exp equivalence or a field port.
    """
    x = np.asarray(values, dtype=np.float32)
    if type(terms) is not int or not 2 <= terms <= 32:
        raise ValueError("invalid continued-fraction depth")
    if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 32):
        raise ValueError("outside public rational domain")
    z = x.astype(np.float64) / 2
    squared = z * z
    a = np.full(x.shape, 2 * terms - 1, np.float64)
    b = np.ones(x.shape, np.float64)
    for index in range(terms - 2, -1, -1):
        a, b = (2 * index + 1) * a + squared * b, a
    return (x.astype(np.float64) * (0.5 + 0.5 * z * b / a)).astype(np.float32)


def rational_silu_tails(values, terms=24):
    """Distinct piecewise numeric candidate, with PUBLIC fixed identity/zero tails.

    This is explicit approximation outside [-32,32], never an implicit fallback
    for rational_silu. The negative-tail real SiLU magnitude is <=32*exp(-32).
    Protected interval selection and full float semantics are not implemented.
    """
    x = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("finite rational-tail input required")
    inside = np.abs(x) <= 32
    result = tropical_silu(x)
    result[inside] = rational_silu(x[inside], terms)
    return result


def batch_inverse(values, prime=65537):
    """Montgomery simultaneous inversion: one inversion and 3(n-1) products."""
    values = np.asarray(values, dtype=np.int64)
    if (
        prime != 65537
        or values.ndim != 1
        or not 1 <= len(values) <= 8192
        or np.any(values <= 0)
        or np.any(values >= prime)
    ):
        raise ValueError("bounded nonzero field elements required")
    prefix = np.empty_like(values)
    prefix[0] = values[0]
    for i in range(1, len(values)):
        prefix[i] = int(prefix[i - 1]) * int(values[i]) % prime
    accumulator = pow(int(prefix[-1]), -1, prime)
    result = np.empty_like(values)
    for i in range(len(values) - 1, 0, -1):
        result[i] = accumulator * int(prefix[i - 1]) % prime
        accumulator = accumulator * int(values[i]) % prime
    result[0] = accumulator
    return result


def masked_inverse_oracle():
    """Alternative algebra: open d*r for fresh nonzero shared r, then scale r.

    This prices one secret product plus one opening per denominator, rather than
    treating Montgomery's clear-arithmetic optimization as optimal MPC. The
    bounded check holds both inputs locally; it is not a distributed protocol.
    """
    p = 65537
    masks = np.arange(1, p, dtype=np.int64)
    for denominator in (1, 3137, p - 1):
        opened = denominator * masks % p
        assert len(np.unique(opened)) == p - 1
    denominators = np.arange(1, 4097, dtype=np.int64)
    random_masks = np.random.default_rng(4907).integers(1, p, len(denominators), dtype=np.int64)
    opened = denominators * random_masks % p
    recovered = np.array([pow(int(x), -1, p) for x in opened]) * random_masks % p
    assert np.all(recovered * denominators % p == 1)
    return {
        "field": p,
        "denominators_checked": len(denominators),
        "exhaustive_nonzero_mask_views_per_checked_denominator": p - 1,
        "nonzero_domain_only": True,
        "scope": "local algebra; fresh shared nonzero masks, secret multiplication and transport absent",
    }


def interpolate(points, values, at, prime=257):
    if len(points) != len(values) or len(set(points)) != len(points) or not points:
        raise ValueError("distinct interpolation points required")
    answer = 0
    for i, (x, y) in enumerate(zip(points, values, strict=True)):
        numerator, denominator = 1, 1
        for j, other in enumerate(points):
            if i != j:
                numerator = numerator * (at - other) % prime
                denominator = denominator * (x - other) % prime
        answer = (answer + int(y) * numerator * pow(denominator, -1, prime)) % prime
    return answer


def ramp_oracle(elements):
    """Test packed multiplication and the concrete all-to-all degree reduction."""
    p, k, t = 257, 2, 1
    degree, workers = k + t - 1, 2 * (k + t - 1) + 1
    beta, alpha = list(range(1, k + t + 1)), list(range(k + t + 1, k + t + workers + 1))
    left, right = [17, 203, 101], [92, 0, 66]  # final entries are test-only masks
    products = [interpolate(beta, left, a) * interpolate(beta, right, a) % p for a in alpha]
    expected = [x * y % p for x, y in zip(left[:k], right[:k], strict=True)]
    assert [interpolate(alpha, products, b) for b in beta[:k]] == expected
    # Party i reshares its contributions to each secret slot with one fresh mask.
    new_shares = np.zeros(workers, np.int64)
    for i, value in enumerate(products):
        unit = [int(j == i) for j in range(workers)]
        slots = [interpolate(alpha, unit, b) * value % p for b in beta[:k]]
        slots.append((19 + 73 * i) % p)
        new_shares += [interpolate(beta, slots, a) for a in alpha]
    new_shares %= p
    assert [
        interpolate(alpha[: degree + 1], list(new_shares[: degree + 1]), b) for b in beta[:k]
    ] == expected
    # Exhaust every mask: a single worker sees all field values for either input.
    for secrets in ([0, 0], [12, 99]):
        views = [interpolate(beta, [*secrets, mask], alpha[0]) for mask in range(p)]
        assert len(set(views)) == p
    records = []
    for packed in (1, 2, 4, 8, 16):
        n = 2 * packed + 1  # one passive corruption, full degree reduction
        batch_count = (elements + packed - 1) // packed
        records.append(
            {
                "packed_secrets": packed,
                "workers": n,
                "public_linear_work_vs_two_offset": n / (2 * packed),
                "degree_reduction_bytes_per_batch": n * (n - 1) * 4,
                "one_gated_product_per_element_peer_bytes": batch_count * n * (n - 1) * 4,
            }
        )
    return {
        "exact_gate_and_reshare": True,
        "one_worker_view_exhausted": True,
        "scope": "test-local dealer, semi-honest one-corruption ramp model; no decoder",
        "layouts": records,
    }


def perm_compose(a, b):
    return tuple(a[b[i]] for i in range(len(a)))


def perm_inverse(a):
    return tuple(a.index(i) for i in range(len(a)))


def permutation_oracle():
    identity = tuple(range(5))

    def is_cycle(p):
        seen, index = set(), 0
        while index not in seen:
            seen.add(index)
            index = p[index]
        return len(seen) == 5 and index == 0

    cycles = [p for p in itertools.permutations(range(5)) if is_cycle(p)]
    first = cycles[0]
    second = next(
        p
        for p in cycles
        if is_cycle(
            perm_compose(perm_compose(perm_compose(first, p), perm_inverse(first)), perm_inverse(p))
        )
    )
    rng = np.random.default_rng(4903)
    for x, y in itertools.product((0, 1), repeat=2):
        word = [
            first if x else identity,
            second if y else identity,
            perm_inverse(first) if x else identity,
            perm_inverse(second) if y else identity,
        ]
        blinds = [identity, *(tuple(rng.permutation(5)) for _ in range(3)), identity]
        clear, masked = identity, identity
        for i, instruction in enumerate(word):
            clear = perm_compose(clear, instruction)
            encoded = perm_compose(
                perm_compose(perm_inverse(blinds[i]), instruction), blinds[i + 1]
            )
            masked = perm_compose(masked, encoded)
        assert clear == masked and (clear != identity) == bool(x and y)
    return {
        "and_truth_table_and_telescoping": True,
        "scope": "group identities only; oblivious branch selection and cryptographic encoding absent",
        "recursive_formula_compiler": [
            {
                "depth": d,
                "instructions": 4**d,
                "two_branch_ideal_7bit_permutations_bytes": (2 * 7 * 4**d + 7) // 8,
                "two_branch_16byte_objects_bytes": 32 * 4**d,
            }
            for d in (4, 8, 12, 24)
        ],
        "bound_kind": "cost of this recursive construction, NOT minimum branching-program size",
    }


def koopman_lift(prime=61):
    """Smallest observed Krylov closure of x under x -> x*x+1 on this field."""
    if prime not in (17, 61, 127):
        raise ValueError("bounded finite state space required")
    state = np.arange(prime, dtype=np.int64)
    transition = (state * state + 1) % prime
    basis = [state]
    for _ in range(prime):
        next_row = basis[-1][transition]
        relation = solve_field(np.array(basis).T, next_row, prime)
        if relation is not None:
            return np.array(basis), relation, transition
        basis.append(next_row)
    raise AssertionError("finite-dimensional closure missing")


def koopman_oracle():
    records = []
    for p in (17, 61, 127):
        basis, relation, transition = koopman_lift(p)
        rng = np.random.default_rng(4904 + p)
        left = rng.integers(0, p, basis.shape, dtype=np.int64)
        right = (basis - left) % p
        clear = np.arange(p, dtype=np.int64)
        steps = 256
        for _ in range(steps):
            clear = transition[clear]
            left = np.vstack((left[1:], relation @ left % p))
            right = np.vstack((right[1:], relation @ right % p))
            assert np.array_equal((left[0] + right[0]) % p, clear)
        records.append(
            {
                "field": p,
                "lift_width": len(basis),
                "all_initial_states": p,
                "exact_steps_per_initial_state": steps,
                "client_public_lift_table_bytes": int(basis.size * 2),
                "two_party_ingress_and_final_scalar_bytes": (2 * len(basis) + 2) * 2,
                "two_workers_nonzero_field_products_per_transition": 2
                * int(np.count_nonzero(relation)),
                "linear_share_updates_without_peer_messages": True,
            }
        )
    # Closure under TWO externally controlled maps, including a pre-square shift.
    # These operators cannot be selected publicly when the control is private.
    p = 61
    states = np.arange(p, dtype=np.int64)
    transitions = [(states * states + 1) % p, ((states + 1) ** 2 + 1) % p]
    controlled = [states]
    cursor = 0
    while cursor < len(controlled):
        for transition in transitions:
            next_row = controlled[cursor][transition]
            relation = solve_field(np.array(controlled).T, next_row, p)
            if relation is None:
                controlled.append(next_row)
        cursor += 1
    features = np.array(controlled)
    for transition in transitions:
        for feature in features:
            relation = solve_field(features.T, feature[transition], p)
            assert relation is not None
            assert np.array_equal(relation @ features % p, feature[transition])
    return {
        "examples": records,
        "controlled_example": {
            "field": p,
            "input_choices": 2,
            "lift_width": len(features),
            "client_public_lift_table_bytes": int(features.size * 2),
            "exact_both_operator_closures": True,
            "private_operator_selection_implemented": False,
        },
        "scope": "autonomous finite-field toy; no new private input, token selection or Qwen state",
        "qwen_lift_width": None,
        "qwen_encoding_cost": None,
    }


def hasse_jet(coefficients, point, multiplicity, prime=65537):
    return np.array(
        [
            sum(
                int(c) * math.comb(i, j) * pow(point, i - j, prime)
                for i, c in enumerate(coefficients)
                if i >= j
            )
            % prime
            for j in range(multiplicity)
        ],
        np.int64,
    )


def hermite_oracle():
    p = 65537
    coefficients = np.array([17, 23], np.int64)
    for _ in range(3):
        coefficients = np.convolve(coefficients, coefficients) % p
    points, multiplicity = (1, 2, 3, 4, 5), 2
    matrix, rhs = [], []
    for point in points:
        rhs.extend(hasse_jet(coefficients, point, multiplicity))
        for j in range(multiplicity):
            matrix.append(
                [
                    math.comb(i, j) * pow(point, i - j, p) % p if i >= j else 0
                    for i in range(len(coefficients))
                ]
            )
    restored = solve_field(matrix, rhs, p)
    assert np.array_equal(restored, coefficients)
    # Revealing two derivatives of a degree-one mask exposes its secret constant.
    x, r, point = 1201, 4309, 13
    view = hasse_jet([x, r], point, 2)
    assert int(view[0] - point * view[1]) % p == x
    # t=s masks are necessary for one worker seeing s jets in this univariate layout.
    return {
        "degree8_hermite_interpolation_exact": True,
        "two_jet_degree1_mask_recovers_secret": True,
        "private_layouts": [
            {
                "quadratic_layers": layers,
                "jet_multiplicity": s,
                "ingress_mask_degree": s,
                "composed_degree": s * 2**layers,
                "workers_for_generic_interpolation": 2**layers + 1,
                "returned_64bit_jet_bytes": (2**layers + 1) * s * 8,
            }
            for layers in (4, 8, 16, 24)
            for s in (1, 2, 4)
        ],
        "scope": "specified no-refresh generic interpolation; not a universal coded-computation lower bound",
    }


def zech_oracle():
    p, generator = 257, 3
    powers = np.array([pow(generator, i, p) for i in range(p - 1)], np.int64)
    assert len(set(powers)) == p - 1
    logs = np.full(p, -1, np.int64)
    logs[powers] = np.arange(p - 1)
    zech = logs[(1 + powers) % p]
    for x, y in itertools.product(range(p), repeat=2):
        if not x or not y:
            product = 0
            total = x or y
        else:
            a, b = int(logs[x]), int(logs[y])
            product = powers[(a + b) % (p - 1)]
            correction = int(zech[(b - a) % (p - 1)])
            total = 0 if correction == -1 else powers[(a + correction) % (p - 1)]
        assert int(product) == x * y % p and int(total) == (x + y) % p
    return {
        "field": p,
        "input_pairs_exhausted": p * p,
        "exact_addition_and_multiplication": True,
        "scope": "local exponent algebra, including zero; private Zech/zero tests and domain conversion unimplemented",
    }


def wallace_counts(weights, bits):
    """Specified unsigned-biased i8 bitheap modulo 2**bits; no synthesis claim.

    z=x+128. W*x=sum_i,b ((W_i<<b) mod q)*z_i,b -128*sum_i W_i.
    Sum bit columns with 3:2 counters, then one ripple adder. Every counter is
    charged two ANDs; public-constant/CSE simplification could lower this count.
    """
    w = np.asarray(weights)
    if w.dtype != np.int8 or w.ndim != 2 or not w.size or not 8 <= bits <= 32:
        raise ValueError("bounded signed-i8 bitheap required")
    if w.size > 64_000_000:
        raise ValueError("public bitheap admission exceeded")
    mask = (1 << bits) - 1
    patterns = np.zeros((256, bits), np.int64)
    for value in range(-128, 128):
        for shift in range(8):
            coefficient = (value << shift) & mask
            for bit in range(bits):
                patterns[value + 128, bit] += (coefficient >> bit) & 1
    hist = np.array([np.bincount(row.astype(np.int16) + 128, minlength=256) for row in w])
    heights = hist @ patterns
    constant = (-128 * w.astype(np.int64).sum(axis=1)) & mask
    for bit in range(bits):
        heights[:, bit] += (constant >> bit) & 1
    full_adders = np.zeros(len(w), np.int64)
    for bit in range(bits):
        counters = np.maximum(0, (heights[:, bit] - 1) // 2)
        full_adders += counters
        heights[:, bit] -= 2 * counters
        if bit + 1 < bits:
            heights[:, bit + 1] += counters
    assert np.all(heights <= 2)
    # Unsimplified fixed-width ripple addition of the two retained bit rows.
    ands = 2 * (full_adders + bits)
    return {"and_gates_per_row": ands, "counter_full_adders_per_row": full_adders}


def wallace_evaluate(weights, values, bits):
    """Bit-level independent evaluator, used only on tiny public test matrices."""
    w, x = np.asarray(weights), np.asarray(values)
    if w.dtype != np.int8 or x.dtype != np.int8 or w.ndim != 1 or x.shape != w.shape:
        raise ValueError("aligned signed-i8 vectors required")
    if len(w) > 64 or not 8 <= bits <= 32:
        raise ValueError("bounded bitheap evaluation required")
    mask, columns = (1 << bits) - 1, [[] for _ in range(bits)]
    for coefficient, value in zip(w, x, strict=True):
        biased = int(value) + 128
        for shift in range(8):
            term = (int(coefficient) << shift) & mask
            for bit in range(bits):
                if (term >> bit) & 1:
                    columns[bit].append((biased >> shift) & 1)
    constant = (-128 * int(w.astype(np.int64).sum())) & mask
    for bit in range(bits):
        if (constant >> bit) & 1:
            columns[bit].append(1)
        while len(columns[bit]) > 2:
            a, b, c = (columns[bit].pop() for _ in range(3))
            columns[bit].append(a ^ b ^ c)
            carry = (a & b) ^ (c & (a ^ b))
            if bit + 1 < bits:
                columns[bit + 1].append(carry)
    return sum(sum(column) << i for i, column in enumerate(columns)) & mask


def algebra_report(gated_elements=8_171_520):
    x = np.linspace(-32, 32, 65537, dtype=np.float32)
    expected = silu32(x)
    rational = rational_silu(x)
    fields = np.arange(1, 4097, dtype=np.int64)
    inverse = batch_inverse(fields)
    assert np.all(fields * inverse % 65537 == 1)
    return {
        "coset": coset_oracle(),
        "span": span_oracle(),
        "saturation": {
            "public_grid_points": len(x),
            "bit_exact_relu_points": int(
                np.count_nonzero(expected.view(np.uint32) == tropical_silu(x).view(np.uint32))
            ),
        },
        "rational": {
            "terms": 24,
            "public_range": [-32, 32],
            "grid_points": len(x),
            "bit_exact_grid_points": int(
                np.count_nonzero(expected.view(np.uint32) == rational.view(np.uint32))
            ),
            "maximum_grid_absolute_error": float(
                np.max(np.abs(expected.astype(np.float64) - rational))
            ),
            "field_batch_inverses_checked": len(fields),
            "batch_inverse_products": 3 * (len(fields) - 1),
            "hypothetical_inverse_products_for_gated_elements": 3 * (gated_elements - 1),
            "ordinary_two_party_beaver_inverse_openings_bytes": 3 * (gated_elements - 1) * 16,
            "masked_inverse_alternative": masked_inverse_oracle(),
            "masked_inverse_four_byte_field_projection_bytes": gated_elements * (16 + 8),
            "inverse_projection_elements": gated_elements,
            "inverse_projection_word_bytes": 4,
            "scope": "grid is not an error certificate; finite-field inverse is not IEEE float division",
        },
        "ramp": ramp_oracle(gated_elements),
        "permutation": permutation_oracle(),
        "koopman": koopman_oracle(),
        "hermite": hermite_oracle(),
        "zech": zech_oracle(),
    }

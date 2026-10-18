"""Public deterministic TEST fixtures, never production code/noise generation.

Affine permutations and literal coefficients are intentionally not samples from
the source ensemble. Their expanded descriptions are available via --emit.
Golden expectations are separately frozen in vectors.json.
"""

import hashlib
import json
from math import log2

from oracle import (BABYBEAR, PROPOSED_Q31_PLUS, Raa, batch_checks,
                    batch_publication, distance_dimension_two, lemma_bound,
                    matvec, privacy_weight, protocol3, sparse_vector)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                    sort_keys=True).encode("ascii")).hexdigest()


def code(q, n, tag):
    N = 8 * n
    return Raa(q, n, 8,
               tuple((5 * j + tag) % N for j in range(N)),
               tuple((7 * j + 3 * tag) % N for j in range(N)),
               tuple(q - 1 - j if j % 7 == 0 else 17 * j + tag + 1 for j in range(N)),
               tuple(q - 1 - j if j % 5 == 0 else 31 * j + tag + 1 for j in range(N)))


def challenge(q, N, weight, tag):
    return [((3 * j + tag) % N, q - 1 - j if j % 2 else 7919 * (j + 1))
            for j in range(weight)]


def small_case(q):
    gx, gy, gp = code(q, 3, 1), code(q, 4, 4), code(q, 6, 9)
    matrices = [
        [[q - 1, 257, 0], [65536, q - 256, 3], [129, 2, q - 1], [0, q // 2, 2]],
        [[0, 513, q - 1], [q - 65536, 5, 256], [1, q - 129, 7], [257, 0, q - 2]],
    ]
    xs = [[q - 1, 123456789, q // 2], [42, q - 257, 65537]]
    exs = [[(0, 1), (1, q - 1), (5, 257), (9, 129), (22, q // 2)],
           [(2, q - 2), (4, 17), (7, 65536), (11, q - 9), (23, 3)]]
    es = [challenge(q, gy.N, 31, j) for j in (0, 1)]
    # Step 5 is coprime to 48. These supports remain without replacement.
    eps = [[((5 * j + tag) % gp.N, 65537 * (j + 1)) for j in range(31)]
           for tag in (2, 7)]
    return gx, gy, gp, matrices, xs, exs, es, eps


def summary(q):
    gx, gy, gp, matrices, xs, exs, es, eps = small_case(q)
    claims = [protocol3(m, x, gx, gy, ex, es) for m, x, ex in zip(matrices, xs, exs)]
    stack, aux = batch_publication(matrices, gy, gp)
    primary = [sparse_vector(gy.N, e, q) for e in es]
    secondary = [sparse_vector(gp.N, e, q) for e in eps]
    responses = [matvec(stack, e, q) for e in primary]
    checks = batch_checks([c["masked_x"] for c in claims], [c["masked_y"] for c in claims],
                          gy, gp, aux, primary, secondary, responses)
    cert_code = code(q, 2, 3)
    bound = lemma_bound(q)
    return {
        "q": q,
        "fixture_inputs_sha256": digest({"codes": [c.record() for c in (gx, gy, gp)],
                                         "matrices": matrices, "x": xs, "ex": exs,
                                         "e": es, "e_prime": eps}),
        "protocol3": [{k: v for k, v in c.items() if k not in ("P", "Q", "z")}
                      | {"P_sha256": digest(c["P"]), "Q_sha256": digest(c["Q"]),
                         "z_sha256": digest(c["z"])} for c in claims],
        "batch": {"Qstack_sha256": digest(stack), "Qaux_sha256": digest(aux),
                  "auxiliary_responses": responses, "checks": checks},
        "dimension_two_certificate": distance_dimension_two(cert_code.dense(), q),
        "lemma_31_two_checks": {"numerator": bound.numerator, "denominator": bound.denominator,
                                "minus_log2_diagnostic": -log2(bound)},
        "windows_65536_batch_bound": {"numerator": (2**17 * bound).numerator,
                                      "denominator": (2**17 * bound).denominator},
    }


def source64_material(tag=0):
    """Two precisely defined public field matrices; second stresses wide sums."""
    q, n = BABYBEAR, 64
    gx, gy = code(q, n, 2 + tag), code(q, n, 11)
    base = [[((i + 1) * 65537 - (j + 1) * 257) % q for j in range(n)] for i in range(n)]
    matrix = base if tag == 0 else [[q - 1 - a for a in row] for row in base]
    x = [(q - 1 - j if j % 3 == 0 else j * 1234567) for j in range(n)]
    if tag:
        x = [q - 1 - a for a in x]
    entries = challenge(q, gx.N, privacy_weight(n), 5 + tag)
    return gx, gy, matrix, x, entries


def source64():
    """64x64, source BabyBear/RAA8/exact-weight 165/31x2 arithmetic vector."""
    gx, gy, matrix, x, entries = source64_material()
    q, n = gx.q, gx.n
    ex = sparse_vector(gx.N, entries, q, privacy_weight(n))
    mask = gx.syndrome(ex)
    correction = [sum(a * b for a, b in zip(gx.encode(row), ex)) % q for row in matrix]
    assert correction == matvec(matrix, mask, q)
    masked_x = [(a + b) % q for a, b in zip(x, mask)]
    masked_y = matvec(matrix, masked_x, q)
    y = [(a - b) % q for a, b in zip(masked_y, correction)]
    assert y == matvec(matrix, x, q)
    # Column encodings form Q; the dot products are unreduced Python integers.
    projection = list(zip(*(gy.encode(col) for col in zip(*matrix))))
    checks = []
    for tag in (0, 1):
        ey = sparse_vector(gy.N, challenge(q, gy.N, 31, tag), q, 31)
        left = sum(ey[i] * sum(a * b for a, b in zip(row, masked_x))
                   for i, row in enumerate(projection)) % q
        right = sum(a * b for a, b in zip(ey, gy.encode(masked_y))) % q
        assert left == right
        checks.append([left, right])
    return {"q": q, "shape": [n, n], "Nx": gx.N, "Ny": gy.N,
            "tx": len(entries), "ty": 31, "ell": 2,
            "inputs_sha256": digest({"gx": gx.record(), "gy": gy.record(),
                                      "matrix": matrix, "x": x, "ex": entries}),
            "mask_sha256": digest(mask), "correction_sha256": digest(correction),
            "masked_x_sha256": digest(masked_x), "masked_y_sha256": digest(masked_y),
            "y": y, "checks": checks}


def source64_batch():
    """Full specified B=2 capsule; no dense G, at most 512x1024 Qaux."""
    q, n = BABYBEAR, 64
    gy, gp = code(q, n, 11), code(q, 2 * n, 13)
    inputs, outputs, matrices, projections = [], [], [], []
    for tag in (0, 1):
        gx, _, m, x, entries = source64_material(tag)
        mask = gx.syndrome(sparse_vector(gx.N, entries, q, 165))
        xhat = [(a + b) % q for a, b in zip(x, mask)]
        matrices.append(m)
        inputs.append(xhat)
        outputs.append(matvec(m, xhat, q))
        projections.append(list(zip(*(gy.encode(col) for col in zip(*m)))))
    stack = [list(row) for projection in projections for row in zip(*projection)]
    aux = [gp.encode(row) for row in zip(*stack)]
    es = [sparse_vector(512, challenge(q, 512, 31, tag), q, 31) for tag in (0, 1)]
    eps = [sparse_vector(1024, challenge(q, 1024, 31, tag), q, 31) for tag in (2, 7)]
    responses = [matvec(stack, e, q) for e in es]
    secondary_checks, original_checks = [], []
    for e, u in zip(es, responses):
        for ep in eps:
            # Sparse bilinear Qaux form vs an independently associated syndrome.
            left = sum(e[i] * ep[j] * aux[i][j]
                       for i in range(512) if e[i] for j in range(1024) if ep[j]) % q
            right = sum(a * b for a, b in zip(gp.syndrome(ep), u)) % q
            assert left == right
            secondary_checks.append([left, right])
        for b, (xhat, yhat) in enumerate(zip(inputs, outputs)):
            left = sum(a * x for a, x in zip(u[b * n:(b + 1) * n], xhat)) % q
            right = sum(a * y for a, y in zip(e, gy.encode(yhat))) % q
            assert left == right
            original_checks.append([left, right])
    return {"q": q, "B": 2, "shape": [64, 64], "N": 512, "Nprime": 1024,
            "t": 31, "ell": 2, "tprime": 31, "ellprime": 2,
            "Qstack_sha256": digest(stack), "Qaux_sha256": digest(aux),
            "masked_inputs_sha256": digest(inputs), "masked_outputs_sha256": digest(outputs),
            "auxiliary_responses_sha256": digest(responses),
            "secondary_checks": secondary_checks, "original_checks": original_checks}


def all_vectors():
    return {"format": "pllm.maverick.scalar-conformance.v1",
            "security_status": "test-only; no ensemble or dual-LPN certification",
            "small_cases": [summary(q) for q in (BABYBEAR, PROPOSED_Q31_PLUS)],
            "source64": source64(), "source64_batch": source64_batch()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="emit expanded small fixture inputs")
    parser.add_argument("--emit64", action="store_true", help="emit both native 64x64 capsule inputs")
    args = parser.parse_args()
    if args.emit64:
        materials = []
        for tag in (0, 1):
            gx, gy, m, x, ex = source64_material(tag)
            materials.append({"gx": gx.record(), "gy": gy.record(), "matrix": m, "x": x, "ex": ex})
        print(json.dumps({"claims": materials, "secondary_code": code(BABYBEAR, 128, 13).record(),
                          "e": [challenge(BABYBEAR, 512, 31, t) for t in (0, 1)],
                          "e_prime": [challenge(BABYBEAR, 1024, 31, t) for t in (2, 7)]}, indent=2))
    elif args.emit:
        inputs = []
        for q in (BABYBEAR, PROPOSED_Q31_PLUS):
            gx, gy, gp, matrices, xs, exs, es, eps = small_case(q)
            inputs.append({"codes": [c.record() for c in (gx, gy, gp)], "matrices": matrices,
                           "x": xs, "ex": exs, "e": es, "e_prime": eps})
        print(json.dumps(inputs, indent=2))
    else:
        print(json.dumps(all_vectors(), indent=2))

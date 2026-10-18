"""Reproduce PUBLIC all-party vectors; never import into a production owner.

This offline fixture deliberately knows all public example keys/shares. Real
roles must hold only their own private key and two legitimate adjacent keys.
Prints canonical JSON to stdout; writes no files and downloads nothing.
"""
from dataclasses import asdict

from aes_reference import source_fill
from oracle import (Cursor, Interval, Q18, U64_MAX, admit_product, canonical_bytes,
                    digest, literal_signed, local_mul, local_trunc_piece,
                    multiply_batch, schedule)

N = 1 << 64
GAP = 1 << 62


def hex_words(values):
    return [f"{word:016x}" for word in values]


def components(values, sequence, right=False):
    if right:
        first = [(0xfedcba9876543210 - i + 11*sequence) % N for i in range(len(values))]
        second = [(0x0102030405060708 + 29*i - 23*sequence) % N for i in range(len(values))]
    else:
        first = [(0x1020304050607080 + i + 13*sequence) % N for i in range(len(values))]
        second = [(0x8877665544332211 - 17*i + 37*sequence) % N for i in range(len(values))]
    return [first, second, [(x - a - b) % N for x, a, b in zip(values, first, second)]]


def specifications():
    one = Q18
    return [
        ([-9, 1 << 31, one - 1], [1, -(1 << 31), one + 17],
         [Interval(-11, -7), Interval(1 << 31, 1 << 31), Interval(one - 2, one)],
         [Interval(1, 2), Interval(-(1 << 31), -(1 << 31)), Interval(one, one + 17)]),
        ([GAP - 1], [1], [Interval(GAP - 2, GAP - 1)], [Interval(1, 1)]),
        ([0, one + 1, -one - 1, (1 << 31) - 1],
         [(1 << 63) - 1, one - 1, one + 1, (1 << 31) - 1],
         [Interval(0, 0), Interval(one, one + 2), Interval(-one - 2, -one),
          Interval(-(1 << 31) + 1, (1 << 31) - 1)],
         [Interval(-(1 << 63), (1 << 63) - 1), Interval(one - 2, one),
          Interval(one, one + 2), Interval(-(1 << 31) + 1, (1 << 31) - 1)]),
    ]


def build_vectors(fill=source_fill):
    pair_keys = [(11*(i + 1)).to_bytes(16, "little") for i in range(3)]
    private_keys = [(101 + i).to_bytes(16, "little") for i in range(3)]
    public_key = (1 + 2 + 3).to_bytes(16, "little")
    cursor, cases = Cursor(), []
    for xs, ys, x_bounds, y_bounds in specifications():
        n = len(xs)
        for xb, yb in zip(x_bounds, y_bounds):
            admit_product(xb, yb)  # before PRSS expansion or modular arithmetic
        pivot = fill(public_key, cursor.public, 1)[0]
        plan, after = schedule(cursor, n, pivot)
        a, b, h = (plan["roles"][role] for role in ("a", "b", "h"))
        edges = [fill(key, cursor.pair, n) for key in pair_keys]
        x_parts = components(xs, cursor.sequence)
        y_parts = components(ys, cursor.sequence, True)
        z = [[local_mul((x_parts[i][j], x_parts[(i+1) % 3][j]),
                        (y_parts[i][j], y_parts[(i+1) % 3][j]),
                        edges[i][j], edges[(i+1) % 3][j]) for j in range(n)]
             for i in range(3)]
        # Ring congruence PLUS independently admitted unwrapped products.
        assert all(sum(z[i][j] for i in range(3)) % N == xs[j]*ys[j] % N for j in range(n))
        ra = fill(pair_keys[a], plan["trunc_mask_start"], n)
        rb = fill(pair_keys[h], plan["trunc_mask_start"], n)
        ya = fill(pair_keys[a], plan["trunc_output_start"], n)
        yb = fill(pair_keys[h], plan["trunc_output_start"], n)
        private = fill(private_keys[h], cursor.private[h], 2*n)
        masks = [(x + y) % N for x, y in zip(ra, rb)]
        ca = [(private[j] + (masks[j] >> 63)) % N for j in range(n)]
        ca += [(private[n+j] + ((masks[j] & (N//2 - 1)) >> 18)) % N for j in range(n)]
        cb = [(-value) % N for value in private]
        oa = [(z[a][j] + z[b][j] + GAP + ra[j]) % N for j in range(n)]
        ob = [(z[h][j] + rb[j]) % N for j in range(n)]
        c = [(u + v) % N for u, v in zip(oa, ob)]
        za = [local_trunc_piece(c[j], ca[j], ca[n+j], ya[j], True) for j in range(n)]
        zb = [local_trunc_piece(c[j], cb[j], cb[n+j], yb[j], False) for j in range(n)]
        middle = [(u + v) % N for u, v in zip(za, zb)]
        outputs = [None] * 3
        outputs[a], outputs[b], outputs[h] = (ya, middle), (middle, yb), (yb, ya)
        expected = multiply_batch(xs, ys, x_bounds, y_bounds, masks)
        literal = tuple(literal_signed(x*y, r) for x, y, r in zip(xs, ys, masks))
        assert expected == literal
        assert all(sum(outputs[i][0][j] for i in range(3)) % N == expected[j] % N for j in range(n))
        frames = []
        def frame(sender, recipient, phase, tag, words):
            frames.append({"from": sender, "to": recipient, "phase": phase,
                           "source_tag": tag, "words_hex": hex_words(words)})
        for i in range(3):
            frame(i, (i + 2) % 3, "mul.rotate", "mulaa", z[i])
        frame(a, b, "trunc.open", "_", oa)
        frame(b, a, "trunc.open", "_", ob)
        frame(h, a, "trunc.correlation", "cr0", ca)
        frame(h, b, "trunc.correlation", "cr1", cb)
        frame(a, b, "trunc.reshare", "2to3", za)
        frame(b, a, "trunc.reshare", "2to3", zb)
        cases.append({"x_signed": [str(v) for v in xs], "y_signed": [str(v) for v in ys],
                      "boxes": [{"x": [str(v.lo), str(v.hi)], "y": [str(w.lo), str(w.hi)]}
                                for v, w in zip(x_bounds, y_bounds)],
                      "plan": plan, "after": asdict(after),
                      "x_components_hex": [hex_words(row) for row in x_parts],
                      "y_components_hex": [hex_words(row) for row in y_parts],
                      "mul_edges_hex": [hex_words(row) for row in edges],
                      "mul_components_hex": [hex_words(row) for row in z],
                      "mask_components_hex": [hex_words(ra), hex_words(rb)],
                      "mask_hex": hex_words(masks), "private_words_hex": hex_words(private),
                      "frames": frames,
                      "output_pairs_hex": [[hex_words(row) for row in output] for output in outputs],
                      "product_signed": [str(x*y) for x, y in zip(xs, ys)],
                      "expected_signed": [str(v) for v in expected]})
        cursor = after
    return {"schema": "puma-mul-trunc-public-vectors.v1",
            "public_key_le128": public_key.hex(),
            "pair_keys_le128": [key.hex() for key in pair_keys],
            "private_keys_le128": [key.hex() for key in private_keys], "cases": cases}


def summary(vectors):
    return {"schema": vectors["schema"], "canonical_vectors_sha256": digest(vectors),
            "cases": [{"sequence": c["plan"]["sequence"], "lanes": c["plan"]["lanes"],
                       "roles": c["plan"]["roles"], "plan": c["plan"], "after": c["after"],
                       "expected_signed": c["expected_signed"],
                       "mul_components_hex": c["mul_components_hex"],
                       "output_pairs_hex": c["output_pairs_hex"],
                       "canonical_case_sha256": digest(c)} for c in vectors["cases"]]}


if __name__ == "__main__":
    # Full deterministic vectors, including all nine frames and every PRSS tape.
    print(canonical_bytes(build_vectors()).decode("ascii"))

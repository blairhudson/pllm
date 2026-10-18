"""Independent public AndBB vectors; historical Q18 fixtures stay frozen."""
import hashlib
import unittest

from aes_reference import encrypt_block, expanded_key


def typed_fill(key, cursor, words, width):
    """Literal source-call boundary with typed extraction; public test keys only."""
    assert width in (1, 2, 4, 8) and 1 <= words <= 2048
    blocks = (words*width + 15)//16
    initial = int.from_bytes(cursor.to_bytes(16, "little"), "big")
    keys = expanded_key(key)
    tape = b"".join(encrypt_block((initial+i).to_bytes(16, "big"), keys)
                    for i in range(blocks))
    return [int.from_bytes(tape[i:i+width], "little")
            for i in range(0, words*width, width)]


def backing(bits):
    return next(width for width in (1, 2, 4, 8) if bits <= 8*width)


def vectors():
    cursor, cases = 0, []
    for seq, (xb, yb, n) in enumerate(((0, 1, 3), (9, 8, 9), (17, 16, 5), (33, 64, 3))):
        def split(bits, right):
            mask = (1 << (8*backing(bits))) - 1
            values = [((i*0x10203 + 17) if right else (i+1)*0x9e3779b9) & ((1 << bits)-1)
                      for i in range(n)]
            a = [((0x12345678 if right else 0x89abcdef) + i + 13*seq) & mask for i in range(n)]
            b = [((0x87654321 if right else 0xfedcba98) - i + 7*seq) & mask for i in range(n)]
            return values, [a, b, [v ^ u ^ w for v, u, w in zip(values, a, b)]]
        x, xs = split(xb, False)
        y, ys = split(yb, True)
        width = backing(max(xb, yb))
        edges = [typed_fill(k.to_bytes(16, "little"), cursor, n, width) for k in (11, 22, 33)]
        z = [[(xs[r][i] & ys[r][i]) ^ (xs[r][i] & ys[(r+1)%3][i]) ^
              (xs[(r+1)%3][i] & ys[r][i]) ^ edges[r][i] ^ edges[(r+1)%3][i]
              for i in range(n)] for r in range(3)]
        assert [a ^ b ^ c for a, b, c in zip(*z)] == [a & b for a, b in zip(x, y)]
        body = b"".join(v.to_bytes(8, "little") for row in z for v in row)
        cases.append((cursor, hashlib.sha256(body).hexdigest()))
        cursor += (n*width + 15)//16
    return cases


def conversion_vectors():
    """Scalar three-party source equations, with full backing-word transcripts."""
    mask = (1 << 64) - 1
    values = [1 << 63, mask, (1 << 63) - 1]
    n = len(values)
    a = [(0xfedcba9876543210 + 17*i) & mask for i in range(n)]
    b = [(0x123456789abcdef0 - 5*i) & mask for i in range(n)]
    inputs = [a, b, [(x-y-z) & mask for x, y, z in zip(values, a, b)]]
    cursor = 0
    cases = []

    def zero(lanes, width):
        nonlocal cursor
        e = [typed_fill(k.to_bytes(16, "little"), cursor, lanes, width) for k in (11, 22, 33)]
        cursor += (lanes*width+15)//16
        return [[x ^ y for x, y in zip(e[r], e[(r+1)%3])] for r in range(3)]

    for msb in (False, True):
        transcript = hashlib.sha256()

        def record(words):
            for row in words:
                for word in row:
                    transcript.update(word.to_bytes(8, "little"))

        def and_gate(x, y, bits):
            z = zero(len(x[0]), backing(bits))
            for r in range(3):
                for i in range(len(x[0])):
                    z[r][i] ^= ((x[r][i] & y[r][i]) ^ (x[r][i] & y[(r+1)%3][i]) ^
                                (x[(r+1)%3][i] & y[r][i]))
            record(z)
            return z

        def xor(x, y):
            return [[a ^ b for a, b in zip(u, v)] for u, v in zip(x, y)]

        m = zero(n, 8)
        m[0] = [x ^ ((a+b) & mask) for x, a, b in zip(m[0], inputs[0], inputs[1])]
        record(m)
        other = [[0]*n, [0]*n, inputs[2]]
        initial = xor(m, other)
        p = [row[:] for row in initial]
        g = and_gate(m, other, 64)
        width = 64
        for level in range(6):
            if not msb:
                rhs = [[(x << (1 << level)) & mask for x in pr+gr] for pr, gr in zip(p, g)]
                out = and_gate([row+row for row in p], rhs, 64)
                p = [row[:n] for row in out]
                g = xor(g, [row[n:] for row in out])
            else:
                if level == 0:
                    p = [[(v << 1) & mask for v in row] for row in p]
                    g = [[(v << 1) & mask for v in row] for row in g]

                def halves(v):
                    # Independent bit-string permutation of each backing block.
                    binary = f"{v:0{8*backing(width)}b}"[::-1]
                    blocks = [binary[i:i+width] for i in range(0, len(binary), width)]
                    shuffled = ''.join(block[::2]+block[1::2] for block in blocks)
                    word = int(shuffled[::-1], 2)
                    return word & ((1 << (width//2))-1), word >> (width//2)

                ps = [[halves(v) for v in row] for row in p]
                gs = [[halves(v) for v in row] for row in g]
                lhs = [[x[0] for x in pr+gr] for pr, gr in zip(ps, gs)]
                rhs = [[x[1] for x in pr]*2 for pr in ps]
                width //= 2
                out = and_gate(lhs, rhs, width)
                p = [row[:n] for row in out]
                g = xor([[x[1] for x in gr] for gr in gs], [row[n:] for row in out])
        result = xor([[x >> 63 for x in row] for row in initial], g) if msb else xor(initial, [[(x << 1) & mask for x in row] for row in g])
        assert [x ^ y ^ z for x, y, z in zip(*result)] == ([v >> 63 for v in values] if msb else values)
        cases.append((cursor, transcript.hexdigest()))
    return cases


def ot_conversion_vectors():
    """Three-party OT equations with independent AES and scalar bit composition."""
    ring = (1 << 64)-1
    cursor = public = 0
    cases = []
    n = 3
    for bits in (0, 1, 4, 8):
        maximum = (1 << bits)-1
        values = [0, maximum, maximum//2]
        x = [[0xa5]*n, [0x33]*n, [v ^ 0xa5 ^ 0x33 for v in values]]
        transcript = hashlib.sha256()

        def record(words):
            for v in words:
                transcript.update(v.to_bytes(8, "little"))

        def compose(words):
            return [sum(words[i*bits+b] << b for b in range(bits)) & ring for i in range(n)]

        helper = 0
        if bits == 0:
            output = [[0]*n for _ in range(3)]
        else:
            helper = typed_fill((6).to_bytes(16, "little"), public, 1, 8)[0] % 3
            public += 1
            receiver, sender = (helper+1)%3, (helper+2)%3
            length = n*bits
            q = (length+1)//2
            random = [typed_fill(k.to_bytes(16, "little"), cursor, length, 8) for k in (11, 22, 33)]
            m0, m1 = [typed_fill(((helper+1)*11).to_bytes(16, "little"), cursor+j*q, length, 8) for j in (1, 2)]
            choices = [(x[receiver][i] >> b) & 1 for i in range(n) for b in range(bits)]
            masks = [m1[j] if c else m0[j] for j, c in enumerate(choices)]
            combined = [(x[sender][i] ^ x[helper][i]) >> b & 1 for i in range(n) for b in range(bits)]
            sums = [(a+b) & ring for a, b in zip(random[helper], random[sender])]
            messages = [[((v ^ candidate)-sums[j]) & ring for j, v in enumerate(combined)] for candidate in (0, 1)]
            messages = [[v ^ r for v, r in zip(message, mask)] for message, mask in zip(messages, (m0, m1))]
            c2 = compose([messages[c][j] ^ masks[j] for j, c in enumerate(choices)])
            for frame in (masks, messages[0], messages[1], c2):
                record(frame)
            output = [None]*3
            output[helper], output[receiver], output[sender] = compose(random[helper]), c2, compose(random[sender])
            cursor += 3*q
        assert [sum(v) & ring for v in zip(*output)] == values
        for row in output:
            record(row)
        cases.append((helper, cursor, transcript.hexdigest()))
    return cases


class BooleanVectors(unittest.TestCase):
    def test_typed_aes_transcripts(self):
        cases = vectors()
        self.assertEqual([start for start, _ in cases], [0, 1, 3, 5])
        self.assertEqual([digest for _, digest in cases], [
            "d7a29a8832c467364dd0de07f9b4705500b4a3857583d75808e6b705502bcdbd",
            "89c04bcf6a19de44bbf89222e119d4a939cc541f73d4cc28e2bc2d6be6c7d647",
            "81d204c6a02f85e84aaa3a590e656eedf82a20c5ceb4554947a35aaac593ec22",
            "3833690eb4f33adc9543b95e33d79f231cf19e9c5da90156a93c7fbe2e5ceb13",
        ])

    def test_conversion_carry_and_typed_padding(self):
        self.assertEqual(conversion_vectors(), [
            (22, "31af625820c93e1a53a26ae326a80094496df7e8267315978ee8b76471308d53"),
            (33, "8c54ac188fbaaf4b096ee2bb945fd1c8ca35cc01a6f4fc3086a843531614eeab"),
        ])

    def test_ot_bit_composition_and_none_cursor_advances(self):
        self.assertEqual(ot_conversion_vectors(), [
            (0, 0, "834a709ba2534ebe3ee1397fd4f7bd288b2acc1d20a08d6c862dcd99b6f04400"),
            (2, 6, "c9aace3e1031119812976b935fa7c65afc4946a8c09f2b9c9e9fe7f37c3c98fb"),
            (0, 24, "1b13d3dae62b3f0870479cc4e90c41b39f18befdf5bc1f3f171e486fcbeb3806"),
            (0, 60, "84847a2cda0df6efffa74cdd9c8d47bb3a92179fa8b82fb59c8ae89efc9ac966"),
        ])

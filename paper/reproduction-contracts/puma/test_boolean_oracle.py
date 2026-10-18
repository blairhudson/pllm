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

"""Independent, slow AES-128 math for PUBLIC reproducible fixtures only.

Standard library only; no upstream code, external crypto executable, seed
provisioning, side-channel claim, secure erasure or production PRSS interface.
S-box is derived from the GF(2^8) inverse/affine definition, not a copied table.
"""
from oracle import Rejected, U64_MAX, MAX_LANES, integer


def xtime(x):
    return ((x << 1) ^ (0x11b if x & 128 else 0)) & 255


def field_mul(a, b):
    out = 0
    for _ in range(8):
        if b & 1:
            out ^= a
        a, b = xtime(a), b >> 1
    return out


def substitute(x):
    inverse, base, exponent = 1, x, 254
    while exponent:
        if exponent & 1:
            inverse = field_mul(inverse, base)
        base = field_mul(base, base)
        exponent >>= 1
    value = inverse ^ 0x63
    for shift in range(1, 5):
        value ^= ((inverse << shift) | (inverse >> (8 - shift))) & 255
    return value


SBOX = tuple(substitute(i) for i in range(256))


def expanded_key(key):
    if type(key) is not bytes or len(key) != 16:
        raise Rejected("AES key shape")
    words = [list(key[i:i+4]) for i in range(0, 16, 4)]
    rcon = 1
    for i in range(4, 44):
        temp = words[-1][:]
        if i % 4 == 0:
            temp = [SBOX[v] for v in temp[1:] + temp[:1]]
            temp[0] ^= rcon
            rcon = xtime(rcon)
        words.append([a ^ b for a, b in zip(words[i - 4], temp)])
    return [sum(words[4*i:4*i+4], []) for i in range(11)]


def encrypt_block(block, keys):
    if type(block) is not bytes or len(block) != 16:
        raise Rejected("AES block shape")
    state = [a ^ b for a, b in zip(block, keys[0])]
    for round_number in range(1, 11):
        state = [SBOX[state[4*((column + row) % 4) + row]]
                 for column in range(4) for row in range(4)]
        if round_number != 10:
            mixed = []
            for col in range(4):
                values = state[4*col:4*col+4]
                total = values[0] ^ values[1] ^ values[2] ^ values[3]
                mixed.extend(values[i] ^ total ^ xtime(values[i] ^ values[(i + 1) % 4])
                             for i in range(4))
            state = mixed
        state = [a ^ b for a, b in zip(state, keys[round_number])]
    return bytes(state)


def source_fill(key, cursor, words):
    """Exact LE128 call reset, BE within-fill counter, LE64 word extraction."""
    integer(words, 1, 2 * MAX_LANES)
    integer(cursor, 0, U64_MAX)
    blocks = (words + 1) // 2
    integer(cursor + blocks, 0, U64_MAX)
    keys = expanded_key(key)
    initial = int.from_bytes(cursor.to_bytes(16, "little"), "big")
    out = []
    for offset in range(blocks):
        block = encrypt_block((initial + offset).to_bytes(16, "big"), keys)
        for at in (0, 8):
            if len(out) < words:
                out.append(int.from_bytes(block[at:at+8], "little"))
    return tuple(out)

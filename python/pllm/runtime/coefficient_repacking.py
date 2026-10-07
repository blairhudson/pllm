"""Bounded SEAL BFV coefficient projection/interleaving, using public ciphertexts.

Independent implementation of trace algebra and BumbleBee Figure 6. Native RNS
division precedes the complete trace; plaintext-ring division is never used.
"""
from __future__ import annotations

import struct

DEGREE = 4096


def geometry(stride: int, count: int) -> tuple[int, tuple[int, ...], int]:
    if (type(stride) is not int or stride < 1 or stride > DEGREE or stride & (stride - 1)
            or type(count) is not int or not 1 <= count <= min(stride, 256)):
        raise ValueError("invalid bounded coefficient repacking geometry")
    if count == 1:
        return 1, (), 0
    padded = 1 << (count - 1).bit_length()
    elements = tuple(DEGREE // (1 << i) + 1 for i in range(stride.bit_length() - 1))
    return padded, elements, padded * (stride.bit_length() - padded.bit_length()) + padded - 1


def _words(context, ciphertext, header):
    # The binding has no mutable coefficient view. Construct only an uncompressed
    # SEAL 4.1/4.3 size-two BFV frame, then use SEAL's checked context-bound loader.
    if (len(header) != 16 or header[:4] != b"\x5e\xa1\x10\x04" or header[4] not in (1, 3)
            or ciphertext.is_ntt_form() or ciphertext.size() != 2
            or ciphertext.poly_modulus_degree() != DEGREE or ciphertext.scale != 1.0
            or ciphertext.parms_id() != context.first_parms_id()):
        raise ValueError("unsupported SEAL coefficient representation")
    moduli = [q.value() for q in context.first_context_data().parms().coeff_modulus()]
    if ciphertext.coeff_modulus_size() != len(moduli) or not 1 <= len(moduli) <= 2:
        raise ValueError("unsupported SEAL RNS geometry")
    data = ciphertext.dyn_array()
    count = 2 * DEGREE * len(moduli)
    if data.size() != count:
        raise ValueError("noncanonical SEAL coefficient buffer")
    words = struct.pack(f"<{count}Q", *(data[i] for i in range(count)))
    return words, moduli


def _load_words(seal, context, template, words, moduli, header, path):
    count = 2 * DEGREE * len(moduli)
    if len(words) != count * 8:
        raise ValueError("noncanonical rebuilt RNS frame")

    def frame(body):
        return header[:5] + bytes(3) + struct.pack("<Q", 16 + len(body)) + body

    body = struct.pack("<4QBQQQdQ", *template.parms_id(), 0, 2, DEGREE, len(moduli), 1.0, 1)
    body += frame(struct.pack("<Q", count) + words)
    path.write_bytes(frame(body))
    try:
        result = seal.Ciphertext()
        result.load(context, str(path))
    finally:
        path.unlink(missing_ok=True)
    return result


def _divide(seal, native, context, ciphertext, divisor, header, path):
    words, moduli = _words(context, ciphertext, header)
    scaled = native.coefficient_repacking_divide(words, moduli, divisor)
    return _load_words(seal, context, ciphertext, scaled, moduli, header, path)


def pack_lwes(seal, native, context, evaluator, encryptor, keys, ciphertexts, positions, header, path):
    """Sample-extract arbitrary RLWE coefficients and pack their LWE messages.

    Extracted (b,a) contains no secret key or plaintext. Its RLWE embedding has
    the correct constant term; trace/interleaving removes the other terms.
    """
    geometry(DEGREE, len(positions))
    if any(type(owner) is not int or type(coefficient) is not int
           or not 0 <= owner < len(ciphertexts) or not 0 <= coefficient < DEGREE
           for owner, coefficient in positions):
        raise ValueError("invalid LWE extraction address")
    embedded = []
    for owner, coefficient in positions:
        template = ciphertexts[owner]
        words, moduli = _words(context, template, header)
        lwe = native.coefficient_repacking_extract(words, moduli, coefficient)
        words = native.coefficient_repacking_embed(lwe, moduli)
        embedded.append(_load_words(seal, context, template, words, moduli, header, path))
    out, _, operations = interleave(seal, native, context, evaluator, encryptor, keys,
        embedded, DEGREE, header, path, source_offset=0)
    return out, [(0, i) for i in range(len(positions))], operations


def interleave(seal, native, context, evaluator, encryptor, keys, ciphertexts, stride, header, path, *, source_offset=None):
    """Retain input j's coefficients at k*stride+stride-1 at k*stride+j.

    All other coefficients vanish. Caller masks every coefficient before sending
    the result. Public geometry bounds all allocations and Galois operations.
    """
    count = len(ciphertexts)
    padded, elements, operations = geometry(stride, count)
    offset = stride - 1 if source_offset is None else source_offset
    if type(offset) is not int or not 0 <= offset < stride:
        raise ValueError("invalid public coefficient offset")
    if not elements:
        return ciphertexts[0], lambda owner, coefficient: coefficient, operations

    def shift(x, exponent):
        exponent %= 2 * DEGREE
        negative = exponent >= DEGREE
        exponent %= DEGREE
        out = seal.Ciphertext()
        evaluator.multiply_plain(x, seal.Plaintext("1" + (f"x^{exponent}" if exponent else "")), out)
        # Negate in the ciphertext ring. Multiplying by plaintext t-1 is also
        # congruent to -1 after decryption, but unnecessarily amplifies noise.
        if negative:
            evaluator.negate_inplace(out)
        return out

    def auto(x, element):
        out = seal.Ciphertext()
        evaluator.apply_galois(x, element, keys, out)
        return out

    def combine(x, y, subtract=False):
        out = seal.Ciphertext()
        (evaluator.sub if subtract else evaluator.add)(x, y, out)
        return out

    values = []
    for index in range(padded):
        if index < count:
            x = shift(ciphertexts[index], -offset) if offset else ciphertexts[index]
        else:
            x = seal.Ciphertext()
            encryptor.encrypt_zero(x)
        x = _divide(seal, native, context, x, stride, header, path)
        h = padded
        while h < stride:
            x = combine(x, auto(x, DEGREE // h + 1))
            h *= 2
        values.append(x)
    h = padded // 2
    while h:
        next_values = []
        for j in range(h):
            y = shift(values[j + h], h)
            next_values.append(combine(combine(values[j], y), auto(combine(values[j], y, True), DEGREE // h + 1)))
        values = next_values
        h //= 2
    return values[0], lambda owner, coefficient: coefficient - offset + owner, operations

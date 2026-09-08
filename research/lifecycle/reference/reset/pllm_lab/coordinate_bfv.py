"""Exact public-weight linear evaluation, packing independent masks by coordinate.

Each input coordinate is one BFV plaintext polynomial. Its coefficients are
that coordinate in independent future masks. Weights are SMALL PUBLIC INTEGERS,
so ciphertext addition, subtraction and doubling suffice. No encrypted
multiplication, rotations, evaluation keys, or model approximation are used.
This reference is not a malicious-client model privacy protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tempfile
import numpy as np
import _sealapi_cpp as seal


@dataclass(frozen=True)
class Profile:
    degree: int = 2048
    modulus: int = 65537

    def context(self):
        parameters = seal.EncryptionParameters(seal.SCHEME_TYPE.BFV)
        parameters.set_poly_modulus_degree(self.degree)
        parameters.set_coeff_modulus(seal.CoeffModulus.BFVDefault(self.degree, seal.SEC_LEVEL_TYPE.TC128))
        parameters.set_plain_modulus(self.modulus)
        context = seal.SEALContext(parameters, True, seal.SEC_LEVEL_TYPE.TC128)
        if not context.parameters_set():
            raise ValueError(context.parameters_error_message())
        return context


def plaintext(coefficients, modulus: int):
    # TenSEAL's low-level binding only exposes string polynomial construction.
    # Formatting is included in client timing, not hidden in server throughput.
    terms = []
    for i, value in enumerate(coefficients):
        v = int(value) % modulus
        if v:
            terms.append(f'{v:X}' if i == 0 else f'{v:X}x^{i}')
    return seal.Plaintext(' + '.join(reversed(terms)) if terms else '0')


def dumps(value) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'object.seal'
        value.save(str(p))
        return p.read_bytes()


def loads(payload: bytes, context):
    ct = seal.Ciphertext()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'object.seal'
        p.write_bytes(payload)
        ct.load(context, str(p))
    return ct


class Client:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.context = profile.context()
        gen = seal.KeyGenerator(self.context)
        self.secret_key = gen.secret_key()
        self.encryptor = seal.Encryptor(self.context, self.secret_key)
        self.decryptor = seal.Decryptor(self.context, self.secret_key)

    def encrypt(self, masks: np.ndarray):
        if masks.ndim != 2 or not 1 <= masks.shape[0] <= self.profile.degree:
            raise ValueError('masks must have shape (1..degree, input_width)')
        if masks.dtype.kind not in 'iu':
            raise TypeError('integer masks required')
        cts = []
        for column in masks.T:
            ct = seal.Ciphertext()
            self.encryptor.encrypt_symmetric(plaintext(column, self.profile.modulus), ct)
            cts.append(ct)
        # A separate zero is needed for an all-zero public output row.
        zero = seal.Ciphertext()
        self.encryptor.encrypt_symmetric(plaintext([0], self.profile.modulus), zero)
        return cts, zero

    def decrypt(self, cts, count: int):
        result = np.empty((count, len(cts)), dtype=np.int64)
        budget = []
        for j, ct in enumerate(cts):
            pt = seal.Plaintext()
            self.decryptor.decrypt(ct, pt)
            available = int(pt.coeff_count())
            result[:,j] = [int(pt[i]) if i < available else 0 for i in range(count)]
            budget.append(self.decryptor.invariant_noise_budget(ct))
        return result, min(budget)


class Server:
    """Server state contains public weights and parameters, and no client key."""
    def __init__(self, profile: Profile, weight: np.ndarray):
        if weight.ndim != 2 or weight.dtype.kind not in 'iu':
            raise ValueError('weight must be an integer matrix')
        if np.any(weight < -7) or np.any(weight > 7):
            raise ValueError('weights outside supported signed W4 range [-7,7]')
        self.profile = profile
        self.context = profile.context()
        self.evaluator = seal.Evaluator(self.context)
        self.weight = weight.astype(np.int8, copy=True)
        self.last_counts = {}

    def evaluate(self, cts, encrypted_zero):
        if len(cts) != self.weight.shape[1]:
            raise ValueError('input width mismatch')
        # Build only needed multiples, using addition rather than multiplying
        # a ciphertext by an encoded dense plaintext. Tables are reusable for
        # every output row IN THIS REQUEST, never for a different input.
        tables = []
        doubles = 0
        sums = 0
        for i, ct in enumerate(cts):
            needed = set(abs(int(w)) for w in self.weight[:,i]) - {0}
            table = {1: ct}
            def get(k):
                nonlocal doubles, sums
                if k not in table:
                    v = seal.Ciphertext()
                    if k % 2 == 0:
                        src = get(k//2)
                        self.evaluator.add(src, src, v)
                        doubles += 1
                    else:
                        self.evaluator.add(get(k-1), ct, v)
                        sums += 1
                    table[k] = v
                return table[k]
            for k in needed:
                get(k)
            tables.append(table)
        outputs = []
        for row in self.weight:
            positive, negative = [], []
            for i, w in enumerate(row):
                if w > 0:
                    positive.append(tables[i][int(w)])
                elif w < 0:
                    negative.append(tables[i][-int(w)])
            def add_terms(terms):
                nonlocal sums
                if not terms:
                    return None
                if len(terms) == 1:
                    return terms[0]
                v = seal.Ciphertext()
                self.evaluator.add_many(terms, v)
                sums += len(terms)-1
                return v
            pos, neg = add_terms(positive), add_terms(negative)
            if pos is not None and neg is not None:
                v = seal.Ciphertext()
                self.evaluator.sub(pos, neg, v)
                sums += 1
            elif pos is not None:
                v = pos
            elif neg is not None:
                v = seal.Ciphertext()
                self.evaluator.negate(neg, v)
            else:
                v = encrypted_zero
            outputs.append(v)
        self.last_counts = {'doublings': doubles, 'additions_subtractions': sums,
                            'rotations': 0, 'multiplications': 0}
        return outputs

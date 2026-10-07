"""Local, bounded CKKS ownership and arithmetic for encrypted operator research.

Only Client owns a decryptor. Provider imports serialized public evaluation
material into a separate SEAL context. Context/sequence framing is a lifecycle
check, not authenticated transport or malicious-computation verification.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import secrets
import struct
import time

from .seal_frames import load_frame, save_frame

HEADER_BYTES = 81


def _context(seal, parameters):
    context = seal.SEALContext(parameters, True, seal.SEC_LEVEL_TYPE.TC128)
    if not context.parameters_set():
        raise ValueError("SEAL rejected bounded CKKS parameters at TC128")
    return context


def _header(session, binding, ordinal, direction):
    return b"PLCKKS01" + session + binding + struct.pack("<QB", ordinal, direction)


def _capacity(context, ciphertext):
    limbs = context.get_context_data(ciphertext.parms_id()).parms().coeff_modulus()
    return 2 * context.key_context_data().parms().poly_modulus_degree() * len(limbs) * 8 + 4096


class Arithmetic:
    """Public evaluator operations. Never receives a client or a secret key."""

    def __init__(self, seal, context, public, galois, relin):
        self.seal, self.context = seal, context
        self.encoder = seal.CKKSEncoder(context)
        self.evaluator = seal.Evaluator(context)
        self.encryptor = seal.Encryptor(context, public)
        self.galois, self.relin = galois, relin
        self.counts = Counter()
        self.scale = float(2**40)
        self.minimum_remaining_depth = context.first_context_data().chain_index()

    def level(self, x):
        return self.context.get_context_data(x.parms_id()).chain_index()

    def switch(self, x, target):
        if x.parms_id() == target.parms_id():
            return x
        result = self.seal.Ciphertext()
        self.evaluator.mod_switch_to(x, target.parms_id(), result)
        self.counts["modulus_switches"] += 1
        return result

    def _align(self, x, y):
        return (self.switch(x, y), y) if self.level(x) > self.level(y) else (x, self.switch(y, x))

    def _rescale(self, x):
        result = self.seal.Ciphertext()
        self.evaluator.rescale_to_next(x, result)
        if abs(result.scale / self.scale - 1) > 0.0001:
            raise ValueError("CKKS scale drift exceeds the declared normalization bound")
        # Standard approximate CKKS scale normalization. Its numeric error is
        # included in the independent decoded-output comparison, never exactness.
        result.scale = self.scale
        self.minimum_remaining_depth = min(self.minimum_remaining_depth, self.level(result))
        self.counts["rescales"] += 1
        self.counts["scale_normalizations"] += 1
        return result

    def add(self, x, y):
        x, y = self._align(x, y)
        result = self.seal.Ciphertext()
        self.evaluator.add(x, y, result)
        self.counts["ciphertext_additions"] += 1
        return result

    def add_plain(self, x, value):
        plain, result = self.seal.Plaintext(), self.seal.Ciphertext()
        self.encoder.encode(value, x.parms_id(), self.scale, plain)
        self.evaluator.add_plain(x, plain, result)
        self.counts["plaintext_additions"] += 1
        return result

    def multiply_plain(self, x, value):
        if self.level(x) < 1:
            raise ValueError("CKKS multiplication requires an unused rescale level")
        plain, result = self.seal.Plaintext(), self.seal.Ciphertext()
        self.encoder.encode(value, x.parms_id(), self.scale, plain)
        self.evaluator.multiply_plain(x, plain, result)
        self.counts["plaintext_multiplications"] += 1
        return self._rescale(result)

    def multiply(self, x, y):
        x, y = self._align(x, y)
        if self.relin is None or self.level(x) < 1:
            raise ValueError("CKKS product requires relinearization keys and a rescale level")
        product, result = self.seal.Ciphertext(), self.seal.Ciphertext()
        self.evaluator.multiply(x, y, product)
        self.evaluator.relinearize(product, self.relin, result)
        self.counts["ciphertext_multiplications"] += 1
        self.counts["relinearizations"] += 1
        return self._rescale(result)

    def rotate(self, x, steps):
        if self.galois is None:
            raise ValueError("CKKS rotation requires declared evaluation keys")
        result = self.seal.Ciphertext()
        self.evaluator.rotate_vector(x, steps, self.galois, result)
        self.counts["rotations"] += 1
        return result

    def zero(self):
        result = self.seal.Ciphertext()
        self.encryptor.encrypt_zero(result)
        result.scale = self.scale
        self.counts["public_zero_encryptions"] += 1
        return result


class Provider:
    def __init__(self, seal, context, public, galois, relin, directory,
                 parameters, session, binding, queries):
        self.seal, self.context, self.directory = seal, context, directory
        self.arithmetic = Arithmetic(seal, context, public, galois, relin)
        self.parameters, self.session, self.binding = parameters, session, binding
        self.remaining, self.next_ordinal, self.alive = queries, 0, True
        self.request_bytes = self.reply_bytes = 0

    def evaluate(self, frame, operation):
        if not self.alive or self.remaining == 0:
            raise ValueError("CKKS reference session is retired")
        ordinal = self.next_ordinal
        self.next_ordinal += 1
        self.remaining -= 1
        try:
            capacity = self.parameters["ciphertext_frame_bytes"]
            if (len(frame) != HEADER_BYTES + capacity
                    or frame[:HEADER_BYTES] != _header(self.session, self.binding, ordinal, 0)):
                raise ValueError("CKKS request context, sequence or shape differs")
            x = load_frame(self.seal.Ciphertext(), self.context, frame[HEADER_BYTES:],
                           self.directory / "request", capacity)
            if x.size() != 2 or x.parms_id() != self.context.first_parms_id() or x.scale != 2**40:
                raise ValueError("CKKS request has an unadmitted size, level or scale")
            self.request_bytes += len(frame)
            y = operation(self.arithmetic, x)
            if y.size() != 2:
                raise ValueError("CKKS reply must be relinearized")
            reply = _header(self.session, self.binding, ordinal, 1) + save_frame(
                y, self.directory / "reply", _capacity(self.context, y))
            self.reply_bytes += len(reply)
            if self.remaining == 0:
                self.alive = False
            return reply
        except BaseException:
            self.retire()
            raise

    def retire(self):
        self.alive, self.remaining = False, 0


class Client:
    def __init__(self, seal, context, public, secret, directory, parameters, session, binding, queries):
        self.seal, self.context, self.directory = seal, context, directory
        self.encoder, self.encryptor = seal.CKKSEncoder(context), seal.Encryptor(context, public)
        self.decryptor = seal.Decryptor(context, secret)
        self.parameters, self.session, self.binding = parameters, session, binding
        self.remaining, self.next_ordinal, self.pending, self.alive = queries, 0, None, True

    def request(self, values):
        if not self.alive or self.remaining == 0 or self.pending is not None:
            raise ValueError("CKKS client is retired or already has a pending request")
        ordinal = self.next_ordinal
        self.next_ordinal += 1
        self.remaining -= 1
        self.pending = ordinal
        try:
            if (not 1 <= len(values) <= self.encoder.slot_count()
                    or any(not math.isfinite(v) or abs(v) > 1 for v in values)):
                raise ValueError("CKKS input requires bounded finite [-1,1] slots")
            plain, cipher = self.seal.Plaintext(), self.seal.Ciphertext()
            self.encoder.encode(values, float(2**40), plain)
            self.encryptor.encrypt(plain, cipher)
            return _header(self.session, self.binding, ordinal, 0) + save_frame(
                cipher, self.directory / "client-request", self.parameters["ciphertext_frame_bytes"])
        except BaseException:
            self.retire()
            raise

    def receive(self, frame, output_depth):
        if not self.alive or self.pending is None:
            raise ValueError("CKKS client has no live response")
        ordinal, self.pending = self.pending, None
        try:
            if type(output_depth) is not int or not 0 <= output_depth <= self.parameters["depth"]:
                raise ValueError("CKKS response depth is outside the public schedule")
            capacity = 2 * self.parameters["degree"] * (self.parameters["depth"] + 1 - output_depth) * 8 + 4096
            if (len(frame) != HEADER_BYTES + capacity
                    or frame[:HEADER_BYTES] != _header(self.session, self.binding, ordinal, 1)):
                raise ValueError("CKKS response context, sequence or shape differs")
            cipher = load_frame(self.seal.Ciphertext(), self.context, frame[HEADER_BYTES:],
                                self.directory / "client-reply", capacity)
            level = self.context.get_context_data(cipher.parms_id()).chain_index()
            if cipher.size() != 2 or level != self.parameters["depth"] - output_depth or cipher.scale != 2**40:
                raise ValueError("CKKS response has an unadmitted size, level or scale")
            plain = self.seal.Plaintext()
            self.decryptor.decrypt(cipher, plain)
            values = self.encoder.decode_double(plain)
            if any(not math.isfinite(value) for value in values):
                raise ValueError("CKKS response decoded non-finite values")
            if self.remaining == 0:
                self.retire()
            return values
        except BaseException:
            self.retire()
            raise

    def retire(self):
        self.alive, self.remaining, self.pending = False, 0, None
        self.decryptor = self.encryptor = None


def setup(seal, native, directory: Path, *, depth, rotation_steps, relinearization,
          live_ciphertexts, queries, binding):
    if (type(queries) is not int or not 1 <= queries <= 64 or len(binding) != 32
            or len(set(rotation_steps)) != len(rotation_steps)
            or any(type(step) is not int or not 1 <= step <= 31 for step in rotation_steps)):
        raise ValueError("invalid CKKS reference session geometry")
    metadata = json.loads(native.ckks_parameters(depth, len(rotation_steps), relinearization, live_ciphertexts))
    p = metadata["parameters"]
    started = time.perf_counter()
    params = seal.EncryptionParameters(seal.SCHEME_TYPE.CKKS)
    params.set_poly_modulus_degree(p["degree"])
    params.set_coeff_modulus(seal.CoeffModulus.Create(p["degree"], p["coefficient_bits"]))
    local = _context(seal, params)
    kg, public = seal.KeyGenerator(local), seal.PublicKey()
    kg.create_public_key(public)
    frames = {}
    frames["parameters"] = save_frame(params, directory / "parameters", 512)
    frames["public-key"] = save_frame(public, directory / "public", p["public_key_frame_bytes"])
    remote_params = load_frame(seal.EncryptionParameters(seal.SCHEME_TYPE.CKKS), None,
        frames["parameters"], directory / "remote-parameters", 512)
    remote = _context(seal, remote_params)
    remote_public = load_frame(seal.PublicKey(), remote, frames["public-key"],
        directory / "remote-public", p["public_key_frame_bytes"])
    galois, relin = None, None
    if rotation_steps:
        keys = seal.GaloisKeys()
        kg.create_galois_keys([pow(3, step, 2 * p["degree"]) for step in rotation_steps], keys)
        capacity = len(rotation_steps) * p["switching_key_frame_bytes"] + 4096
        frames["galois-keys"] = save_frame(keys, directory / "galois", capacity)
        galois = load_frame(seal.GaloisKeys(), remote, frames["galois-keys"], directory / "remote-galois", capacity)
        del keys
    if relinearization:
        keys = seal.RelinKeys()
        kg.create_relin_keys(keys)
        capacity = p["switching_key_frame_bytes"] + 4096
        frames["relinearization-keys"] = save_frame(keys, directory / "relin", capacity)
        relin = load_frame(seal.RelinKeys(), remote, frames["relinearization-keys"], directory / "remote-relin", capacity)
        del keys
    session = secrets.token_bytes(32)
    binding = hashlib.sha256(binding + json.dumps(p, sort_keys=True).encode()).digest()
    client = Client(seal, local, public, kg.secret_key(), directory, p, session, binding, queries)
    provider = Provider(seal, remote, remote_public, galois, relin, directory, p, session, binding, queries)
    metadata["setup_frame_bytes"] = {name: len(frame) for name, frame in frames.items()}
    metadata["secret_polynomial_bytes"] = p["degree"] * len(p["coefficient_bits"]) * 8
    metadata["setup_seconds"] = time.perf_counter() - started
    return client, provider, metadata


def diagonal_linear(arithmetic, x, diagonals, key_policy):
    width, result, shifted = len(diagonals), None, x
    slots = arithmetic.encoder.slot_count()
    for index, diagonal in enumerate(diagonals):
        if index:
            shifted = arithmetic.rotate(shifted if key_policy == "single" else x,
                                        1 if key_policy == "single" else index)
        if any(diagonal):
            term = arithmetic.multiply_plain(shifted, diagonal * (slots // width))
            result = term if result is None else arithmetic.add(result, term)
    if result is None:
        # Public zero weights still follow the same output-level contract.
        result = arithmetic.multiply_plain(arithmetic.zero(), 1.0)
    return result

"""Local, bounded CKKS ownership and arithmetic for encrypted operator research.

Only Client owns a decryptor. Provider imports serialized public evaluation
material into a separate SEAL context. Context/sequence framing is a lifecycle
check, not authenticated transport or malicious-computation verification.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
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

    def negate(self, x):
        result = self.seal.Ciphertext()
        self.evaluator.negate(x, result)
        self.counts["ciphertext_negations"] += 1
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
          live_ciphertexts, queries, binding, softmax_width=None, rotation_stride=1):
    if (type(queries) is not int or not 1 <= queries <= 64 or len(binding) != 32
            or type(rotation_stride) is not int or rotation_stride not in (1, 2, 4, 8, 16, 32, 64)
            or len(set(rotation_steps)) != len(rotation_steps)
            or any(type(step) is not int or step % rotation_stride != 0
                   or not 1 <= step // rotation_stride <= 31 for step in rotation_steps)):
        raise ValueError("invalid CKKS reference session geometry")
    if softmax_width is None:
        metadata = json.loads(native.ckks_parameters(depth, len(rotation_steps), relinearization, live_ciphertexts))
    else:
        metadata = json.loads(native.ckks_softmax_parameters(softmax_width))
        if (depth != 14 or relinearization is not True or live_ciphertexts != 32 or rotation_stride != 1
                or rotation_steps != ((1,) if softmax_width > 1 else ())):
            raise ValueError("CKKS softmax requires its fixed admitted schedule")
    p = metadata["parameters"]
    if any(step >= p["degree"] // 2 for step in rotation_steps):
        raise ValueError("CKKS rotation exceeds the admitted slot count")
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


def diagonal_linear(arithmetic, x, diagonals, key_policy, *, batch_size=1):
    width, result, shifted = len(diagonals), None, x
    slots = arithmetic.encoder.slot_count()
    if (type(batch_size) is not int or batch_size not in (1, 2, 4, 8, 16, 32, 64)
            or width * batch_size > slots or slots % (width * batch_size)):
        raise ValueError("CKKS interleaved batch exceeds admitted slots")
    for index, diagonal in enumerate(diagonals):
        if index:
            shifted = arithmetic.rotate(shifted if key_policy == "single" else x,
                                         batch_size if key_policy == "single" else index * batch_size)
        if any(diagonal):
            interleaved = [coefficient for coefficient in diagonal for _ in range(batch_size)]
            term = arithmetic.multiply_plain(shifted, interleaved * (slots // (width * batch_size)))
            result = term if result is None else arithmetic.add(result, term)
    if result is None:
        # Public zero weights still follow the same output-level contract.
        result = arithmetic.multiply_plain(arithmetic.zero(), 1.0)
    return result


# Unqualified, source-only metadata contract. Deliberately not used by Arithmetic,
# Client, Provider, setup, or diagonal_linear above. Legacy normalization stays put.
# Public source basis, NOT installed-library/build identity:
# TenSEAL v0.3.17 tenseal/sealapi/sealapi_encrypt.cpp (overloads and has_key);
# SEAL v4.3.3 native/src/seal/{evaluator.cpp,util/galois.cpp,kswitchkeys.h}.
# Multiplication updates double scale first; rescaling divides that double by
# double(coeff_modulus.back().value()). Fraction arithmetic is a separate oracle.
# No tolerance, CKKS parameter set, key generation, or backend adapter is adopted.
SCALE_STATE_QUALIFICATION = "Unqualified"

# S=4096, N=8192, generator 3 modulo 2*N. Positive means left rotation.
# Tuple entries: (signed step, residue modulo S, uint32 Galois element).
# Future public-binding call: create_galois_keys([-3,-2,-1,1,2,3], dst).
# Negative entries reject vector<uint32_t> conversion, selecting vector<int>.
# has_key receives ELEMENT, not step/residue/index; size counts nonempty entries,
# not the outer data vector length. No call or key allocation occurs here.
SIGNED_ROTATION_MANIFEST = (
    (-3, 4093, 6675), (-2, 4094, 3641), (-1, 4095, 10923),
    (1, 1, 3), (2, 2, 9), (3, 3, 27),
)


def _scale_float(value):
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError("Unqualified scale metadata requires a positive finite float")
    return value


def _parms_id(value):
    if (type(value) is not tuple or len(value) != 4
            or any(type(word) is not int or not 0 <= word < 2**64 for word in value)
            or not any(value)):
        raise ValueError("Unqualified parms_id requires four nonzero-as-a-whole uint64 words")


@dataclass(frozen=True)
class ScaleLevel:
    """Unqualified snapshot of one DATA context, never the key context.

    Moduli are metadata, not asserted primes or an admitted parameter set.
    A future binding must read parms_id, chain_index and coefficient moduli from
    successive first_context_data()/next_context_data() records, not bit counts
    or symbolic placeholders. Snapshot validation cannot prove that happened.
    """

    parms_id: tuple[int, int, int, int]
    chain_index: int
    coefficient_moduli: tuple[int, ...]

    def __post_init__(self):
        _parms_id(self.parms_id)
        if type(self.chain_index) is not int or self.chain_index < 0:
            raise ValueError("Unqualified invalid chain index")
        moduli = self.coefficient_moduli
        if (type(moduli) is not tuple or not moduli
                or any(type(p) is not int or not 2 < p < 2**64 or p % 2 == 0 for p in moduli)
                or len(set(moduli)) != len(moduli)):
            raise ValueError("Unqualified malformed coefficient modulus metadata")


@dataclass(frozen=True)
class ScaleContext:
    """Unqualified context binding; caller identity is NOT authority/provenance.

    No ciphertext, keys, numeric/security qualification or native allocation
    guarantee. Workspace accounting can only describe LOGICAL reference storage:
    evaluator pools and Python/vector conversions remain hidden allocations.
    """

    identity: bytes
    delta: float
    levels: tuple[ScaleLevel, ...]

    def __post_init__(self):
        if type(self.identity) is not bytes or not self.identity:
            raise ValueError("Unqualified context identity must be nonempty bytes")
        _scale_float(self.delta)
        if (type(self.levels) is not tuple or not self.levels
                or any(type(level) is not ScaleLevel for level in self.levels)):
            raise ValueError("Unqualified data context chain required")
        if len({level.parms_id for level in self.levels}) != len(self.levels):
            raise ValueError("Unqualified repeated parms_id")
        for current, following in zip(self.levels, self.levels[1:]):
            if (following.chain_index != current.chain_index - 1
                    or following.coefficient_moduli != current.coefficient_moduli[:-1]):
                raise ValueError("Unqualified chain must drop exactly its last modulus")
        if self.levels[-1].chain_index != 0 or len(self.levels[-1].coefficient_moduli) != 1:
            raise ValueError("Unqualified complete data context chain required")

    def initial_state(self, parms_id, actual_scale):
        """Unqualified observed metadata gate before immutable publication."""
        if parms_id != self.levels[0].parms_id:
            raise ValueError("Unqualified initial level mismatch")
        _parms_id(parms_id)
        return ScaleState(self, 0, ScaleExpression(1, ()), actual_scale)


@dataclass(frozen=True)
class ScaleExpression:
    """Unqualified symbolic Delta**power / product(ordered dropped moduli).

    Narrow reusable contract: repeated multiply-at-Delta then rescale, without
    manual retagging. Not a general expression simplifier or noise certificate.
    """

    delta_power: int
    dropped_primes: tuple[int, ...]

    def __post_init__(self):
        if (type(self.delta_power) is not int or self.delta_power < 1
                or type(self.dropped_primes) is not tuple
                or self.delta_power != len(self.dropped_primes) + 1
                or any(type(p) is not int or not 2 < p < 2**64 or p % 2 == 0
                       for p in self.dropped_primes)):
            raise ValueError("Unqualified invalid symbolic scale expression")


@dataclass(frozen=True)
class ScaleState:
    """Unqualified immutable metadata, never ciphertext authority.

    actual_scale is checked against operation-ordered floating metadata, NOT the
    rounded exact Fraction. Exact equality here is intentionally stricter than
    SEAL's approximate scale compatibility. Neither establishes numeric accuracy.
    Constructor validation also rejects forged level/expression/scale pairings;
    it cannot authenticate observations, ciphertext ancestry or backend execution.
    """

    context: ScaleContext
    level: int
    expression: ScaleExpression
    actual_scale: float

    def __post_init__(self):
        if (type(self.context) is not ScaleContext or type(self.level) is not int
                or not 0 <= self.level < len(self.context.levels)
                or type(self.expression) is not ScaleExpression):
            raise ValueError("Unqualified invalid typed scale state")
        dropped = tuple(level.coefficient_moduli[-1]
                        for level in self.context.levels[:self.level])
        if self.expression != ScaleExpression(self.level + 1, dropped):
            raise ValueError("Unqualified level or symbolic scale mismatch")
        expected = self.context.delta
        for prime in dropped:
            expected = _scale_float(expected * self.context.delta)
            expected = _scale_float(expected / float(prime))
        if _scale_float(self.actual_scale) != expected:
            raise ValueError("Unqualified operation-ordered scale mismatch; retagging forbidden")

    @property
    def parms_id(self):
        return self.context.levels[self.level].parms_id

    @property
    def exact_scale(self):
        """Unqualified rational oracle for binary-float Delta, not CKKS error."""
        return (Fraction.from_float(self.context.delta) ** self.expression.delta_power
                / math.prod(self.expression.dropped_primes))

    def multiply_delta_rescale(self, observed_parms_id, observed_scale):
        """Unqualified publication after BOTH next-level and float checks pass.

        Represents multiply_plain at Delta (or a matching-scale first product)
        then rescale_to_next. No native operation, implicit switch or retag.
        Parents remain unchanged on all failures; no native-output atomicity claim.
        """
        following = self.level + 1
        if following >= len(self.context.levels):
            raise ValueError("Unqualified rescale chain exhausted")
        _parms_id(observed_parms_id)
        if observed_parms_id != self.context.levels[following].parms_id:
            raise ValueError("Unqualified rescale must use the exact next parms_id")
        prime = self.context.levels[self.level].coefficient_moduli[-1]
        expression = ScaleExpression(self.expression.delta_power + 1,
                                     self.expression.dropped_primes + (prime,))
        return ScaleState(self.context, following, expression, observed_scale)

    def require_add_compatible(self, other):
        """Unqualified refusal gate, not SEAL's are_close numeric certificate."""
        if (type(other) is not ScaleState or self.context != other.context
                or self.parms_id != other.parms_id or self.level != other.level
                or self.expression != other.expression or self.actual_scale != other.actual_scale):
            raise ValueError("Unqualified branch context, level, expression or scale mismatch")


@dataclass(frozen=True)
class ProductScaleBranches:
    """Unqualified support/wrap/remainder and SEPARATE full-support a0 states.

    Each child must descend one scale transition from ORIGINAL product P, never
    from a previously masked child. This checks declared metadata only; it cannot
    prove which ciphertext an evaluator actually masked. No support geometry or
    mask coefficient validity is inferred from scale metadata.
    """

    product: ScaleState
    support: ScaleState
    wrap: ScaleState
    remainder: ScaleState
    a0_full_support: ScaleState

    def __post_init__(self):
        if type(self.product) is not ScaleState:
            raise ValueError("Unqualified original product scale state required")
        for child in (self.support, self.wrap, self.remainder, self.a0_full_support):
            if type(child) is not ScaleState:
                raise ValueError("Unqualified typed branch required")
            expected = self.product.multiply_delta_rescale(child.parms_id, child.actual_scale)
            expected.require_add_compatible(child)

    def bias_encoding_state(self):
        """Unqualified encode bias at resulting beta parms_id AND actual scale.

        Returned metadata is an encoding requirement, not an encoded plaintext.
        In particular, do not encode at Delta and overwrite plaintext.scale.
        """
        return self.a0_full_support


def publish_product_scale_branches(product, observations):
    """Unqualified atomic metadata publication; all four branches or no result.

    Observations are a tuple of (parms_id, actual_scale), ordered support, wrap,
    remainder, a0_full_support. All derive independently from original P.
    At P=alpha=Delta**2/p1 every child is beta=Delta**3/(p1*p2).
    No ciphertext operations or externally visible partial publication occur.
    """
    if (type(product) is not ScaleState or type(observations) is not tuple
            or len(observations) != 4
            or any(type(item) is not tuple or len(item) != 2 for item in observations)):
        raise ValueError("Unqualified four branch metadata observations required")
    children = tuple(product.multiply_delta_rescale(parms_id, scale)
                     for parms_id, scale in observations)
    return ProductScaleBranches(product, *children)

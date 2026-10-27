"""Bounded ROSETTA phase-index *adaptation*, not an admitted HE capability.

Paper-inferred contract: 2609.16915v1 §§2.2.2–2.2.3, 3.1–3.3, Algorithm 1.
The missing HomComp-to-rotation rounding is implemented as a Boolean circuit
over an EXTRA, independently keyed BFV encryption of binary LWE secret bits.
This is not source-author-faithful HomComp, trained inference, or a security
claim. Joint-key/circuit privacy and BFV noise remain unproved. No HomComp,
modulus normalization, key switch, or BFV-to-RGSW bridge is implemented here.

Reuses coefficient_conversion's canonical-before-down contract,
coefficient_layout's signed negacyclic law, segmented_affine_reference's grid,
and encrypted_level_schedule's rule that unknown costs are illegal, not zero.
Those Rust contracts are not exposed to Python; the small reference helpers
below are explicit adaptations, not new generic HE infrastructure. Existing
party/source reservations cannot authorize this new circuit. No SDK exports,
protected routing, production-binding API, or caller-minted admission tokens.

Compilation and clear circuit interpretation are executable. Public encrypted
evaluation currently refuses: exact backend memory/time/noise bounds are absent.
The private SEAL gate implementation is retained for independent source review;
it is NOT permission to execute an unadmitted circuit.

The 8GiB/one-thread/30s constants are unchanged policy ceilings, NOT enforcement.
Hard backend execution limits, source import/consumption/replay retirement and
BFV profile authority are unimplemented. Environment thread settings would not
prove enforcement. Width and dimension ceilings are separate, not joint support.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

GRID = 1 << 20
INDEX_BITS = 12
CELLS = 1 << INDEX_BITS
MAX_MEMORY_BYTES = 8 << 30
MAX_THREADS = 1
MAX_SECONDS = 30  # policy only, not enforced or measured HE time
MAX_NODES = 100_000
MAX_WIDTH = 63
MAX_DIMENSION = 1024
MAX_INTEGER_BITS = 256
COMPONENT_ERROR_CELLS = Fraction(1025, 2)  # n=1024: (n+1)/2


class PhaseIndexRefused(RuntimeError):
    """A missing proof/cost is a refusal, never a zero-cost capability."""


class PhaseIndexBackendUnavailable(RuntimeError):
    pass


def _integer(value, name, bits=MAX_INTEGER_BITS):
    if type(value) is not int or value.bit_length() > bits:
        raise ValueError(f"{name}: bounded integer required")
    return value


@dataclass(frozen=True, slots=True)
class PhaseIndexSpec:
    q: int
    n: int
    key_epoch: int = 0
    phase_sign: str = "plus"
    scale: int = GRID
    level: int = 0

    def __post_init__(self):
        _integer(self.q, "q", MAX_WIDTH)
        _integer(self.n, "n", 16)
        _integer(self.key_epoch, "key_epoch", 64)
        if self.q < 3 or not self.q & 1 or not 1 <= self.n <= MAX_DIMENSION:
            raise ValueError("odd q >= 3 and 1 <= n <= 1024 required")
        if (self.key_epoch < 0 or self.phase_sign != "plus"
                or type(self.scale) is not int or self.scale != GRID
                or type(self.level) is not int or self.level != 0):
            raise ValueError("requires plus phase, grid scale and level zero")


def nearest_even(numerator: int, denominator: int) -> int:
    """Signed rational RNE; unlike modular quantization, ties are possible."""
    _integer(numerator, "numerator")
    _integer(denominator, "denominator")
    if denominator <= 0:
        raise ValueError("positive denominator required")
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator
                          or (2 * remainder == denominator and quotient & 1))


def quantize_phase_reference(phase: int, q: int) -> int:
    """Whole phase first: floor((8192*(phase mod q)+q)/(2*q)) mod 4096."""
    _integer(phase, "phase")
    _integer(q, "q", MAX_WIDTH)
    if q < 3 or not q & 1:
        raise ValueError("odd q >= 3 required")
    return ((8192 * (phase % q) + q) // (2 * q)) % CELLS


def centered_phase(phase: int, q: int) -> int:
    quantize_phase_reference(phase, q)  # same domain, canonicalize FIRST
    p = phase % q
    return p if p <= q // 2 else p - q


def segment_index(numerator: int, denominator: int = 1) -> tuple[int, int]:
    """Exact rational x -> (segment, local index), never clamp or round."""
    _integer(numerator, "numerator")
    _integer(denominator, "denominator")
    if denominator <= 0 or numerator < 0:
        raise ValueError("invalid rational grid input")
    x, remainder = divmod(numerator * GRID, denominator)
    if remainder or not GRID <= x < 16 * GRID:
        raise ValueError("off-grid or outside [1,16)")
    segment = sum(x >= GRID << s for s in (1, 2, 3))
    return segment, (x - (GRID << segment)) // (1 << (9 + segment))


def local_center(x: int, segment: int) -> Fraction:
    _integer(x, "X", 32)
    _integer(segment, "segment", 3)
    if not 0 <= segment < 4 or not GRID << segment <= x < GRID << (segment + 1):
        raise ValueError("X outside segment")
    width = 1 << (9 + segment)
    return Fraction(2 * (x - (GRID << segment)) - width + 1, 2 * width)


def require_index_error(*, normalized_phase_error: Fraction,
                        local_coordinate_error: Fraction, segment: int) -> None:
    """Check TWO independently stated errors; no implicit unit conversion.

    normalized_phase_error is in phase turns (one full period = 1).
    local_coordinate_error is in z_s index-coordinate units (one entry = 1).
    These named quantities are not interchangeable, even if numerically equal.
    Strict bounds are numerical contracts, not BFV admission/noise evidence.
    """
    _integer(segment, "segment", 3)
    if (type(normalized_phase_error) is not Fraction
            or type(local_coordinate_error) is not Fraction or not 0 <= segment < 4):
        raise ValueError("two named exact rational errors and segment required")
    if abs(normalized_phase_error) >= Fraction(1, 8192):
        raise PhaseIndexRefused("uniform normalized error must be < 1/8192")
    if abs(local_coordinate_error) >= Fraction(1, 2 * (1 << (9 + segment))):
        raise PhaseIndexRefused("actual per-segment error must be < 1/(2B_s)")


def negacyclic_rotation(index: int) -> tuple[int, int]:
    """Y^-J mod (Y^2048+1): return monomial exponent and integer sign."""
    _integer(index, "index", 13)
    if not 0 <= index < CELLS:
        raise ValueError("12-bit index required")
    quotient, exponent = divmod(-index, 2048)
    return exponent, -1 if quotient & 1 else 1


def homcomp_error_cells(q: int, summed_error: int) -> Fraction:
    """Three phases beta*h+e, |h| <= 1. This does not implement HomComp."""
    quantize_phase_reference(0, q)
    _integer(summed_error, "summed_error")
    beta = nearest_even(q, CELLS)
    return abs(Fraction(CELLS * summed_error, q)) + 3 * abs(Fraction(CELLS * beta, q) - 1)


def require_quarter_cell(q: int, summed_error: int) -> None:
    # Preserve stronger existing quarter-cell promise; half-cell alone insufficient.
    if homcomp_error_cells(q, summed_error) >= Fraction(1, 4):
        raise PhaseIndexRefused("HomComp quarter-cell bound not established")


@dataclass(frozen=True, slots=True)
class CircuitQuote:
    """Exact static gate schedule, NOT an HE allocation/admission quote.

    live_bit_planes counts ONLY peak entries in _interpret's values dictionary,
    after insertion and before last-consumer deletion. It excludes input tuple,
    owner/frames, persistent ciphertext references, result local scope (including
    dead results), output tuple construction, and multiply/relinearize/XOR
    scratch. It therefore does NOT bound backend ciphertext peak or memory.
    ir_heap_estimate_bytes is an unproved 256-byte-per-node shape heuristic;
    not a Python heap upper bound, capacity check, reservation or authority.
    """
    nodes: int
    and_gates: int
    xor_gates: int
    not_gates: int
    products: int
    relinearizations: int
    multiplicative_depth: int
    live_bit_planes: int
    ir_heap_estimate_bytes: int
    # None explicitly means unknown, NOT zero or an accepted caller estimate.
    context_bytes: int | None = None
    evaluation_key_bytes: int | None = None
    serialization_bytes: int | None = None
    import_overlap_bytes: int | None = None
    working_set_bytes: int | None = None
    elapsed_seconds: int | None = None
    noise_margin_bits: int | None = None
    tc128_parameters_checked: bool = False
    joint_key_privacy_checked: bool = False

    def require_admission(self):
        # No public override: neither caller labels nor supplied noise numbers
        # can authorize a new cryptographic capability. Even a tiny shape refuses.
        raise PhaseIndexRefused(
            "unadmitted BFV Boolean circuit: exact noise/working-set/time, TC128 "
            "parameter check, joint-key privacy and retained source reservation "
            "missing; caps remain 8GiB, one thread, 30s; no HE allocation allowed")


@dataclass(frozen=True, slots=True)
class _Circuit:
    spec: PhaseIndexSpec
    b: int
    a: tuple[int, ...]
    nodes: tuple[tuple[str, int, int], ...]
    outputs: tuple[int, ...]
    uses: tuple[int, ...]
    quote: CircuitQuote
    digest: bytes


class _Builder:
    """Bounded SSA, no paper-specific parallel scheduling/HE framework."""

    def __init__(self):
        self.nodes = []
        self.depths = []
        self.zero = self.emit("const", 0)
        self.one = self.emit("const", 1)

    def emit(self, op, left, right=-1):
        if len(self.nodes) >= MAX_NODES:
            raise PhaseIndexRefused("static IR capacity exceeded before next node")
        depth = 0
        if op in ("and", "xor"):
            depth = max(self.depths[left], self.depths[right]) + 1
        elif op == "not":
            depth = self.depths[left]
        self.nodes.append((op, left, right))
        self.depths.append(depth)
        return len(self.nodes) - 1

    def const(self, value, width):
        return [self.one if value >> i & 1 else self.zero for i in range(width)]

    def add(self, left, right, carry=None):
        carry = self.zero if carry is None else carry
        result = []
        for a, b in zip(left, right, strict=True):
            pair = self.emit("xor", a, b)
            result.append(self.emit("xor", pair, carry))
            carry = self.emit("xor", self.emit("and", a, b),
                              self.emit("and", pair, carry))
        return result, carry

    def subtract_constant(self, bits, value):
        # Unsigned subtraction via two's complement, carry means bits >= value.
        return self.add(bits, self.const(((1 << len(bits)) - 1) ^ value, len(bits)), self.one)

    def choose(self, flag, yes, no):
        return [self.emit("xor", n, self.emit("and", flag, self.emit("xor", y, n)))
                for y, n in zip(yes, no, strict=True)]

    def modular_add(self, left, right, q):
        total, _ = self.add(left + [self.zero], right + [self.zero])
        difference, ge = self.subtract_constant(total, q)
        return self.choose(ge, difference, total)[:-1]

    def quotient(self, phase, q):
        # Actual constant quotient, not component-wise quantization. Numerator
        # < 8193*q fits w+14 bits. Restoring division never branches on secrets.
        width = len(phase) + 14
        numerator, _ = self.add([self.zero] * 13 + phase + [self.zero],
                                self.const(q, width))
        remainder = [self.zero] * (q.bit_length() + 1)
        quotient = []
        for bit in reversed(numerator):
            trial = [bit] + remainder  # width w+2, handles shifted remainder
            difference, ge = self.subtract_constant(trial, 2 * q)
            remainder = self.choose(ge, difference, trial)[:-1]
            quotient.append(ge)
        return tuple(reversed(quotient))[:INDEX_BITS]


def _parents(node):
    op, left, right = node
    return (left, right) if op in ("and", "xor") else (left,) if op == "not" else ()


def compile_phase_index(spec: PhaseIndexSpec, b: int, a: tuple[int, ...]) -> _Circuit:
    """Quote BEFORE backend import, context creation, HE allocation or keys.

    Public coefficients are canonical and fixed into this request's SSA. Shape
    preflight bounds IR construction before any potentially giant node list.
    This quote is NOT a reservation or an encrypted_level_schedule capability.
    """
    if type(spec) is not PhaseIndexSpec:
        raise ValueError("typed phase spec required")
    _integer(b, "b", MAX_WIDTH)
    if type(a) is not tuple or len(a) != spec.n:
        raise ValueError("canonical tuple a with exactly n residues required")
    if not 0 <= b < spec.q:
        raise ValueError("noncanonical b")
    for value in a:
        _integer(value, "a", MAX_WIDTH)
        if not 0 <= value < spec.q:
            raise ValueError("noncanonical a")
    w = spec.q.bit_length()
    # add: 5 gates/bit; reduction: 5+3 gates/bit. Division: same
    # subtraction/mux schedule, w+14 iterations. No secret/value pruning.
    # This bounds NODE COUNT, not Python heap bytes or backend working set.
    upper = (2 + spec.n + 13 * spec.n * (w + 1)
             + 5 * (w + 14) + 8 * (w + 14) * (w + 2))
    if upper > MAX_NODES:
        raise PhaseIndexRefused(
            f"static shape {upper} nodes exceeds {MAX_NODES}-node IR limit; no HE allocation")
    builder = _Builder()
    terms = [builder.const(b, w)]
    for ordinal, value in enumerate(a):
        secret = builder.emit("input", ordinal)
        terms.append([secret if value >> i & 1 else builder.zero for i in range(w)])
    # Balanced modular-add tree, not an n-long carry chain across terms.
    while len(terms) > 1:
        terms = [builder.modular_add(terms[i], terms[i + 1], spec.q)
                 if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
    outputs = builder.quotient(terms[0], spec.q)
    nodes = tuple(builder.nodes)
    uses = [0] * len(nodes)
    for node in nodes:
        for parent in _parents(node):
            uses[parent] += 1
    for output in outputs:
        uses[output] += 1
    remaining = uses.copy()
    live = peak = 0
    for index, node in enumerate(nodes):
        live += 1
        peak = max(peak, live)
        for parent in _parents(node):
            remaining[parent] -= 1
            if remaining[parent] == 0:
                live -= 1
        if remaining[index] == 0:
            live -= 1
    ands = sum(op == "and" for op, _, _ in nodes)
    xors = sum(op == "xor" for op, _, _ in nodes)
    nots = sum(op == "not" for op, _, _ in nodes)
    quote = CircuitQuote(len(nodes), ands, xors, nots, ands + xors, ands + xors,
                         max(builder.depths), peak,
                         upper * 256)
    # Public structural digest; absolutely not source/key authority.
    description = [spec.q, spec.n, spec.key_epoch, spec.phase_sign, spec.scale,
                   spec.level, b, a, nodes, outputs]
    digest = hashlib.sha256(json.dumps(description, separators=(",", ":")).encode()).digest()
    return _Circuit(spec, b, a, nodes, outputs, tuple(uses), quote, digest)


def _interpret_clear_bits(circuit: _Circuit, inputs, operations):
    """Portable clear-only harness: reject malformed bits BEFORE any operation.

    This check cannot validate encrypted bits. Future ciphertexts require an
    honest binary-source assumption or separately reviewed validity mechanism;
    evaluator never decrypts or inspects private plaintext bits to check it.
    """
    if type(inputs) is not tuple or len(inputs) != circuit.spec.n:
        raise ValueError("clear binary-key tuple shape differs")
    if any(type(bit) is not int or bit not in (0, 1) for bit in inputs):
        raise ValueError("clear inputs must be integer binary bits")
    return _interpret(circuit, inputs, operations)


def _interpret(circuit: _Circuit, inputs, operations):
    """Private SSA engine; caller establishes bit domain, never by decrypting.

    Dictionary liveness only; inputs/owners/frames/scratch remain outside quote.
    No decrypt operation exists. This engine is not an admission/owner API.
    """
    if len(inputs) != circuit.spec.n:
        raise ValueError("encrypted binary-key input count differs")
    values = {}
    remaining = list(circuit.uses)
    for index, (op, left, right) in enumerate(circuit.nodes):
        if op == "input":
            result = inputs[left]
        elif op == "const":
            result = operations.constant(left)
        elif op == "not":
            result = operations.negate_bit(values[left])
        elif op == "and":
            result = operations.product(values[left], values[right])
        elif op == "xor":
            result = operations.xor(values[left], values[right])
        else:
            raise ValueError("unknown circuit operation")
        values[index] = result
        for parent in _parents((op, left, right)):
            remaining[parent] -= 1
            if remaining[parent] == 0:
                del values[parent]
        if remaining[index] == 0:
            del values[index]
    return tuple(values[output] for output in circuit.outputs)


class _SealBooleanOperations:
    """Real BFV operations; public-only owner, NEVER a client or decryptor.

    Deliberately private and not an admission bypass API. No rotations, noise
    refresh, modulus normalization or clear phase are smuggled into evaluation.
    Independent binary-key validity remains a client research assumption.
    """

    def __init__(self, seal, context, public_key, relin_keys):
        if not context.parameters_set():
            raise PhaseIndexRefused("invalid SEAL context")
        self.seal = seal
        self.context = context
        self.public_key = public_key
        self.relin_keys = relin_keys
        self.evaluator = seal.Evaluator(context)
        self.encryptor = seal.Encryptor(context, public_key)

    def constant(self, bit):
        result = self.seal.Ciphertext()
        self.encryptor.encrypt(self.seal.Plaintext(str(bit)), result)
        return result

    def product(self, left, right):
        product, result = self.seal.Ciphertext(), self.seal.Ciphertext()
        self.evaluator.multiply(left, right, product)
        self.evaluator.relinearize(product, self.relin_keys, result)
        return result

    def xor(self, left, right):
        product = self.product(left, right)
        twice, total, result = (self.seal.Ciphertext() for _ in range(3))
        self.evaluator.add(product, product, twice)
        self.evaluator.add(left, right, total)
        self.evaluator.sub(total, twice, result)
        return result

    def negate_bit(self, value):
        negative, result = self.seal.Ciphertext(), self.seal.Ciphertext()
        self.evaluator.negate(value, negative)
        self.evaluator.add_plain(negative, self.seal.Plaintext("1"), result)
        return result


@dataclass(frozen=True, slots=True)
class EncryptedIndexBits:
    """Research-local 12-bit result; no production owner/authority is asserted.

    A future admitted harness must derive context/key/source/request identities
    from retained serialized owners with seal_frames, not supplied labels.
    No constructor factory for such authority is provided by this candidate.
    """
    bits: tuple[object, ...]
    _retained_owner: object
    _local_binding: bytes

    def __post_init__(self):
        if len(self.bits) != INDEX_BITS or len(self._local_binding) != 32:
            raise ValueError("exactly twelve encrypted index bits required")


@dataclass(frozen=True, slots=True)
class _LocalPublicOwner:
    """Research-only retained public objects, not a production reservation.

    Kept private: even possession of these objects cannot authorize evaluation.
    No caller context/source/key/request strings are accepted as authority.
    """
    seal: object
    context: object
    public_key: object
    relin_keys: object
    encrypted_secret_bits: tuple[object, ...]
    retained_frames: tuple[bytes, ...]
    circuit: _Circuit
    binding: bytes


def _retain_local_public_owner(circuit, seal, context, public_key, relin_keys,
                               encrypted_secret_bits, directory):
    """Future private harness hook; admission precedes serialization/import.

    Future implementation must reuse seal_frames and admitted retained owners;
    neither framing nor source import/consumption is implemented here. Capacities
    cannot be guessed: they must come from a reviewed BFV profile. With no such
    profile this MUST refuse before serialization. No copied CKKS authority,
    fake capacities or caller-provided labels.
    """
    circuit.quote.require_admission()
    raise PhaseIndexRefused("bounded public-material frame geometry not supplied")


def _execute_retained(circuit, owner):
    """Real gate executor, unreachable without missing admission/exact owner hook."""
    circuit.quote.require_admission()
    if type(owner) is not _LocalPublicOwner or owner.circuit is not circuit:
        raise PhaseIndexRefused("research source owner/request differs")
    # Hash the retained serialized owners, not user-supplied descriptive labels.
    digest = hashlib.sha256(b"pllm.local-phase-index.v1\0" + circuit.digest)
    for frame in owner.retained_frames:
        digest.update(len(frame).to_bytes(8, "little"))
        digest.update(frame)
    if digest.digest() != owner.binding:
        raise PhaseIndexRefused("retained public-material identity differs")
    if len(owner.encrypted_secret_bits) != circuit.spec.n:
        raise PhaseIndexRefused("binary-key source shape differs")
    for cipher in owner.encrypted_secret_bits:
        if cipher.size() != 2 or cipher.parms_id() != owner.context.first_parms_id():
            raise PhaseIndexRefused("binary-key source size or level differs")
    operations = _SealBooleanOperations(owner.seal, owner.context, owner.public_key, owner.relin_keys)
    bits = _interpret(circuit, owner.encrypted_secret_bits, operations)
    return EncryptedIndexBits(bits, owner, owner.binding)


def _backend():
    try:
        import _sealapi_cpp as seal
    except ImportError as exc:
        raise PhaseIndexBackendUnavailable("TenSEAL low-level SEAL bindings required") from exc
    return seal


def evaluate_encrypted_phase_index(circuit: _Circuit, encrypted_secret_bits) -> EncryptedIndexBits:
    """Fail closed BEFORE importing backend or receiving any decryptor.

    Encrypted source bits alone cannot establish a retained source/key owner,
    request freshness, noise margin or resource reservation. There is deliberately
    no 'trusted=True', caller-provided admission quote, or clear fallback.
    """
    if type(circuit) is not _Circuit:
        raise ValueError("compiled phase-index circuit required")
    circuit.quote.require_admission()
    return _execute_retained(circuit, encrypted_secret_bits)

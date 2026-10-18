"""NON-PRODUCTION bounded FuseFSS descriptor/compiler/interpreter specification.

Everything, including masks, thresholds and payloads, is clear reference data.
No FSS keys, shares, randomness, transport, native execution or SDK API exists.
The identity Phi is the only admitted post-circuit in this initial slice.
"""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import struct

PROFILE = "fusefss.source-derived-ring50-q12.v1"
TOY_PROFILE = "fusefss.synthetic-conformance.v1"
SCHEMA = "pllm.fusefss.scalar-descriptor.v1"
GELU_GATE = "sigma-cpu-gelu.affine-q0-q12.identity.v1"
TABLE_HASH = "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b"
MAX_SOURCE_INTERVALS = 512
MAX_PADDED_INTERVALS = 513
MAX_DESCRIPTOR_BYTES = 256 << 10
MAX_INSTANCE_BYTES = 1 << 20
MAX_BOOLEAN_OUTPUTS = 4
MAX_EXPRESSION_NODES = 8192
MAX_EXPRESSION_DEPTH = 12
MAX_QUERIES = 32768
MAX_VM_OPS = 65536
HERE = Path(__file__).resolve().parent


class Invalid(ValueError):
    pass


def integer(value, lower, upper):
    # bool is an int subclass: accepting it would erase A_n/B distinctions.
    if type(value) is not int or not lower <= value <= upper:
        raise Invalid("integer outside descriptor bounds")
    return value


def fingerprint(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Invalid("canonical SHA-256 required")
    return value


@dataclass(frozen=True)
class ArithmeticType:
    bits: int
    fractional: int
    signed: bool = True

    def validate(self):
        integer(self.bits, 1, 50)
        integer(self.fractional, 0, self.bits - 1)
        if type(self.signed) is not bool:
            raise Invalid("explicit signedness required")


@dataclass(frozen=True)
class AffinePiece:
    constant: int
    linear: int
    booleans: tuple = ()


def _expression_nodes(expr, bits, depth=0):
    if type(expr) is not tuple or not 1 <= len(expr) <= 3 or depth > MAX_EXPRESSION_DEPTH:
        raise Invalid("bounded typed Boolean expression required")
    tag, *args = expr
    modulus = 1 << bits
    if tag == "const" and len(args) == 1:
        integer(args[0], 0, 1)
    elif tag == "lt" and len(args) == 1:
        integer(args[0], 0, modulus)
    elif tag == "low_lt" and len(args) == 2:
        integer(args[0], 1, bits)
        integer(args[1], 0, 1 << args[0])
    elif tag == "slt" and len(args) == 1:
        integer(args[0], -modulus // 2, modulus // 2)
    elif tag == "msb" and len(args) == 1:
        integer(args[0], 0, modulus - 1)
    elif (tag == "not" and len(args) == 1) or (tag in ("xor", "and", "or") and len(args) == 2):
        return 1 + sum(_expression_nodes(arg, bits, depth + 1) for arg in args)
    else:
        raise Invalid("unsupported or ill-typed Boolean expression")
    return 1


@dataclass(frozen=True)
class Descriptor:
    word: ArithmeticType
    boundaries: tuple
    pieces: tuple
    profile: str = TOY_PROFILE
    gate: str = "synthetic-affine"
    source_lock_sha256: str = "0" * 64
    phi: str = "identity"

    def validate(self):
        if type(self.word) is not ArithmeticType:
            raise Invalid("arithmetic type required")
        self.word.validate()
        if self.profile not in (PROFILE, TOY_PROFILE):
            raise Invalid("unsupported profile")
        if type(self.gate) is not str or not self.gate.isascii() or not 1 <= len(self.gate) <= 96:
            raise Invalid("bounded ASCII gate identity required")
        fingerprint(self.source_lock_sha256)
        if self.phi != "identity":
            raise Invalid("non-identity Phi is blocked in this slice")
        if self.profile == PROFILE and (self.word != ArithmeticType(50, 12) or self.gate != GELU_GATE):
            raise Invalid("D8 slice admits only ring50/Q12 CPU GELU")
        if type(self.boundaries) is not tuple or type(self.pieces) is not tuple:
            raise Invalid("immutable descriptor sequences required")
        count = len(self.pieces)
        if not 1 <= count <= MAX_SOURCE_INTERVALS or len(self.boundaries) != count + 1:
            raise Invalid("source interval cap or partition shape")
        modulus = 1 << self.word.bits
        for boundary in self.boundaries:
            integer(boundary, 0, modulus)  # N is a wide public sentinel, not a ring word.
        if self.boundaries[0] != 0 or self.boundaries[-1] != modulus:
            raise Invalid("complete canonical partition required")
        if any(a >= b for a, b in zip(self.boundaries, self.boundaries[1:])):
            raise Invalid("strict half-open boundaries required")
        arity = None
        nodes = 0
        for piece in self.pieces:
            if type(piece) is not AffinePiece or type(piece.booleans) is not tuple:
                raise Invalid("typed affine piece required")
            integer(piece.constant, 0, modulus - 1)
            integer(piece.linear, 0, modulus - 1)
            if len(piece.booleans) > MAX_BOOLEAN_OUTPUTS:
                raise Invalid("Boolean output cap")
            if arity is None:
                arity = len(piece.booleans)
            if len(piece.booleans) != arity:
                raise Invalid("uniform Boolean output arity required")
            for expression in piece.booleans:
                nodes += _expression_nodes(expression, self.word.bits)
                if nodes > MAX_EXPRESSION_NODES:
                    raise Invalid("Boolean expression budget")
        if self.profile == PROFILE and arity != 0:
            raise Invalid("GELU slice has no Boolean outputs")

    def object(self):
        self.validate()
        return {
            "schema": SCHEMA,
            "profile": self.profile,
            "gate": self.gate,
            "source_lock_sha256": self.source_lock_sha256,
            "input": {"domain": "A", "bits": self.word.bits, "fractional": self.word.fractional, "signed": self.word.signed},
            "output": {"arithmetic": "same-as-input", "boolean_arity": len(self.pieces[0].booleans)},
            "external_encoding": "nearest-ties-even-input-and-weight",
            "internal_rounding": "signed-floor-only",
            "partition_order": "unsigned-canonical-half-open",
            "degree": 1,
            "coefficient_order": ["constant", "linear"],
            "coefficient_fractional": [self.word.fractional, 0],
            "selector_semantics": "floor-x-div-64-before-abs-clip255" if self.profile == PROFILE else "descriptor-defined",
            "source_schedule": {
                "effective_input": [38, 12],
                "early_tr_output": [32, 6],
                "clipped_index": [8, 6],
                "lut_and_output": [50, 12],
                "required_signed_input_range": [-(1 << 37), 1 << 37],
                "range_status": "requires-sign-extension-provenance-not-a-model-proof",
            } if self.profile == PROFILE else {},
            "phi": self.phi,
            "boundaries": self.boundaries,
            "pieces": [{"coefficients": [p.constant, p.linear], "booleans": p.booleans} for p in self.pieces],
        }

    def canonical_bytes(self):
        result = json.dumps(self.object(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        if len(result) > MAX_DESCRIPTOR_BYTES:
            raise Invalid("descriptor byte budget")
        return result

    def digest(self):
        return sha256(b"FuseFSS-reference-descriptor-v1\0" + self.canonical_bytes()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Invalid("duplicate JSON key")
        result[key] = value
    return result


def _reject_number(_):
    raise Invalid("JSON floating/nonfinite values are not descriptor integers")


def _parse_integer(text):
    if len(text) > 17:
        raise Invalid("descriptor integer token too wide")
    return int(text)


def _tuple_expression(expr, depth=0):
    if type(expr) is not list or not 1 <= len(expr) <= 3 or depth > MAX_EXPRESSION_DEPTH:
        raise Invalid("bounded JSON expression required")
    return tuple(_tuple_expression(v, depth + 1) if type(v) is list else v for v in expr)


def parse_descriptor(encoded):
    if type(encoded) is not bytes or not 0 < len(encoded) <= MAX_DESCRIPTOR_BYTES:
        raise Invalid("descriptor byte budget")
    try:
        obj = json.loads(encoded, object_pairs_hook=_unique_object, parse_int=_parse_integer,
                         parse_float=_reject_number, parse_constant=_reject_number)
        if type(obj) is not dict or type(obj.get("pieces")) is not list or not 1 <= len(obj["pieces"]) <= MAX_SOURCE_INTERVALS:
            raise Invalid("descriptor envelope or source cap")
        if type(obj.get("boundaries")) is not list or len(obj["boundaries"]) != len(obj["pieces"]) + 1:
            raise Invalid("partition shape")
        pieces = []
        for p in obj["pieces"]:
            if type(p) is not dict or set(p) != {"coefficients", "booleans"}:
                raise Invalid("piece fields")
            if type(p["coefficients"]) is not list or len(p["coefficients"]) != 2:
                raise Invalid("fixed degree-one coefficient shape")
            if type(p["booleans"]) is not list or len(p["booleans"]) > MAX_BOOLEAN_OUTPUTS:
                raise Invalid("Boolean output cap")
            pieces.append(AffinePiece(*p["coefficients"], tuple(_tuple_expression(e) for e in p["booleans"])))
        desc = Descriptor(ArithmeticType(obj["input"]["bits"], obj["input"]["fractional"], obj["input"]["signed"]),
                          tuple(obj["boundaries"]), tuple(pieces), obj["profile"], obj["gate"], obj["source_lock_sha256"], obj["phi"])
        # Re-encoding checks every metadata field, type, extra key and normalization.
        if encoded != desc.canonical_bytes():
            raise Invalid("noncanonical or unsupported descriptor metadata")
        return desc
    except (KeyError, TypeError, RecursionError, UnicodeError, json.JSONDecodeError) as error:
        raise Invalid("malformed descriptor") from error


def encode_external(value, word):
    """Exact rational external encoding only; no implicit binary32/64 bridge."""
    word.validate()
    if type(value) is not Fraction:
        raise Invalid("explicit Fraction required; external tensor format remains a gate")
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
        raise Invalid("external rational representation budget")
    scaled = value * (1 << word.fractional)
    lo, remainder = divmod(scaled.numerator, scaled.denominator)
    code = lo + int(2 * remainder > scaled.denominator or (2 * remainder == scaled.denominator and lo % 2 != 0))
    lower = -(1 << (word.bits - 1)) if word.signed else 0
    upper = (1 << (word.bits - 1)) if word.signed else (1 << word.bits)
    if not lower <= code < upper:
        raise Invalid("external encoded value overflows its declared type")
    return code % (1 << word.bits)


def signed(word, bits):
    integer(bits, 1, 50)
    integer(word, 0, (1 << bits) - 1)
    return word if word < 1 << (bits - 1) else word - (1 << bits)


def ars(word, bits, drop):
    integer(drop, 0, bits - 1)
    return (signed(word, bits) // (1 << drop)) % (1 << bits)


def tr(word, bits, drop):
    integer(bits, 1, 50)
    integer(drop, 0, bits - 1)
    integer(word, 0, (1 << bits) - 1)
    return word >> drop  # result type is A_(bits-drop), NOT A_bits


@dataclass(frozen=True)
class Query:
    bits: int
    shift: int
    threshold: int


@dataclass(frozen=True)
class Shape:
    source_intervals: int
    padded_intervals: int
    boolean_outputs: int
    queries: int
    vm_ops: int
    boolean_ands: int
    boolean_xors: int
    reference_bytes: int
    degree: int = 1
    coefficients_per_interval: int = 2
    horner_beaver_products: int = 1
    horner_ring_additions: int = 1
    recover_x_local_subtractions: int = 1
    lookup_interface_calls: int = 1
    phi_beaver_products: int = 0
    phi_b2a: int = 0
    phi_a2b: int = 0

    @property
    def predicate_interface_calls(self):
        return int(self.queries != 0)


class _Emitter:
    def __init__(self, bits, mask, keep):
        self.bits, self.mask, self.keep = bits, mask, keep
        self.query_count = self.ops = self.ands = self.xors = 0
        self.queries = [] if keep else None
        self.programs = [] if keep else None
        self.current = [] if keep else None

    def emit(self, opcode, argument=0):
        self.ops += 1
        if self.ops > MAX_VM_OPS:
            raise Invalid("normalized Boolean work cap")
        self.ands += opcode in ("and", "or")
        self.xors += (opcode == "xor") + 2 * (opcode == "or")
        if self.keep:
            self.current.append((opcode, argument))

    def query(self, bits, shift, threshold):
        index = self.query_count
        self.query_count += 1
        if self.query_count > MAX_QUERIES:
            raise Invalid("comparison query cap")
        if self.keep:
            self.queries.append(Query(bits, shift, threshold))
        self.emit("query", index)

    def lt(self, bits, shift, threshold):
        modulus = 1 << bits
        if threshold in (0, modulus):  # descriptor-level sentinels only
            self.emit("bit", int(threshold == modulus))
            return
        r = self.mask % modulus
        self.query(bits, shift, (r + threshold) % modulus)
        self.query(bits, shift, r)
        self.emit("xor")
        self.emit("bit", int(r + threshold >= modulus))  # secret instance carry in a real backend
        self.emit("xor")

    def expression(self, expression):
        tag, *args = expression
        if tag == "const":
            self.emit("bit", args[0])
        elif tag == "lt":
            self.lt(self.bits, 0, args[0])
        elif tag == "low_lt":
            self.lt(args[0], 0, args[1])
        elif tag == "slt":
            half = 1 << (self.bits - 1)
            self.lt(self.bits, half, args[0] + half)
        elif tag == "msb":
            self.lt(self.bits, args[0], 1 << (self.bits - 1))
            self.emit("not")
        else:
            for arg in args:
                self.expression(arg)
            self.emit(tag)

    def normalize(self, descriptor):
        for output in range(len(descriptor.pieces[0].booleans)):
            for i, piece in enumerate(descriptor.pieces):
                self.lt(self.bits, 0, descriptor.boundaries[i + 1])
                self.lt(self.bits, 0, descriptor.boundaries[i])
                self.emit("xor")
                self.expression(piece.booleans[output])
                self.emit("and")
                if i:
                    self.emit("xor")
            if self.keep:
                self.programs.append(tuple(self.current))
                self.current = []


def admission(descriptor):
    """Static reference layout/work bound, NOT native allocation/RSS admission."""
    descriptor.canonical_bytes()
    n = descriptor.word.bits
    m = len(descriptor.pieces)
    padded = min(m + 1, 1 << n)
    if padded > MAX_PADDED_INTERVALS:
        raise Invalid("padded interval cap")
    dry = _Emitter(n, 0, False)
    dry.normalize(descriptor)  # count work before creating normalized/query arrays
    arity = len(descriptor.pieces[0].booleans)
    word_bytes, boundary_bytes = (n + 7) // 8, (n + 8) // 8
    size = (56 + word_bytes + (padded + 1) * boundary_bytes + padded * (2 + 2 * word_bytes)
            + dry.query_count * (1 + 2 * word_bytes) + arity * 4 + dry.ops * 5)
    if size > MAX_INSTANCE_BYTES:
        raise Invalid("reference instance byte cap")
    return Shape(m, padded, arity, dry.query_count, dry.ops, dry.ands, dry.xors, size)


@dataclass(frozen=True)
class CompiledReference:
    descriptor_digest: str
    word: ArithmeticType
    mask: int
    boundaries: tuple
    ordinals: tuple
    payloads: tuple
    queries: tuple
    programs: tuple
    shape: Shape

    def canonical_bytes(self):
        """Clear diagnostic format; NOT a network/FSS key format. Contains mask."""
        n = self.word.bits
        w, b = (n + 7) // 8, (n + 8) // 8
        chunks = [b"FFSREF01", bytes.fromhex(self.descriptor_digest),
                  struct.pack("<BBBHHIBI", n, self.word.fractional, self.word.signed,
                              self.shape.source_intervals, self.shape.padded_intervals,
                              self.shape.queries, self.shape.boolean_outputs, self.shape.vm_ops),
                  self.mask.to_bytes(w, "little")]
        chunks.extend(x.to_bytes(b, "little") for x in self.boundaries)
        for ordinal, payload in zip(self.ordinals, self.payloads):
            chunks.append(ordinal.to_bytes(2, "little"))
            chunks.extend(x.to_bytes(w, "little") for x in payload)
        for q in self.queries:
            chunks.extend((bytes([q.bits]), q.shift.to_bytes(w, "little"), q.threshold.to_bytes(w, "little")))
        opcodes = {"bit": 0, "query": 1, "not": 2, "xor": 3, "and": 4, "or": 5}
        for program in self.programs:
            chunks.append(len(program).to_bytes(4, "little"))
            chunks.extend(struct.pack("<BI", opcodes[op], argument) for op, argument in program)
        encoded = b"".join(chunks)
        if len(encoded) != self.shape.reference_bytes:
            raise Invalid("reference layout mismatch")
        return encoded


def compile_descriptor(descriptor, mask):
    shape = admission(descriptor)
    modulus = 1 << descriptor.word.bits
    integer(mask, 0, modulus - 1)
    # Protocol 3: ordinal is retained only in this clear reference transcript.
    starts = sorted(((a + mask) % modulus, i) for i, a in enumerate(descriptor.boundaries[:-1]))
    boundaries = [v for v, _ in starts]
    ordinals = [i for _, i in starts]
    if boundaries[0] == 0:
        boundaries.append(modulus)
        if len(ordinals) < modulus:
            j = next(i for i in range(len(ordinals)) if boundaries[i + 1] - boundaries[i] >= 2)
            boundaries.insert(j + 1, boundaries[j] + 1)
            ordinals.insert(j + 1, ordinals[j])
    else:
        boundaries = [0] + boundaries + [modulus]
        ordinals.insert(0, ordinals[-1])
    emitter = _Emitter(descriptor.word.bits, mask, True)
    emitter.normalize(descriptor)
    if (len(ordinals), emitter.query_count, emitter.ops) != (shape.padded_intervals, shape.queries, shape.vm_ops):
        raise Invalid("mask-dependent public shape")
    payloads = tuple((descriptor.pieces[i].constant, descriptor.pieces[i].linear) for i in ordinals)
    return CompiledReference(descriptor.digest(), descriptor.word, mask, tuple(boundaries), tuple(ordinals),
                             payloads, tuple(emitter.queries), tuple(emitter.programs), shape)


def interpret_compiled(instance, masked_word, expected_descriptor_digest):
    """Clear ideal-backend evaluation. Digest binds identity, not authenticity."""
    fingerprint(expected_descriptor_digest)
    if instance.descriptor_digest != expected_descriptor_digest:
        raise Invalid("descriptor substitution")
    modulus = 1 << instance.word.bits
    integer(masked_word, 0, modulus - 1)
    index = bisect_right(instance.boundaries, masked_word) - 1
    coefficients = instance.payloads[index]
    x = (masked_word - instance.mask) % modulus
    result = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = (result * x + coefficient) % modulus  # original x, NOT xhat
    bits = [int((masked_word + q.shift) % (1 << q.bits) < q.threshold) for q in instance.queries]
    outputs = []
    for program in instance.programs:
        stack = []
        for op, argument in program:
            if op == "bit":
                stack.append(argument)
            elif op == "query":
                stack.append(bits[argument])
            elif op == "not":
                stack[-1] ^= 1
            else:
                right, left = stack.pop(), stack.pop()
                stack.append({"xor": left ^ right, "and": left & right, "or": left | right}[op])
        if len(stack) != 1 or stack[0] not in (0, 1):
            raise Invalid("ill-typed Boolean VM result")
        outputs.append(stack[0])
    return result, tuple(outputs)  # identity Phi: no conversions or rescale


def table_digest(table):
    return sha256(b"".join(struct.pack("<Q", value) for value in table)).hexdigest()


def load_table():
    obj = json.loads((HERE / "gelu-table.json").read_bytes())
    values = obj["values"]
    if len(values) != 256 or any(type(x) is not int or not 0 <= x < 4096 for x in values):
        raise Invalid("GELU table shape")
    if obj["sha256_u64le"] != TABLE_HASH or table_digest(values) != TABLE_HASH:
        raise Invalid("GELU table substitution")
    return tuple(values)


def gelu_descriptor(table=None):
    table = load_table() if table is None else table
    if len(table) != 256 or table_digest(table) != TABLE_HASH:
        raise Invalid("GELU table identity")
    modulus = 1 << 50
    # Maximal intervals on which (D,J) is constant, not value-based merging.
    boundaries = (0, *range(64, 256 * 64, 64), modulus // 2,
                  *(modulus - 64 * j for j in range(254, 0, -1)), modulus)
    pieces = []
    for lower in boundaries[:-1]:
        value = lower if lower < modulus // 2 else lower - modulus
        reduced = value // 64
        pieces.append(AffinePiece((-table[min(abs(reduced), 255)]) % modulus, int(reduced >= 0)))
    lock = sha256((HERE / "source-lock.json").read_bytes()).hexdigest()
    descriptor = Descriptor(ArithmeticType(50, 12), boundaries, tuple(pieces), PROFILE, GELU_GATE, lock)
    descriptor.validate()
    return descriptor

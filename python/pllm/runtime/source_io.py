"""Bounded source reads, not model semantics or protected funding authority.

Limits reuse the forward source envelope: 16-MiB metadata, 4-GiB snapshots,
65,536 lexical/inventory nodes, and 1-MiB streaming chunks. Default-free callers
keep their existing I/O behavior; a policy is explicitly passed on strict paths.
"""
from __future__ import annotations

import builtins
import fnmatch
import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import BuiltinFunctionType, CodeType, FunctionType, ModuleType
from typing import Any, Callable, Iterator

from .bundle_storage import CHUNK_BYTES


class SourceReadError(ValueError):
    pass


def _identity(sample):
    return (sample.st_dev, sample.st_ino, sample.st_size, sample.st_mtime_ns,
            sample.st_ctime_ns, sample.st_mode)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SourceReadError(f"duplicate source JSON key {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise SourceReadError(f"nonfinite source JSON value {value!r}")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise SourceReadError("nonfinite source JSON number")
    return result


_SOURCE_INT = int
_SOURCE_TYPE = type
_SOURCE_DICT = dict
_SOURCE_LEN = len
_SOURCE_MAX = max
_SOURCE_VALUE_ERROR = ValueError
_SOURCE_MODULE_GET = ModuleType.__getattribute__
_SOURCE_OBJECT_GET = object.__getattribute__
_SOURCE_TYPE_GET = type.__getattribute__
_ERROR_ABSENT = object()
_JSON_SPACE = (" ", "\t", "\r", "\n")


def _bounded_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SourceReadError("duplicate source JSON key")
        result[key] = value
    return result


def _bounded_constant(value):
    raise SourceReadError("nonfinite source JSON value")


class _ScannerContext:
    """Eight fixed slots; six C Scanner inputs, two trusted backend anchors.

    No JSONDecoder, Python scanner, regex, user hooks or mutable stdlib scanner
    lookup. The factory/converter anchors refer to startup-captured C objects.
    """
    __slots__ = ("strict", "object_hook", "object_pairs_hook", "parse_int",
                 "parse_float", "parse_constant", "_factory", "_slot_bound")

    def __init__(self, factory, converter):
        self.strict = True
        self.object_hook = None
        self.object_pairs_hook = _BOUNDED_OBJECT
        self.parse_int = _SOURCE_INT
        self.parse_float = converter
        self.parse_constant = _BOUNDED_CONSTANT
        self._factory = factory
        self._slot_bound = 137


_BOUNDED_OBJECT = _bounded_object
_BOUNDED_CONSTANT = _bounded_constant


def _bounded_loads(decoded):
    _require_cached_json_error_path()
    factory, converter = _BOUNDED_BACKEND
    context = _ScannerContext(factory, converter)
    scanner = factory(context)  # Fresh C memo per document, never cached input.
    start = 0
    # startswith reads existing Unicode storage directly: unlike decoded[i], it
    # does not populate an input-selected Latin-1 character cache. No slicing.
    while start < len(decoded) and decoded.startswith(_JSON_SPACE, start):
        start += 1
    value, end = scanner(decoded, start)
    while end < len(decoded) and decoded.startswith(_JSON_SPACE, end):
        end += 1
    if end != len(decoded):
        raise SourceReadError("trailing source JSON data")
    return value


def _capture_cached_json_error_path():
    """Capture only the already-loaded stdlib error path at ordinary startup.

    No import priming through a test/host wrapper: __import__ must itself be the
    genuine builtin, not a Python importlib or instrumentation replacement.
    Ordinary startup trust includes sys.modules being the interpreter's actual
    module cache at capture; later rebinding is refused. No C-cache attestation
    or support for pre-startup hostile interpreter surgery is claimed.
    These anchors describe trusted host configuration, not security attestation.
    """
    try:
        modules = sys.modules
        namespace = builtins.__dict__
        importer = namespace.get("__import__")
        decoder = json.decoder
        decoder_dict = decoder.__dict__
        error = decoder_dict["JSONDecodeError"]
        error_dict = _SOURCE_TYPE_GET(error, "__dict__")
        initializer = error_dict.get("__init__")
        spec_type = _SOURCE_TYPE(decoder.__spec__)
        spec_dict = _SOURCE_TYPE_GET(spec_type, "__dict__")
        if (_SOURCE_TYPE(importer) is not BuiltinFunctionType
                or importer.__module__ != "builtins" or importer.__name__ != "__import__"
                or importer.__self__ is not builtins
                or _SOURCE_TYPE(modules) is not _SOURCE_DICT
                or _SOURCE_TYPE(error) is not _SOURCE_TYPE
                or _SOURCE_TYPE(initializer) is not FunctionType
                or error.__mro__[1:] != _SOURCE_VALUE_ERROR.__mro__
                or _SOURCE_TYPE(spec_type) is not _SOURCE_TYPE
                or spec_type.__mro__ != (spec_type, object)):
            return None
        records = tuple((name, module, module.__dict__, module.__spec__, module.__spec__.__dict__)
                        for name, module in (("json", json), ("json.decoder", decoder)))
        return (modules, builtins, namespace, importer,
                _bounded_loads.__globals__.get("__builtins__"), _bounded_loads.__globals__, _bounded_loads,
                decoder, decoder_dict, error, initializer, initializer.__code__,
                decoder_dict.get("__builtins__"), error.__mro__, error_dict,
                spec_type, spec_type.__mro__, spec_dict, spec_dict.get("__dict__"), records)
    except (AttributeError, KeyError, TypeError):
        return None


def _cached_json_module_complete(record, spec_type):
    name, module, namespace, spec, state = record
    # Exact types precede attribute access; fixed-key dict reads cannot dispatch
    # a module __getattr__ or custom spec truthiness/property under this predicate.
    return (_SOURCE_TYPE(module) is ModuleType
            and _CACHED_JSON_ERROR_PATH[0].get(name) is module
            and _SOURCE_MODULE_GET(module, "__dict__") is namespace
            and _SOURCE_TYPE(namespace) is _SOURCE_DICT
            and namespace.get("__spec__") is spec
            and namespace.get("__getattr__", _ERROR_ABSENT) is _ERROR_ABSENT
            and _SOURCE_TYPE(namespace.get("__name__")) is str
            and namespace.get("__name__") == name
            and _SOURCE_TYPE(spec) is spec_type
            and _SOURCE_OBJECT_GET(spec, "__dict__") is state
            and _SOURCE_TYPE(state) is _SOURCE_DICT
            and state.get("_initializing") is False
            and _SOURCE_TYPE(state.get("name")) is str and state.get("name") == name)


def _cached_json_error_path_supported():
    """Fixed-key condition for CPython 3.11--3.13's cached error lookup.

    Inspected Modules/_json.c: 3.11 uses _PyImport_GetModuleId; 3.12/13
    _PyImport_GetModuleAttr calls PyImport_Import. Python/import.c's latter
    reads current scanner-frame globals['__builtins__']['__import__'], passes
    an EMPTY list and level zero, then fetches json.decoder from sys.modules.
    Genuine builtin import takes the exact-dict cached branch for json.decoder,
    sees spec._initializing is exactly False, and recursively returns cached
    parent json for the empty fromlist. Parent's completed spec is checked too.
    Thus no find/load, module locks, _handle_fromlist or Python importlib helper
    is reached. This permits cached import invocation; it is NOT zero imports.

    Host setup must remain stable during the call. Arbitrary host mutation,
    import instrumentation, tracers/profilers/signals/GC hooks are not logical
    input-data owners, nor does this predicate attest a hostile interpreter.
    """
    if _CACHED_JSON_ERROR_PATH is None:
        return False
    (modules, builtin_module, namespace, importer, source_view, source_globals, scanner_frame,
     decoder, decoder_dict, error, initializer, code, decoder_view, error_mro, error_dict,
     spec_type, spec_mro, spec_dict, spec_descriptor, records) = _CACHED_JSON_ERROR_PATH
    if (_SOURCE_TYPE(sys) is not ModuleType or _SOURCE_TYPE(builtin_module) is not ModuleType
            or _SOURCE_MODULE_GET(sys, "__dict__").get("modules") is not modules
            or _SOURCE_TYPE(modules) is not _SOURCE_DICT
            or modules.get("builtins") is not builtin_module
            or _SOURCE_MODULE_GET(builtin_module, "__dict__") is not namespace
            or _SOURCE_TYPE(namespace) is not _SOURCE_DICT
            or namespace.get("__import__") is not importer
            or namespace.get("ValueError") is not _SOURCE_VALUE_ERROR
            or namespace.get("type") is not _SOURCE_TYPE or namespace.get("dict") is not _SOURCE_DICT
            or namespace.get("len") is not _SOURCE_LEN or namespace.get("max") is not _SOURCE_MAX
            or namespace.get("int") is not _SOURCE_INT
            or source_globals.get("__builtins__") is not source_view
            or scanner_frame.__globals__ is not source_globals or scanner_frame.__builtins__ is not namespace
            or (source_view is not namespace and source_view is not builtin_module)
            or decoder_dict.get("__builtins__") is not decoder_view
            or (decoder_view is not namespace and decoder_view is not builtin_module)):
        return False
    # ModuleSpec uses object attribute access and its original __dict__ slot;
    # no hook/descriptor can replace the completed _initializing dictionary fact.
    if (_SOURCE_TYPE(spec_type) is not _SOURCE_TYPE
            or _SOURCE_TYPE_GET(spec_type, "__mro__") is not spec_mro
            or spec_dict.get("__dict__") is not spec_descriptor):
        return False
    for name in ("__getattribute__", "__getattr__", "_initializing"):
        if spec_dict.get(name, _ERROR_ABSENT) is not _ERROR_ABSENT:
            return False
    if not _cached_json_module_complete(records[0], spec_type) or not _cached_json_module_complete(records[1], spec_type):
        return False
    if (source_globals.get("json") is not records[0][1]
            or records[0][2].get("decoder") is not decoder
            or decoder_dict.get("JSONDecodeError") is not error
            or _SOURCE_TYPE(error) is not _SOURCE_TYPE
            or _SOURCE_TYPE_GET(error, "__mro__") is not error_mro
            or error_dict.get("__init__") is not initializer
            or initializer.__code__ is not code or initializer.__globals__ is not decoder_dict
            or initializer.__builtins__ is not namespace
            or initializer.__defaults__ is not None or initializer.__kwdefaults__ is not None
            or initializer.__closure__ is not None
            or decoder_dict.get("ValueError", _ERROR_ABSENT) is not _ERROR_ABSENT):
        return False
    # The unchanged constructor calls builtin ValueError.__init__ and stores
    # these five attributes. Forbid constructor/allocation/store redirection;
    # the unchanged MRO supplies only immutable builtin base implementations.
    for name in ("__new__", "__setattr__", "__getattribute__", "__getattr__", "__del__",
                 "msg", "doc", "pos", "lineno", "colno", "args"):
        if error_dict.get(name, _ERROR_ABSENT) is not _ERROR_ABSENT:
            return False
    return True


def _require_cached_json_error_path():
    if not _cached_json_error_path_supported():
        raise SourceReadError("bounded source JSON cached error configuration unavailable")


def _code_slot_inventory(functions):
    """Source-owned Python frames, including comprehensions/generator bodies.

    Specialist CPython 3.11--3.13 bound: S=9+locals+cells+free+stack <=137
    fits one <=16-KiB interpreter stack chunk. Materialized frame/generator plus
    two traceback objects remain <32 KiB/frame. Machine stacks are NOT priced.
    """
    result = []
    pending = [function.__code__ for function in functions]
    while pending:
        code = pending.pop()
        slots = 9 + code.co_nlocals + len(code.co_cellvars) + len(code.co_freevars) + code.co_stacksize
        result.append((getattr(code, "co_qualname", code.co_name), slots))
        pending.extend(value for value in code.co_consts if type(value) is CodeType)
    return tuple(result)


def _json_frames_supported(functions):
    try:
        return all(slots <= 137 for _, slots in _code_slot_inventory(functions))
    except (AttributeError, TypeError):
        return False


def _capture_inventory_getsizeof():
    primitive = _SOURCE_MODULE_GET(sys, "__dict__").get("getsizeof")
    if (_SOURCE_TYPE(primitive) is not BuiltinFunctionType
            or primitive.__module__ != "sys" or primitive.__name__ != "getsizeof"
            or primitive.__self__ is not sys):
        return None
    return primitive


def _reported_owner_bytes(value):
    """Shallow storage of trusted INTERNAL inventory instances only.

    Internally constructed exact builtins and known source/stdlib classes have
    trusted __sizeof__; no caller objects/hooks or class objects are inventory
    owners. The genuine interpreter primitive includes GC AND managed-instance
    preheaders. Args, dictionaries and other referents are charged separately;
    this is neither recursive storage nor allocator/RSS measurement.
    """
    if _INVENTORY_GETSIZEOF is None:
        raise SourceReadError("source JSON inventory sizing unavailable")
    return _INVENTORY_GETSIZEOF(value)


def _bounded_control_inventory(factory, converter):
    """Measure actual fixed control owners, not a dummy parse or input sample.

    Input-dependent Unicode, DOM, memo entries, pairs, numbers and Python frames
    have separate named census components. Price eight simultaneous <=160-char
    fixed error/control messages, four exception alternatives plus their dicts/
    args, six two-argument/result tuples, three small keyword/attribute tables,
    and 64 address-sized live counters: <=24 lexical/census locals/intermediates,
    <=16 decode/start/end/length/admission values, <=24 error positions/line/column
    and context/control values. Shared hooks are charged in full.
    Error messages never interpolate an input key/token. Caller callbacks and
    host exception retention after API return are explicitly outside this owner.
    """
    context = _ScannerContext(factory, converter)
    scanner = factory(context)
    error_class = _CACHED_JSON_ERROR_PATH[9]
    errors = (SourceReadError("invalid strict source JSON"), ValueError("invalid source JSON"),
              UnicodeDecodeError("utf-8", b"", 0, 1, "invalid source JSON"),
              error_class("invalid source JSON", "", 0))
    sizes = [("context", _reported_owner_bytes(context)),
             ("scanner", _reported_owner_bytes(scanner)),
             ("census", _reported_owner_bytes(_JSONCensus(1, 1, 0, 1, 0, 0, 1))),
             ("fixed_hooks", _reported_owner_bytes(_BOUNDED_OBJECT) + _reported_owner_bytes(_BOUNDED_CONSTANT)),
             ("converter", _reported_owner_bytes(converter)),
             ("ascii_whitespace_constants", _reported_owner_bytes(_JSON_SPACE)
              + sum(_reported_owner_bytes(value) for value in _JSON_SPACE)),
             ("unicode_prefix_method", _reported_owner_bytes("".startswith)),
             ("counter_slots_64", 64 * _reported_owner_bytes(sys.maxsize)),
             ("call_result_tuples_6", 6 * _reported_owner_bytes((None, sys.maxsize))),
             ("small_tables_3", 3 * _reported_owner_bytes({"a": None, "b": None, "c": None})),
             ("fixed_messages_8", 8 * _reported_owner_bytes("x" * 160))]
    for error in errors:
        name = _SOURCE_TYPE(error).__name__
        sizes.append((name, _reported_owner_bytes(error)))
        sizes.append((name + "_args", _reported_owner_bytes(error.args)))
        sizes.append((name + "_attributes", _reported_owner_bytes(error.__dict__)))
    # Actual Python/import.c cached branch: PyList_New(0), OOOOi call tuple,
    # fixed module name/parent substring/attribute name. Referents of the import
    # args are existing globals/modules, not copies of their dictionaries.
    sizes.extend((
        ("cached_import_empty_fromlist", _reported_owner_bytes([])),
        ("cached_import_five_args", _reported_owner_bytes((None, None, None, None, 0))),
        ("cached_import_names", sum(_reported_owner_bytes(value)
                                    for value in ("json.decoder", "json", "JSONDecodeError"))),
        ("error_constructor_three_args", _reported_owner_bytes((None, None, 0))),
        ("error_initializer_self_args", _reported_owner_bytes((None, None, None, 0))),
        ("error_bound_initializer", _reported_owner_bytes(errors[-1].__init__)),
        ("error_base_initializer", _reported_owner_bytes(_SOURCE_VALUE_ERROR.__init__)),
        ("error_initializer_result", _reported_owner_bytes(None)),
        ("cached_host_anchors", _reported_owner_bytes(_CACHED_JSON_ERROR_PATH)
         + _reported_owner_bytes(_CACHED_JSON_ERROR_PATH[19])
         + sum(_reported_owner_bytes(record) for record in _CACHED_JSON_ERROR_PATH[19])
         + _reported_owner_bytes(_CACHED_JSON_ERROR_PATH[14])
         + _reported_owner_bytes(_CACHED_JSON_ERROR_PATH[17]) + _reported_owner_bytes(_ERROR_ABSENT)),
    ))
    # JSONDecodeError result/instance, args and attribute dict are ALREADY priced
    # in errors above; positions/line/column in counter_slots_64, messages above.
    return tuple(sizes)


def _capture_bounded_backend():
    """Ordinary startup trust, NOT security attestation of a hostile interpreter.

    Capture actual stdlib C factory, never json.scanner.make_scanner (which can
    be Python fallback). Native converter's fixed 0.0 call validates its cached
    ABI before any bounded parse. Missing/unsupported builds remain importable
    for legacy SDK users; bounded mode fails closed without lazy module loading.
    The supported, already-cached error-module import lookup is allowed.
    """
    try:
        if _INVENTORY_GETSIZEOF is None:
            return None
        if not _SOURCE_JSON_FRAMES_SUPPORTED:
            return None
        if not _cached_json_error_path_supported():
            return None
        import _json
        from pllm import _native
        factory = _json.make_scanner
        converter = _native.source_json_f64
        if (type(factory) is not type or factory.__module__ != "_json"
                or factory.__name__ != "Scanner"
                or type(converter) is not BuiltinFunctionType
                or sys.implementation.name != "cpython" or sys.abiflags
                or sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13))
                or sys.maxsize != (1 << 63) - 1
                or hasattr(sys, "getobjects") or hasattr(sys, "gettotalrefcount")
                or type(converter("0.0")) is not float):
            return None
        if sum(size for _, size in _bounded_control_inventory(factory, converter)) > 8192:
            return None
        return factory, converter
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        return None


@dataclass(frozen=True, slots=True)
class _JSONCensus:
    """Pre-decode capacity bounds for 64-bit GIL CPython 3.11--3.13 release ABI.

    B=raw bytes, N=all lexical occurrences including keys, C=containers,
    D=depth, T=longest scalar <=1024. Only explicit bounded C mode has this proof.
    R includes raw+Unicode/scalar backing, headers, dict resize and list spare
    capacity, without memo/sharing credits. Writer prices overlapping old/new C
    Unicode buffers, each <=1.5*4B, including widening/finish; no Python fallback
    chunks. Memo and hook pairs get independent resizing/backing components.
    L(T)=32+4*ceil(4T/30) bounds a PyLong. Numeric covers bounded substring/ASCII,
    old/new PyLong and result float. No SRE/dtoa/_pylong input cache on this path.
    Parse has <=2D+12 source-owned Python frames; freeze <=2D+1 (recursive helper
    plus generator/comprehension). Startup code-slot guard checks <=137 slots
    including nine prefix slots, giving <32 KiB/frame with two tracebacks.
    These are logical heap/frame owners, NOT C/Rust machine stacks, allocator
    RSS, existing caller state, arbitrary admission/tracer/profiler/signal/GC
    callbacks, or exception retention past API return. Ordinary callback owners
    and upstream source lifetimes have their own allowances. No margin here.
    """
    raw_bytes: int
    nodes: int
    strings: int
    tokens: int
    containers: int
    max_depth: int
    longest_token: int = 0

    @property
    def retained_bytes(self):
        return 33 + 5 * self.raw_bytes + 384 * self.nodes + 256 * self.containers

    @property
    def freeze_bytes(self):
        return 256 * (self.nodes + self.containers) + 32768 * (2 * self.max_depth + 1)

    @property
    def decoded_bytes(self):
        return 80 + 4 * self.raw_bytes

    @property
    def writer_bytes(self):
        return 256 + 12 * self.raw_bytes

    @property
    def memo_bytes(self):
        return 256 + 256 * self.nodes

    @property
    def hook_pairs_bytes(self):
        return 128 * self.nodes + 256 * self.containers

    @property
    def integer_bytes(self):
        return 32 + 4 * ((4 * self.longest_token + 29) // 30)

    @property
    def numeric_bytes(self):
        return 256 + 4 * self.longest_token + 2 * self.integer_bytes

    @property
    def parse_frame_bytes(self):
        return 32768 * (2 * self.max_depth + 12)

    @property
    def fixed_control_bytes(self):
        return 8192

    @property
    def parse_peak_bytes(self):
        return (self.retained_bytes + self.decoded_bytes + self.writer_bytes
                + self.memo_bytes + self.hook_pairs_bytes + self.numeric_bytes
                + self.fixed_control_bytes + self.parse_frame_bytes)


def _legacy_json_allowance(census):
    """Preserve historical SDK callbacks, NOT a bounded-parser certificate.

    The legacy decoder remains json.loads with its original optional policy
    allowance. Forward never selects this heuristic or Python decoder branch.
    """
    retained = 33 + 5 * census.raw_bytes + 256 * census.nodes + 128 * census.containers
    peak = (retained + 80 + 8 * census.raw_bytes + 384 * census.nodes
            + 128 * census.containers + 8192 * (census.max_depth + 1))
    return retained, peak


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceReadPolicy:
    metadata_bytes: int = 16 * CHUNK_BYTES
    file_bytes: int = 4 << 30
    inventory_nodes: int = 65536
    # Opt-in only: legacy/default decoding is NOT covered by the C-path census.
    bounded_json: bool = False
    # Optional ordinary RAM admission, never protected-source authorization.
    # Receives a conservative byte estimate before allocation, not an owner.
    metadata_admit: Callable[[int], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if type(self.bounded_json) is not bool:
            raise SourceReadError("invalid bounded source JSON mode")
        for name, ceiling in (("metadata_bytes", 16 * CHUNK_BYTES),
                              ("file_bytes", 4 << 30), ("inventory_nodes", 65536)):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= ceiling:
                raise SourceReadError(f"invalid source {name} limit")

    def _require_bounded_backend(self):
        if self.bounded_json and (_BOUNDED_BACKEND is None or not _SOURCE_JSON_FRAMES_SUPPORTED):
            raise SourceReadError("bounded source JSON C/native backend unavailable")
        if self.bounded_json:
            _require_cached_json_error_path()

    def _admit(self, size):
        if type(size) is not int or not 0 <= size <= sys.maxsize:
            raise SourceReadError("source storage exceeds addressable byte envelope")
        if self.metadata_admit is not None:
            self.metadata_admit(size)

    @property
    def max_metadata_bytes(self):
        return self.metadata_bytes

    @property
    def max_file_bytes(self):
        return self.file_bytes

    @property
    def max_files(self):
        return min(self.inventory_nodes, self.metadata_bytes // 256)

    @property
    def max_name_bytes(self):
        return self.metadata_bytes

    def _limit(self, limit):
        if limit is None:
            return self.metadata_bytes
        if type(limit) is not int or limit <= 0:
            raise SourceReadError("invalid source metadata read limit")
        return min(limit, self.metadata_bytes)

    def chunks(self, path: str | Path) -> Iterator[bytes]:
        """One bounded regular-file descriptor; identity/length checked at EOF.

        Consumers must exhaust this iterator before accepting its contents.
        Snapshot import and hash_file do so; no timestamp cache authenticates it.
        """
        self._require_bounded_backend()
        path = Path(path)
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= self.file_bytes:
                raise SourceReadError("source file exceeds regular-file snapshot envelope")
            expected, count = before.st_size, 0
            self._admit(3 * CHUNK_BYTES)
            while True:
                block = stream.read(min(CHUNK_BYTES, expected - count + 1))
                if not block:
                    break
                count += len(block)
                if count > expected or count > self.file_bytes:
                    raise SourceReadError("source file grew while reading")
                yield block
            if (count != expected or _identity(before) != _identity(os.fstat(stream.fileno()))
                    or _identity(before) != _identity(path.stat())):
                raise SourceReadError("source file changed or truncated while reading")

    def read_bytes(self, path: str | Path, *, limit: int | None = None) -> bytes:
        self._require_bounded_backend()
        limit = self._limit(limit)
        path = Path(path)
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= min(limit, self.file_bytes):
                raise SourceReadError("source metadata exceeds bounded regular-file envelope")
            self._admit(16 * before.st_size + 3 * CHUNK_BYTES)
            raw = stream.read(min(limit, before.st_size) + 1)
            if (len(raw) > limit or len(raw) != before.st_size
                    or _identity(before) != _identity(os.fstat(stream.fileno()))
                    or _identity(before) != _identity(path.stat())):
                raise SourceReadError("source metadata changed, truncated, or exceeded limit")
            return raw

    def _json_census(self, raw: bytes, *, limit: int | None = None) -> _JSONCensus:
        """Constant-state lexical bounds, not a second JSON grammar."""
        limit = self._limit(limit)
        if type(raw) is not bytes or not 0 < len(raw) <= limit:
            raise SourceReadError("invalid bounded source JSON bytes")
        nodes = depth = maximum = strings = tokens = containers = 0
        token_length = longest = 0
        quoted = escaped = token = False
        for byte in raw:
            if quoted:
                if escaped:
                    escaped = False
                elif byte == 92:
                    escaped = True
                elif byte == 34:
                    quoted = False
                continue
            if self.bounded_json and byte >= 128:
                raise SourceReadError("non-ASCII source JSON outside string")
            if byte in (32, 9, 10, 13, 44, 58):
                token = False
                token_length = 0
                continue
            if byte in (93, 125):
                depth -= 1
                if depth < 0:
                    raise SourceReadError("invalid source JSON nesting")
                token = False
                token_length = 0
                continue
            if byte in (91, 123):
                depth += 1
                maximum = max(maximum, depth)
                containers += 1
                nodes += 1
                token = False
                token_length = 0
            elif byte == 34:
                strings += 1
                nodes += 1
                quoted = True
                token = False
                token_length = 0
            else:
                if not token:
                    tokens += 1
                    nodes += 1
                    token = True
                # Count first character, minus, decimal and exponent bytes too;
                # malformed scalars get the SAME limit, without a second grammar.
                token_length += 1
                longest = max(longest, token_length)
                if self.bounded_json and token_length > 1024:
                    raise SourceReadError("source JSON scalar exceeds 1024-byte envelope")
            if nodes > self.inventory_nodes or depth > 17:
                raise SourceReadError("source JSON exceeds lexical node/depth envelope")
        census = _JSONCensus(len(raw), nodes, strings, tokens, containers, maximum, longest)
        if max(census.parse_peak_bytes, census.retained_bytes + census.freeze_bytes) > sys.maxsize:
            raise SourceReadError("source JSON exceeds addressable byte envelope")
        return census

    def _parse_json(self, raw: bytes, *, limit: int | None = None,
                    retain: Callable[[int], None] | None = None) -> tuple[Any, _JSONCensus]:
        """Admit before decode/hook; transfer retained charge before scratch dies.

        Native getter bytes already exist: native transfer admission, not this
        scanner, must cover that earlier allocation. Callback adds live owners.
        Only bounded mode uses the C-path proof; legacy keeps its old heuristic.
        """
        self._require_bounded_backend()
        census = self._json_census(raw, limit=limit)
        if self.bounded_json:
            retained_bytes, parse_bytes = census.retained_bytes, census.parse_peak_bytes
        else:
            retained_bytes, parse_bytes = _legacy_json_allowance(census)
        self._admit(parse_bytes)
        if self.bounded_json:
            # Admission callback may have changed host setup. Recheck directly
            # before UTF-8 decode, then again at genuine scanner entry.
            _require_cached_json_error_path()
        try:
            decoded = raw.decode("utf-8")
            if self.bounded_json:
                value = _bounded_loads(decoded)
            else:
                # Compatibility path only. No bounded C-path allocation claim.
                value = json.loads(decoded, object_pairs_hook=_object,
                                   parse_constant=_constant, parse_float=_float)
        except (ValueError, UnicodeError, RecursionError, StopIteration) as exc:
            raise SourceReadError("invalid strict source JSON") from exc
        if retain is not None:
            retain(retained_bytes)
        # decoded/raw remain live until retained transfer succeeds. Pair/memo
        # scratch lives only inside loads, covered by the admitted phase peak.
        return value, census

    def parse_json(self, raw: bytes, *, limit: int | None = None) -> Any:
        """Parse exact bytes; only bounded_json=True selects the C-path proof."""
        return self._parse_json(raw, limit=limit)[0]

    def read_json(self, path: str | Path, *, limit: int | None = None) -> Any:
        return self.parse_json(self.read_bytes(path, limit=limit), limit=limit)

    def hash_file(self, path: str | Path) -> str:
        digest = hashlib.sha256()
        for block in self.chunks(path):
            digest.update(block)
        return digest.hexdigest()

    def directory_files(self, root: str | Path, *, patterns: tuple[str, ...]) -> tuple[Path, ...]:
        """Cap all scanned names/nodes before accumulating selected paths."""
        self._require_bounded_backend()
        root = Path(root)
        selected, count, name_bytes = [], 0, 0
        with os.scandir(root) as directory:
            for entry in directory:
                count += 1
                name_bytes += len(os.fsencode(entry.name))
                if count > self.max_files or name_bytes + 256 * count > self.metadata_bytes:
                    raise SourceReadError("source inventory exceeds metadata envelope")
                self._admit(16 * name_bytes + 256 * count + 3 * CHUNK_BYTES)
                if any(fnmatch.fnmatchcase(entry.name, pattern) for pattern in patterns) and entry.is_file():
                    selected.append(root / entry.name)
        return tuple(sorted(selected, key=lambda path: path.name))


# Capture/check at ordinary module initialization, never inside admitted parsing.
# A later monkeypatch of json.loads/json.scanner/_json globals cannot select a
# different backend. This is startup trust, not resistance to hostile module edits.
_INVENTORY_GETSIZEOF = _capture_inventory_getsizeof()
_CACHED_JSON_ERROR_PATH = _capture_cached_json_error_path()
_SOURCE_JSON_FRAME_FUNCTIONS = (
    _bounded_object, _bounded_constant, _ScannerContext.__init__, _bounded_loads,
    _JSONCensus.__init__, SourceReadPolicy.__init__, SourceReadPolicy.__post_init__,
    SourceReadPolicy._require_bounded_backend, SourceReadPolicy._limit,
    SourceReadPolicy._json_census, SourceReadPolicy._parse_json,
    SourceReadPolicy.parse_json, SourceReadPolicy.read_json, SourceReadPolicy._admit,
    _identity, SourceReadPolicy.chunks, SourceReadPolicy.read_bytes,
    SourceReadPolicy.hash_file, SourceReadPolicy.directory_files,
    *[value.fget for value in vars(_JSONCensus).values() if type(value) is property],
    *[value.fget for value in vars(SourceReadPolicy).values() if type(value) is property],
    *((_CACHED_JSON_ERROR_PATH[10],) if _CACHED_JSON_ERROR_PATH is not None else ()),
    _capture_cached_json_error_path, _cached_json_module_complete,
    _cached_json_error_path_supported, _require_cached_json_error_path,
    _capture_inventory_getsizeof, _reported_owner_bytes, _bounded_control_inventory,
)
_SOURCE_JSON_FRAMES_SUPPORTED = _json_frames_supported(_SOURCE_JSON_FRAME_FUNCTIONS)
_BOUNDED_BACKEND = _capture_bounded_backend()

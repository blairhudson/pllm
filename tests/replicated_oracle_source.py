"""TEST ONLY: bounded mock-publication consumer, never execution qualification.

No filesystem/native/source-authority integration exists here. A PublicationView
is metadata, NOT a credential. Its mock commitments prove consistency with the
supplied mock bytes, not authentic origin, canonical publication admission, or
executed-source identity. Numeric/schedule identities explicitly name this mock.
Limits bound serialized buffers/word counts, NOT physical Python heap: object
integers, NumPy temporaries, callbacks, replay and transport need separate funded
owner contracts. Trained layout mapping, independent RoPE and native capture
remain unavailable; require_execution_qualification always refuses.

The separately admitted fixed-cardinality frame fingerprint index is ephemeral
mock integrity metadata, not a payload cache, receipt archive or authority proof.
Its SHA256 records bind rereads to the exact partitions hashed at verification.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import struct

import numpy as np


class SourceError(ValueError):
    pass


ORIGINAL = "pllm.test.mock.original-le.v1"
CANONICAL = "pllm.test.mock.canonical-f32-le.v1"
ENCODED = "pllm.test.mock.encoded-q18-i64-le.v1"
NUMERIC = "mock-public-q18-epsilon-one-not-source-epsilon"
SCHEDULE = "mock-consumer-decoder-order-not-native-layout"
_WIDTH = {"F32": 4, "F16": 2, "BF16": 2}


def _int(n, low=0, high=(1 << 63)-1):
    if type(n) is not int or not low <= n <= high:
        raise SourceError("exact bounded source integer required")
    return n


def _text(s):
    if type(s) is not str or not s:
        raise SourceError("nonempty source identity required")


def _digest(s):
    if type(s) is not str or len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise SourceError("mock digest syntax")


def _hash(domain):
    # Explicit MOCK framing: ASCII domain, NUL, concatenated C-order bytes.
    return hashlib.sha256(domain.encode("ascii") + b"\0")


@dataclass(frozen=True)
class Descriptor:
    name: str
    source_id: str
    shard_id: str
    dtype: str
    shape: tuple
    byte_offset: int
    byte_length: int
    shard_extent: int
    original_digest: str
    canonical_id: str
    canonical_length: int
    canonical_digest: str
    encoded_id: str
    encoded_length: int
    encoded_digest: str
    order: str = "C"
    endian: str = "LE"
    original_domain: str = ORIGINAL
    canonical_domain: str = CANONICAL
    encoded_domain: str = ENCODED

    def validate(self):
        for s in (self.name, self.source_id, self.shard_id, self.canonical_id, self.encoded_id):
            _text(s)
        if (type(self.dtype) is not str or self.dtype not in _WIDTH
                or type(self.order) is not str or self.order != "C"
                or type(self.endian) is not str or self.endian != "LE"):
            raise SourceError("source dtype/order/endian")
        if type(self.shape) is not tuple or len(self.shape) not in (1, 2):
            raise SourceError("source rank/immutable shape")
        for n in self.shape:
            _int(n, 1, 65536 if len(self.shape) == 1 else (1 << 31)-1)
        if self.shape[-1] > 65536:
            raise SourceError("source columns")
        words = math.prod(self.shape)
        for n in (self.byte_offset, self.byte_length, self.shard_extent,
                  self.canonical_length, self.encoded_length):
            _int(n)
        if (words > ((1 << 63)-1)//8 or self.byte_length != words*_WIDTH[self.dtype]
                or self.canonical_length != words*4 or self.encoded_length != words*8
                or self.byte_offset + self.byte_length > self.shard_extent):
            raise SourceError("source extent/range mismatch")
        if (self.original_domain, self.canonical_domain, self.encoded_domain) != (ORIGINAL, CANONICAL, ENCODED):
            raise SourceError("unsupported non-mock commitment domains")
        for d in (self.original_digest, self.canonical_digest, self.encoded_digest):
            _digest(d)
        if len({self.source_id, self.canonical_id, self.encoded_id}) != 3:
            raise SourceError("original/canonical/encoded identities must be distinct")


@dataclass(frozen=True)
class Geometry:
    config_bytes: bytes
    input_rows: int

    def config(self):
        if type(self.config_bytes) is not bytes or not 0 < len(self.config_bytes) <= 65536:
            raise SourceError("bounded frozen config bytes")
        _int(self.input_rows, 1, 32)
        c = json.loads(self.config_bytes)
        if type(c) is not dict or c.get("model_type") not in ("qwen2", "qwen3", "llama"):
            raise SourceError("source model family")
        for k in ("hidden_size", "intermediate_size", "num_attention_heads",
                  "num_key_value_heads", "vocab_size", "num_hidden_layers"):
            _int(c.get(k), 1, 64 if k == "num_hidden_layers" else (1 << 31)-1)
        h, k, H = c["num_attention_heads"], c["num_key_value_heads"], c["hidden_size"]
        d = c.get("head_dim", H//h)
        _int(d, 2, 128)
        if h % k or d % 2 or max(H, c["intermediate_size"], h*d) > 65536:
            raise SourceError("source geometry")
        for key, default in (("tie_word_embeddings", False), ("attention_bias", c["model_type"] == "qwen2")):
            if type(c.get(key, default)) is not bool:
                raise SourceError("source alias/bias policy")
        if (c.get("rope_scaling") or c.get("use_sliding_window")
                or c.get("partial_rotary_factor", 1) != 1
                or c.get("attention_dropout", 0) != 0 or c.get("hidden_act", "silu") != "silu"):
            raise SourceError("unsupported source operators")
        return c

    def roles(self):
        c = self.config()
        H, M, h, k = (c[x] for x in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"))
        d, V = c.get("head_dim", H//h), c["vocab_size"]
        yield "model.embed_tokens.weight", (V, H)
        for l in range(c["num_hidden_layers"]):
            p = f"model.layers.{l}."
            yield p + "input_layernorm.weight", (H,)
            for kind, heads in (("q", h), ("k", k), ("v", k)):
                if c.get("attention_bias", c["model_type"] == "qwen2"):
                    yield p + f"self_attn.{kind}_proj.bias", (heads*d,)
                yield p + f"self_attn.{kind}_proj.weight", (heads*d, H)
                if c["model_type"] == "qwen3" and kind != "v":
                    yield p + f"self_attn.{kind}_norm.weight", (d,)
            yield p + "self_attn.o_proj.weight", (H, h*d)
            yield p + "post_attention_layernorm.weight", (H,)
            yield p + "mlp.gate_proj.weight", (M, H)
            yield p + "mlp.up_proj.weight", (M, H)
            yield p + "mlp.down_proj.weight", (H, M)
        yield "model.norm.weight", (H,)
        yield "lm_head.weight", (V, H)


@dataclass(frozen=True)
class Limits:
    """Logical quotas only; index_bytes counts 32-byte fingerprints, not heap.

    Offset-table cardinality is separately bounded by descriptor count, which
    cannot exceed index_records (each descriptor has at least one partition).
    No physical size of dict entries, ints, allocators or hash objects is claimed.
    """
    frame_words: int = 65536
    scratch_bytes: int = 65536*32
    vector_words: int = 65536
    gather_words: int = 65536
    index_records: int = 65536
    index_bytes: int = 65536*32

    def validate(self):
        for n in (self.frame_words, self.scratch_bytes, self.vector_words, self.gather_words,
                  self.index_records, self.index_bytes):
            _int(n, 1)
        if self.frame_words > 65536 or self.index_records > 65536 or self.index_bytes > 65536*32:
            raise SourceError("source frame cap")


@dataclass(frozen=True)
class PublicationView:
    revision: str
    publication_id: str
    canonical_publication_id: str
    config_bytes: bytes
    generation: int
    descriptors: tuple
    aliases: tuple
    owner: object
    numeric_identity: str = NUMERIC
    schedule_identity: str = SCHEDULE


def widen_le(raw, dtype):
    """Exact IEEE widening only. Caller preflights frame before allocation."""
    if (type(dtype) is not str or dtype not in _WIDTH
            or type(raw) not in (bytes, bytearray, memoryview) or not raw
            or len(raw) % _WIDTH[dtype] or len(raw)//_WIDTH[dtype] > 65536
            or type(raw) is memoryview and (raw.ndim != 1 or raw.format != "B" or not raw.c_contiguous)):
        raise SourceError("complete LE IEEE frame required")
    out = bytearray(4*(len(raw)//_WIDTH[dtype]))
    for i in range(len(out)//4):
        at = i*_WIDTH[dtype]
        bits = int.from_bytes(raw[at:at+_WIDTH[dtype]], "little")
        if dtype == "BF16":
            bits <<= 16
        elif dtype == "F16":
            sign, exponent, fraction = (bits & 0x8000) << 16, (bits >> 10) & 31, bits & 1023
            if exponent == 31:
                raise SourceError("nonfinite original IEEE value")
            if exponent:
                bits = sign | ((exponent+112) << 23) | (fraction << 13)
            elif fraction:
                lead = fraction.bit_length()-1
                bits = sign | ((lead+103) << 23) | ((fraction-(1 << lead)) << (23-lead))
            else:
                bits = sign
        if bits & 0x7f800000 == 0x7f800000:
            raise SourceError("nonfinite original IEEE value")
        struct.pack_into("<I", out, 4*i, bits)
    return out


try:
    _EXCEPTION_GROUP = BaseExceptionGroup
except NameError:  # Grouped exceptions were introduced in Python 3.11.
    _EXCEPTION_GROUP = ()


def _scrub_error(error):
    """Discard subordinate payload-bearing traceback frames, including contexts.

    Only used on mock-source paths. Preserve exception type, message and chain;
    never promise to sanitize arbitrary exception attributes or callback heaps.
    Group children participate in the same identity-visited graph traversal as
    cause/context, so shared children and cyclic context links terminate.
    """
    pending, seen = [error], set()
    while pending:
        caught = pending.pop()
        if id(caught) not in seen:
            seen.add(id(caught))
            caught.__traceback__ = None
            pending.extend(e for e in (caught.__cause__, caught.__context__) if e is not None)
            if isinstance(caught, _EXCEPTION_GROUP):
                pending.extend(caught.exceptions)
    return error


class SourceWeights(Mapping):
    """Scoped synchronous pull owner; closures retain descriptors, never payloads.

    Owner must implement generation, readinto(id,tensor_relative_offset,dest),
    close(). This is a trusted MOCK owner protocol, not a production lease proof.
    Callback must not retain destination. We release its memoryview on all paths.
    """

    def __init__(self, view, geometry, limits, *, source_f32, matrix_type, array):
        self._view = None
        self._state = "preflight"
        self._busy = False
        self._poisoned = False
        self._entered = self._decoder_running = self._decoder_completed = False
        self._gathering = False
        if type(view) is not PublicationView or type(geometry) is not Geometry or type(limits) is not Limits:
            raise SourceError("immutable source contracts required")
        limits.validate()
        if not all(callable(helper) for helper in (source_f32, matrix_type, array)):
            raise SourceError("injected existing codec/array/matrix helpers required")
        c = geometry.config()
        if (type(view.config_bytes) is not bytes or view.config_bytes != geometry.config_bytes or view.numeric_identity != NUMERIC
                or view.schedule_identity != SCHEDULE):
            raise SourceError("source config/profile disagreement")
        for s in (view.revision, view.publication_id, view.canonical_publication_id):
            _text(s)
        if view.publication_id == view.canonical_publication_id:
            raise SourceError("distinct publication identities required")
        _int(view.generation)
        if (type(view.descriptors) is not tuple or type(view.aliases) is not tuple
                or not callable(getattr(view.owner, "readinto", None))
                or not callable(getattr(view.owner, "close", None))):
            raise SourceError("pinned rereadable mock owner required")
        roles = tuple(geometry.roles())
        tied = c.get("tie_word_embeddings", False)
        expected_aliases = (("lm_head.weight", "model.embed_tokens.weight"),) if tied else ()
        if view.aliases != expected_aliases:
            raise SourceError("source alias disagreement")
        required = roles[:-1] if tied else roles
        if len(view.descriptors) != len(required):
            raise SourceError("missing/extra source bindings")
        sources, ranges, canonical_ids, encoded_ids = {}, {}, set(), set()
        records = 0
        for desc, (name, shape) in zip(view.descriptors, required):
            if type(desc) is not Descriptor:
                raise SourceError("immutable source descriptor required")
            desc.validate()
            if desc.name != name or desc.shape != shape or desc.source_id in sources:
                raise SourceError("source binding/shape/order disagreement")
            if desc.canonical_id in canonical_ids or desc.encoded_id in encoded_ids:
                raise SourceError("duplicate canonical/encoded source")
            canonical_ids.add(desc.canonical_id)
            encoded_ids.add(desc.encoded_id)
            for start, stop, extent in ranges.get(desc.shard_id, ()):
                if extent != desc.shard_extent or max(start, desc.byte_offset) < min(stop, desc.byte_offset+desc.byte_length):
                    raise SourceError("overlapping physical sources/shard extent")
            ranges.setdefault(desc.shard_id, []).append((desc.byte_offset, desc.byte_offset+desc.byte_length, desc.shard_extent))
            sources[desc.source_id] = desc
            words = min(desc.shape[0], 128, 65536//desc.shape[1])*desc.shape[1] if len(shape) == 2 else shape[0]
            self._admit(desc, words, limits)
            if len(shape) == 1 and shape[0] > limits.vector_words:
                raise SourceError("source vector budget")
            step = 1 if len(shape) == 1 else min(128, 65536//shape[1])
            records += 1 if len(shape) == 1 else (shape[0]+step-1)//step
        # Separate metadata admission. Cardinality counts actual canonical read
        # partitions ONCE per descriptor, not roles, selections, words or bytes.
        # Fixed 32-byte records contain no payload and confer no authority.
        if records > limits.index_records or 32*records > limits.index_bytes:
            raise SourceError("source fingerprint index budget")
        if (geometry.input_rows+1)*c["hidden_size"] > limits.gather_words:
            raise SourceError("source gather budget")
        all_ids = [d.source_id for d in view.descriptors] + list(canonical_ids) + list(encoded_ids)
        if len(set(all_ids)) != len(all_ids):
            raise SourceError("cross-domain source identity alias")
        self._descs = {d.name: d for d in view.descriptors}
        self._roles, self._geometry, self._limits = roles, geometry, limits
        self._quantize, self._matrix, self._array = source_f32, matrix_type, array
        self._view, self._state = view, "verifying"
        try:
            self._check_owner()
            self._frame_hashes = bytearray(32*records)
            self._index_offsets = {}
            offset = 0
            for desc in view.descriptors:
                self._index_offsets[desc.source_id] = offset
                if len(desc.shape) == 1:
                    offset += 1
                else:
                    step = min(128, 65536//desc.shape[1])
                    offset += (desc.shape[0]+step-1)//step
            self._verify()
            self._cursor = iter(self._demands())
            self._next = next(self._cursor, None)
            self._selection = 0
            self._state = "active"
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _admit(desc, words, limits):
        # Raw + canonical + optional encoded hash frame + two serialized i64
        # owners. Python list/object/NumPy heap is explicitly NOT admitted.
        if words > limits.frame_words or words*(_WIDTH[desc.dtype]+28) > limits.scratch_bytes:
            raise SourceError("source frame/scratch budget")

    def _check_owner(self):
        generation = None if self._view is None else self._view.owner.generation
        if self._view is None or type(generation) is not int or generation != self._view.generation:
            raise SourceError("closed/changed source generation")

    def _active(self):
        if self._busy:
            self._poisoned, self._state = True, "failed"
            raise SourceError("reentrant source demand")
        if self._state != "active":
            raise SourceError("source scope not active")
        try:
            self._check_owner()
        except BaseException:
            self.close()
            raise

    def _frames(self, desc):
        if len(desc.shape) == 1:
            yield 0, desc.shape[0]
        else:
            step = min(128, 65536//desc.shape[1])
            for start in range(0, desc.shape[0], step):
                yield start*desc.shape[1], min(start+step, desc.shape[0])*desc.shape[1]

    def _frame(self, desc, start, stop):
        if self._busy:
            self._poisoned, self._state = True, "failed"
            raise SourceError("reentrant source demand")
        self._busy = True
        raw = canonical = values = destination = count = None
        failure = None
        try:
            self._check_owner()
            _int(start)
            _int(stop, 1)
            words = stop-start
            if words <= 0 or stop > math.prod(desc.shape):
                raise SourceError("source flat range")
            self._admit(desc, words, self._limits)
            stride = desc.shape[0] if len(desc.shape) == 1 else min(128, 65536//desc.shape[1])*desc.shape[1]
            if start % stride or stop != min(start+stride, math.prod(desc.shape)):
                raise SourceError("noncanonical source fingerprint partition")
            record = self._index_offsets[desc.source_id] + start//stride
            raw = bytearray(words*_WIDTH[desc.dtype])
            destination = memoryview(raw)
            count = self._view.owner.readinto(desc.source_id, start*_WIDTH[desc.dtype], destination)
            if type(count) is not int or count != len(raw):
                raise SourceError("short/oversized source readinto")
            if self._poisoned:
                raise SourceError("poisoned source callback")
            self._check_owner()
            destination.release()
            destination = None
            digest = _hash("pllm.test.mock.reread-frame.v1")
            digest.update(raw)
            fingerprint = digest.digest()
            if self._state == "verifying":
                self._frame_hashes[32*record:32*(record+1)] = fingerprint
            elif self._frame_hashes[32*record:32*(record+1)] != fingerprint:
                raise SourceError("same-generation source reread content mismatch")
            canonical = widen_le(raw, desc.dtype)
            values = self._array(self._quantize(canonical))
            if self._poisoned:
                raise SourceError("poisoned source callback")
            if values.shape != (words,):
                raise SourceError("injected source codec flat shape")
        except BaseException as error:
            failure = _scrub_error(error)
            self._poisoned, self._state = True, "failed"
        finally:
            if destination is not None:
                destination.release()
            # A rejected return is arbitrary callback data, potentially an
            # ndarray owning source bytes. Never retain it in a new traceback.
            destination = count = None
            self._busy = False
        if failure is not None:
            # No callback or destination borrow is live at owner cleanup. A
            # throwing close cannot bypass scrubbing of the primary failure.
            raw = canonical = values = None
            self.abort_decoder(failure)
        return raw, canonical, values

    def _verify(self):
        for desc in self._view.descriptors:
            original, canonical_hash, encoded = (_hash(d) for d in (ORIGINAL, CANONICAL, ENCODED))
            for start, stop in self._frames(desc):
                raw, canonical, values = self._frame(desc, start, stop)
                try:
                    original.update(raw)
                    canonical_hash.update(canonical)
                    for word in values.flat:
                        encoded.update(struct.pack("<q", int(word)))
                finally:
                    del raw, canonical, values
            if (original.hexdigest(), canonical_hash.hexdigest(), encoded.hexdigest()) != (
                    desc.original_digest, desc.canonical_digest, desc.encoded_digest):
                raise SourceError("mock original/canonical/encoded commitment mismatch")

    def _demands(self):
        for selection in range(2):
            for name, shape in self._roles:
                desc = self._resolve(name)
                for start, stop in self._frames(desc):
                    yield selection, name, start, stop

    def _resolve(self, name):
        if name == "lm_head.weight" and self._view.aliases:
            name = "model.embed_tokens.weight"
        return self._descs[name]

    def _take(self, selection, name, start, stop):
        try:
            self._active()
            if name == "model.embed_tokens.weight" and not self._gathering:
                raise SourceError("embedding role reserved for semantic gather")
            if self._next != (selection, name, start, stop):
                raise SourceError("mock public source demand order/range")
            desc = self._resolve(name)
            raw, canonical, values = self._frame(desc, start, stop)
            del raw, canonical
            self._next = next(self._cursor, None)
            return values
        except BaseException:
            self.close()
            raise
        finally:
            values = None

    def __iter__(self):
        self._active()
        return iter(name for name, _ in self._roles)

    def __len__(self):
        return len(self._roles)

    def __contains__(self, name):
        self._active()
        return name in self._descs or name == "lm_head.weight" and bool(self._view.aliases)

    def __getitem__(self, name):
        self._active()
        desc = self._resolve(name)
        if len(desc.shape) == 1:
            return self._take(self._selection, name, 0, desc.shape[0])
        logical = "lm_head.weight" if name == "model.embed_tokens.weight" and self._view.aliases else name
        def rows(start, stop):
            try:
                self._active()
                _int(start)
                _int(stop, 1)
                cols = desc.shape[1]
                return self._take(self._selection, logical, start*cols, stop*cols).reshape(stop-start, cols)
            except BaseException:
                self.close()
                raise
        return self._matrix(desc.shape, rows)

    def gather_embedding(self, ids):
        """Public all-V scan; retain only n*H selected words, no selected offsets."""
        result = block = None
        try:
            self._active()
            if type(ids) is not list or len(ids) != self._geometry.input_rows+self._selection:
                raise SourceError("source actual input count/two selections")
            desc = self._descs["model.embed_tokens.weight"]
            V, H = desc.shape
            if any(type(i) is not int or not 0 <= i < V for i in ids):
                raise SourceError("source private token domain")
            if len(ids)*H > self._limits.gather_words:
                raise SourceError("source gather budget")
            self._gathering = True
            result = np.empty((len(ids), H), dtype=object)
            for start, stop in self._frames(desc):
                block = self._take(self._selection, desc.name, start, stop).reshape(-1, H)
                for index, token in enumerate(ids):
                    if start//H <= token < stop//H:
                        result[index] = block[token-start//H]
                del block
            return result
        except BaseException:
            self.close()
            raise
        finally:
            self._gathering = False
            result = block = None

    def _begin_decoder(self, config, ids, outputs):
        """Private validation for run_decoder's fixed-body source wrapper.

        This is a consumer mock lifecycle hook, NOT native execution authority.
        Run with a fresh typed config copy parsed from the frozen bytes.
        """
        self._active()
        expected = self._geometry.config()
        def same_typed(a, b):
            if type(a) is not type(b):
                return False
            if type(a) is dict:
                return (all(type(k) is str for k in a) and a.keys() == b.keys()
                        and all(same_typed(a[k], b[k]) for k in b))
            if type(a) is list:
                return len(a) == len(b) and all(same_typed(x, y) for x, y in zip(a, b))
            return type(a) in (str, int, float, bool, type(None)) and a == b
        first = self._descs["model.embed_tokens.weight"]
        first_stop = min(128, 65536//first.shape[1], first.shape[0])*first.shape[1]
        if (self._entered or self._selection != 0
                or self._next != (0, first.name, 0, first_stop)
                or type(outputs) is not int or outputs != 2
                or type(ids) is not list or len(ids) != self._geometry.input_rows
                or not same_typed(config, expected)):
            self.close()
            raise SourceError("one two-selection decoder entry with frozen typed config/prefix required")
        self._entered = self._decoder_running = True
        return expected

    def _end_decoder(self):
        # Only the existing fixed decoder wrapper calls this after body return.
        # These private transitions are not a Python introspection security wall.
        self._active()
        if not self._decoder_running or self._selection != 2 or self._next is not None:
            self.close()
            raise SourceError("unfinished actual mock decoder invocation")
        self._decoder_running, self._decoder_completed = False, True

    def abort_decoder(self, error):
        # Called by the source-aware decoder error boundary too: dropping only
        # adapter locals cannot release bias/gamma held in consumer tracebacks.
        error = _scrub_error(error)
        self._decoder_running = False
        try:
            self.close()
        except BaseException as cleanup:
            _scrub_error(cleanup)
            raise error.with_traceback(None) from cleanup
        raise error.with_traceback(None)

    def selection_finished(self):
        self._active()
        if self._selection >= 2 or self._next is not None and self._next[0] == self._selection:
            self.close()
            raise SourceError("unfinished mock source selection")
        self._selection += 1

    def finish(self):
        """Close a completed mock consumer call, not qualify executed source.

        A balanced row transcript alone is insufficient. The public run_decoder
        wrapper validates, invokes its FIXED existing body, then privately marks
        completion on successful return. No public begin/end markers or caller-
        selected computation callback are offered by this mapping.
        """
        self._active()
        if self._next is not None or self._selection != 2 or not self._decoder_completed:
            self.close()
            raise SourceError("unfinished mock source grammar")
        self.close()

    def require_execution_qualification(self):
        raise SourceError("mock only: missing borrowed source authority, native layout, RoPE and physical admission")

    def close(self):
        if self._busy:
            # Nested demand/explicit close poisons immediately, but only the
            # outer frame may release the borrow and then clean up the owner.
            self._poisoned, self._state = True, "failed"
            return
        view, self._view = self._view, None
        self._state = "closed"
        self._cursor = self._next = None
        self._frame_hashes = self._index_offsets = None
        try:
            if view is not None:
                view.owner.close()
        except BaseException as error:
            raise _scrub_error(error).with_traceback(None)
        finally:
            view = None

    def __del__(self):
        # Best effort only; callers must use the scope for deterministic release.
        if getattr(self, "_view", None) is not None:
            try:
                self.close()
            except BaseException:
                pass

    def __enter__(self):
        self._active()
        return self

    def __exit__(self, *exc):
        error = exc[1]
        exc = None  # Do not retain the separately supplied traceback argument.
        if error is not None:
            self.abort_decoder(error)
        self.close()


def open_weights(view, geometry, limits, *, source_f32, matrix_type, array):
    return SourceWeights(view, geometry, limits, source_f32=source_f32,
                         matrix_type=matrix_type, array=array)

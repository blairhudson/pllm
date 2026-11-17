"""Independent, bounded, clear-fixture PUMA semantic decoder specification.

TEST ONLY. Never import production arithmetic or read runtime tensors here.
Provenance: paper/reproduction-contracts/{shared_profile_oracle.py,puma/oracle.py}
specify TruncAPr and pinned HAL polynomials; D9/D10/D11 and the semantic adapter
specify the composition. Equations are rederived with Python unbounded integers.

Integration contract (all arrays C-order, batch dimension omitted):
* ``Tape.batches`` maps unique semantic operation names to ``{shift, residues}``.
  Residues have EXACT output shape, integers in [0,2**shift). Capture reconstructed
  TruncAPr mask low bits in a TEST-ONLY party harness, not multiplication masks,
  output masks, or masked openings. ``adapt_tapes`` maps physical invocation/lane
  coordinates to semantic batches (including tiled/streamed products). Every
  physical truncation lane must be consumed exactly once. Nontruncating PRSS draws
  are deliberately outside this arithmetic tape; this is not a PRSS schedule proof.
* ``run_decoder`` returns ``Result.trace`` mapping boundary names to signed ring64
  integer arrays. Prefix ``s{selection}/``; layer ``layer.{index}/``; attention
  tensors [query_heads, prefix, feature/key], K/V [kv_heads,prefix,feature].
  ``compare_trace`` requires ALL boundaries, shapes and values, including complete
  logits [vocabulary], last_hidden [hidden], selected scalar and feedback
  [prefix+1]. Native LastToken precedes the head; other prefix rows are NOT
  evaluated by the native output head. No None,
  missing values, subset comparisons, digests or final IDs imply conformance.
* Public ``rotary`` is a mapping position -> (cosine,sine) Q18 half-width vectors,
  frozen independently alongside source/config. Production's pinned libm binary32
  coefficient generation needs separate conformance evidence; host math is NOT an
  exact substitute. Scale coefficient uses explicit binary32 operations.
* Weights are Q18 arrays or ``PublicMatrix`` readers in HuggingFace [out,in]
  layout. Readers return bounded row blocks; never load full checkpoint here.
  ``source_f32``
  independently encodes bounded little-endian source frames, never a checkpoint
  loader. Test harness must bind config, sources, coefficient rows and trace to
  same test execution. Reconstruction is authorized only in that harness.

Supported: unscaled split-half RoPE, Qwen2 biases, Qwen3 Q/K RMSNorm, Llama,
GQA, causal full-prefix recomputation, epsilon-one RMSNorm, guarded SiLU/softmax,
full vocabulary stable argmax, private D9 feedback. Unsupported: scaled/partial
RoPE, sliding windows, persistent KV, other activations, private-weight arithmetic,
and guards outside RMSNorm inputs. Explicit checked-range schedules model safe
selection and final acceptance, not cryptographic guard-circuit correctness.
No production export is implemented.
"""

from dataclasses import dataclass
import math
import struct
import importlib.util
from bisect import bisect_right
from pathlib import Path

import numpy as np

Q = 1 << 18
GAP = 1 << 62
TRACE_SCHEMA = "pllm.test.puma.semantic-decoder-trace.v1"
TAPE_SCHEMA = "pllm.test.puma.semantic-low-residue-tape.v1"
CHECKED_RANGE_LIMIT = 256 << 18
MAX_VALUES = 2_000_000
MAX_TRACE_VALUES = 8_000_000
MAX_PREFIX = 72


class OracleError(ValueError):
    """Missing evidence, unsupported semantics, or unsafe unwrapped arithmetic."""


@dataclass(frozen=True)
class RangeCheck:
    """Frozen inclusive signed-word endpoints; no subtract-and-wrap predicate.

    Compiler's initial RMSNorm profile uses [-256*Q,256*Q], fallback=0.
    General endpoints permit independent native guard boundary controls.
    """
    lower: int = -CHECKED_RANGE_LIMIT
    upper: int = CHECKED_RANGE_LIMIT
    fallback: int = 0

    def __post_init__(self):
        if any(type(v) is not int or not -(1 << 63) <= v < 1 << 63
               for v in (self.lower, self.upper, self.fallback)):
            raise OracleError("exact signed integer guard endpoints required")
        if not self.lower <= self.fallback <= self.upper:
            raise OracleError("guard fallback must belong to inclusive accepted range")

    def apply(self, incoming):
        incoming = array(incoming)
        valid = np.array([int(self.lower <= int(v) <= self.upper)
                          for v in incoming.flat], dtype=object).reshape(incoming.shape)
        safe = np.where(valid == 1, incoming, self.fallback)
        return safe, valid


@dataclass(frozen=True)
class CheckedRangeSchedule:
    """Immutable public placement, frozen before evaluating any private fixture.

    entries: tuple of (semantic RMSNorm input name, RangeCheck). Names are norm
    boundary plus '/input', e.g. 's0/layer.0/input_norm/input',
    's1/layer.0/q_norm/input', 's1/final_norm/input'. Only listed inputs are guarded,
    regardless of fixture magnitude. Supply compiler PUBLIC placement; oracle
    cannot establish that source-bound analysis selected the right sites.
    Unused/unknown sites fail. Guard traces append '/checked_range/{safe,valid,
    all_valid,sticky}' to input name; incoming uses that input name itself.
    No truncation tape entries correspond to these guard operations.
    """
    entries: tuple = ()

    def __post_init__(self):
        if type(self.entries) is not tuple:
            raise OracleError("immutable tuple guard schedule required")
        names = set()
        for entry in self.entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise OracleError("immutable guard schedule entry required")
            name, rule = entry
            if not isinstance(name, str) or not name.endswith('/input') or not isinstance(rule, RangeCheck):
                raise OracleError("RMSNorm input name and frozen RangeCheck required")
            if name in names:
                raise OracleError("duplicate guard placement")
            names.add(name)

    @classmethod
    def from_mapping(cls, placements):
        return cls(tuple(sorted(placements.items())))


@dataclass(frozen=True)
class PublicMatrix:
    """Public source reader, NOT runtime tensor access.

    ``read_rows(start, stop)`` returns Q18 [stop-start,columns] array. Caller binds
    reader to frozen source commitment. Each read is bounded to at most 128 rows
    and 65536 words. For F32 source bytes, encode each frame with ``source_f32``.
    This interface permits full vocabulary/projection sources without unbounded
    oracle weight allocation. Trace/output limits remain explicit test limits,
    unrelated to production admission; fail rather than omit large tensors.
    """
    shape: tuple
    read_rows: object

    def __post_init__(self):
        if len(self.shape) != 2 or any(type(n) is not int or n <= 0 for n in self.shape) or self.shape[1] > 65536:
            raise OracleError("bounded public matrix geometry")
        if not callable(self.read_rows):
            raise OracleError("public matrix reader required")

    def rows(self, start, stop):
        if not 0 <= start < stop <= self.shape[0] or (stop-start)*self.shape[1] > 65536 or stop-start > 128:
            raise OracleError("bounded public source read")
        value = array(self.read_rows(start, stop))
        if value.shape != (stop-start, self.shape[1]):
            raise OracleError("public source reader shape")
        return value

    def blocks(self):
        step = min(128, 65536//self.shape[1])
        for start in range(0, self.shape[0], step):
            yield start, self.rows(start, min(start+step, self.shape[0]))


def array(value):
    a = np.asarray(value, dtype=object)
    if not 0 < a.size <= MAX_VALUES:
        raise OracleError("bounded nonempty array required")
    for x in a.flat:
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
            raise OracleError("integer array required")
        if not -(1 << 63) <= int(x) < 1 << 63:
            raise OracleError("signed ring64 representative required")
    return a


def ring(value):
    """Exact local ring64 affine arithmetic; never use to admit a raw product."""
    a = np.asarray(value, dtype=object)
    return ((a + (1 << 63)) % (1 << 64)) - (1 << 63)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def source_f32(raw):
    if not raw or len(raw) % 4 or len(raw) // 4 > MAX_VALUES:
        raise OracleError("bounded complete float32 source frame required")
    values = []
    for (v,) in struct.iter_unpack("<f", raw):
        try:
            scaled = f32(v * Q)
        except OverflowError as error:
            raise OracleError("source Q18 range") from error
        if not math.isfinite(scaled) or abs(scaled) > 1 << 40:
            raise OracleError("source Q18 range")
        values.append(int(scaled))
    return array(values)


class Tape:
    """Explicit per-operation low-residue tape. No implicit/random/zero fallback."""

    def __init__(self, batches):
        self.batches = dict(batches)
        self.used = set()

    def truncate(self, name, raw, shift=18):
        raw = np.asarray(raw, dtype=object)
        if not 1 <= shift <= 62 or not 0 < raw.size <= MAX_VALUES:
            raise OracleError("truncation geometry")
        # Check BEFORE ring reduction; wrapped inputs cannot prove quarter gap.
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in raw.flat):
            raise OracleError("integer truncation input required")
        if any(not -GAP <= int(x) < GAP for x in raw.flat):
            raise OracleError("unwrapped truncation outside quarter-gap")
        if name in self.used or name not in self.batches:
            raise OracleError(f"missing or repeated tape: {name}")
        batch = self.batches[name]
        if set(batch) != {"shift", "residues"} or batch["shift"] != shift:
            raise OracleError(f"tape shift/schema: {name}")
        residues = array(batch["residues"])
        if residues.shape != raw.shape or any(not 0 <= x < 1 << shift for x in residues.flat):
            raise OracleError(f"tape shape/residue: {name}")
        self.used.add(name)
        # Independent signed law, equivalent to source high-bit correction.
        return np.asarray((raw + residues) // (1 << shift), dtype=object)

    def finish(self):
        extra = self.batches.keys() - self.used
        if extra:
            raise OracleError(f"excess tape batches: {sorted(extra)}")


class StreamingTape(Tape):
    """Bounded source-present audit adapter; consume/discard one mapped batch.

    Provider yields (semantic_name, {shift,residues}) in oracle operation order.
    Native public-span/lane mapper must arrange this order with bounded queues;
    never obtain residues from compared tensor values. All checks of Tape apply;
    finish rejects extra batches. This is not a native tiled schedule generator.
    """
    def __init__(self, batches):
        super().__init__({})
        self.provider = iter(batches)

    def truncate(self, name, raw, shift=18):
        try:
            observed_name, batch = next(self.provider)
        except StopIteration as error:
            raise OracleError(f"missing streamed tape: {name}") from error
        if observed_name != name:
            raise OracleError(f"streamed tape order mismatch: {name}")
        self.batches[name] = batch
        try:
            return super().truncate(name, raw, shift)
        finally:
            self.batches.pop(name, None)

    def finish(self):
        super().finish()
        sentinel = object()
        if next(self.provider, sentinel) is not sentinel:
            raise OracleError("excess streamed tape batches")


def adapt_tapes(physical, layout):
    """Explicit physical-to-semantic permutation; no inferred schedule.

    physical: {invocation_id: {shift: int, masks: [unsigned ring64 words]}}.
    layout: {semantic_name: {shift: int, shape: [...], lanes: [[id,index],...]}}.
    Invocation IDs may be strings or integers; list order gives semantic C order.
    Includes discarded guarded-tail lanes and causal placeholder scaling lanes.
    """
    available = {}
    for invocation, batch in physical.items():
        if set(batch) != {"shift", "masks"} or not 1 <= batch["shift"] <= 62:
            raise OracleError("physical tape schema")
        if not 0 < len(batch["masks"]) <= MAX_VALUES:
            raise OracleError("physical tape bounds")
        for lane, mask in enumerate(batch["masks"]):
            if type(mask) is not int or not 0 <= mask < 1 << 64:
                raise OracleError("unsigned mask required")
            available[invocation, lane] = (batch["shift"], mask)
    batches = {}
    for name, spec in layout.items():
        shape = tuple(spec["shape"])
        if any(type(n) is not int or n <= 0 for n in shape):
            raise OracleError("semantic tape shape")
        count = math.prod(shape)
        if not 0 < count <= MAX_VALUES or len(spec["lanes"]) != count:
            raise OracleError("semantic tape lane count")
        residues = []
        for coordinate in spec["lanes"]:
            key = tuple(coordinate)
            if key not in available:
                raise OracleError("missing or reused physical draw")
            shift, mask = available.pop(key)
            if shift != spec["shift"]:
                raise OracleError("physical/semantic shift mismatch")
            residues.append(mask % (1 << shift))
        batches[name] = {"shift": spec["shift"], "residues": np.array(residues, dtype=object).reshape(shape)}
    if available:
        raise OracleError("excess physical draws")
    return Tape(batches)


def native_mask_tapes(events, pair_keys, public_seed, shifts):
    """Replay ACTUAL native TruncAPr masks from test-only setup/frame evidence.

    Events are emitted by compiler oracle_trace.rs, immediately after native
    transitions. Uses source-pinned independent AES implementation, not native
    expansion, ciphertext payloads, output tensors, or inferred rounding choices.
    pair_keys are three 16-byte own-edge seeds captured from existing test setup
    seed messages; public_seed is wrapping sum of public contributions. Never
    persist these all-party test secrets or include them in benchmark records.

    shifts maps each native binding to public semantic shift (from reviewed
    scalar lane layout). It is NOT recovered from arithmetic results. Native
    wire binds but does not disclose shift; complete tensor equality also checks
    that claimed shift matches actual execution. Unmapped invocations fail.
    """
    replay = NativeMaskReplay(pair_keys, public_seed, max_invocations=len(shifts))
    physical = {}
    for event in events:
        binding = event["binding"]
        if binding not in shifts:
            raise OracleError("complete native truncation binding/shift map required")
        completed = replay.ingest(event, shifts[binding])
        if completed is not None:
            physical[binding] = completed
    replay.finish()
    if not physical or set(physical) != set(shifts):
        raise OracleError("complete native truncation binding/shift map required")
    return physical


def _independent_aes_reference():
    path = Path(__file__).resolve().parents[1] / "paper/reproduction-contracts/puma/aes_reference.py"
    spec = importlib.util.spec_from_file_location("independent_puma_aes", path)
    reference = importlib.util.module_from_spec(spec)
    # Frozen standalone fixture uses an unqualified `from oracle import ...`.
    # Bind that import to its sibling independent fixture, not this module or a
    # production package. Restore module registry immediately after offline load.
    import sys

    oracle_spec = importlib.util.spec_from_file_location("oracle", path.with_name("oracle.py"))
    fixture = importlib.util.module_from_spec(oracle_spec)
    previous = sys.modules.get("oracle")
    sys.modules["oracle"] = fixture
    try:
        oracle_spec.loader.exec_module(fixture)
        spec.loader.exec_module(reference)
    finally:
        if previous is None:
            sys.modules.pop("oracle", None)
        else:
            sys.modules["oracle"] = previous
    return reference


class NativeMaskReplay:
    """Scoped actual-seed replay; ingest in physical order, consume masks later.

    Only incomplete four-route groups retain event headers. Completed identities
    and per-role cursors are bounded public metadata, never an event/mask spool.
    The independent AES fixture is loaded once per replay owner.
    """
    def __init__(self, pair_keys, public_seed, *, max_invocations):
        if len(pair_keys) != 3 or any(type(k) is not bytes or len(k) != 16 for k in pair_keys):
            raise OracleError("three test-local 128-bit edge keys required")
        if type(public_seed) is not int or not 0 <= public_seed < 1 << 128:
            raise OracleError("test public seed required")
        if type(max_invocations) is not int or max_invocations <= 0:
            raise OracleError("bounded native invocation census required")
        self.keys, self.seed = pair_keys, public_seed
        self.reference = _independent_aes_reference()
        self.limit = max_invocations
        self.pending, self.previous_by_role = {}, {}
        self.seen_bindings, self.seen_cursors = set(), set()
        self.closed = False

    def ingest(self, event, shift):
        if self.closed or (shift is not None and (type(shift) is not int or not 1 <= shift <= 62)):
            raise OracleError("native replay state/shift")
        role = event["role"]
        if type(role) is not int or role not in (0, 1, 2):
            raise OracleError("native truncation sender role")
        binding = event["binding"]
        if binding in self.seen_bindings:
            raise OracleError("duplicate native truncation binding")
        lanes, public_end, pair_end = (event[key] for key in ("lanes", "public_end", "pair_end"))
        if any(type(value) is not int for value in (lanes, public_end, pair_end)) or not 0 < lanes <= 1024:
            raise OracleError("native truncation event geometry")
        reserved_blocks = 2*((lanes+1)//2)
        if not 1 <= public_end < 1 << 64 or not reserved_blocks <= pair_end < 1 << 64:
            raise OracleError("native truncation cursor")
        reservation_start = pair_end-reserved_blocks
        previous = self.previous_by_role.get(role)
        if previous is not None and event["binding"] != previous["binding"]:
            if public_end <= previous["public_end"] or pair_end <= previous["pair_end"]:
                raise OracleError("native truncation schedule order/cursor reuse")
            # begin_inner burns BOTH ceil(lanes/2)-block fills. Increasing
            # endpoints alone permits a rewind into the preceding output fill.
            # Gaps belong to other protocols and are valid; the helper's second
            # route for this same binding is not a fresh reservation.
            if reservation_start < previous["pair_end"]:
                raise OracleError("overlapping native PRSS pair reservation")
        self.previous_by_role[role] = event
        if binding not in self.pending:
            if len(self.pending) + len(self.seen_bindings) >= self.limit:
                raise OracleError("native invocation census exceeded")
            self.pending[binding] = (shift, [])
        old_shift, group = self.pending[binding]
        if shift != old_shift:
            raise OracleError("one native invocation assigned inconsistent shifts")
        if group and any(event.get(k) != group[0].get(k) for k in
                         ("node", "loop_offset", "tensor_step", "invocation", "lanes")):
            raise OracleError("native truncation owner metadata differs across parties")
        if any((e["role"], e["to"], e["phase"]) ==
               (role, event["to"], event["phase"]) for e in group):
            raise OracleError("incomplete or duplicate actual truncation routes")
        group.append(event)
        if len(group) < 4:
            return None
        first = group[0]
        lanes, public_end, pair_end = (first[key] for key in ("lanes", "public_end", "pair_end"))
        if any(type(value) is not int for value in (lanes, public_end, pair_end)) or not 0 < lanes <= 1024:
            raise OracleError("native truncation event geometry")
        q = (lanes+1)//2
        if not 1 <= public_end < 1 << 64 or not 2*q <= pair_end < 1 << 64:
            raise OracleError("native truncation cursor")
        if (public_end, pair_end) in self.seen_cursors:
            raise OracleError("duplicate native PRSS invocation")
        if any((event["lanes"], event["public_end"], event["pair_end"]) != (lanes, public_end, pair_end) for event in group):
            raise OracleError("native parties disagree on truncation cursors")
        pivot = self.reference.source_fill(self.seed.to_bytes(16, "little"), public_end-1, 1)[0]
        if pivot > (1 << 64)-3:
            raise OracleError("source pivot rejected top aliases")
        a = pivot % 3
        b, h = (a+1)%3, (a+2)%3
        observed = {(event["role"], event["to"], event["phase"]) for event in group}
        expected = {(a, b, 10), (b, a, 10), (h, a, 11), (h, b, 11)}
        if len(group) != 4 or observed != expected:
            raise OracleError("incomplete or duplicate actual truncation routes")
        # Source begin reserves mask and output fills before sending messages.
        # Both fills consume ceil(lanes/2) blocks; discarded odd halves stay lost.
        mask_start = pair_end-2*q
        left = self.reference.source_fill(self.keys[a], mask_start, lanes)
        right = self.reference.source_fill(self.keys[h], mask_start, lanes)
        self.seen_bindings.add(binding)
        self.seen_cursors.add((public_end, pair_end))
        del self.pending[binding]
        return {"shift": shift, "masks": [(x+y) % (1 << 64) for x, y in zip(left, right)]}

    def finish(self):
        if self.closed or self.pending:
            raise OracleError("incomplete native truncation replay")
        self.closed = True


_ROOT_ALIASES = {"token_lookup": "embedding", "input.tokens": "input.tokens",
                 "final_norm": "final_norm", "last_hidden": "last_hidden",
                 "output_head": "logits", "token_selection": "selected",
                 "token_feedback": "selected", "feedback.history": "feedback"}
_LAYER_ALIASES = {"q_linear": "q_proj", "k_linear": "k_proj", "v_linear": "v_proj",
                  "input_norm": "input_norm", "q_heads": "q_heads", "k_heads": "k_heads",
                  "v_heads": "v_heads", "q_norm": "q_norm", "k_norm": "k_norm",
                  "rope_q": "q_rotary", "rope_k": "k_rotary", "key_append": "full_prefix_k",
                  "key_view": "full_prefix_k", "value_append": "full_prefix_v",
                  "value_view": "full_prefix_v", "attention_scores": "scores",
                  "attention_scale": "scaled_scores", "causal_mask": "causal_mask",
                  "softmax": "probabilities", "attention_values": "context",
                  "attention_hidden": "merged", "o_proj": "o_proj",
                  "attention_residual": "attention_residual", "post_norm": "post_attention_norm",
                  "gate_proj": "gate_proj", "up_proj": "up_proj", "silu": "silu",
                  "gated_multiply": "gate_up", "down_proj": "down_proj", "mlp_residual": "output"}


def _native_checkpoint_name(selection, native_id, oracle_name=None):
    if type(selection) is not int or not 0 <= selection < 8:
        raise OracleError("native checkpoint selection domain")
    if type(native_id) is not str or not native_id:
        raise OracleError("native checkpoint identity")
    if oracle_name is not None:
        if type(oracle_name) is not str or not oracle_name:
            raise OracleError("native diagnostic name")
        return oracle_name
    if native_id in _ROOT_ALIASES:
        suffix = _ROOT_ALIASES[native_id]
    elif native_id.startswith("layer.") and native_id.count(".") == 2:
        _, layer, operator = native_id.split(".")
        if (not layer.isascii() or not layer.isdecimal() or str(int(layer)) != layer
                or not 0 <= int(layer) < 64 or operator not in _LAYER_ALIASES):
            raise OracleError("unmapped native semantic operator/layer")
        suffix = f"layer.{layer}/" + _LAYER_ALIASES[operator]
    else:
        raise OracleError(f"unmapped native checkpoint: {native_id}")
    return f"s{selection}/" + suffix


def _native_checkpoint_shape(shape, batch_dimension, native_id):
    if (type(shape) not in (list, tuple) or type(batch_dimension) is not bool
            or any(type(n) is not int or n <= 0 for n in shape)
            or math.prod(shape) > MAX_VALUES):
        raise OracleError("native checkpoint bounded shape metadata")
    shape = tuple(shape)
    if batch_dimension and (not shape or shape[0] != 1):
        raise OracleError("native checkpoint requires batch-one shape")
    normalized = shape[1:] if batch_dimension else shape
    if native_id == "token_feedback":
        if math.prod(normalized) != 1:
            raise OracleError("native feedback scalar shape")
        normalized = ()
    return shape, normalized


def _native_flat_words(words, count):
    # Validate flat input/count BEFORE NumPy conversion can expand nested input.
    if (not isinstance(words, (list, tuple, np.ndarray))
            or isinstance(words, np.ndarray) and words.ndim != 1
            or len(words) != count
            or any(isinstance(word, (bool, np.bool_)) or not isinstance(word, (int, np.integer))
                   for word in words)):
        raise OracleError("native flat integer word count/schema")


def _native_checkpoint_value(checkpoint):
    """Validate one existing capture record; reconstruction does not prove origin.

    A native producer must independently check overlapping replicated pairs.
    Generated records are synthetic controls, not evidence that producer ran.
    """
    required = {"selection", "id", "shape"}
    optional = {"batch_dimension", "oracle_name", "values", "components"}
    if (type(checkpoint) is not dict or not required <= checkpoint.keys()
            or checkpoint.keys() - required - optional):
        raise OracleError("native checkpoint schema")
    _native_checkpoint_name(checkpoint["selection"], checkpoint["id"], checkpoint.get("oracle_name"))
    shape, normalized = _native_checkpoint_shape(
        checkpoint["shape"], checkpoint.get("batch_dimension", True), checkpoint["id"])
    if ("components" in checkpoint) == ("values" in checkpoint):
        raise OracleError("exactly one native reconstructed/shared checkpoint representation required")
    if "components" in checkpoint:
        components = checkpoint["components"]
        if (not isinstance(components, (list, tuple, np.ndarray))
                or isinstance(components, np.ndarray) and components.ndim != 2
                or len(components) != 3):
            raise OracleError("native checkpoint component shape")
        for part in components:
            _native_flat_words(part, math.prod(shape))
        value = reconstruct_test_shares(components)
    else:
        _native_flat_words(checkpoint["values"], math.prod(shape))
        value = array(checkpoint["values"])
        if value.shape != (math.prod(shape),):
            raise OracleError("native reconstructed checkpoint shape")
    return value.reshape(normalized)


def _check_native_final(result, outputs, checked_profile, final_components):
    if (type(outputs) is not int or not 1 <= outputs <= 8
            or type(checked_profile) is not bool or type(result.accepted) is not bool
            or type(result.tokens) is not list
            or len(result.tokens) != (outputs if result.accepted else 0)
            or any(type(token) is not int or token < 0 for token in result.tokens)):
        raise OracleError("native final result geometry")
    if final_components is None:
        raise OracleError("native final response components required")
    expected = result.tokens if result.accepted else [0]*outputs
    if checked_profile:
        expected = [*expected, int(result.accepted)]
    if (not isinstance(final_components, (list, tuple, np.ndarray))
            or isinstance(final_components, np.ndarray) and final_components.ndim != 2
            or len(final_components) != 3):
        raise OracleError("native final component shape")
    for part in final_components:
        _native_flat_words(part, len(expected))
    final = reconstruct_test_shares(final_components)
    if final.shape != (len(expected),) or final.tolist() != expected:
        raise OracleError("native final token masking/aggregate acceptance mismatch")


class StreamingTraceComparator:
    """Finite semantic-ordered comparator, NOT a native capture/source verifier.

    Public declaration/provenance must be frozen before private iteration. No
    payload/array/history is retained by this object after a callback; only the
    one-shot provider, public tuples, schedule, counters and terminal state remain.
    Provider itself must not buffer private payloads. No search-ahead or queue.
    Budgets count every alias. Peak counters are logical words, not Python RSS:
    components transiently need three vectors plus a reconstructed array, beside
    the expected oracle array and decoder/operator working set. Declaration and
    run_decoder.emitted/StreamingTape.used still scale with scheduled boundaries.
    """

    def __init__(self, checkpoints, *, declaration, source_provenance,
                 checked_ranges, outputs, max_records, max_total_values):
        if (any(type(n) is not int or n <= 0 for n in (outputs, max_records, max_total_values))
                or outputs > 8):
            raise OracleError("positive exact integer stream budgets/outputs required")
        if checked_ranges is not None and type(checked_ranges) is not CheckedRangeSchedule:
            raise OracleError("immutable checked-range schedule required")

        def immutable(value):
            return type(value) in (str, int, bool, type(None)) or (
                type(value) is tuple and all(immutable(item) for item in value))

        if type(source_provenance) is not tuple or not source_provenance:
            raise OracleError("immutable public source provenance required")
        keys = set()
        for entry in source_provenance:
            if (type(entry) is not tuple or len(entry) != 2
                    or type(entry[0]) is not str or not entry[0] or entry[0] in keys
                    or not immutable(entry[1])):
                raise OracleError("immutable public source provenance schema")
            keys.add(entry[0])
        if type(declaration) is not tuple or not declaration:
            raise OracleError("immutable nonempty stream declaration required")
        diagnostics = set()
        if checked_ranges is not None:
            diagnostics.add("acceptance.valid")
            for name, _ in checked_ranges.entries:
                diagnostics.add(name)
                diagnostics.update(name + "/checked_range/" + suffix for suffix in
                                   ("safe", "valid", "all_valid", "tile_sticky", "sticky"))
        names, identities = set(), set()
        records = values = 0
        for group in declaration:
            if type(group) is not tuple or len(group) != 2:
                raise OracleError("immutable stream group required")
            name, specs = group
            if (type(name) is not str or name in names or type(specs) is not tuple or not specs):
                raise OracleError("duplicate or mutable stream group")
            names.add(name)
            for spec in specs:
                if type(spec) is not tuple or len(spec) != 5 or type(spec[2]) is not tuple:
                    raise OracleError("immutable native record declaration required")
                selection, native_id, shape, batch, oracle_name = spec
                mapped = _native_checkpoint_name(selection, native_id, oracle_name)
                if selection >= outputs or mapped != name or (selection, native_id) in identities:
                    raise OracleError("stream declaration identity/name/selection")
                if oracle_name is not None and (name not in diagnostics or (
                        name != "acceptance.valid" and not name.startswith(f"s{selection}/"))):
                    raise OracleError("undeclared diagnostic name/placement")
                _native_checkpoint_shape(shape, batch, native_id)
                identities.add((selection, native_id))
                records += 1
                values += math.prod(shape)
        if checked_ranges is not None:
            if declaration[-1][0] != "acceptance.valid":
                raise OracleError("checked stream requires terminal acceptance declaration")
            for spec in declaration[-1][1]:
                if _native_checkpoint_shape(spec[2], spec[3], spec[1])[1] != ():
                    raise OracleError("scalar acceptance declaration required")
        if records > max_records or values > max_total_values:
            raise OracleError("declared stream budget exceeded")
        self._declaration = declaration
        self._source_provenance = source_provenance
        self._checked_ranges = checked_ranges
        self._outputs = outputs
        self._max_records, self._max_total_values = max_records, max_total_values
        self.declared_records, self.declared_values = records, values
        self.consumed_records = self.consumed_values = 0
        self.peak_record_values = self.peak_record_words = 0
        self._index, self._state = 0, "active"
        # Iteration starts only after all public validation/budget checks.
        self._checkpoints = iter(checkpoints)

    @property
    def declaration(self):
        return self._declaration

    @property
    def source_provenance(self):
        return self._source_provenance

    def _active(self):
        if self._state != "active":
            raise OracleError("stream comparator terminal")

    def __call__(self, name, value):
        self._active()
        try:
            if self._index >= len(self.declaration) or name != self.declaration[self._index][0]:
                raise OracleError("stream boundary order mismatch")
            expected = array(value)
            specs = self.declaration[self._index][1]
            # Validate all public shapes before consuming any private alias.
            for _, native_id, shape, batch, _ in specs:
                if _native_checkpoint_shape(shape, batch, native_id)[1] != expected.shape:
                    raise OracleError("stream declared boundary shape")
            for spec in specs:
                count = math.prod(spec[2])
                if (self.consumed_records + 1 > self._max_records
                        or self.consumed_values + count > self._max_total_values):
                    raise OracleError("stream budget exceeded")
                sentinel = object()
                checkpoint = next(self._checkpoints, sentinel)
                if checkpoint is sentinel:
                    raise OracleError("missing streamed checkpoint")
                if type(checkpoint) is not dict:
                    raise OracleError("native checkpoint schema")
                # Avoid Python equality accepting bool as integer metadata.
                selection = checkpoint.get("selection")
                native_id = checkpoint.get("id")
                _native_checkpoint_name(selection, native_id, checkpoint.get("oracle_name"))
                shape, _ = _native_checkpoint_shape(checkpoint.get("shape"),
                    checkpoint.get("batch_dimension", True), native_id)
                actual_spec = (selection, native_id, shape,
                    checkpoint.get("batch_dimension", True), checkpoint.get("oracle_name"))
                if actual_spec != spec:
                    raise OracleError("stream checkpoint identity/order/metadata mismatch")
                self.consumed_records += 1
                self.consumed_values += count
                self.peak_record_values = max(self.peak_record_values, count)
                self.peak_record_words = max(self.peak_record_words,
                                             count*(4 if "components" in checkpoint else 1))
                actual = _native_checkpoint_value(checkpoint)
                compare_trace({name: expected}, {name: actual})
                del actual, checkpoint
            self._index += 1
        except BaseException:
            self._state, self._checkpoints = "failed", None
            raise
        finally:
            # Do not leave payload references in this callback's traceback frame.
            value = expected = actual = checkpoint = None

    def finish(self, result, *, final_components):
        self._active()
        try:
            if self._checked_ranges is not None and not self._checked_ranges.entries:
                # Expected empty conjunction ONLY; actual checkpoint remains mandatory.
                if result.accepted is not True:
                    raise OracleError("empty checked conjunction must accept")
                self("acceptance.valid", array(int(result.accepted)))
            if self._index != len(self.declaration):
                raise OracleError("unfinished stream declaration")
            sentinel = object()
            if next(self._checkpoints, sentinel) is not sentinel:
                raise OracleError("excess streamed checkpoint")
            _check_native_final(result, self._outputs, self._checked_ranges is not None, final_components)
        except BaseException:
            self._state, self._checkpoints = "failed", None
            raise
        self._state, self._checkpoints = "finished", None


def audit_native_capture(config, weights, input_ids, outputs, rotary, *,
                         checkpoints, events, pair_keys, public_seed, layout,
                         checked_ranges=None, final_components=None,
                         gqa_controls=False, dense_head_control=False):
    """Complete native/independent audit; never accepts capture-only evidence.

    ``layout`` is adapt_tapes' explicit semantic -> physical lane map. Every
    actual truncation lane must appear exactly once, including guarded tails.
    Checkpoints use compiler graph IDs; native batch dimension must be one.
    Additional input.tokens/feedback.history hooks capture D9 state. No missing
    boundary is synthesized from oracle output or public input metadata.
    This bounded collector is for generated fixtures. For full-source auditing,
    drive independent reference/checkpoint comparison incrementally and retire
    each matched tile; do not serialize/persist a complete all-party dump.
    """
    shifts = {}
    for batch in layout.values():
        for binding, _ in batch["lanes"]:
            if binding in shifts and shifts[binding] != batch["shift"]:
                raise OracleError("one native invocation assigned inconsistent shifts")
            shifts[binding] = batch["shift"]
    physical = native_mask_tapes(events, pair_keys, public_seed, shifts)
    tape = adapt_tapes(physical, layout)
    result = run_decoder(config, weights, input_ids, outputs,
                         tape, rotary, checked_ranges=checked_ranges)
    if checked_ranges is not None and "acceptance.valid" not in result.trace:
        # Checked profile with no placed guards still emits final aggregate one.
        # This follows public empty schedule, not an unobserved native value.
        result.trace["acceptance.valid"] = array(1)
    actual, seen = {}, set()
    for checkpoint in checkpoints:
        selection, native_id = checkpoint["selection"], checkpoint["id"]
        identity = (selection, native_id)
        if identity in seen:
            raise OracleError("duplicate native checkpoint")
        seen.add(identity)
        name = _native_checkpoint_name(selection, native_id, checkpoint.get("oracle_name"))
        value = _native_checkpoint_value(checkpoint)
        if name not in result.trace:
            raise OracleError(f"native boundary absent from oracle: {name}")
        compare_trace({name: result.trace[name]}, {name: value})
        actual[name] = value
    compare_trace(result.trace, actual)
    _check_native_final(result, outputs, checked_ranges is not None, final_components)
    if gqa_controls:
        _assert_fixture_gqa_coverage(config, result.trace, outputs, tape.batches)
    if dense_head_control:
        batch = tape.batches["s0/head"]
        bad = Tape({"s0/head": {"shift": batch["shift"], "residues": np.roll(batch["residues"], 1)}})
        wrong = linear(result.trace["s0/last_hidden"], weights["lm_head.weight"], bad, "s0/head")
        bad.finish()
        try:
            compare_trace({"logits": result.trace["s0/logits"]}, {"logits": wrong})
        except OracleError:
            pass
        else:
            raise OracleError("dense full head cannot distinguish actual rounding-tape permutation")
    return result


def generated_native_layout(config, input_rows, outputs, events, spans):
    """Pinned native physical schedule, independent of private tensor VALUES.

    Public compiler spans identify semantic owners; actual native frames identify
    physical invocations. Multiple layers, causal GQA, row-wise RMSNorm, linear
    C-order tiles, bounded rotary rows and 128-lane activation tails are explicit.
    Alternate native layouts must add public metadata; never guess from values.
    """
    h, kv = config["num_attention_heads"], config["num_key_value_heads"]
    hidden, mid, layers = config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]
    if any(type(n) is not int or n <= 0 for n in (h, kv, hidden, mid, layers, input_rows, outputs)):
        raise OracleError("native lane mapper unsupported geometry")
    d = config.get("head_dim", hidden//h)
    if (type(d) is not int or not 1 <= layers <= 64 or not 1 <= kv <= h <= 64 or h % kv or
            not 1 <= hidden <= 1024 or not 2 <= d <= 128 or d % 2 or
            not 1 <= mid <= 65536 or not 1 <= input_rows or not 1 <= outputs <= 8 or
            input_rows+outputs > MAX_PREFIX):
        raise OracleError("native lane mapper unsupported geometry")
    unique = {}
    for event in events:
        binding = event["binding"]
        if binding in unique:
            old = unique[binding]
            if any(event[key] != old[key] for key in ("node", "tensor_step", "invocation", "loop_offset", "lanes")):
                raise OracleError("native truncation owner metadata differs across parties")
        else:
            unique[binding] = event
    ordered_spans = sorted((s for s in spans if s["start"] < s["end"]), key=lambda s: s["start"])
    if any(left["end"] > right["start"] for left, right in zip(ordered_spans, ordered_spans[1:])):
        raise OracleError("overlapping native semantic compiler spans")
    starts = [s["start"] for s in ordered_spans]
    owners = {}
    for event in unique.values():
        at = bisect_right(starts, event["node"])-1
        if at < 0 or not event["node"] < ordered_spans[at]["end"]:
            raise OracleError("native truncation requires one exact semantic compiler span")
        span = ordered_spans[at]
        owners.setdefault((span["selection"], span["id"]), []).append(event)
    return _generated_owner_layout(config, input_rows, owners)


def _generated_owner_layout(config, input_rows, owners):
    """Shared owner/stage recipe, used by collector and single-owner bridge."""
    h, kv = config["num_attention_heads"], config["num_key_value_heads"]
    hidden, mid, layers = config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]
    d = config.get("head_dim", hidden//h)
    layout = {}

    def put(name, shape, draws, shift=18):
        lanes = [(event["binding"], lane) for event in draws for lane in range(event["lanes"])]
        if len(lanes) != math.prod(shape) or name in layout:
            raise OracleError(f"native scalar recipe lane count: {name}")
        layout[name] = {"shift": shift, "shape": list(shape), "lanes": lanes}

    def staged(name, draws, stages, rows, scalar_shape, vector_shape, widths=None):
        if len(draws) != rows*len(stages):
            raise OracleError(f"native scalar recipe stage count: {name}")
        for index, (suffix, scalar, shift) in enumerate(stages):
            stage_draws = draws[index::len(stages)]
            expected_widths = ([1]*rows if scalar else widths if widths is not None
                               else [math.prod(vector_shape)//rows]*rows)
            if [draw["lanes"] for draw in stage_draws] != expected_widths:
                raise OracleError(f"native scalar recipe row/tile widths: {name}{suffix}")
            put(name + suffix, scalar_shape if scalar else vector_shape,
                stage_draws, shift)

    norm_stages = [("/squares", False, 18), ("/mean", True, 18)]
    norm_stages += [("/rsqrt/" + suffix, True, 18) for suffix in ("u", "u2", "u3", "u4", "polynomial", "out")]
    norm_stages += [("/normalize", False, 18), ("/gamma", False, 18)]
    sm_stages = [("/reduce", False, 5)]
    sm_stages += [(f"/square.{i}", False, 18) for i in range(5)]
    sm_stages += [("/reciprocal/" + suffix, True, 18) for suffix in ("c", "w", "xw", "r", "e2", "out")]
    sm_stages += [("/probabilities", False, 18)]
    silu_stages = [("/" + suffix, False, 18) for suffix in
                   ("x2", "x4", "x6", "low.x", "low.x2", "high.x", "high.x2", "high.x4", "high.x6")]
    for (selection, native_id), draws in owners.items():
        draws.sort(key=lambda event: (event["node"], event["loop_offset"], event["invocation"] or 0, event["tensor_step"] or 0))
        n = input_rows + selection
        root = f"s{selection}/"
        if native_id.startswith("layer."):
            _, layer, op = native_id.split(".")
            if not 0 <= int(layer) < layers:
                raise OracleError("native layer outside frozen geometry")
            p = root + f"layer.{layer}/"
        else:
            p, op = root, native_id
        if op in {"input_norm", "post_norm", "q_norm", "k_norm", "final_norm"}:
            name = root + "final_norm" if op == "final_norm" else p + {"post_norm": "post_attention_norm"}.get(op, op)
            heads = h if op == "q_norm" else kv
            scalar = (heads, n) if op in {"q_norm", "k_norm"} else (n,)
            vector = (*scalar, d if op in {"q_norm", "k_norm"} else hidden)
            staged(name, draws, norm_stages, math.prod(scalar), scalar, vector)
        elif op in {"q_linear", "k_linear", "v_linear", "gate_proj", "up_proj", "down_proj", "o_proj"}:
            name = {"q_linear": "q_proj", "k_linear": "k_proj", "v_linear": "v_proj"}.get(op, op)
            width = (mid if op in {"gate_proj", "up_proj"} else
                     h*d if op == "q_linear" else kv*d if op in {"k_linear", "v_linear"} else hidden)
            put(p + name, (n, width), draws)
        elif op == "output_head":
            put(root + "head", (config["vocab_size"],), draws)
        elif op in {"rope_q", "rope_k"}:
            heads = h if op == "rope_q" else kv
            if len(draws) != heads*n*2:
                raise OracleError("native rotary scalar recipe stage count")
            kind = "q" if op == "rope_q" else "k"
            for head in range(heads):
                for row in range(n):
                    i = head*n+row
                    put(p + f"{kind}_rotary/h{head}/p{row}/cos", (d,), draws[2*i:2*i+1])
                    put(p + f"{kind}_rotary/h{head}/p{row}/sin", (d,), draws[2*i+1:2*i+2])
        elif op == "attention_scale":
            put(p + "scale", (h, n, n), draws)
        elif op in {"attention_scores", "attention_values"}:
            at = 0
            for head in range(h):
                for row in range(n):
                    kind, width = ("scores", row+1) if op == "attention_scores" else ("context", d)
                    start, count = at, 0
                    while count < width and at < len(draws):
                        count += draws[at]["lanes"]
                        at += 1
                    if count != width:
                        raise OracleError("native attention contraction tile crossing")
                    put(p + f"{kind}/h{head}/r{row}", (width,), draws[start:at])
            if at != len(draws):
                raise OracleError("excess native attention contraction tiles")
        elif op == "softmax":
            if len(draws) != h*n*13:
                raise OracleError("native softmax scalar recipe stage count")
            for head in range(h):
                for row in range(n):
                    i = head*n+row
                    staged(p + f"softmax/h{head}/r{row}", draws[13*i:13*(i+1)], sm_stages, 1, (), (row+1,))
        elif op == "silu":
            width = config["intermediate_size"]
            chunks = (n*width+127)//128
            widths = [min(128, n*width-start) for start in range(0, n*width, 128)]
            staged(p + "silu", draws, silu_stages, chunks, (), (n, width), widths)
        elif op == "gated_multiply":
            put(p + "gate_up", (n, config["intermediate_size"]), draws)
        else:
            raise OracleError(f"native truncation on unsupported semantic operation: {native_id}")
    return layout


def mul(x, y, tape, name):
    return tape.truncate(name, array(x) * array(y))


def linear(x, weight, tape, name, bias=None):
    """Complete contraction, ONE truncate, regardless of native tile width."""
    x = array(x)
    weight = weight if isinstance(weight, PublicMatrix) else array(weight)
    if x.ndim < 1 or len(weight.shape) != 2 or x.shape[-1] != weight.shape[1]:
        raise OracleError("linear shape")
    shape = (*x.shape[:-1], weight.shape[0])
    if math.prod(shape) > MAX_VALUES:
        raise OracleError("bounded linear output")
    if isinstance(weight, PublicMatrix):
        raw = np.empty(shape, dtype=object)
        for start, block in weight.blocks():
            try:
                raw[..., start:start+len(block)] = x @ block.T
            finally:
                # Release caller's previous block before generator requests next.
                del block
    else:
        raw = x @ weight.T  # dtype=object: unwrapped integer accumulation
    out = tape.truncate(name, raw)
    if bias is not None:
        bias = array(bias)
        if bias.shape != (weight.shape[0],):
            raise OracleError("bias shape")
        out = ring(out + bias)
    return out


def reciprocal(x, tape, name):
    x = array(x)
    if any(not 0 < v < 1 << 36 for v in x.flat):
        raise OracleError("reciprocal positive bit-reversal domain")
    factor = np.array([1 << (35 - int(v).bit_length() + 1) for v in x.flat], dtype=object).reshape(x.shape)
    c = mul(x, factor, tape, name + "/c")
    w = mul(763940 - 2 * c, factor, tape, name + "/w")
    e = Q - mul(x, w, tape, name + "/xw")
    r = mul(w, Q + e, tape, name + "/r")
    e2 = mul(e, e, tape, name + "/e2")
    return mul(r, Q + e2, tape, name + "/out")


def rsqrt(x, tape, name, partitioned=True):
    """Partitions affect public range proofs only; all six truncations retained."""
    if type(partitioned) is not bool:
        raise OracleError("explicit RMSNorm proof profile required")
    x = array(x)
    if any(not 0 < v < 1 << 35 for v in x.flat):
        raise OracleError("rsqrt positive bit-reversal domain")
    j = [int(v).bit_length() for v in x.flat]
    factor = np.array([1 << (35 - e) for e in j], dtype=object).reshape(x.shape)
    u = mul(x, factor, tape, name + "/u")
    u2 = mul(u, u, tape, name + "/u2")
    u3 = mul(u2, u, tape, name + "/u3")
    u4 = mul(u3, u, tape, name + "/u4")
    p = tape.truncate(name + "/polynomial", -4057974*u + 10085068*u2 - 13072088*u3 + 6823457*u4) + 1086023
    comp = np.array([(724 if e % 2 == 0 else 512) * (1 << (17-e//2)) for e in j], dtype=object).reshape(x.shape)
    return mul(p, comp, tape, name + "/out")


def rmsnorm(x, gamma, tape, name, partitioned=True):
    x, gamma = array(x), array(gamma)
    width = x.shape[-1]
    if gamma.shape != (width,) or not 1 <= width <= 262144:
        raise OracleError("RMSNorm width/gamma")
    squares = mul(x, x, tape, name + "/squares")
    coefficient = int(f32(f32(1.0 / f32(width)) * Q))
    total = squares.sum(axis=-1)
    # RowMap Sum is ring-local; reject unsafe unwrapped sum, never silently wrap.
    array(total)
    denominator = tape.truncate(name + "/mean", total * coefficient) + 1
    inv = rsqrt(denominator, tape, name + "/rsqrt", partitioned)
    normalized = mul(x, np.expand_dims(inv, -1), tape, name + "/normalize")
    return mul(normalized, gamma, tape, name + "/gamma")


def silu(x, tape, name):
    x = array(x)
    safe = np.where((x >= -8*Q) & (x <= 4*Q), x, 0)
    x2 = mul(safe, safe, tape, name + "/x2")
    x4 = mul(x2, x2, tape, name + "/x4")
    x6 = mul(x2, x4, tape, name + "/x6")
    # Activation polynomials truncate EACH coefficient term, unlike rsqrt's
    # single accumulated polynomial truncate and linear's complete-dot truncate.
    low = -80413 + tape.truncate(name + "/low.x", -21489*safe)
    low = low + tape.truncate(name + "/low.x2", -1453*x2)
    high = 2229 + tape.truncate(name + "/high.x", 131072*safe)
    high = high + tape.truncate(name + "/high.x2", 59806*x2)
    high = high + tape.truncate(name + "/high.x4", -2913*x4)
    high = high + tape.truncate(name + "/high.x6", 71*x6)
    return np.where(x < -8*Q, 0, np.where(x < -4*Q, low, np.where(x <= 4*Q, high, x)))


def softmax(x, tape, name):
    """One visible causal row. Tail still consumes all guarded exponent draws."""
    x = array(x)
    if x.ndim != 1:
        raise OracleError("softmax row required")
    z = x - max(x)
    array(z)  # subtraction/sign precondition: no wrapped cutoff decision
    keep = z > -14*Q
    safe = np.where(keep, z, 0)
    exp = Q + tape.truncate(name + "/reduce", safe, 5)
    for i in range(5):
        exp = mul(exp, exp, tape, name + f"/square.{i}")
    exp = exp * keep
    total = array(exp.sum())
    inverse = reciprocal(total, tape, name + "/reciprocal")
    return mul(exp, inverse, tape, name + "/probabilities")


def rotary_half(x, cosine, sine, tape, name):
    x, cosine, sine = array(x), array(cosine), array(sine)
    width = x.shape[-1]
    if width % 2 or cosine.shape != (width//2,) or sine.shape != cosine.shape:
        raise OracleError("split-half rotary shape")
    if any(abs(v) > Q for v in (*cosine, *sine)):
        raise OracleError("rotary coefficient range")
    cos = np.concatenate((cosine, cosine))
    sin = np.concatenate((-sine, sine))
    swapped = np.concatenate((x[..., width//2:], x[..., :width//2]), axis=-1)
    a = mul(x, cos, tape, name + "/cos")
    b = mul(swapped, sin, tape, name + "/sin")
    return ring(a + b)


def stable_argmax(logits):
    logits = array(logits)
    if logits.ndim != 1:
        raise OracleError("argmax full vocabulary row required")
    return max(range(len(logits)), key=lambda i: (logits[i], -i))


@dataclass
class Result:
    tokens: list
    trace: dict
    accepted: bool = True


def compare_trace(expected, actual):
    """Complete evidence only. No missing-private-value equality fiction."""
    if not expected or actual is None:
        raise OracleError("nonempty complete trace evidence required")
    if set(expected) != set(actual):
        raise OracleError(f"trace keys differ: missing={sorted(expected.keys()-actual.keys())}, excess={sorted(actual.keys()-expected.keys())}")
    for name in expected:
        a, b = array(expected[name]), array(actual[name])
        if a.shape != b.shape or not np.array_equal(a, b):
            raise OracleError(f"trace mismatch: {name}")


def reconstruct_test_shares(shares):
    """Three additive components ONLY, not six overlapping replicated copies.

    Caller is the authorized test harness. This function has no runtime access.
    """
    if len(shares) != 3:
        raise OracleError("three additive components required")
    parts = []
    for share in shares:
        a = np.asarray(share, dtype=object)
        if not 0 < a.size <= MAX_VALUES or any(type(x) is not int or not 0 <= x < 1 << 64 for x in a.flat):
            raise OracleError("unsigned additive components required")
        parts.append(a)
    if any(a.shape != parts[0].shape for a in parts):
        raise OracleError("share shape")
    return ring(sum(parts))


def run_decoder(config, weights, input_ids, outputs, tape, rotary, *, partitioned=True,
                checked_ranges=None, trace_sink=None):
    """Optional mock-source lifecycle/error boundary; ordinary callers unchanged.

    Source-backed exceptions keep their type/message, but subordinate traceback
    locals are retired so an actual bias/gamma cannot outlive the failed scope.
    This is the only public source-completion route: validate privately, invoke
    the fixed existing body (not a caller-selected callback), then mark complete.
    It is not a native/source-authority attestation or an introspection barrier.
    """
    begin = getattr(weights, "_begin_decoder", None)
    if begin is None:
        return _run_decoder(config, weights, input_ids, outputs, tape, rotary,
            partitioned=partitioned, checked_ranges=checked_ranges, trace_sink=trace_sink)
    try:
        frozen_config = begin(config, input_ids, outputs)
        ids = list(input_ids)  # Source hook first bounds the exact list length.
        result = _run_decoder(frozen_config, weights, ids, outputs, tape, rotary,
            partitioned=partitioned, checked_ranges=checked_ranges, trace_sink=trace_sink)
        weights._end_decoder()
        return result
    except BaseException as error:
        weights.abort_decoder(error)


def _run_decoder(config, weights, input_ids, outputs, tape, rotary, *, partitioned=True,
                checked_ranges=None, trace_sink=None):
    """Recompute COMPLETE prefix per selection. No KV reuse, no early EOS exit.

    See module docstring. Trace names below are the integration interface. Each
    arithmetic tape name is its boundary name plus documented primitive suffixes.
    Q/K norm runs per [head,position] row and consumes row-shaped scalar tapes.
    Checked-range failure does not change public execution length or return early.
    Result.tokens is empty on rejection; Result.trace contains TEST-ONLY diagnostic
    values (including internal selection/feedback), never accepted model outputs.
    Default API returns no intermediate IDs or per-site validity before completion.
    ``trace_sink`` is an optional TEST-ONLY boundary comparator, called with
    (name,array); it must compare/retire the corresponding native checkpoint.
    In this mode Result.trace is empty and no complete trace is retained. Sink
    observes diagnostic values even on rejection, not production/client output.
    """
    family = config["model_type"]
    if family not in {"qwen2", "qwen3", "llama"}:
        raise OracleError("unsupported model family")
    if (config.get("rope_scaling") or config.get("use_sliding_window")
            or config.get("partial_rotary_factor", 1) != 1
            or config.get("attention_dropout", 0) != 0
            or config.get("hidden_act", "silu") != "silu"):
        raise OracleError("unsupported decoder operator")
    h, kv = config["num_attention_heads"], config["num_key_value_heads"]
    hidden, mid = config["hidden_size"], config["intermediate_size"]
    layers, vocab = config["num_hidden_layers"], config["vocab_size"]
    if any(type(n) is not int or n <= 0 for n in (h, kv, hidden, mid, layers, vocab, outputs)):
        raise OracleError("decoder geometry")
    d = config.get("head_dim", hidden//h)
    dims = (h, kv, hidden, mid, layers, vocab, d, outputs)
    if any(type(n) is not int or n <= 0 for n in dims) or h % kv or d % 2 or d > 128:
        raise OracleError("decoder geometry")
    ids = list(input_ids)
    if not ids or len(ids) + outputs > MAX_PREFIX or outputs > 8:
        raise OracleError("bounded D9 envelope")
    if any(type(i) is not int or not 0 <= i < vocab for i in ids):
        raise OracleError("input token domain")
    trace, selected, trace_size = {}, [], 0
    if trace_sink is not None and not callable(trace_sink):
        raise OracleError("test-only trace comparator must be callable")
    emitted = set()
    if checked_ranges is not None and not isinstance(checked_ranges, CheckedRangeSchedule):
        raise OracleError("explicit immutable checked-range schedule required")
    guards = dict(checked_ranges.entries) if checked_ranges is not None else {}
    guards_used, sticky = set(), True

    def emit(name, value):
        nonlocal trace_size
        value = array(value)
        if name in emitted:
            raise OracleError("duplicate boundary")
        emitted.add(name)
        if trace_sink is None:
            trace_size += value.size
            if trace_size > MAX_TRACE_VALUES:
                raise OracleError("bounded trace exhausted")
            trace[name] = value.copy()
        else:
            trace_sink(name, value)
        return value

    def weight(name, shape):
        if name not in weights:
            raise OracleError(f"missing source: {name}")
        w = weights[name]
        w = w if isinstance(w, PublicMatrix) else array(w)
        if w.shape != shape:
            raise OracleError(f"source shape: {name}")
        return w

    def normalize(value, gamma, name):
        nonlocal sticky
        input_name = name + "/input"
        if input_name in guards:
            guards_used.add(input_name)
            emit(input_name, value)
            value, valid = guards[input_name].apply(value)
            prefix = input_name + "/checked_range/"
            emit(prefix + "safe", value)
            emit(prefix + "valid", valid)
            bits = list(valid.flat)
            reductions, sticky_tiles = [], []
            for start in range(0, len(bits), 128):
                tile_valid = all(v == 1 for v in bits[start:start+128])
                reductions.append(int(tile_valid))
                sticky = sticky and tile_valid
                sticky_tiles.append(int(sticky))
            # Native checked_view creates a reduction and sticky product PER
            # tile. There is no extra site-wide scalar instruction to fabricate.
            emit(prefix + "all_valid", reductions[0] if len(reductions) == 1 else reductions)
            emit(prefix + "tile_sticky", sticky_tiles)
            emit(prefix + "sticky", int(sticky))
        return emit(name, rmsnorm(value, gamma, tape, name, partitioned))

    embedding = weight("model.embed_tokens.weight", (vocab, hidden))
    scale = int(f32(f32(1.0 / f32(math.sqrt(f32(d)))) * Q))
    for selection in range(outputs):
        root = f"s{selection}/"
        emit(root + "input.tokens", ids)
        gather = getattr(weights, "gather_embedding", None)
        if gather is not None:
            embedded = gather(ids)
        elif isinstance(embedding, PublicMatrix):
            # Clear fixture IDs only. This is not a private runtime lookup API.
            embedded = np.concatenate([embedding.rows(i, i+1) for i in ids], axis=0)
        else:
            embedded = embedding[ids]
        x = emit(root + "embedding", embedded)
        del embedded
        n = len(ids)
        if h*n*n > MAX_VALUES or h*n*d > MAX_VALUES:
            raise OracleError("bounded attention geometry")
        for layer in range(layers):
            p, src = root + f"layer.{layer}/", f"model.layers.{layer}."
            residual = x
            norm = normalize(x, weight(src + "input_layernorm.weight", (hidden,)), p + "input_norm")
            qkv = []
            for kind, heads in (("q", h), ("k", kv), ("v", kv)):
                name = p + kind + "_proj"
                bias = weight(src + f"self_attn.{kind}_proj.bias", (heads*d,)) if config.get("attention_bias", family == "qwen2") else None
                projected = emit(name, linear(norm, weight(src + f"self_attn.{kind}_proj.weight", (heads*d, hidden)), tape, name, bias))
                del bias
                shaped = emit(p + kind + "_heads", projected.reshape(n, heads, d).transpose(1, 0, 2))
                if family == "qwen3" and kind in {"q", "k"}:
                    shaped = normalize(shaped, weight(src + f"self_attn.{kind}_norm.weight", (d,)), p + kind + "_norm")
                if kind in {"q", "k"}:
                    rotated = np.empty_like(shaped)
                    for head in range(heads):
                        for pos in range(n):
                            if pos not in rotary:
                                raise OracleError("missing frozen public rotary coefficients")
                            cos, sin = rotary[pos]
                            rotated[head, pos] = rotary_half(shaped[head, pos], cos, sin, tape, p + f"{kind}_rotary/h{head}/p{pos}")
                    shaped = emit(p + kind + "_rotary", rotated)
                qkv.append(shaped)
            q, k, v = qkv
            emit(p + "full_prefix_k", k)
            emit(p + "full_prefix_v", v)
            scores = np.zeros((h, n, n), dtype=object)
            for head in range(h):
                for row in range(n):
                    scores[head, row, :row+1] = linear(q[head, row], k[head//(h//kv), :row+1], tape, p + f"scores/h{head}/r{row}")
            emit(p + "scores", scores)
            scores = emit(p + "scaled_scores", tape.truncate(p + "scale", scores*scale))
            emit(p + "causal_mask", scores)  # future placeholders are zero, NOT softmax inputs
            probabilities = np.zeros((h, n, n), dtype=object)
            for head in range(h):
                for row in range(n):
                    probabilities[head, row, :row+1] = softmax(scores[head, row, :row+1], tape, p + f"softmax/h{head}/r{row}")
            emit(p + "probabilities", probabilities)
            context = np.zeros((h, n, d), dtype=object)
            for head in range(h):
                for row in range(n):
                    context[head, row] = linear(probabilities[head, row, :row+1], v[head//(h//kv), :row+1].T, tape, p + f"context/h{head}/r{row}")
            emit(p + "context", context)
            merged = emit(p + "merged", context.transpose(1, 0, 2).reshape(n, h*d))
            projected = emit(p + "o_proj", linear(merged, weight(src + "self_attn.o_proj.weight", (hidden, h*d)), tape, p + "o_proj"))
            x = emit(p + "attention_residual", ring(residual + projected))
            norm = normalize(x, weight(src + "post_attention_layernorm.weight", (hidden,)), p + "post_attention_norm")
            gate = emit(p + "gate_proj", linear(norm, weight(src + "mlp.gate_proj.weight", (mid, hidden)), tape, p + "gate_proj"))
            up = emit(p + "up_proj", linear(norm, weight(src + "mlp.up_proj.weight", (mid, hidden)), tape, p + "up_proj"))
            activated = emit(p + "silu", silu(gate, tape, p + "silu"))
            product = emit(p + "gate_up", mul(activated, up, tape, p + "gate_up"))
            down = emit(p + "down_proj", linear(product, weight(src + "mlp.down_proj.weight", (hidden, mid)), tape, p + "down_proj"))
            x = emit(p + "output", ring(x + down))
        x = normalize(x, weight("model.norm.weight", (hidden,)), root + "final_norm")
        last = emit(root + "last_hidden", x[-1])
        head = embedding if config.get("tie_word_embeddings", False) else weight("lm_head.weight", (vocab, hidden))
        logits = emit(root + "logits", linear(last, head, tape, root + "head"))
        token = stable_argmax(logits)
        emit(root + "selected", token)
        selected.append(token)
        ids.append(token)
        emit(root + "feedback", ids)
        selection_finished = getattr(weights, "selection_finished", None)
        if selection_finished is not None:
            selection_finished()
    if guards.keys() != guards_used:
        raise OracleError(f"unused guard placements: {sorted(guards.keys()-guards_used)}")
    tape.finish()
    if guards:
        emit("acceptance.valid", int(sticky))
    return Result(selected if sticky else [], trace, accepted=sticky)


def _assert_fixture_gqa_coverage(config, trace, outputs, batches=None):
    """Adversarial public-fixture check, not a replacement for full comparison.

    Constant/repeated KV groups can hide wrong-group routing even in an otherwise
    complete native trace. Require a value-routing change larger than an entire
    Q18 rounding unit, hence distinguishable under EVERY low-residue mask.
    """
    h, kv = config["num_attention_heads"], config["num_key_value_heads"]
    if kv < 2 or h <= kv:
        raise OracleError("adversarial GQA fixture requires multiple distinct KV groups")
    for selection in range(outputs):
        for layer in range(config["num_hidden_layers"]):
            p = f"s{selection}/layer.{layer}/"
            values, probabilities = trace[p+"full_prefix_v"], trace[p+"probabilities"]
            if np.array_equal(values[0], values[1]):
                raise OracleError("degenerate native GQA fixture repeats KV groups")
            for head in range(h):
                correct, incorrect = head//(h//kv), (head//(h//kv)+1) % kv
                difference = probabilities[head, 1, :2] @ (values[correct, :2]-values[incorrect, :2])
                if not any(abs(int(v)) >= Q for v in difference.flat):
                    raise OracleError("native GQA fixture cannot distinguish wrong-group value routing")
                if batches is not None:
                    name = p+f"context/h{head}/r1"
                    bad = Tape({name: batches[name]})
                    wrong = linear(probabilities[head, 1, :2], values[incorrect, :2].T, bad, name)
                    bad.finish()
                    altered = trace[p+"context"].copy()
                    altered[head, 1] = wrong
                    try:
                        compare_trace({p+"context": trace[p+"context"]}, {p+"context": altered})
                    except OracleError:
                        pass
                    else:
                        raise OracleError("wrong KV-group routing mutation not rejected using actual mask tape")


def _native_capture_main():
    """Anonymous-stdin-only unit-test bridge. Never writes private evidence."""
    import copy
    import json
    import sys

    bundle = json.load(sys.stdin)
    config = bundle["config"]
    weights = {}
    for name, source in bundle["sources"].items():
        shape, values = tuple(source["shape"]), source["values"]
        if len(values) != math.prod(shape):
            raise OracleError("native public source element count")
        if len(shape) == 2:
            # Public fixture snapshot remains caller-owned; construct at most a
            # bounded row block, never a complete weight ndarray/checkpoint.
            columns = shape[1]
            weights[name] = PublicMatrix(shape, lambda start, stop, values=values, columns=columns:
                                         array(values[start*columns:stop*columns]).reshape(stop-start, columns))
        else:
            weights[name] = array(values).reshape(shape)
    keys = [bytes.fromhex(key) for key in bundle["pair_keys_hex"]]
    public_seed = int.from_bytes(bytes.fromhex(bundle["public_seed_le_hex"]), "little")
    layout = generated_native_layout(config, len(bundle["input_ids"]), bundle["outputs"],
                                     bundle["events"], bundle["spans"])
    checked = None
    if bundle.get("checked_profile", False):
        checked = CheckedRangeSchedule.from_mapping({
            spec["name"]: RangeCheck(spec["lower"], spec["upper"], spec["fallback"])
            for spec in bundle["guard_schedule"]})
    # Frozen two-wide public coefficients: sole frequency is exactly one. This
    # bridge deliberately refuses wider/source-present RoPE coefficient claims.
    half = config.get("head_dim", config["hidden_size"]//config["num_attention_heads"])//2
    if half != 1 and config["rope_theta"] != 1:
        raise OracleError("native bridge requires frozen wider RoPE rows; theta=1 fixture supported")
    rotary = {position: ([cos]*half, [sin]*half) for position, cos, sin in
              ((0, Q, 0), (1, 141637, 220586), (2, -109090, 238366))}

    def audit(checkpoints, events):
        return audit_native_capture(config, weights, bundle["input_ids"], bundle["outputs"], rotary,
                                    checkpoints=checkpoints, events=events, pair_keys=keys,
                                    public_seed=public_seed, layout=layout,
                                    final_components=bundle["final_components"], checked_ranges=checked,
                                    gqa_controls=bundle.get("assert_distinct_gqa", False),
                                    dense_head_control=config["vocab_size"] == 151936)

    result = audit(bundle["checkpoints"], bundle["events"])
    if result.accepted != bundle["expected_accepted"]:
        raise OracleError("frozen native fixture acceptance outcome differs")
    controls = dict.fromkeys(("tensor_mutation", "missing_frame", "cursor_order", "missing_trace",
                              "rounding_shift", "final_response_mutation"), 1)
    if bundle.get("assert_distinct_gqa", False):
        controls["wrong_kv_group_routing"] = config["num_attention_heads"]*config["num_hidden_layers"]*bundle["outputs"]

    def receipt():
        print(json.dumps({"family": config["model_type"], "mode": bundle["fixture_mode"],
                          "lookup_tiled": bundle["lookup_tiled"], "selection_tiled": bundle["selection_tiled"],
                          "layers": config["num_hidden_layers"], "heads": config["num_attention_heads"],
                          "kv_heads": config["num_key_value_heads"], "head_dim": config["head_dim"],
                          "hidden": config["hidden_size"], "intermediate": config["intermediate_size"],
                          "vocab": config["vocab_size"], "checkpoints": len(bundle["checkpoints"]),
                          "truncation_invocations": len({e["binding"] for e in bundle["events"]}),
                          "negative_controls": controls}, sort_keys=True))
    if config["num_hidden_layers"] >= 2 and bundle.get("checked_profile", False):
        name = "s1/layer.0/input_norm/input/checked_range/tile_sticky"
        tiles = math.ceil((len(bundle["input_ids"])+1)*config["hidden_size"]/128)
        if tiles < 2 or name not in result.trace or result.trace[name].shape != (tiles,):
            raise OracleError("multilayer GQA fixture must exercise complete actual guard tiles")
        if not result.accepted and result.trace[name].tolist() != [1]*(tiles-1)+[0]:
            raise OracleError("guard tile tail must fail after a valid first tile")
        native = next(c for c in bundle["checkpoints"] if c.get("oracle_name") == name)
        mutated = array(native["values"]).copy()
        mutated[-1] = 1-mutated[-1]
        try:
            compare_trace({name: result.trace[name]}, {name: mutated})
        except OracleError:
            pass
        else:
            raise OracleError("guard tile sticky mutation control failed")
        controls["guard_tile_sticky_mutation"] = 1
    if config["vocab_size"] == 151936:
        if result.tokens != [151935, 0]:
            raise OracleError("full-vocabulary frozen tail selection/feedback outcome")
        head = next(c for c in bundle["checkpoints"] if c["selection"] == 0 and c["id"] == "output_head")
        controls = {"head_rounding_tape_rotation": 1, "head_tail_corruption": 1,
                    "head_dense_middle_corruption": 1, "missing_head": 1}
        middle = config["vocab_size"]//2
        if not any(v != 0 for v in weights["lm_head.weight"].rows(middle, middle+1).flat):
            raise OracleError("full-vocabulary middle-row source must be nonzero")
        for row in (-1, middle):
            values = array(head["values"]).copy()
            values[row] += 1
            try:
                compare_trace({"logits": result.trace["s0/logits"]}, {"logits": values})
            except OracleError:
                pass
            else:
                raise OracleError("full-vocabulary logit mutation control failed")
        try:
            compare_trace({"logits": result.trace["s0/logits"]}, {})
        except OracleError:
            pass
        else:
            raise OracleError("full-vocabulary missing-head control failed")
        receipt()
        return  # Full trace/tape audited; avoid six repeated huge AES replays.
    if not result.accepted:
        if result.trace["s0/selected"] != 2 or any(
                any(v != 1 for v in result.trace[spec["name"] + "/checked_range/all_valid"].flat)
                for spec in bundle["guard_schedule"] if spec["name"].startswith("s0/")):
            raise OracleError("later-feedback-invalid fixture failed before feedback")
    corrupted = copy.deepcopy(bundle["checkpoints"])
    if "components" in corrupted[0]:
        corrupted[0]["components"][0][0] = (corrupted[0]["components"][0][0]+1) % (1 << 64)
    else:
        corrupted[0]["values"][0] = int(ring(corrupted[0]["values"][0]+1))
    try:
        audit(corrupted, bundle["events"])
    except OracleError:
        pass
    else:
        raise OracleError("native tensor corruption negative control failed")
    try:
        audit(bundle["checkpoints"], bundle["events"][1:])
    except OracleError:
        pass
    else:
        raise OracleError("native truncation evidence corruption negative control failed")
    bindings = list(dict.fromkeys(event["binding"] for event in bundle["events"]))
    if len(bindings) < 2:
        raise OracleError("native order corruption control needs two truncations")
    reordered = ([event for event in bundle["events"] if event["binding"] == bindings[1]]
                 + [event for event in bundle["events"] if event["binding"] == bindings[0]]
                 + [event for event in bundle["events"] if event["binding"] not in bindings[:2]])
    try:
        audit(bundle["checkpoints"], reordered)
    except OracleError:
        pass
    else:
        raise OracleError("native PRSS order corruption negative control failed")
    try:
        audit(bundle["checkpoints"][1:], bundle["events"])
    except OracleError:
        pass
    else:
        raise OracleError("native missing checkpoint negative control failed")
    changed_layout = copy.deepcopy(layout)
    first_name = next(iter(changed_layout))
    changed_layout[first_name]["shift"] = 17 if changed_layout[first_name]["shift"] == 18 else 18
    try:
        audit_native_capture(config, weights, bundle["input_ids"], bundle["outputs"], rotary,
                             checkpoints=bundle["checkpoints"], events=bundle["events"],
                             pair_keys=keys, public_seed=public_seed, layout=changed_layout,
                             final_components=bundle["final_components"], checked_ranges=checked)
    except OracleError:
        pass
    else:
        raise OracleError("native rounding-shift corruption negative control failed")
    final_corrupted = copy.deepcopy(bundle["final_components"])
    final_corrupted[0][0] = (final_corrupted[0][0]+1) % (1 << 64)
    try:
        audit_native_capture(config, weights, bundle["input_ids"], bundle["outputs"], rotary,
                             checkpoints=bundle["checkpoints"], events=bundle["events"],
                             pair_keys=keys, public_seed=public_seed, layout=layout,
                             final_components=final_corrupted, checked_ranges=checked)
    except OracleError:
        pass
    else:
        raise OracleError("native final response mutation negative control failed")
    receipt()


# Anonymous-pipe adapter, generated expanded fixtures ONLY. No collector bundle,
# private stdout diagnostics, alternate evaluator, or source authority is added.
_PIPE_HEADER = struct.Struct("<4sBBHIIII")
_CAPTURE_WORDS = 1 << 20


def _pipe_exact(reader, count, *, raw_limit=None):
    """Per-read requested payload envelope, NOT allocated backing or total memory.

    3n covers result plus prior/replacement read returns, assuming read(requested)
    obeys its size bound. Buffered IO internals, JSON/Python objects, native/RSS
    and raw payloads retained by callers/callbacks are outside this policy.
    """
    if type(count) is not int or not 0 <= count <= 64 << 20:
        raise OracleError("pipe allocation bound")
    if raw_limit is not None and (type(raw_limit) is not int or raw_limit < 0):
        raise OracleError("pipe raw requested-payload limit")
    if raw_limit is not None and 3*count > raw_limit:
        raise OracleError("pipe raw requested-payload envelope")
    result = bytearray(count)
    at = 0
    while at < count:
        requested = count-at
        chunk = reader.read(requested)
        if not chunk:
            raise OracleError("truncated pipe frame")
        # An excessive external return already exists; reject before assignment
        # can grow our destination. No preallocation claim over that reader.
        if len(chunk) > requested:
            raise OracleError("oversized pipe reader return")
        result[at:at+len(chunk)] = chunk
        at += len(chunk)
    return result


def _pipe_receive(reader, sequence, expected, *, event=None, spans=(), raw_limit=None):
    """Validate fixed header and exact per-kind geometry BEFORE payload allocation."""
    while True:
        magic, version, kind, reserved, seq, index, count, length = _PIPE_HEADER.unpack(
            _pipe_exact(reader, _PIPE_HEADER.size, raw_limit=raw_limit))
        if magic != b"POT1" or version != 1 or reserved or seq != sequence:
            raise OracleError("pipe header/order")
        if kind == 13:
            if (index, count, length) != (0, 1, 2):
                raise OracleError("pipe failure schema")
            code = struct.unpack("<H", _pipe_exact(reader, 2, raw_limit=raw_limit))[0]
            if not 1 <= code <= 8:
                raise OracleError("pipe failure code")
            raise OracleError("native pipe failure")
        if kind == 6 and event is not None:
            if not 0 <= index < len(spans) or count != 1 or length != 92:
                raise OracleError("pipe event bounds")
            event(index, _pipe_exact(reader, 92, raw_limit=raw_limit))
            continue
        if (kind, index, count, length) != expected:
            raise OracleError("unexpected pipe response geometry")
        return _pipe_exact(reader, length, raw_limit=raw_limit)


def _metadata_raw_remaining(mbytes, raw_limit):
    """Pure metadata requested-payload admission, before allocation/read/JSON.

    Exact policy threshold M + max(72, 3*min(M,4096)); not a JSON expansion,
    allocator capacity or total-owner proof. Generated <=64 MiB stays HOLD.
    """
    if type(mbytes) is not int or not 0 < mbytes <= 64 << 20:
        raise OracleError("metadata raw length")
    if type(raw_limit) is not int or raw_limit < 0:
        raise OracleError("metadata raw requested-payload limit")
    if mbytes + max(72, 3*min(mbytes, 4096)) > raw_limit:
        raise OracleError("metadata raw requested-payload envelope")
    return raw_limit-mbytes


def _pipe_send(writer, kind, seq, index=0):
    if kind not in range(32, 38) or not 1 <= seq < 1 << 32 or not 0 <= index < 1 << 32:
        raise OracleError("pipe command geometry")
    writer.write(_PIPE_HEADER.pack(b"POT1", 1, kind, 0, seq, index, 0, 0))
    writer.flush()


def _freeze_public(value):
    if type(value) is dict:
        return tuple((k, _freeze_public(v)) for k, v in sorted(value.items()))
    if type(value) is list:
        return tuple(_freeze_public(v) for v in value)
    if type(value) not in (str, int, bool, type(None), float):
        raise OracleError("public metadata type")
    # Comparator provenance permits only exact immutable leaves.
    return repr(value) if type(value) is float else value


@dataclass(frozen=True, slots=True)
class JSONLexicalPolicy:
    """Optional lexical refusal only; ordinary json.loads has NO byte quote.

    requested_copy_bytes admits only the NEW immutable config payload, not its
    header, outer graph, decoding, total memory or parser workspace. copy_admit
    observes (requested_bytes, held_outer_meta) before that one conversion.
    """
    byte_limit: int = 16_777_216
    node_limit: int = 65_536
    depth_limit: int = 17
    token_limit: int = 1024
    requested_copy_bytes: int = 4096
    copy_admit: object = None

    def __post_init__(self):
        _validate_json_policy(self)


def _validate_json_policy(policy):
    if policy is None:
        return
    if type(policy) is not JSONLexicalPolicy:
        raise OracleError("invalid JSON lexical policy")
    try:
        limits = ((policy.byte_limit, 16_777_216), (policy.node_limit, 65_536),
                  (policy.depth_limit, 17), (policy.token_limit, 1024),
                  (policy.requested_copy_bytes, 4096))
        hook = policy.copy_admit
    except AttributeError:
        raise OracleError("invalid JSON lexical policy") from None
    for value, ceiling in limits:
        if type(value) is not int or not 0 < value <= ceiling:
            raise OracleError("invalid JSON lexical policy")
    if hook is not None and not callable(hook):
        raise OracleError("invalid JSON lexical policy")


_JSON_CENSUS_REFERENCE = None


def _json_census_reference():
    """Load neutral file once, before INIT/admitted work; no package startup."""
    global _JSON_CENSUS_REFERENCE
    if _JSON_CENSUS_REFERENCE is None:
        path = Path(__file__).resolve().parents[1] / "python/pllm/_json_census.py"
        spec = importlib.util.spec_from_file_location("independent_json_census", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        _JSON_CENSUS_REFERENCE = reference
    return _JSON_CENSUS_REFERENCE


def _lexical_json_census(raw, policy, reference):
    try:
        return reference.json_census(raw, byte_limit=policy.byte_limit,
            node_limit=policy.node_limit, depth_limit=policy.depth_limit,
            token_limit=policy.token_limit)
    except reference.JSONCensusError:
        raise OracleError("JSON lexical envelope refused") from None


def _stream_public_schema(meta, init, *, json_policy=None):
    import json

    _validate_json_policy(json_policy)
    reference = _json_census_reference() if json_policy is not None else None
    required = {"config", "config_bytes", "policy", "provenance", "recipe", "program", "sources",
                "boundaries", "spans", "guards", "demands", "frontiers", "terminal",
                "capture_peak", "pending_peak", "tape_words", "source_words",
                "checkpoint_words", "total_tape", "invocations", "logical_len", "outputs", "input_rows",
                "fixture_mode", "expected_accepted"}
    if type(meta) is not dict or set(meta) != required or meta["provenance"] != "generated":
        raise OracleError("generated public metadata schema")
    c = meta["config"]
    raw = meta["config_bytes"]
    if (type(raw) is not list or not 0 < len(raw) <= 4096
            or any(type(n) is not int or not 0 <= n <= 255 for n in raw)):
        raise OracleError("actual generated config bytes required")
    if json_policy is not None:
        n = len(raw)
        if n > json_policy.byte_limit or n > json_policy.requested_copy_bytes:
            raise OracleError("JSON requested immutable-copy envelope refused")
        if json_policy.copy_admit is not None:
            try:
                json_policy.copy_admit(n, meta)
            except Exception:
                # Only caller copy admission; decoder exceptions below unchanged.
                raise OracleError("JSON requested immutable-copy envelope refused") from None
            if type(meta) is not dict or set(meta) != required or meta["provenance"] != "generated":
                raise OracleError("generated public metadata schema")
            _validate_json_policy(json_policy)
            if (meta.get("config_bytes") is not raw or meta.get("config") is not c
                    or type(raw) is not list or len(raw) != n
                    or not 0 < n <= min(4096, json_policy.byte_limit, json_policy.requested_copy_bytes)
                    or any(type(value) is not int or not 0 <= value <= 255 for value in raw)):
                raise OracleError("JSON requested immutable-copy envelope refused")
    config_bytes = bytes(raw)
    if json_policy is not None:
        _lexical_json_census(config_bytes, json_policy, reference)
    if json.loads(config_bytes) != c:
        raise OracleError("actual generated config bytes required")
    del config_bytes
    # This first native selector is deliberately narrower than the collector.
    if (type(c) is not dict or c.get("model_type") != "qwen3"
            or (c.get("hidden_size"), c.get("intermediate_size"), c.get("num_hidden_layers"))
            not in ((2, 2, 1), (72, 131, 2))
            or meta["outputs"] != 2 or meta["input_rows"] != 2):
        raise OracleError("generated stream geometry")
    for key in ("capture_peak", "pending_peak", "source_words", "checkpoint_words", "total_tape",
                "invocations", "logical_len"):
        if type(meta[key]) is not int or not 0 < meta[key] <= 8_000_000:
            raise OracleError("public finite census")
    if max(meta["capture_peak"], meta["pending_peak"]) > _CAPTURE_WORDS:
        raise OracleError("public frontier bound")
    boundaries, spans, demands = (meta[k] for k in ("boundaries", "spans", "demands"))
    if any(type(v) is not list or not 0 < len(v) <= _CAPTURE_WORDS for v in (boundaries, spans, demands)):
        raise OracleError("public metadata counts")
    _, r, s, d, inv, ws, wc, wt, outputs, rows = init
    if (r, s, d, inv, ws, wc, wt, outputs, rows) != (len(boundaries), len(spans), len(demands),
            meta["invocations"], meta["source_words"], meta["checkpoint_words"], meta["total_tape"], 2, 2):
        raise OracleError("public INIT census disagreement")
    if type(meta["sources"]) is not dict or not meta["sources"]:
        raise OracleError("generated source descriptors required")
    source_words = 0
    for name, source in meta["sources"].items():
        if (type(name) is not str or not name or type(source) is not dict
                or "public_q18" in source or "values" in source):
            raise OracleError("public source descriptor schema")
        shape = source["shape"]
        if (type(shape) is not list or not 1 <= len(shape) <= 2
                or any(type(n) is not int or n <= 0 for n in shape)
                or math.prod(shape) > _CAPTURE_WORDS or len(shape) == 2 and shape[1] > 65536
                or type(source["digest"]) is not list or len(source["digest"]) != 32
                or any(type(n) is not int or not 0 <= n <= 255 for n in source["digest"])):
            raise OracleError("public source shape/commitment")
        source_words += math.prod(shape)
    if source_words != ws:
        raise OracleError("public source word census")
    checked = meta["policy"].get("range_policy") == "checked"
    if meta["policy"].get("range_policy") not in ("static", "checked"):
        raise OracleError("public checked policy")
    guards = CheckedRangeSchedule.from_mapping({g["name"]: RangeCheck(g["lower"], g["upper"], g["fallback"])
                                                for g in meta["guards"]}) if checked else None
    if len(meta["guards"]) != (len(guards.entries) if guards is not None else 0):
        raise OracleError("duplicate/unchecked guard placement")
    declaration, used, tape_names = [], set(), set()
    record_words = tape_words = 0
    owner_words = [0]*s
    for demand in demands:
        if type(demand) is not dict or demand.get("kind") not in ("C", "T"):
            raise OracleError("public demand kind")
        shape = demand["shape"]
        if type(shape) is not list or any(type(n) is not int or n <= 0 for n in shape):
            raise OracleError("public demand shape")
        count = math.prod(shape)
        if not 0 < count <= _CAPTURE_WORDS or type(demand["end"]) is not int or not 0 < demand["end"] <= meta["logical_len"]:
            raise OracleError("public demand endpoint/word count")
        if demand["kind"] == "T":
            j = demand["span"]
            if (type(j) is not int or not 0 <= j < len(spans) or demand["end"] != spans[j]["end"]
                    or demand["name"] in tape_names or demand["shift"] not in (5, 18)):
                raise OracleError("public tape owner/order")
            tape_names.add(demand["name"])
            tape_words += count
            owner_words[j] += count
        else:
            specs = []
            for index in demand["aliases"]:
                if type(index) is not int or not 0 <= index < len(boundaries) or index in used:
                    raise OracleError("public checkpoint alias coverage")
                used.add(index)
                b = boundaries[index]
                spec = (b["selection"], b["id"], tuple(b["shape"]), b["batch_dimension"], b["oracle_name"])
                if (_native_checkpoint_name(spec[0], spec[1], spec[4]) != demand["name"]
                        or _native_checkpoint_shape(spec[2], spec[3], spec[1])[1] != tuple(shape)):
                    raise OracleError("public alias name/shape")
                coords = b["coordinates"]
                if len(coords) != count or any(type(w) is not list or len(w) != 2
                        or any(type(n) is not int or n < 0 for n in w) or w[0] >= meta["logical_len"] for w in coords):
                    raise OracleError("public boundary coordinates")
                specs.append(spec)
                record_words += count
            declaration.append((demand["name"], tuple(specs)))
    if len(used) != r or record_words != wc or tape_words != wt or len(meta["frontiers"]) != d:
        raise OracleError("public complete demand census")
    if owner_words != meta["tape_words"]:
        raise OracleError("public owner word census")
    previous = 0
    for frontier in meta["frontiers"]:
        if (type(frontier) is not list or len(frontier) != 3 or any(type(n) is not int for n in frontier)
                or not previous <= frontier[0] <= meta["logical_len"]
                or not 0 <= frontier[1] <= meta["capture_peak"] or not 0 <= frontier[2] <= meta["pending_peak"]):
            raise OracleError("public finite frontier table")
        previous = frontier[0]
    if (max(f[1] for f in meta["frontiers"]) != meta["capture_peak"]
            or max(f[2] for f in meta["frontiers"]) != meta["pending_peak"]
            or meta["pending_peak"] >= wt):
        raise OracleError("public frontier peak/release census")
    ordered = sorted((v for v in spans if v["start"] < v["end"]), key=lambda v: v["start"])
    if any(a["end"] > b["start"] for a, b in zip(ordered, ordered[1:])):
        raise OracleError("overlapping public spans")
    if meta["terminal"] is not None:
        j = meta["terminal"]
        if (not checked or guards.entries or type(j) is not int or declaration[-1] !=
                ("acceptance.valid", ((0, "acceptance.valid", (), False, "acceptance.valid"),))
                or demands[-1]["aliases"] != [j]):
            raise OracleError("exact checked-empty terminal group required")
    elif checked and not guards.entries:
        raise OracleError("missing genuine checked-empty terminal slot")
    provenance = (("kind", "generated"), ("fixture", _freeze_public({k: meta[k] for k in
                  ("config", "config_bytes", "policy", "recipe", "program", "sources")})))
    return tuple(declaration), provenance, guards


class _StreamProvider:
    """Iterator without a suspended generator frame retaining its last array."""
    def __init__(self, client, tape):
        self.client, self.tape = client, tape

    def __iter__(self):
        return self

    def __next__(self):
        return self.client.take_tape() if self.tape else self.client.take_checkpoint()


class GeneratedNativeStream:
    def __init__(self, reader, writer, meta, replay, *, mutation="none", raw_limit=None):
        if raw_limit is not None and (type(raw_limit) is not int or raw_limit < 0):
            raise OracleError("pipe raw requested-payload limit")
        self.reader, self.writer, self.meta, self.replay = reader, writer, meta, replay
        self.raw_limit = raw_limit
        self.sequence, self.at, self.alias_at, self.frontier = 1, 0, 0, 0
        self.pending, self.mapped, self.binding_owner = {}, {}, {}
        self.pending_words = self.peak_words = self.events = 0
        self.span_words = [0]*len(meta["spans"])
        self.last_tape = [-1]*len(meta["spans"])
        for i, demand in enumerate(meta["demands"]):
            if demand["kind"] == "T":
                self.last_tape[demand["span"]] = i
        self.tape_eof = self.checkpoint_eof = False
        self.state, self.mutation = "running", mutation

    def census(self):
        """Logical payload/last-consumer census, not Python RSS/secure erasure."""
        words = sum(e["lanes"] for events, _ in self.pending.values() for e in events.values())
        words += sum(batch["residues"].size for batches in self.mapped.values() for batch in batches.values())
        if words != self.pending_words or any(self.last_tape[j] < self.at for j in self.pending.keys() | self.mapped.keys()):
            raise OracleError("old operator survived its last semantic consumer")
        if self.pending_words > self.meta.get("pending_peak", self.pending_words):
            raise OracleError("pending payload high-water exceeded")

    def _event(self, index, payload):
        if self.tape_eof or self.state != "running" or self.events >= 4*self.meta["invocations"]:
            raise OracleError("event after tape EOF/bound")
        binding = bytes(payload[:32]).hex()
        role, to, phase, reserved = struct.unpack_from("<BBBB", payload, 32)
        node, offset, step, invocation, public, pair, lanes, span = struct.unpack_from("<QQQQQQII", payload, 36)
        s = self.meta["spans"][index]
        if (role not in (0, 1, 2) or to not in (0, 1, 2) or phase not in (10, 11) or reserved
                or span != index or not s["start"] <= node < s["end"] or not 1 <= lanes <= 1024):
            raise OracleError("physical event owner/geometry")
        event = {"binding": binding, "role": role, "to": to, "phase": phase, "node": node,
                 "loop_offset": offset, "tensor_step": None if step == (1 << 64)-1 else step,
                 "invocation": None if invocation == (1 << 64)-1 else invocation,
                 "public_end": public, "pair_end": pair, "lanes": lanes}
        if binding not in self.binding_owner:
            # Reserve lookahead before AES expansion/array mapping, including a
            # physically incomplete group. It is not one fictitious batch.
            if (self.pending_words+lanes > self.meta["pending_peak"]
                    or self.span_words[index]+lanes > self.meta["tape_words"][index]):
                raise OracleError("runtime pending operator bound")
            self.binding_owner[binding] = index
            self.pending_words += lanes
            self.span_words[index] += lanes
            self.peak_words = max(self.peak_words, self.pending_words)
            self.pending.setdefault(index, ({}, {}))[0][binding] = event
        elif self.binding_owner[binding] != index:
            raise OracleError("native invocation crossed owners")
        # AES replay is shift-independent; the public owner recipe assigns the
        # exact shift only when the complete owner group is mapped below.
        complete = self.replay.ingest(event, None)
        if complete is not None:
            self.pending[index][1][binding] = complete
        self.events += 1

    def request(self, kind, index, response, count=0, length=0):
        if self.state not in ("running", "components") or self.sequence >= 1 << 32:
            raise OracleError("pipe terminal request")
        _pipe_send(self.writer, kind, self.sequence, index)
        payload = _pipe_receive(self.reader, self.sequence, (response, index if kind in (32, 33) else 0, count, length),
                                event=self._event if self.state == "running" else None, spans=self.meta["spans"],
                                raw_limit=self.raw_limit)
        self.sequence += 1
        self.census()
        return payload

    def _demand(self, kind):
        demands = self.meta["demands"]
        if self.at >= len(demands) or demands[self.at]["kind"] != kind:
            raise OracleError("semantic demand order")
        return demands[self.at]

    def take_tape(self):
        if self.tape_eof:
            raise StopIteration
        if self.at == len(self.meta["demands"]) or self.meta["demands"][self.at]["name"] == "acceptance.valid":
            if self.pending_words or self.pending or self.mapped:
                raise OracleError("unused pending native operators")
            self.replay.finish()
            self.request(35, 0, 10)
            self.tape_eof = True
            raise StopIteration
        d = self._demand("T")
        span = d["span"]
        if self.frontier < d["end"]:
            self.request(32, span, 7)
            self.frontier = max(self.frontier, d["end"])
        if span not in self.mapped:
            events, physical = self.pending.pop(span, ({}, {}))
            if len(events) != len(physical):
                raise OracleError("incomplete physical owner")
            # Existing per-owner branch/stage tables; never an all-run event list.
            layout = generated_native_layout(self.meta["config"], 2, 2, events.values(), [self.meta["spans"][span]])
            shifts = {}
            for spec in layout.values():
                for binding, _ in spec["lanes"]:
                    if binding in shifts and shifts[binding] != spec["shift"]:
                        raise OracleError("inconsistent native owner shift")
                    shifts[binding] = spec["shift"]
            if set(shifts) != set(physical):
                raise OracleError("complete owner invocation coverage")
            for binding, shift in shifts.items():
                physical[binding]["shift"] = shift
                del self.binding_owner[binding]
            mapped = adapt_tapes(physical, layout)
            self.mapped[span] = mapped.batches
            del mapped, physical, events, layout
        batches = self.mapped[span]
        if not batches or next(iter(batches)) != d["name"]:
            raise OracleError("mapped semantic tape order")
        batch = batches.pop(d["name"])
        if tuple(batch["residues"].shape) != tuple(d["shape"]) or batch["shift"] != d["shift"]:
            raise OracleError("mapped semantic tape geometry")
        words = math.prod(d["shape"])
        self.pending_words -= words
        self.span_words[span] -= words
        if not batches:
            del self.mapped[span]
            if self.span_words[span]:
                raise OracleError("last-consumer operator residue")
        self.at += 1
        self.census()
        return d["name"], batch

    def take_checkpoint(self):
        if self.at == len(self.meta["demands"]):
            if self.state != "components" or self.pending_words or self.binding_owner or not self.tape_eof:
                raise OracleError("premature checkpoint EOF")
            if not self.checkpoint_eof:
                self.request(36, 0, 11)
                self.checkpoint_eof = True
            raise StopIteration
        d = self._demand("C")
        index = d["aliases"][self.alias_at]
        if self.state == "components" and index != self.meta["terminal"]:
            raise OracleError("nonterminal checkpoint after FINAL")
        if index == self.meta["terminal"] and self.state != "components":
            raise OracleError("checked-empty acceptance must follow real finish")
        b = self.meta["boundaries"][index]
        count = math.prod(b["shape"])
        payload = self.request(33, index, 8, count, 8*count)
        self.frontier = max(self.frontier, d["end"])
        values = [n[0] for n in struct.iter_unpack("<q", payload)]
        if index == self.meta["terminal"]:
            if self.mutation == "acceptance":
                values[0] = 1-values[0]
            elif self.mutation == "missing_acceptance":
                raise OracleError("mutated missing actual acceptance slot")
        self.alias_at += 1
        if self.alias_at == len(d["aliases"]):
            self.alias_at = 0
            self.at += 1
        self.census()
        return {k: b[k] for k in ("selection", "id", "shape", "oracle_name", "batch_dimension")} | {"values": values}

    def final(self):
        if not self.tape_eof or self.state != "running":
            raise OracleError("premature or duplicate FINAL")
        if self.at < len(self.meta["demands"]):
            d = self._demand("C")
            if d["aliases"] != [self.meta["terminal"]] or self.alias_at:
                raise OracleError("FINAL missing nonterminal checkpoint")
        width = 2+int(self.meta["policy"]["range_policy"] == "checked")
        payload = self.request(34, 0, 9, 3*width, 24*width)
        self.state = "components"
        components = [list(struct.unpack_from("<"+"Q"*width, payload, 8*width*role)) for role in range(3)]
        if self.mutation == "final_validity":
            components[0][-1] = (components[0][-1]+1) % (1 << 64)
        elif self.mutation == "pipe_after_final":
            raise OracleError("mutated pipe loss after FINAL")
        return components

    def success(self):
        if self.state != "components" or not self.checkpoint_eof or self.pending_words or self.binding_owner:
            raise OracleError("unfinished native stream")
        self.request(37, 0, 12)
        self.state = "closed"


def _native_stream_main(backing_limit, mutation, *, json_policy=None):
    import hashlib
    import json
    import sys

    if type(backing_limit) is not int or not 0 < backing_limit <= 64 << 20:
        raise OracleError("fixed test raw requested-payload policy limit")
    if mutation not in ("none", "acceptance", "final_validity", "missing_acceptance", "pipe_after_final"):
        raise OracleError("fixed native stream control")
    _validate_json_policy(json_policy)
    reference = _json_census_reference() if json_policy is not None else None
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    # CLI backing_limit name is historical: this is a raw per-call refusal
    # policy, never a certificate for JSON, Python/native owners or total RSS.
    init = struct.unpack("<10Q", _pipe_receive(reader, 0, (1, 0, 10, 80), raw_limit=backing_limit))
    mbytes, r, spans, demands, inv, ws, wc, wt, outputs, rows = init
    if (not 0 < mbytes <= backing_limit or any(not 0 < n <= 8_000_000 for n in (r, spans, demands, inv, ws, wc, wt))
            or outputs != 2 or rows != 2 or rows+outputs > MAX_PREFIX):
        raise OracleError("fixed INIT bounds")
    metadata_raw_limit = _metadata_raw_remaining(mbytes, backing_limit)
    metadata = bytearray(mbytes)
    for offset in range(0, mbytes, 4096):
        count = min(4096, mbytes-offset)
        # Direct assignment: no old named part survives into the next receive.
        metadata[offset:offset+count] = _pipe_receive(reader, 0, (2, offset, count, count),
                                                     raw_limit=metadata_raw_limit)
    if json_policy is not None:
        # Exact assembly is exclusively owned here: no external alias/callback
        # during synchronous scan. Readonly alone is NOT general alias safety.
        view = memoryview(metadata).toreadonly()
        try:
            _lexical_json_census(view, json_policy, reference)
        finally:
            view.release()
        del view
    meta = json.loads(metadata)
    del metadata
    if json_policy is None:
        declaration, provenance, guards = _stream_public_schema(meta, init)
    else:
        declaration, provenance, guards = _stream_public_schema(meta, init, json_policy=json_policy)
    weights, received = {}, 0
    for index, (name, descriptor) in enumerate(sorted(meta["sources"].items())):
        if "public_q18" in descriptor or "values" in descriptor:
            raise OracleError("source values forbidden in public metadata")
        shape = tuple(descriptor["shape"])
        if not 1 <= len(shape) <= 2 or any(type(n) is not int or n <= 0 for n in shape) or math.prod(shape) > _CAPTURE_WORDS:
            raise OracleError("bounded generated source shape")
        total = math.prod(shape)
        if received+total > ws:
            raise OracleError("generated source reservation exceeded")
        values = [None]*total
        step = min(128, 65536//shape[1])*shape[1] if len(shape) == 2 else 65536
        digest = hashlib.sha256()
        for offset in range(0, total, step):
            count = min(step, total-offset)
            payload = _pipe_receive(reader, 0, (3, index, count, 8+8*count), raw_limit=backing_limit)
            if struct.unpack_from("<Q", payload)[0] != offset:
                raise OracleError("source offset/order")
            digest.update(payload[8:])
            values[offset:offset+count] = [n[0] for n in struct.iter_unpack("<q", payload[8:])]
            del payload
        if list(digest.digest()) != descriptor["digest"]:
            raise OracleError("generated source commitment mismatch")
        received += total
        if received > ws:
            raise OracleError("generated source census exceeded")
        if len(shape) == 2:
            columns = shape[1]
            weights[name] = PublicMatrix(shape, lambda start, stop, v=values, cols=columns:
                                         array(v[start*cols:stop*cols]).reshape(stop-start, cols))
        else:
            weights[name] = array(values).reshape(shape)
        del values
    if received != ws:
        raise OracleError("incomplete generated source census")
    ids = list(struct.unpack("<2Q", _pipe_receive(reader, 0, (4, 0, rows, 8*rows), raw_limit=backing_limit)))
    if any(i >= meta["config"]["vocab_size"] for i in ids):
        raise OracleError("input token domain")
    setup = _pipe_receive(reader, 0, (5, 0, 4, 64), raw_limit=backing_limit)
    replay = NativeMaskReplay([bytes(setup[i:i+16]) for i in (0, 16, 32)],
                              int.from_bytes(setup[48:], "little"), max_invocations=inv)
    del setup
    client = GeneratedNativeStream(reader, writer, meta, replay, mutation=mutation, raw_limit=backing_limit)
    comparator = StreamingTraceComparator(_StreamProvider(client, False), declaration=declaration,
        source_provenance=provenance, checked_ranges=guards, outputs=2, max_records=r, max_total_values=wc)
    half = meta["config"]["head_dim"]//2
    if half != 1 and meta["config"]["rope_theta"] != 1:
        raise OracleError("generated rotary recipe")
    rotary = {p: ([cos]*half, [sin]*half) for p, cos, sin in
              ((0, Q, 0), (1, 141637, 220586), (2, -109090, 238366))}
    result = run_decoder(meta["config"], weights, ids, 2, StreamingTape(_StreamProvider(client, True)),
                         rotary, checked_ranges=guards, trace_sink=comparator)
    if type(meta["expected_accepted"]) is not bool or result.accepted != meta["expected_accepted"]:
        raise OracleError("generated fixture acceptance outcome")
    components = client.final()
    comparator.finish(result, final_components=components)
    client.success()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "--audit-native-stream":
        try:
            _native_stream_main(int(sys.argv[2]), sys.argv[3])
        except BaseException:
            # Private operands/keys/tracebacks never reach stdout or stderr.
            raise SystemExit(1) from None
    elif sys.argv[1:] != ["--audit-native-capture"]:
        raise SystemExit("test-only anonymous-stdin native capture audit required")
    else:
        _native_capture_main()

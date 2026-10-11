"""Thread-safe, client-local ownership of *pending* completed prefills.

No executor, provider material, response session, or completed cache lives here.
An elected producer executes in its caller's worker and MUST call
``finish_producer`` after resetting its runtime, even after cancellation/failure.
Admission precedes runtime construction and snapshot allocation. Every acquired
branch must be released AFTER its private runtime is reset. ``close`` signals
cooperative cancellation and drains those scopes; it never zeroizes live work.

Publication copies caller-owned arrays. Caller still owns/erases originals.
Copies retain existing SnapshotStateBasis seals, not a new provenance scheme.
Private payload commitments detect trusted-process mutation, not attestation.
Only an entire pending producer prompt may match a request prefix. There is no
speculative split and no retained-result lookup, hence no cache-hit double gain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import secrets
import threading
from typing import Callable, Sequence

import numpy as np

from .transformer_client import LayerCache, RuntimeSnapshot, SnapshotStateBasis


class InflightCancelled(RuntimeError):
    """This request was cancelled or its client closed."""


class InflightProducerFailed(RuntimeError):
    """Producer failed; caller may retry with fresh provider material."""


def _compiled_metadata(compiled):
    """Revalidate canonical metadata only, never construct an online runtime."""
    from pllm.configuration import Pipeline
    from pllm.profiles import resolve_runtime_composition
    from .model_binding import CompiledRuntimeModel
    from .semantic_executor import SemanticDecoderRuntime

    if not isinstance(compiled, CompiledRuntimeModel):
        raise TypeError("in-flight metadata requires CompiledRuntimeModel")
    schedule = compiled._validated_schedule()
    composition = Pipeline.from_spec(json.loads(compiled._canonical_composition))
    options = resolve_runtime_composition(composition)
    graphs = compiled._plan.to_dict()
    if (options is None or options.causal_reduction != "prefix_f32"
            or options.verification_component is not None or options.verification_target_failure_bits
            or options.nonlinear_layout is not None
            or {"scheduler", "state", "nonlinear", "verification", "public_prefix"} & composition.components.keys()
            or schedule.get("mlp_tiling") is not None or schedule.get("retained_state") is not None
            or compiled._bundle.privacy.get("verification_target_failure_bits", 0)
            or SemanticDecoderRuntime._declared_windows(graphs)
            or SemanticDecoderRuntime._declared_hybrid_state(graphs)):
        raise ValueError("in-flight reuse requires untiled unverified native full-KV prefix_f32")
    if schedule["composition_digest"] != composition.digest():
        raise ValueError("in-flight composition differs from native schedule")
    # Constructing this native metadata contract validates exclusive full-KV,
    # original source/schedule/numeric bindings and supported linear composition.
    # It does not reserve material, evaluate tensors or allocate runtime caches.
    continuation = compiled._plan.continuation_schedule(composition)
    document = continuation.to_dict()
    if (document["mode"] != "full_kv_batched_suffix"
            or document["source_plan_digest"] != compiled._plan.digest
            or document["composition_digest"] != composition.digest()
            or document["source_schedule_digest"] != compiled.runtime_schedule_digest
            or document["token_bound"] != graphs["prefill"]["maximum_key_sequence"]):
        raise ValueError("in-flight native continuation binding mismatch")
    return schedule, composition, graphs


def _compiled_geometry(compiled, graphs):
    """Reject heterogeneous/windowed/hybrid/shared geometry, not guess a family."""
    capacity = graphs["decode"]["maximum_key_sequence"]
    if type(capacity) is not int or not 1 <= capacity <= 1_000_000:
        raise ValueError("invalid compiled decode KV capacity")
    owners = None
    widths = set()
    for phase, direction in (("prefill", "state_outputs"), ("decode", "state_inputs"),
                             ("decode", "state_outputs")):
        rows = graphs[phase][direction]
        current = set()
        for row in rows:
            shape = row.get("shape")
            layer, kind, maximum = row.get("layer"), row.get("kind"), row.get("maximum_sequence")
            if (type(layer) is not int or layer < 0 or kind not in {"key", "value"}
                    or type(shape) is not list or len(shape) != 4
                    or any(type(n) is not int or n < 1 for n in shape)
                    or shape[0] != 1 or type(maximum) is not int or shape[2] != maximum
                    or maximum < graphs[phase]["maximum_key_sequence"]
                    or phase == "decode" and maximum != capacity
                    or (layer, kind) in current):
                raise ValueError("in-flight requires exclusive homogeneous full-KV geometry")
            current.add((layer, kind))
            widths.add((shape[1], shape[3]))
        if not current or owners is not None and current != owners:
            raise ValueError("in-flight KV ownership differs across phases")
        owners = current
    layers = {layer for layer, _ in owners}
    if (layers != set(range(len(layers)))
            or owners != {(layer, kind) for layer in layers for kind in ("key", "value")}
            or len(widths) != 1 or graphs["prefill"]["state_inputs"]):
        raise ValueError("in-flight requires exclusive homogeneous full-KV geometry")
    heads, dim = widths.pop()
    cfg = compiled._bundle.cfg
    vocab = cfg["vocab_size"]  # Source/bundle validated by _validated_schedule.
    for name, expected in (("num_hidden_layers", len(layers)), ("num_key_value_heads", heads), ("head_dim", dim)):
        if name in cfg and (type(cfg[name]) is not int or cfg[name] != expected):
            raise ValueError("in-flight bundle differs from canonical KV geometry")
    return len(layers), heads, dim, vocab, capacity


@dataclass(frozen=True, slots=True)
class InflightBinding:
    source_digest: str
    composition_digest: str
    compiled_digest: str
    state_binding_digest: str
    native_prefill_digest: str

    def __post_init__(self) -> None:
        for value in (self.source_digest, self.composition_digest, self.compiled_digest,
                      self.state_binding_digest, self.native_prefill_digest):
            if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid in-flight source/runtime binding")

    @classmethod
    def from_compiled(cls, compiled) -> InflightBinding:
        """Derive source/state/original-prefill identity before online allocation."""
        from .model_binding import _state_binding_digest

        schedule, composition, graphs = _compiled_metadata(compiled)
        _compiled_geometry(compiled, graphs)
        native = hashlib.sha256(json.dumps({
            "source_plan": compiled._plan.digest, "graph": graphs["prefill"],
            "schedule": schedule["prefill"],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(compiled.bundle_fingerprint, composition.digest(), compiled.digest,
                   _state_binding_digest(compiled._plan, compiled._bundle, composition.digest()), native)

    @classmethod
    def from_runtime(cls, runtime, *, source_digest: str, composition_digest: str,
                     compiled_digest: str) -> InflightBinding:
        """Inspect an already admitted native runtime; no tensor allocation.

        Parent may derive equivalent metadata before constructing request runtime.
        Runtime itself is not retained by this binding.
        """
        if (runtime._causal_reduction != "prefix_f32" or runtime._mlp_tiling is not None
                or runtime.nonlinear_evaluator is not None or runtime._verification_bits
                or any(getattr(runtime, name, None) for name in
                       ("_capsule", "_tiered", "_retained", "_window_contracts", "_hybrid_contracts"))):
            raise ValueError("in-flight reuse requires untiled unverified native full-KV prefix_f32")
        if runtime._schedule["composition_digest"] != composition_digest:
            raise ValueError("in-flight composition differs from native runtime")
        native = hashlib.sha256(json.dumps({
            "source_plan": runtime._source_plan.digest, "graph": runtime._graphs["prefill"],
            "schedule": runtime._schedule["prefill"],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(source_digest, composition_digest, compiled_digest,
                   runtime._state_binding_digest, native)


@dataclass(frozen=True, slots=True)
class InflightResources:
    """State-payload bounds, NOT model/workspace/RSS admission.

    capacity_tokens must cover the compiled maximum allocation, not active rows.
    Double maximum KV covers restored/growing runtime allocation overlap.
    Producer also prices caller snapshot + owner publication copy simultaneously.
    Native continuation/workspace and host resource admission remain mandatory.
    """
    layers: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    capacity_tokens: int

    @classmethod
    def from_compiled(cls, compiled) -> InflightResources:
        """Read validated canonical state geometry, never current prompt length."""
        _, _, graphs = _compiled_metadata(compiled)
        return cls(*_compiled_geometry(compiled, graphs))

    def __post_init__(self) -> None:
        if any(type(n) is not int or not 1 <= n <= 1_000_000 for n in
               (self.layers, self.kv_heads, self.head_dim, self.vocab_size, self.capacity_tokens)):
            raise ValueError("invalid in-flight resource geometry")
        if self.runtime_overlap_bytes > (1 << 63) - 1:
            raise ValueError("in-flight resource geometry overflow")

    @property
    def kv_row_bytes(self) -> int:
        return self.layers * 2 * self.kv_heads * self.head_dim * 4

    @property
    def logits_bytes(self) -> int:
        return self.vocab_size * 4

    @property
    def runtime_overlap_bytes(self) -> int:
        return 2 * self.capacity_tokens * self.kv_row_bytes + self.logits_bytes

    def payload_bytes(self, prefix_tokens: int) -> int:
        if type(prefix_tokens) is not int or not 1 <= prefix_tokens <= self.capacity_tokens:
            raise ValueError("prefix exceeds compiled runtime capacity")
        return prefix_tokens * self.kv_row_bytes + self.logits_bytes

    def producer_bytes(self, prefix_tokens: int) -> int:
        return 2 * self.payload_bytes(prefix_tokens) + self.runtime_overlap_bytes

    def branch_bytes(self, prefix_tokens: int) -> int:
        return self.payload_bytes(prefix_tokens) + self.runtime_overlap_bytes


@dataclass(eq=False, slots=True)
class _Node:
    key: bytes
    binding: InflightBinding
    resources: InflightResources
    tokens: np.ndarray
    reservation: int
    cancel: threading.Event = field(default_factory=threading.Event)
    requests: set = field(default_factory=set)
    active: bool = True
    status: str = "pending"
    result: tuple[RuntimeSnapshot, np.ndarray] | None = None
    commitment: bytes | None = None
    owned_arrays: tuple[np.ndarray, ...] = ()


def _arrays(snapshot: RuntimeSnapshot, logits: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(value for row in snapshot.caches for value in (row.key, row.value)) + (logits,)


def _erase_arrays(arrays: tuple[np.ndarray, ...]) -> None:
    # Retain actual owned roots: mutable snapshot descriptors may be replaced.
    for value in arrays:
        value.setflags(write=True)
        value.fill(0)


def _erase(snapshot: RuntimeSnapshot, logits: np.ndarray) -> None:
    _erase_arrays(_arrays(snapshot, logits))


class InflightLease:
    """Independent branch KV/logits; release only after branch runtime reset."""

    def __init__(self, request: InflightRequest, snapshot: RuntimeSnapshot, logits: np.ndarray):
        self._request = request
        self.snapshot, self.logits = snapshot, logits
        self._owned_arrays = _arrays(snapshot, logits)

    @property
    def prefix_tokens(self) -> int:
        return self._request.prefix_tokens

    def close(self) -> None:
        self._request.release()

    def __enter__(self) -> InflightLease:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class InflightRequest:
    """One admitted request scope. Methods are safe across worker threads.

    Producer cancellation event is shared only for stopping producer work. This
    request's cancellation is independent. A producer request may be cancelled
    while its worker still publishes for other waiters.
    """

    def __init__(self, owner, node, producer: bool, reservation: int):
        self._owner, self._node = owner, node
        self.is_producer = producer
        self.prefix_tokens = int(node.tokens.size)
        self._reservation = reservation
        self._released = self._cancelled = False
        self._lease = None

    @property
    def producer_cancel(self) -> threading.Event:
        return self._node.cancel

    def publish(self, snapshot: RuntimeSnapshot, logits: np.ndarray, *,
                validate_snapshot: Callable[[RuntimeSnapshot], None]) -> None:
        """Copy caller result; validator must be native runtime._validate_full_snapshot.

        Caller must not mutate input during this call and must erase input after.
        Failure broadcasts generic error without persisting exception/token data.
        """
        self._owner._publish(self, snapshot, logits, validate_snapshot)

    def fail(self) -> None:
        """Broadcast failure without retaining provider exception or private data."""
        self._owner._fail(self)

    def finish_producer(self) -> None:
        """Mandatory finally hook AFTER producer runtime/original copies are erased."""
        self._owner._finish(self)

    def wait(self, *, timeout: float | None = None) -> InflightLease:
        """Wait for private branch; timeout cancels only this request."""
        return self._owner._wait(self, timeout)

    def cancel(self) -> None:
        """Wake this waiter. Existing branch stays owned until explicit release."""
        self._owner._cancel(self)

    def release(self) -> None:
        """Idempotent; erase owned branch, release reservation, possibly retire node."""
        self._owner._release(self)


class InflightPrefixOwner:
    """One client capability; bounded pending namespace, waiters and payload RAM.

    ``lookup_or_register`` returns None on completed-cache bypass or cap rejection.
    Identity/type errors and native admission failures raise before allocation.
    Pending requests count through branch release; nodes count through safe
    producer completion AND final branch release, including retiring producers.
    """

    def __init__(self, client: object, *, max_requests: int = 8,
                 max_bytes: int = 64 << 20, max_nodes: int | None = None):
        nodes = max_requests if max_nodes is None else max_nodes
        if (type(max_requests) is not int or not 1 <= max_requests <= 64
                or type(nodes) is not int or not 1 <= nodes <= 64
                or type(max_bytes) is not int or not 1 <= max_bytes <= 256 << 20):
            raise ValueError("invalid in-flight owner bounds")
        self._client = client
        self.max_requests, self.max_nodes, self.max_bytes = max_requests, nodes, max_bytes
        self._secret = bytearray(secrets.token_bytes(32))
        self._condition = threading.Condition(threading.RLock())
        self._pending: dict[bytes, _Node] = {}
        self._nodes: set[_Node] = set()
        self._closed = False
        self._stats = dict(reserved_bytes=0, peak_reserved_bytes=0, requests=0,
                           producer_registrations=0, publications=0, joins=0, failures=0,
                           cancellations=0, cancelled_producers=0, retired=0,
                           copy_bytes=0, bound_rejections=0, cache_bypasses=0)

    def counters(self) -> dict[str, int]:
        """Public counts only: never keys, tokens, digests, logits or exceptions."""
        with self._condition:
            return dict(self._stats, nodes=len(self._nodes), pending_nodes=len(self._pending),
                        active_producers=sum(n.active for n in self._nodes))

    def _key(self, binding: InflightBinding, tokens: np.ndarray) -> bytes:
        digest = hmac.new(self._secret, b"pllm.client.inflight-prefix.v1\0", "sha256")
        digest.update(repr(binding).encode())
        digest.update(tokens.size.to_bytes(4, "big"))
        digest.update(tokens.astype("<u4").tobytes())
        return digest.digest()

    def lookup_or_register(self, client: object, binding: InflightBinding,
                           token_ids: Sequence[int], resources: InflightResources, *,
                           admit: Callable[[int, int], None],
                           completed_cache_hit: bool = False) -> InflightRequest | None:
        """Join longest complete pending prefill, otherwise elect full-prompt producer.

        ``admit(prefix_tokens, suffix_tokens)`` MUST perform existing native
        continuation + client resource/workspace admission before any runtime,
        material reservation or snapshot allocation. Called under owner lock;
        metadata-only, nonblocking, non-reentrant. Zero suffix needs no split.
        If caller already chose completed cache (even proper-prefix hit), bypass.
        """
        if client is not self._client or type(binding) is not InflightBinding or type(resources) is not InflightResources:
            raise ValueError("in-flight trusted client/binding mismatch")
        if type(completed_cache_hit) is not bool or not callable(admit):
            raise ValueError("invalid in-flight admission")
        # Check length before allocating; reject coercions/floats/bools/overflow.
        if not 1 <= len(token_ids) <= min(4096, resources.capacity_tokens):
            raise ValueError("invalid bounded in-flight tokens")
        if any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
               or not 0 <= int(t) < min(resources.vocab_size, 1 << 32) for t in token_ids):
            raise ValueError("invalid in-flight token IDs")
        values = np.array(token_ids, dtype="<i8", copy=True)
        if values.ndim != 1:
            raise ValueError("invalid in-flight token shape")
        try:
            with self._condition:
                if self._closed:
                    raise InflightCancelled("in-flight client closed")
                if completed_cache_hit:
                    self._stats["cache_bypasses"] += 1
                    return None
                # O(max_nodes * tokens), no unbounded prefix-key namespace.
                candidates = [n for n in self._pending.values() if n.binding == binding
                              and n.resources == resources and n.tokens.size <= values.size
                              and np.array_equal(n.tokens, values[:n.tokens.size])]
                node = max(candidates, key=lambda n: n.tokens.size, default=None)
                prefix = int(node.tokens.size) if node else int(values.size)
                branch = resources.branch_bytes(prefix)
                producer = resources.producer_bytes(prefix) if node is None else 0
                if (self._stats["requests"] >= self.max_requests
                        or node is None and len(self._nodes) >= self.max_nodes
                        or self._stats["reserved_bytes"] + branch + producer > self.max_bytes):
                    self._stats["bound_rejections"] += 1
                    return None
                admit(prefix, int(values.size) - prefix)
                if node is None:
                    key = self._key(binding, values)
                    node = _Node(key, binding, resources, values.copy(), producer)
                    self._nodes.add(node)
                    self._pending[key] = node
                    self._stats["producer_registrations"] += 1
                else:
                    self._stats["joins"] += 1
                request = InflightRequest(self, node, bool(producer), branch)
                node.requests.add(request)
                self._stats["requests"] += 1
                self._stats["reserved_bytes"] += producer + branch
                self._stats["peak_reserved_bytes"] = max(self._stats["peak_reserved_bytes"], self._stats["reserved_bytes"])
                return request
        finally:
            values.fill(0)

    def _producer(self, request) -> _Node:
        if request._owner is not self or not request.is_producer or request._node not in self._nodes:
            raise RuntimeError("invalid or retired in-flight producer")
        return request._node

    def _validate(self, node, snapshot, logits, validator) -> None:
        r, n, b = node.resources, int(node.tokens.size), node.binding
        if type(snapshot) is not RuntimeSnapshot or type(snapshot.state_basis) is not SnapshotStateBasis:
            raise ValueError("in-flight snapshot lacks native execution basis")
        basis = snapshot.state_basis
        execution = hashlib.sha256(b"pllm.decoder.execution-basis.v1\0" + b.native_prefill_digest.encode()
                                   + np.arange(n, dtype="<i8").tobytes() + node.tokens.tobytes()).hexdigest()
        if (not basis.completed_prefill() or basis.verification_failure_bits != 0
                or basis.source_binding_digest != b.state_binding_digest
                or basis.native_phase_digest != b.native_prefill_digest
                or basis.execution_digest != execution or basis.prefill_extent != n
                or basis.position != n or snapshot.position != n or snapshot.shared_kv
                or snapshot.semantic_binding_digest != b.state_binding_digest
                or len(snapshot.caches) != r.layers):
            raise ValueError("in-flight snapshot provenance mismatch")
        arrays = []
        for row in snapshot.caches:
            if type(row) is not LayerCache or row.length != n:
                raise ValueError("in-flight full-KV owner mismatch")
            for value in (row.key, row.value):
                if (type(value) is not np.ndarray or value.dtype != np.float32
                        or value.shape != (n, r.kv_heads, r.head_dim)
                        or not value.flags.owndata or not value.flags.c_contiguous
                        or not np.all(np.isfinite(value))):
                    raise ValueError("in-flight full-KV shape/numeric/ownership mismatch")
                arrays.append(value)
        if (type(logits) is not np.ndarray or logits.dtype != np.float32
                or logits.shape != (r.vocab_size,) or not logits.flags.owndata
                or not logits.flags.c_contiguous or not np.all(np.isfinite(logits))):
            raise ValueError("in-flight logits shape/numeric/ownership mismatch")
        arrays.append(logits)
        if len({id(a) for a in arrays}) != len(arrays):
            raise ValueError("in-flight arrays alias")
        if sum(a.nbytes for a in arrays) != r.payload_bytes(n):
            raise ValueError("in-flight payload exceeds admitted geometry")
        validator(snapshot)

    def _commit(self, snapshot, logits) -> bytes:
        digest = hmac.new(self._secret, digestmod="sha256")
        digest.update(repr((snapshot.position, snapshot.semantic_binding_digest, snapshot.state_basis)).encode())
        for row in snapshot.caches:
            digest.update(str(row.length).encode())
            for value in (row.key, row.value):
                digest.update(repr((value.shape, value.dtype.str)).encode())
                digest.update(memoryview(value).cast("B"))
        digest.update(repr((logits.shape, logits.dtype.str, snapshot.shared_kv)).encode())
        digest.update(memoryview(logits).cast("B"))
        return digest.digest()

    @staticmethod
    def _copy(snapshot, logits):
        caches = []
        scores = None
        try:
            for row in snapshot.caches:
                key = row.key.copy()
                try:
                    value = row.value.copy()
                except BaseException:
                    key.fill(0)
                    raise
                caches.append(LayerCache(key, value, row.length))
            scores = logits.copy()
            return RuntimeSnapshot(snapshot.position, caches, {}, snapshot.semantic_binding_digest,
                                   snapshot.state_basis), scores
        except BaseException:
            for row in caches:
                row.key.fill(0)
                row.value.fill(0)
            if scores is not None:
                scores.fill(0)
            raise

    def _unindex(self, node) -> None:
        # A newer same-key producer can exist while this node drains.
        if self._pending.get(node.key) is node:
            del self._pending[node.key]

    def _mark_failed(self, node) -> None:
        if node.status != "failed":
            node.status = "failed"
            self._stats["failures"] += 1
        self._unindex(node)
        self._condition.notify_all()

    def _publish(self, request, snapshot, logits, validator) -> None:
        with self._condition:
            node = self._producer(request)
            if not node.active or node.status != "pending" or node.cancel.is_set():
                raise InflightCancelled("in-flight producer no longer publishable")
            saved = None
            try:
                if not callable(validator):
                    raise ValueError("native snapshot validator required")
                self._validate(node, snapshot, logits, validator)
                before = self._commit(snapshot, logits)
                saved = self._copy(snapshot, logits)
                self._validate(node, *saved, validator)
                if not hmac.compare_digest(before, self._commit(*saved)) or not hmac.compare_digest(before, self._commit(snapshot, logits)):
                    raise ValueError("in-flight snapshot mutated during publication")
                for row in saved[0].caches:
                    row.key.setflags(write=False)
                    row.value.setflags(write=False)
                saved[1].setflags(write=False)
                node.result, node.commitment = saved, before
                node.owned_arrays = _arrays(*saved)
                node.status = "published"
                self._unindex(node)
                self._stats["publications"] += 1
                self._stats["copy_bytes"] += node.resources.payload_bytes(int(node.tokens.size))
                self._condition.notify_all()
            except BaseException:
                if saved is not None:
                    _erase(*saved)
                self._mark_failed(node)
                raise

    def _fail(self, request) -> None:
        with self._condition:
            node = self._producer(request)
            if not node.active:
                raise RuntimeError("in-flight producer already finished")
            self._mark_failed(node)

    def _finish(self, request) -> None:
        with self._condition:
            # Idempotent even after node retirement.
            if request._owner is not self or not request.is_producer:
                raise RuntimeError("not an in-flight producer")
            node = request._node
            if not node.active:
                return
            if node.status == "pending":
                self._mark_failed(node)
            node.active = False
            # Keep publication reservation until retirement: it owns saved arrays.
            self._retire(node)
            self._condition.notify_all()

    def _wait(self, request, timeout) -> InflightLease:
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("invalid in-flight wait timeout")
        with self._condition:
            node = request._node
            if request._released or request._cancelled or self._closed:
                raise InflightCancelled("in-flight request cancelled")
            if request._lease is not None:
                return request._lease
            ready = self._condition.wait_for(lambda: node.status != "pending" or request._cancelled
                                             or request._released or self._closed, timeout)
            if not ready:
                self._cancel(request)
                raise TimeoutError("in-flight prefix wait timed out")
            if request._released or request._cancelled or self._closed:
                raise InflightCancelled("in-flight request cancelled")
            if request._lease is not None:
                return request._lease
            if node.status == "failed":
                self._release(request)
                raise InflightProducerFailed("in-flight prefix producer failed")
            try:
                if not hmac.compare_digest(node.commitment, self._commit(*node.result)):
                    raise ValueError("in-flight published result mutated")
            except BaseException:
                self._mark_failed(node)
                self._release(request)
                raise
            saved = None
            try:
                saved = self._copy(*node.result)
                if not hmac.compare_digest(node.commitment, self._commit(*saved)):
                    raise ValueError("in-flight branch copy mutated")
                request._lease = InflightLease(request, *saved)
                self._stats["copy_bytes"] += node.resources.payload_bytes(request.prefix_tokens)
                return request._lease
            except BaseException:
                if saved is not None:
                    _erase(*saved)
                # A private branch allocation/copy failure does not poison other
                # requests or erase their valid published producer state.
                self._release(request)
                raise

    def _cancel(self, request) -> None:
        with self._condition:
            if request._released or request._cancelled:
                return
            request._cancelled = True
            self._stats["cancellations"] += 1
            if request._lease is None:
                self._release(request)
            self._condition.notify_all()

    def _release(self, request) -> None:
        with self._condition:
            if request._released:
                return
            request._released = True
            node = request._node
            if request._lease is not None:
                _erase_arrays(request._lease._owned_arrays)
            node.requests.remove(request)
            self._stats["requests"] -= 1
            self._stats["reserved_bytes"] -= request._reservation
            if not node.requests:
                self._unindex(node)
                if node.active and not node.cancel.is_set():
                    node.cancel.set()
                    self._stats["cancelled_producers"] += 1
            self._retire(node)
            self._condition.notify_all()

    def _retire(self, node) -> None:
        if node.requests or node.active or node not in self._nodes:
            return
        self._unindex(node)
        if node.result is not None:
            _erase_arrays(node.owned_arrays)
            node.result = None
            node.owned_arrays = ()
        node.tokens.fill(0)
        node.commitment = None
        node.key = b""
        self._stats["reserved_bytes"] -= node.reservation
        self._nodes.remove(node)
        self._stats["retired"] += 1
        if self._closed and not self._nodes:
            self._secret[:] = bytes(len(self._secret))

    def close(self, *, wait: bool = True, timeout: float | None = None) -> bool:
        """Stop admission, wake waiters, signal producers, optionally drain.

        True means all copies erased and all producer/branch scopes released.
        False means active scopes still own counted memory; caller must reset and
        release them. Never call blocking close from its own active scope.
        Timed/nonblocking close can be followed by another close to finish drain.
        """
        if type(wait) is not bool or (timeout is not None and (isinstance(timeout, bool)
                or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0)):
            raise ValueError("invalid in-flight close options")
        with self._condition:
            self._closed = True
            for node in list(self._nodes):
                if node.active and not node.cancel.is_set():
                    node.cancel.set()
                    self._stats["cancelled_producers"] += 1
                self._unindex(node)
                for request in list(node.requests):
                    self._cancel(request)
            self._condition.notify_all()
            drained = not self._nodes
            if wait and not drained:
                drained = self._condition.wait_for(lambda: not self._nodes, timeout)
            if drained:
                self._secret[:] = bytes(len(self._secret))
            return drained


__all__ = ["InflightBinding", "InflightResources", "InflightPrefixOwner", "InflightRequest",
           "InflightLease", "InflightCancelled", "InflightProducerFailed"]

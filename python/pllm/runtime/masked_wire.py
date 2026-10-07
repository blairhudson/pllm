"""Compiler-bound masked-wire regions with client conversion and opaque relay.

The trusted client issues fresh material; each authenticated worker imports only
its own party. This protocol does not move float conversion or attention out of
the client. Application-body counters include material and both relay legs.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import struct
import threading
import time
from typing import Any, Callable

import numpy as np

from pllm import _native

EXECUTOR = "remote_gated_block"
MAX_LANES = 256
MAX_FRAME = 1 << 20
_START = struct.Struct("!QHI")
_ARM_SCHEMA = "pllm.masked_wire_session.v1"


def operation_digest(phase: str, step: dict[str, Any]) -> str:
    return hashlib.sha256(b"pllm.masked_wire_region.v1\0" + json.dumps(
        {"phase": phase, "step": step}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def validate_region(step: dict[str, Any], operations: dict[str, Any]) -> None:
    """Independently check the native region against the source graph."""
    if (step.get("executor") != EXECUTOR or step.get("operators") != ["silu", "multiply"]
        or len(step["operation_ids"]) != 2 or step.get("weight_ids") != []):
        raise ValueError("masked-wire region operators differ")
    silu, product = (operations[key] for key in step["operation_ids"])
    inputs = product["inputs"]
    if (len(inputs) != 2 or inputs.count(silu["id"]) != 1 or len(silu["inputs"]) != 1
        or silu["attributes"] != {} or product["attributes"] != {}
        or silu["output_shape"] != product["output_shape"]
        or step["input_ids"] != [silu["inputs"][0], next(x for x in inputs if x != silu["id"]) ]
        or step["outputs"] != [{"operation_id": product["id"],
            "output_shape": product["output_shape"], "stage_offset": 0,
            "stage_width": product["output_shape"][-1]}]
        or sum(silu["id"] in op["inputs"] for op in operations.values()) != 1):
        raise ValueError("masked-wire region dependencies or representation differ")


def _regions(schedule: dict[str, Any]) -> dict[str, tuple[int, int]]:
    result = {}
    for phase in ("prefill", "decode"):
        for step in schedule[phase]["steps"]:
            if step["executor"] == EXECUTOR:
                shape = step["outputs"][0]["output_shape"]
                result[operation_digest(phase, step)] = (shape[-1], int(np.prod(shape)))
    if not result or not schedule.get("nonlinear"):
        raise ValueError("masked-wire requires a compiled nonlinear schedule")
    return result


def _session_digest(schedule: str, ids: list[str] | tuple[str, str]) -> bytes:
    if (len(ids) != 2 or ids[0] == ids[1]
        or any(type(x) is not str or len(x) != 32
               or any(c not in "0123456789abcdef" for c in x) for x in ids)):
        raise ValueError("masked-wire requires two distinct worker-minted sessions")
    return hashlib.sha256(b"pllm.masked_wire_workers.v1\0" + bytes.fromhex(schedule)
        + bytes.fromhex(ids[0]) + bytes.fromhex(ids[1])).digest()


def encode_input(value: np.ndarray, *, gate: bool) -> np.ndarray:
    """Float32 -> signed Q24, explicit floor, public bounds, no saturation."""
    value = np.asarray(value)
    lower, upper = (-48, 48) if gate else (-128, 128)
    if (value.dtype != np.float32 or value.size == 0 or value.size > MAX_LANES
        or not np.all(np.isfinite(value)) or np.any(value < lower) or np.any(value > upper)):
        raise ValueError("masked-wire float32 input exceeds its public range")
    return np.floor(value.astype(np.float64) * (1 << 24)).astype("<i8")


def decode_output(value: bytes, size: int, shape: tuple[int, ...]) -> np.ndarray:
    if len(value) != size * 8:
        raise ValueError("masked-wire result size differs")
    return (np.frombuffer(value, dtype="<i8").astype(np.float32)
            / np.float32(1 << 12)).reshape(shape)


class WorkerRegions:
    """One pending party-local chunk; monotonic indices burn imported material."""

    def __init__(self, schedule: dict[str, Any], digest: str, party: int, max_new: int) -> None:
        self.digest = digest
        self.plan = bytes.fromhex(digest)
        self.layout = schedule["nonlinear"]["layout"]
        self.regions = _regions(schedule)
        self.limits = {key: size * (max_new - 1 if key in {
            operation_digest("decode", s) for s in schedule["decode"]["steps"]
            if s["executor"] == EXECUTOR
        } else 1) for key, (_, size) in self.regions.items()}
        self.party_id = party
        self.session: bytes | None = None
        self.next_indices: dict[str, int] = {}
        self.pending = None
        self.pending_region: str | None = None
        self.closed = False

    def cancel(self) -> None:
        self.closed = True
        if self.pending is not None:
            self.pending.cancel()
        self.pending = None
        self.pending_region = None

    def arm(self, own_id: str, document: dict[str, Any]) -> dict[str, str]:
        if (self.closed or self.session is not None or type(document) is not dict
            or set(document) != {"schema", "schedule_digest", "sessions"}
            or document["schema"] != _ARM_SCHEMA or document["schedule_digest"] != self.digest
            or type(document["sessions"]) is not list):
            raise ValueError("masked-wire session admission differs")
        session = _session_digest(self.digest, document["sessions"])
        if document["sessions"][self.party_id] != own_id:
            raise ValueError("masked-wire worker session is not its own")
        self.session = session
        return {"schema": _ARM_SCHEMA, "schedule_digest": self.digest, "session": session.hex()}

    def start(self, region: str, payload: bytes) -> bytes:
        if (self.closed or self.session is None or self.pending is not None
            or region not in self.regions or len(payload) < _START.size or len(payload) > MAX_FRAME):
            raise ValueError("masked-wire chunk admission differs")
        first, lanes, length = _START.unpack_from(payload)
        if (not 1 <= lanes <= MAX_LANES or first != self.next_indices.get(region, 0)
            or first + lanes > self.limits[region] or not 0 < length < len(payload) - _START.size):
            raise ValueError("masked-wire chunk is replayed or exceeds its tensor budget")
        self.next_indices[region] = first + lanes
        material = payload[_START.size:_START.size + length]
        self.pending = _native._MaskedWireParty(material, self.plan, self.session,
            bytes.fromhex(region), first, lanes, self.layout, self.party_id)
        self.pending_region = region
        return self.pending.start(payload[_START.size + length:])

    def advance(self, region: str, payload: bytes) -> bytes:
        if (self.closed or self.pending is None or self.pending_region != region
            or not 0 < len(payload) <= MAX_FRAME):
            raise ValueError("masked-wire peer frame has no pending region")
        done, frame = self.pending.advance(payload)
        if done:
            self.pending = None
            self.pending_region = None
        return bytes([done]) + frame


class ClientRegions:
    """Bounded trusted issuance; private values stay within client ownership."""

    def __init__(self, schedule: dict[str, Any], digest: str,
                 sessions: tuple[str, str], post: Callable[..., Any]) -> None:
        self.regions = _regions(schedule)
        self.plan = bytes.fromhex(digest)
        self.session = _session_digest(digest, sessions)
        self.layout = schedule["nonlinear"]["layout"]
        self.sessions = sessions
        self.post = post
        self.dispatch = schedule["nonlinear"].get("dispatch", "serial")
        if self.dispatch not in {"serial", "overlap"}:
            raise ValueError("masked-wire dispatch is unsupported")
        self._state_lock = threading.Lock()
        self._running = False
        self.next_indices: dict[str, int] = {}
        self.stats = {"elements": 0, "chunks": 0, "party_key_bytes": 0, "peer_frame_bytes": 0,
                      "client_issuance_ns": 0, "client_issuance_cpu_ns": 0}
        self.closed = False
        self.current = None
        for index, sid in enumerate(sessions):
            status, response = post(index, f"/v1/offset-reference/sessions/{sid}/nonlinear/arm",
                phase="setup", json={"schema": _ARM_SCHEMA, "schedule_digest": digest,
                                     "sessions": list(sessions)})
            if status != 200 or json.loads(response) != {
                "schema": _ARM_SCHEMA, "schedule_digest": digest, "session": self.session.hex(),
            }:
                raise ValueError("masked-wire worker did not acknowledge its compiled region")

    def cancel(self) -> None:
        # The running call owns native handles while their GIL-released methods
        # execute. Mark terminal now; that call drains/burns both chunks before
        # it returns. Cancellation must not mutably borrow an active PyO3 handle.
        with self._state_lock:
            self.closed = True
            if not self._running and self.current is not None:
                self.current.cancel()
                self.current = None

    def _issue(self, region: str, first: int, lanes: int):
        if self.closed:
            raise ValueError("masked-wire issuance cancelled")
        started, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            return _native._masked_wire_issue(self.plan, self.session,
                bytes.fromhex(region), first, lanes, self.layout)
        finally:
            # Process CPU would also count concurrent HTTP work. Issuance runs
            # on this thread, including its GIL-released native call.
            with self._state_lock:
                self.stats["client_issuance_ns"] += time.perf_counter_ns() - started
                self.stats["client_issuance_cpu_ns"] += time.thread_time_ns() - cpu

    def _pair(self, pool, region: str, action: str, payloads: tuple[bytes, bytes]) -> list[bytes]:
        if pool is None:
            return [self._send(index, region, action, body) for index, body in enumerate(payloads)]
        # One thread issues the next chunk, one sends to worker A, and the
        # calling thread sends to B. Never have two requests to the same worker.
        future = pool.submit(self._send, 0, region, action, payloads[0])
        try:
            right = self._send(1, region, action, payloads[1])
            return [future.result(), right]
        except BaseException:
            self.cancel()
            raise
        finally:
            # A failed peer cannot leave an unjoined request issuing more work.
            future.cancel()
            try:
                future.result()
            except BaseException:
                pass

    def _send(self, index: int, region: str, action: str, payload: bytes) -> bytes:
        if self.closed:
            raise ValueError("masked-wire session is terminal")
        sid = self.sessions[index]
        status, response = self.post(index,
            f"/v1/offset-reference/sessions/{sid}/nonlinear/{region}/{action}",
            phase="online", content=payload, limit=MAX_FRAME)
        if status != 200:
            raise ValueError("masked-wire worker declined its committed frame")
        return response

    def run(self, phase: str, step: dict[str, Any], gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        region = operation_digest(phase, step)
        gate, up = np.asarray(gate), np.asarray(up)
        if (self.closed or region not in self.regions or gate.shape != up.shape or gate.ndim not in (2, 3)
            or gate.dtype != np.float32 or up.dtype != np.float32 or gate.size == 0
            or gate.shape[-1] != self.regions[region][0] or gate.size > self.regions[region][1]):
            raise ValueError("masked-wire tensor differs from the compiled region")
        with self._state_lock:
            if self.closed or self._running:
                raise ValueError("masked-wire session is terminal or already executing")
            self._running = True
        pending: Future | None = None
        pool = None
        try:
            result = np.empty(gate.size, dtype=np.float32)
            gates, ups = gate.reshape(-1), up.reshape(-1)
            if self.dispatch == "overlap":
                pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pllm-nonlinear")
            for offset in range(0, gate.size, MAX_LANES):
                lanes = min(MAX_LANES, gate.size - offset)
                g = encode_input(gates[offset:offset + lanes], gate=True)
                u = encode_input(ups[offset:offset + lanes], gate=False)
                if pending is None:
                    first = self.next_indices.get(region, 0)
                    self.next_indices[region] = first + lanes
                    self.current, left, right = self._issue(region, first, lanes)
                else:
                    first = self.next_indices[region] - lanes
                    self.current, left, right = pending.result()
                    pending = None
                if self.closed:
                    raise ValueError("masked-wire session cancelled during issuance")
                next_lanes = min(MAX_LANES, gate.size - offset - lanes)
                if pool is not None and next_lanes:
                    # Reserve once, before issuance. A failure burns this range;
                    # it is never resubmitted or exported to another session.
                    next_first = self.next_indices[region]
                    self.next_indices[region] = next_first + next_lanes
                    pending = pool.submit(self._issue, region, next_first, next_lanes)
                masked = self.current.encode(g.tobytes(), u.tobytes())
                frames = self._pair(pool, region, "start", tuple(
                    _START.pack(first, lanes, len(key)) + key + masked for key in (left, right)))
                self.stats["party_key_bytes"] += len(left) + len(right)
                for _ in range(8 if self.layout == "fused" else 11):
                    self.stats["peer_frame_bytes"] += sum(map(len, frames))
                    returned = self._pair(pool, region, "advance", (frames[1], frames[0]))
                    if (any(not body or body[0] not in (0, 1) for body in returned)
                        or returned[0][0] != returned[1][0]):
                        raise ValueError("masked-wire workers disagree on their phase")
                    frames = [body[1:] for body in returned]
                    if returned[0][0]:
                        decoded = self.current.decode(*frames)
                        self.current = None
                        result[offset:offset + lanes] = decode_output(decoded, lanes, (lanes,))
                        break
                else:
                    raise ValueError("masked-wire workers exceeded their phase bound")
                self.stats["elements"] += lanes
                self.stats["chunks"] += 1
            if self.closed:
                raise ValueError("masked-wire session cancelled during evaluation")
            return result.reshape(gate.shape)
        except BaseException:
            self.cancel()
            raise
        finally:
            if pending is not None:
                pending.cancel()
                try:
                    unused, _, _ = pending.result()
                except BaseException:
                    pass
                else:
                    unused.cancel()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            with self._state_lock:
                if self.current is not None:
                    self.current.cancel()
                    self.current = None
                self._running = False

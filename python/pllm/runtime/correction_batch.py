"""Bounded local Preparation batching with independent one-use inventory cursors."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import hmac
import json
import secrets
import tempfile
from threading import RLock
import time

import numpy as np

from .native import CompiledMatrix
from .preparation_protocol import PreparationRequest, prepared_mask_rows
from .stage_protocol import pack_residues, unpack_residues


def _header(binding, offset, rows):
    return json.dumps([binding, offset, rows], separators=(",", ":")).encode()


class CorrectionTicket:
    """One ordered, consume-once correction stream owned by its batch."""

    def __init__(self, batch, request, cursor, sink, binding):
        self._batch, self._request = batch, replace(request, seed=b"")
        self._cursor, self._file, self._binding = cursor, sink, binding
        self._state = "pending"
        self._issued = self._received = self._taken = 0

    @property
    def state(self) -> str:
        return self._state

    def take_rows(self, rows: int) -> bytes:
        """Return exact ring-packed rows, once, only after complete publication."""
        with self._batch._lock:
            try:
                if (self._state != "sealed" or type(rows) is not int or not 1 <= rows <= 128
                        or self._taken + rows > self._request.rows):
                    raise ValueError("correction stream is unavailable or row extent differs")
                count = rows * self._request.out_features * (self._request.wire_bits // 8)
                payload = self._file.read(count)
                if len(payload) != count:
                    raise ValueError("correction stream is incomplete")
                self._taken += rows
                if self._taken == self._request.rows:
                    self._retire("consumed")
                return payload
            except BaseException:
                self._batch.close()
                raise

    def _retire(self, state="burned"):
        self._cursor.cancel()
        self._file.close()
        if self._state != "consumed":
            self._state = state

    def close(self):
        with self._batch._lock:
            self._retire()

    def __copy__(self):
        raise TypeError("one-use correction tickets cannot be copied")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce_ex__(self, protocol):
        return self.__copy__()


class CorrectionBatch:
    """Explicit local batching; no provider queue, waiting policy or network lease.

    All requests must bind the same stage and numeric contract. The caller keeps
    its source requests; this owner retains native cursors without root seeds.
    """

    def __init__(self, matrix: CompiledMatrix, *, kernel="native", window_rows=128,
                 min_gpu_rows=64, max_tickets=8, workspace_bytes=64 << 20):
        if type(matrix) is not CompiledMatrix or not matrix.owner.native:
            raise ValueError("correction batches require an immutable native matrix")
        if kernel not in {"native", "digit8", "metal", "adaptive"}:
            raise ValueError("unknown correction batch kernel")
        for value, maximum in ((window_rows, 128), (min_gpu_rows, 128), (max_tickets, 8),
                               (workspace_bytes, 64 << 20)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("correction batch resource bound exceeded")
        out, width = matrix.shape
        if matrix.weight_bytes > 128 << 20 or 32 * window_rows * (out + width) > workspace_bytes:
            raise ValueError("correction batch weight or row workspace bound exceeded")
        if kernel == "digit8":
            from ._native_support import capabilities
            if not capabilities().get("digit8_cpu"):
                raise ValueError("native digit8 kernel is unavailable")
        self._lock = RLock()
        self._matrix, self._kernel, self._window = matrix, kernel, window_rows
        self._minimum = min_gpu_rows
        self._maximum = max_tickets
        weights = matrix.weight_view()
        self._weight_digest = hashlib.sha256(memoryview(weights)).hexdigest()
        bound_rows = max(1, min(64, workspace_bytes // (8 * width)))
        self._weight_bound = max(int(np.abs(weights[start:start + bound_rows].astype(np.int32)).sum(axis=1).max())
                                 for start in range(0, out, bound_rows))
        self._gpu = None
        self._tickets = []
        self._key = secrets.token_bytes(32)
        self._closed = False
        self._disk_bound = 0
        self._products = 0
        self.allocation_bytes_estimate = (matrix.weight_bytes + workspace_bytes
            + ((224 << 20) if kernel == "digit8" else 0)
            + (matrix.weight_bytes + 8 * window_rows * (width + out) if kernel in {"metal", "adaptive"} else 0))
        if kernel in {"metal", "adaptive"}:
            from .metal import MetalGEMM
            self._gpu = MetalGEMM().compile(weights)

    def add(self, request: PreparationRequest) -> CorrectionTicket:
        with self._lock:
            cursor = sink = None
            try:
                if self._closed or len(self._tickets) >= self._maximum:
                    raise ValueError("correction batch is closed or full")
                if (type(request) is not PreparationRequest or type(request.seed) is not bytes
                        or any(type(getattr(request, name)) is not int for name in (
                            "rows", "in_features", "out_features", "weight_bits", "activation_bits",
                            "signed_output_bound", "modulus", "wire_bits"))):
                    raise ValueError("invalid preparation request types")
                request._validate(max_rows=4096, max_tensor_elements=4_000_000)
                if ((request.out_features, request.in_features) != self._matrix.shape
                        or request.weight_digest != self._weight_digest
                        or request.signed_output_bound < self._weight_bound * ((1 << (request.activation_bits - 1)) - 1)):
                    raise ValueError("correction request weight or numeric binding differs")
                weights = self._matrix.weight_view()
                high = (1 << (request.weight_bits - 1)) - 1
                if int(weights.min()) < -high - 1 or int(weights.max()) > high:
                    raise ValueError("correction weights exceed their quantization contract")
                if self._tickets and request.context != self._tickets[0]._request.context:
                    raise ValueError("correction requests must bind the same public stage")
                if any(t._request.attempt_id == request.attempt_id for t in self._tickets):
                    raise ValueError("correction attempt replay")
                disk = request.rows * request.out_features * (request.wire_bits // 8)
                if self._disk_bound + disk > 64 << 20:
                    raise ValueError("correction batch private disk bound exceeded")
                products = request.rows * request.in_features * request.out_features
                if self._products + products > 8_000_000_000:
                    raise ValueError("correction batch product work bound exceeded")
                cursor = prepared_mask_rows(request)
                if cursor is None:
                    raise ValueError("native one-use mask cursor required")
                sink = tempfile.TemporaryFile()
                binding = hashlib.sha256(json.dumps({"context": asdict(request.context),
                    "attempt": request.attempt_id, "session": request.session_id,
                    "rows": request.rows}, sort_keys=True).encode()).hexdigest()
                ticket = CorrectionTicket(self, request, cursor, sink, binding)
                self._tickets.append(ticket)
                self._disk_bound += disk
                self._products += products
                return ticket
            except BaseException:
                if cursor is not None:
                    cursor.cancel()
                if sink is not None:
                    sink.close()
                self.close()
                raise

    def _accept(self, ticket, offset, count, payload, tag):
        expected = hmac.digest(self._key, _header(ticket._binding, offset, count) + payload, "sha256")
        if (ticket._state != "pending" or offset != ticket._received or count <= 0
                or offset + count > ticket._request.rows or not hmac.compare_digest(expected, tag)):
            raise ValueError("correction row attribution or authentication differs")
        unpack_residues(payload, (count, ticket._request.out_features), ticket._request.wire_bits)
        ticket._file.write(payload)
        ticket._received += count

    def execute(self, tickets=None) -> dict:
        """Fill bounded windows across fresh inventories; seal after every row arrives."""
        with self._lock:
            try:
                tickets = self._tickets if tickets is None else tickets
                if (self._closed or type(tickets) not in (list, tuple) or not tickets
                        or len(tickets) > self._maximum or len({id(t) for t in tickets}) != len(tickets)
                        or any(not any(t is owned for owned in self._tickets)
                               or t._state != "pending" or t._issued for t in tickets)):
                    raise ValueError("correction tickets are not fresh or owned")
                metrics = {"windows": 0, "rows": 0, "encoded_bytes": 0,
                           "kernel_calls": {k: 0 for k in ("native", "digit8", "metal")}}
                start, cpu = time.perf_counter(), time.process_time()
                while any(t._issued < t._request.rows for t in tickets):
                    pieces, masks, ledger, capacity = [], [], [], self._window
                    active = [t for t in tickets if t._issued < t._request.rows]
                    for index, ticket in enumerate(active):
                        if not capacity:
                            break
                        count = min(ticket._request.rows - ticket._issued,
                                    max(1, capacity // (len(active) - index)))
                        offset = ticket._issued
                        raw_r, raw_s = ticket._cursor.take(offset, count)
                        pieces.append(np.frombuffer(raw_r, "<u4").reshape(count, ticket._request.in_features))
                        masks.append(np.frombuffer(raw_s, "<u4").reshape(count, ticket._request.out_features))
                        ledger.append((ticket, offset, count))
                        ticket._issued += count
                        capacity -= count
                    inputs, output_masks = np.concatenate(pieces), np.concatenate(masks)
                    backend = self._kernel
                    if backend == "adaptive":
                        backend = "metal" if len(inputs) >= self._minimum else "native"
                    bits = tickets[0]._request.wire_bits
                    result = (self._gpu.wrap32(inputs) if backend == "metal" else
                              self._matrix.digit8(inputs, bits) if backend == "digit8" else self._matrix.wrap32(inputs))
                    correction = np.subtract(result, output_masks, dtype=np.uint32)
                    correction &= np.uint32((1 << bits) - 1)
                    metrics["kernel_calls"][backend] += 1
                    row = 0
                    for ticket, offset, count in ledger:
                        payload = pack_residues(correction[row:row + count], bits)
                        header = _header(ticket._binding, offset, count)
                        tag = hmac.digest(self._key, header + payload, "sha256")
                        self._accept(ticket, offset, count, payload, tag)
                        metrics["encoded_bytes"] += len(header) + len(payload) + len(tag)
                        row += count
                    metrics["windows"] += 1
                    metrics["rows"] += len(inputs)
                if any(t._received != t._request.rows for t in tickets):
                    raise ValueError("incomplete correction stream cannot seal")
                for ticket in tickets:
                    ticket._file.flush()
                    ticket._file.seek(0)
                    ticket._cursor.cancel()
                    ticket._state = "sealed"
                metrics.update(service_seconds=time.perf_counter() - start,
                               service_cpu_seconds=time.process_time() - cpu,
                               private_file_bytes=sum(t._request.rows * t._request.out_features *
                                                      (t._request.wire_bits // 8) for t in tickets))
                return metrics
            except BaseException:
                self.close()
                raise

    def close(self):
        with self._lock:
            self._closed = True
            for ticket in self._tickets:
                ticket._retire()
            self._key = b""
            self._gpu = self._matrix = None

    def __enter__(self):
        if self._closed:
            raise ValueError("correction batch is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def __copy__(self):
        raise TypeError("live correction batches cannot be copied")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce_ex__(self, protocol):
        return self.__copy__()

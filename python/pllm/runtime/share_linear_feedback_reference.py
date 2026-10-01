"""Bounded two-share affine-state/client-nonlinearity *research reference*.

Only the client reconstructs the rank-width cut. Each worker holds one fresh
additive share of its state and performs public linear operations in a field.
No Qwen operator, trained language model, transport authentication, or whole
decoder is implemented here. The field is deliberately small and must not be
mistaken for a Qwen fixed-point or exact-truncation contract.
"""

from __future__ import annotations

import hashlib
import secrets
import struct
from dataclasses import dataclass

import numpy as np

_MODULUS = 65521
_MAGIC = b"PLLMRANK"
_FRAME = struct.Struct("<8s32sBBHHI")
_INPUT, _CUT, _FEEDBACK, _SCORES = range(1, 5)


class FeedbackReferenceError(ValueError):
    pass


def _residue(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(values, dtype=np.int64) % _MODULUS, dtype=np.uint16)


def _uniform(count: int) -> np.ndarray:
    """Unbiased rejection sampling; modulo reduction of random u16 is biased."""
    if type(count) is not int or not 0 < count <= 64:
        raise FeedbackReferenceError("fresh rank share exceeds the 64-element bound")
    chunks: list[np.ndarray] = []
    remaining = count
    while remaining:
        candidates = np.frombuffer(secrets.token_bytes(remaining * 4), dtype="<u2")
        good = candidates[candidates < _MODULUS][:remaining]
        chunks.append(good.copy())
        remaining -= good.size
    return np.concatenate(chunks).astype(np.uint16, copy=False)


def _split(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if value.dtype != np.uint16 or value.ndim != 1 or np.any(value >= _MODULUS):
        raise FeedbackReferenceError("client can share only bounded field vectors")
    first = _uniform(value.size)
    return first, _residue(value.astype(np.int64) - first.astype(np.int64))


def _matrix(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.shape != shape or raw.dtype != np.int16
        or np.any(raw < -2) or np.any(raw > 2)
    ):
        raise FeedbackReferenceError("public linear weights need exact shapes and coefficients in [-2, 2]")
    owned = np.array(raw, dtype=np.int64, order="C", copy=True)
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True)
class FeedbackWeights:
    recurrent: tuple[np.ndarray, ...]
    incoming: tuple[np.ndarray, ...]
    project: tuple[np.ndarray, ...]
    expand: tuple[np.ndarray, ...]
    head: np.ndarray
    token_codes: np.ndarray
    digest: bytes
    layers: int
    hidden: int
    rank: int
    vocab: int

    @classmethod
    def bind(
        cls, *, recurrent: tuple[np.ndarray, ...], incoming: tuple[np.ndarray, ...],
        project: tuple[np.ndarray, ...], expand: tuple[np.ndarray, ...],
        head: np.ndarray, token_codes: np.ndarray,
    ) -> FeedbackWeights:
        layers = len(recurrent)
        raw_head, codes = np.asarray(head), np.asarray(token_codes)
        if (
            not 1 <= layers <= 24 or raw_head.ndim != 2 or codes.ndim != 2
            or codes.dtype != np.uint16 or not 2 <= codes.shape[0] <= 64
            or not 1 <= codes.shape[1] <= 64 or np.any(codes >= _MODULUS)
        ):
            raise FeedbackReferenceError("public model exceeds bounded layer, code or vocabulary contract")
        hidden, rank, vocab = raw_head.shape[1], codes.shape[1], codes.shape[0]
        if not 2 <= hidden <= 896 or raw_head.shape != (vocab, hidden):
            raise FeedbackReferenceError("public head dimensions are invalid")
        if any(len(group) != layers for group in (incoming, project, expand)):
            raise FeedbackReferenceError("public model omits a declared layer operator")
        groups = (
            tuple(_matrix(row, (hidden, hidden)) for row in recurrent),
            tuple(_matrix(row, (hidden, rank if idx == 0 else hidden)) for idx, row in enumerate(incoming)),
            tuple(_matrix(row, (rank, hidden)) for row in project),
            tuple(_matrix(row, (hidden, rank)) for row in expand),
        )
        head_owned = _matrix(raw_head, (vocab, hidden))
        codes_owned = np.array(codes, dtype=np.uint16, order="C", copy=True)
        codes_owned.setflags(write=False)
        digest = hashlib.sha256(b"pllm.share_linear_feedback_reference.v1\0")
        digest.update(struct.pack("<HHHH", layers, hidden, rank, vocab))
        for group in groups:
            for item in group:
                digest.update(np.asarray(item, dtype="<i2").tobytes())
        digest.update(np.asarray(head_owned, dtype="<i2").tobytes())
        digest.update(np.asarray(codes_owned, dtype="<u2").tobytes())
        return cls(*groups, head_owned, codes_owned, digest.digest(), layers, hidden, rank, vocab)


def _pack(
    binding: bytes, kind: int, party: int, step: int, layer: int, values: np.ndarray,
) -> bytes:
    if (
        type(binding) is not bytes or len(binding) != 32 or kind not in range(1, 5)
        or party not in (0, 1) or type(step) is not int or not 0 <= step <= 70
        or type(layer) is not int or not 0 <= layer <= 24
        or values.dtype != np.uint16 or values.ndim != 1
        or not 1 <= values.size <= 64 or np.any(values >= _MODULUS)
    ):
        raise FeedbackReferenceError("rank frame violates the bounded public wire contract")
    return _FRAME.pack(_MAGIC, binding, kind, party, step, layer, values.size) + np.asarray(values, dtype="<u2").tobytes()


def _unpack(
    payload: bytes, binding: bytes, kind: int, party: int, step: int, layer: int, count: int,
) -> np.ndarray:
    if type(payload) is not bytes or len(payload) != _FRAME.size + count * 2:
        raise FeedbackReferenceError("rank frame is missing, oversized or has trailing bytes")
    magic, actual_binding, actual_kind, actual_party, actual_step, actual_layer, actual_count = _FRAME.unpack_from(payload)
    if (magic, actual_binding, actual_kind, actual_party, actual_step, actual_layer, actual_count) != (
        _MAGIC, binding, kind, party, step, layer, count,
    ):
        raise FeedbackReferenceError("rank frame has a different session, step, role or tensor shape")
    values = np.frombuffer(payload, dtype="<u2", count=count, offset=_FRAME.size)
    if np.any(values >= _MODULUS):
        raise FeedbackReferenceError("rank frame contains a non-field residue")
    return values.copy()


class FeedbackWorker:
    """One party's share and committed public matrices; no client token IDs."""

    def __init__(self, weights: FeedbackWeights, binding: bytes, party: int, *, max_rows: int) -> None:
        if (
            type(weights) is not FeedbackWeights or type(binding) is not bytes or len(binding) != 32
            or type(party) is not int or party not in (0, 1)
            or type(max_rows) is not int or not 0 < max_rows <= 70
        ):
            raise FeedbackReferenceError("worker identity or token budget is invalid")
        self.weights = weights
        self.binding = binding
        self.party = party
        self.max_rows = max_rows
        self.position = 0
        self.layer = 0
        self._states = [np.zeros(weights.hidden, dtype=np.uint16) for _ in range(weights.layers)]
        self._input: np.ndarray | None = None
        self._pending: np.ndarray | None = None
        self.closed = False
        self.integer_macs = 0

    def cancel(self) -> None:
        self.closed = True
        self._input = self._pending = None
        for state in self._states:
            state.fill(0)

    def start_token(self, payload: bytes) -> None:
        try:
            if self.closed or self._input is not None or self.position >= self.max_rows:
                raise FeedbackReferenceError("rank worker cannot accept another input")
            self._input = _unpack(payload, self.binding, _INPUT, self.party, self.position, 0, self.weights.rank)
        except BaseException:
            self.cancel()
            raise

    def begin_layer(self) -> bytes:
        try:
            if self.closed or self._input is None or self._pending is not None or self.layer >= self.weights.layers:
                raise FeedbackReferenceError("rank worker has no next ready layer")
            layer = self.layer
            base = _residue(
                self.weights.recurrent[layer] @ self._states[layer].astype(np.int64)
                + self.weights.incoming[layer] @ self._input.astype(np.int64)
            )
            cut = _residue(self.weights.project[layer] @ base.astype(np.int64))
            self.integer_macs += self.weights.hidden ** 2 + self.weights.hidden * self._input.size
            self.integer_macs += self.weights.rank * self.weights.hidden
            self._pending = base
            return _pack(self.binding, _CUT, self.party, self.position, layer, cut)
        except BaseException:
            self.cancel()
            raise

    def finish_layer(self, payload: bytes) -> None:
        try:
            if self.closed or self._pending is None:
                raise FeedbackReferenceError("rank worker has no pending cut")
            share = _unpack(
                payload, self.binding, _FEEDBACK, self.party, self.position,
                self.layer, self.weights.rank,
            )
            result = _residue(
                self._pending.astype(np.int64)
                + self.weights.expand[self.layer] @ share.astype(np.int64)
            )
            self.integer_macs += self.weights.hidden * self.weights.rank
            self._states[self.layer][:] = result
            self._input = result
            self._pending = None
            self.layer += 1
        except BaseException:
            self.cancel()
            raise

    def complete_token(self, *, score: bool) -> bytes | None:
        try:
            if self.closed or self._pending is not None or self.layer != self.weights.layers or self._input is None:
                raise FeedbackReferenceError("rank worker cannot finish an incomplete token")
            scores = None
            if score:
                scores = _residue(self.weights.head @ self._input.astype(np.int64))
                self.integer_macs += self.weights.vocab * self.weights.hidden
            packet = None if scores is None else _pack(
                self.binding, _SCORES, self.party, self.position,
                self.weights.layers, scores,
            )
            self._input = None
            self.layer = 0
            self.position += 1
            return packet
        except BaseException:
            self.cancel()
            raise


class SharedFeedbackReference:
    """Test-local client conductor; accounts for every serialized role body."""

    def __init__(self, weights: FeedbackWeights, *, max_rows: int) -> None:
        if (
            type(weights) is not FeedbackWeights or type(max_rows) is not int
            or not 0 < max_rows <= 70
        ):
            raise FeedbackReferenceError("rank session needs validated weights and a bounded response")
        self.weights = weights
        self.binding = hashlib.sha256(
            b"pllm.share_linear_feedback_session.v1\0" + weights.digest
            + secrets.token_bytes(32) + struct.pack("<H", max_rows)
        ).digest()
        self.workers = (
            FeedbackWorker(weights, self.binding, 0, max_rows=max_rows),
            FeedbackWorker(weights, self.binding, 1, max_rows=max_rows),
        )
        self.bodies = {"client_to_worker_0": 0, "client_to_worker_1": 0,
                       "worker_0_to_client": 0, "worker_1_to_client": 0}
        self.nonlinear_elements = 0
        self.last_scores: np.ndarray | None = None
        self.closed = False

    def cancel(self) -> None:
        self.closed = True
        for worker in self.workers:
            worker.cancel()
        self.last_scores = None

    def _send(self, worker: FeedbackWorker, data: bytes) -> None:
        self.bodies[f"client_to_worker_{worker.party}"] += len(data)

    def _receive(self, worker: FeedbackWorker, data: bytes) -> None:
        self.bodies[f"worker_{worker.party}_to_client"] += len(data)

    def run_token(self, token: int, *, score: bool) -> np.ndarray | None:
        try:
            if self.closed or type(token) is not int or not 0 <= token < self.weights.vocab:
                raise FeedbackReferenceError("client token is unavailable or out of range")
            code_shares = _split(self.weights.token_codes[token])
            for worker, share in zip(self.workers, code_shares, strict=True):
                packet = _pack(self.binding, _INPUT, worker.party, worker.position, 0, share)
                self._send(worker, packet)
                worker.start_token(packet)
            for layer in range(self.weights.layers):
                cuts = []
                for worker in self.workers:
                    packet = worker.begin_layer()
                    self._receive(worker, packet)
                    cuts.append(_unpack(
                        packet, self.binding, _CUT, worker.party, worker.position,
                        layer, self.weights.rank,
                    ))
                selected = _residue(cuts[0].astype(np.int64) + cuts[1].astype(np.int64))
                nonlinear = _residue(selected.astype(np.int64) ** 2 + 3 * selected.astype(np.int64) + 7)
                self.nonlinear_elements += selected.size
                for worker, share in zip(self.workers, _split(nonlinear), strict=True):
                    packet = _pack(
                        self.binding, _FEEDBACK, worker.party, worker.position,
                        layer, share,
                    )
                    self._send(worker, packet)
                    worker.finish_layer(packet)
            packets = [worker.complete_token(score=score) for worker in self.workers]
            if not score:
                return None
            parts = []
            for worker, packet in zip(self.workers, packets, strict=True):
                assert packet is not None
                self._receive(worker, packet)
                parts.append(_unpack(
                    packet, self.binding, _SCORES, worker.party,
                    worker.position - 1, self.weights.layers, self.weights.vocab,
                ))
            self.last_scores = _residue(parts[0].astype(np.int64) + parts[1].astype(np.int64))
            return self.last_scores.copy()
        except BaseException:
            self.cancel()
            raise

    def prefill(self, tokens: list[int]) -> np.ndarray:
        if not tokens or self.workers[0].position + len(tokens) > self.workers[0].max_rows:
            raise FeedbackReferenceError("prefill must fit the declared response budget")
        for token in tokens[:-1]:
            self.run_token(token, score=False)
        result = self.run_token(tokens[-1], score=True)
        assert result is not None
        return result

    def decode_selected(self) -> tuple[int, np.ndarray]:
        if self.last_scores is None:
            raise FeedbackReferenceError("decode requires a preceding scored token")
        token = int(np.argmax(np.where(
            self.last_scores <= _MODULUS // 2,
            self.last_scores.astype(np.int64), self.last_scores.astype(np.int64) - _MODULUS,
        )))
        next_scores = self.run_token(token, score=True)
        assert next_scores is not None
        return token, next_scores

    def ledger(self) -> dict:
        return {
            "schema": "pllm.share_linear_feedback_reference.v1",
            "client_to_workers_body_bytes": sum(self.bodies[key] for key in (
                "client_to_worker_0", "client_to_worker_1")),
            "workers_to_client_body_bytes": sum(self.bodies[key] for key in (
                "worker_0_to_client", "worker_1_to_client")),
            "all_link_body_bytes": sum(self.bodies.values()),
            "worker_to_worker_body_bytes": 0,
            "offline_one_use_body_bytes": 0,
            "remote_integer_macs": sum(worker.integer_macs for worker in self.workers),
            "client_nonlinear_elements": self.nonlinear_elements,
            "dependent_client_cuts": self.nonlinear_elements // self.weights.rank,
            "positions": self.workers[0].position,
        }


def project_feedback_cost(
    *, layers: int, rank: int, rows: int, scored_rows: int,
    hidden: int, vocabulary: int, word_bytes: int,
) -> dict[str, int]:
    """Header-inclusive response body projection, excluding weights and wire framing.

    Only word_bytes=2 is executable by this toy field. Wider words project a
    *different*, unimplemented numeric contract, never a supported topology.
    """
    if (
        type(layers) is not int or not 1 <= layers <= 24
        or type(rank) is not int or not 1 <= rank <= 64
        or type(rows) is not int or not 1 <= rows <= 70
        or type(scored_rows) is not int or not 1 <= scored_rows <= rows
        or type(hidden) is not int or not 2 <= hidden <= 896
        or type(vocabulary) is not int or not 2 <= vocabulary <= 64
        or type(word_bytes) is not int or word_bytes not in (2, 4, 8)
    ):
        raise FeedbackReferenceError("feedback cost projection exceeds bounded shape or ring")
    rank_frame = _FRAME.size + rank * word_bytes
    score_frame = _FRAME.size + vocabulary * word_bytes
    upload = 2 * rows * (layers + 1) * rank_frame
    download = 2 * rows * layers * rank_frame + 2 * scored_rows * score_frame
    per_worker_macs = rows * (
        layers * hidden * hidden
        + hidden * rank + (layers - 1) * hidden * hidden
        + 2 * layers * hidden * rank
    ) + scored_rows * vocabulary * hidden
    return {
        "client_to_workers_body_bytes": upload,
        "workers_to_client_body_bytes": download,
        "all_link_body_bytes": upload + download,
        "worker_to_worker_body_bytes": 0,
        "offline_one_use_body_bytes": 0,
        "remote_integer_macs": 2 * per_worker_macs,
        "client_nonlinear_elements": rows * layers * rank,
        "implemented_dependent_client_cuts": rows * layers,
        "wavefront_cut_round_lower_bound_if_batched": (
            rows - scored_rows + layers + (scored_rows - 1) * layers
        ),
    }


class ClearFeedbackReference:
    """Independent unshared finite-field recurrence for exact parity checks."""

    def __init__(self, weights: FeedbackWeights, *, max_rows: int) -> None:
        if type(weights) is not FeedbackWeights or type(max_rows) is not int or not 0 < max_rows <= 70:
            raise FeedbackReferenceError("clear reference requires valid weights and row budget")
        self.weights = weights
        self.max_rows = max_rows
        self.position = 0
        self.states = [np.zeros(weights.hidden, dtype=np.uint16) for _ in range(weights.layers)]
        self.last_scores: np.ndarray | None = None

    def run_token(self, token: int, *, score: bool) -> np.ndarray | None:
        if type(token) is not int or not 0 <= token < self.weights.vocab or self.position >= self.max_rows:
            raise FeedbackReferenceError("clear token exceeds the model or response budget")
        value = self.weights.token_codes[token]
        for layer in range(self.weights.layers):
            base = _residue(
                self.weights.recurrent[layer] @ self.states[layer].astype(np.int64)
                + self.weights.incoming[layer] @ value.astype(np.int64)
            )
            projected = _residue(self.weights.project[layer] @ base.astype(np.int64))
            nonlinear = _residue(projected.astype(np.int64) ** 2 + 3 * projected.astype(np.int64) + 7)
            value = _residue(base.astype(np.int64) + self.weights.expand[layer] @ nonlinear.astype(np.int64))
            self.states[layer][:] = value
        self.position += 1
        if score:
            self.last_scores = _residue(self.weights.head @ value.astype(np.int64))
            return self.last_scores.copy()
        return None

    def prefill(self, tokens: list[int]) -> np.ndarray:
        if not tokens or self.position + len(tokens) > self.max_rows:
            raise FeedbackReferenceError("clear prefill exceeds the response budget")
        for token in tokens[:-1]:
            self.run_token(token, score=False)
        result = self.run_token(tokens[-1], score=True)
        assert result is not None
        return result

    def decode_selected(self) -> tuple[int, np.ndarray]:
        if self.last_scores is None:
            raise FeedbackReferenceError("clear decode requires a preceding scored token")
        token = int(np.argmax(np.where(
            self.last_scores <= _MODULUS // 2,
            self.last_scores.astype(np.int64), self.last_scores.astype(np.int64) - _MODULUS,
        )))
        next_scores = self.run_token(token, score=True)
        assert next_scores is not None
        return token, next_scores

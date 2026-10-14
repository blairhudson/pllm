"""Client framing, real native arithmetic, partial accounting and rollback.

Mock HTTP admission is not compiled/native context admission evidence. One
explicit legacy-DPF adapter tests record decode/chunking before the parent's
context API is installed; production transport has no such fallback.
"""

import hashlib
import inspect
import threading
import time

import httpx
import numpy as np
import pytest

from pllm import _native
from pllm.runtime.native import MaskedGEMM
from pllm.runtime.private_boundary_protocol import (
    BoundaryAdmission, BoundaryDescriptor, BoundaryError, ROLES, SCHEMA, canonical,
    context_bytes, contextual_lookup_digest, pack_frame, parse_control, unpack_frame,
)
from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row


KEYS = ("private-boundary-A-secret-92837465", "private-boundary-B-secret-19283746")


class Harness:
    def __init__(self, bits=8):
        self.weights = np.asarray([[2, -3, 4], [-7, 1, 0], [0, 0, 0], [127, -127, 3]], np.int8)
        self.scales = np.asarray([0.5, 0.25, 1.0, 0.125], np.float32)
        records = b"".join(w.tobytes() + self.scales[i:i + 1].tobytes() for i, w in enumerate(self.weights))
        self.roots = [_native.PrivatePageServer(records, 7, p) for p in (0, 1)]
        self.matrix = MaskedGEMM(threads=1).compile(self.weights)
        self.d = BoundaryDescriptor(
            **{k: "a" * 64 for k in ("source_digest", "checkpoint_digest", "model_config_digest",
                "composition_digest", "schedule_digest")},
            head_weight_digest=hashlib.sha256(self.weights.tobytes()).hexdigest(),
            head_scales_digest=hashlib.sha256(self.scales.tobytes()).hexdigest(),
            lookup_digest=self.roots[0].descriptor()[0].hex(), vocabulary=4, hidden=3,
            activation_bits=bits, signed_output_bound=257 * ((1 << (bits - 1)) - 1))
        self.admission = BoundaryAdmission(self.d.digest, "c" * 64, 150, 4)
        self.requests = []
        self.fail_open = self.bad_reply = self.overflow = False
        self.clients = [httpx.Client(base_url=f"http://127.0.0.1:{8001 + p}",
                                    transport=httpx.MockTransport(self.handle)) for p in (0, 1)]

    def handle(self, request):
        p = int(request.url.port) - 8001
        role = ROLES[p]
        self.requests.append(request)
        assert request.headers["authorization"] == "Bearer " + KEYS[p]
        assert KEYS[1 - p].encode() not in request.content
        if request.url.path.endswith("/sessions"):
            row = parse_control(request.content)
            assert row["credential_digests"] == {r: hashlib.sha256(k.encode()).hexdigest()
                                                  for r, k in zip(ROLES, KEYS, strict=True)}
            a = BoundaryAdmission.from_dict(row["admission"])
            context = context_bytes(self.d, a, row["session"], row["epoch"])
            result = {"schema": SCHEMA + ".opened", "role": role, "session": row["session"],
                      "epoch": row["epoch"], "admission": a.to_dict(),
                      "credential_digests": row["credential_digests"],
                      "lookup_digest": contextual_lookup_digest(self.d, context).hex()}
            if self.fail_open and p == 1:
                result["role"] = ROLES[0]
            return httpx.Response(200, content=canonical(result))
        if request.url.path.endswith("/execute"):
            row = unpack_frame(request.content)
            assert "activation_scales" not in row and "token_ids" not in row
            if row["operation"] == "head":
                assert len(row["frames"]) == 1 and len(row["frames"][0]) == 12
                values = np.frombuffer(row["frames"][0], "<u4").reshape(1, 3)
                output = self.matrix.wrap32(values)
                if self.overflow:
                    output = np.full((1, 4), 1 << 29, dtype="<u4")
                frames = [output.astype("<u4", copy=False).tobytes()]
            else:
                frames = self.roots[p].evaluate_batch(row["frames"])
            return httpx.Response(200, content=pack_frame(
                role=ROLES[1 - p] if self.bad_reply else role,
                paired_session=row["session"], epoch=row["epoch"], sequence=row["sequence"],
                operation=row["operation"], frames=frames, response=True))
        row = parse_control(request.content)
        action = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=canonical({"schema": SCHEMA + ".finished",
            "role": role, "session": row["session"], "epoch": row["epoch"], "status": action}))

    def transport(self, **overrides):
        return PrivateBoundaryTransport(**{
            "worker_a": self.clients[0], "worker_b": self.clients[1],
            "api_key_a": KEYS[0], "api_key_b": KEYS[1], "descriptor": self.d,
            "head_scales": self.scales, **overrides})

    def close(self):
        for client in self.clients:
            client.close()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(3,), (1, 3), (1, 1, 3)])
def test_head_exact_bits_shape_random_shares_and_no_clear_scales(bits, shape):
    h = Harness(bits)
    try:
        t = h.transport()
        with pytest.raises(BoundaryError):
            t.linear("lm_head", np.ones(shape, np.float32))
        # A failed operation before open retires the one-use transport.
        with pytest.raises(BoundaryError):
            t.open(h.admission)
        t = h.transport().open(h.admission)
        x = np.asarray([-0.25, 3.75, -1.0], np.float32).reshape(shape)
        q = quantize_activation_per_row(x, bits=bits)
        expected = dequantize_matmul(h.matrix.clear(q.values), q.scales, h.scales,
                                    output_shape=shape[:-1] + (4,))
        for _ in range(2):
            actual = t.linear("lm_head", x)
            assert actual.dtype == np.float32 and actual.shape == expected.shape
            np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
        requests = [unpack_frame(r.content) for r in h.requests if r.url.path.endswith("/execute")]
        shares = [np.frombuffer(r["frames"][0], "<u4").astype(np.int64) for r in requests]
        np.testing.assert_array_equal((shares[0] + shares[1]) % (1 << 32), q.values[0].astype(np.int64) % (1 << 32))
        assert not np.array_equal(shares[1], shares[3])
        assert not np.array_equal(shares[0], q.values[0].astype(np.int64) % (1 << 32))
        t.complete()
        assert t.body_coverage_complete
        assert t.operation_counters["head"]["rows"] == 2
        bodies = t.role_link_bodies
        assert sum(v["online_upload_bytes"] for v in bodies.values()) == t.operation_counters["head"]["upload_bytes"]
        assert sum(v["online_download_bytes"] for v in bodies.values()) == t.operation_counters["head"]["download_bytes"]
        t.close()
        assert not any(c.is_closed for c in h.clients)
    finally:
        h.close()


def test_legacy_native_dpf_decode_chunks_and_scale_bits(monkeypatch):
    h = Harness()
    original_issue = _native.private_page_issue
    t = h.transport().open(h.admission)
    # Explicitly adapt legacy query domain ONLY in this pre-integration test.
    # Real context rejection belongs to context-enabled worker/native tests.
    def legacy_issue(digest, records, width, index, *, leaf_bits):
        assert digest == t._lookup_digest and leaf_bits == 128
        if "leaf_bits" in inspect.signature(original_issue).parameters:
            return original_issue(bytes.fromhex(h.d.lookup_digest), records, width, index, leaf_bits=leaf_bits)
        # The installed pre-acceleration extension supports scalar DPF only.
        return original_issue(bytes.fromhex(h.d.lookup_digest), records, width, index)
    monkeypatch.setattr(_native, "private_page_issue", legacy_issue)
    try:
        ids = np.arange(150, dtype=np.int64) % 4
        actual = t.token_lookup(ids)
        expected = h.weights[ids].astype(np.float32) * h.scales[ids, None]
        np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
        frames = [unpack_frame(r.content) for r in h.requests if r.url.path.endswith("/execute")]
        assert [len(row["frames"]) for row in frames] == [64, 64, 64, 64, 22, 22]
        assert t.operation_counters["lookup"]["calls"] == 3
        assert t.operation_counters["lookup"]["rows"] == 150
        with pytest.raises(BoundaryError):
            t.token_lookup([0])
        assert sum(r.url.path.endswith("/cancel") for r in h.requests) == 2
    finally:
        t.close()
        h.close()


def test_partial_open_rolls_back_both_roles_and_counts_all_control_bodies():
    h = Harness()
    h.fail_open = True
    t = h.transport()
    try:
        with pytest.raises(BoundaryError):
            t.open(h.admission)
        assert [r.url.path.rsplit("/", 1)[-1] for r in h.requests] == ["sessions", "sessions", "cancel", "cancel"]
        for p, role in enumerate(ROLES):
            ledger = t.role_link_bodies[role]
            requests = [r for r in h.requests if r.url.port == 8001 + p]
            assert ledger["setup_upload_bytes"] + ledger["teardown_upload_bytes"] == sum(len(r.content) for r in requests)
            assert ledger["setup_download_bytes"] > 0 and ledger["teardown_download_bytes"] > 0
        assert t.body_coverage_complete  # Rejection was fully observed, not unknown delivery.
        with pytest.raises(BoundaryError):
            t.open(h.admission)
    finally:
        t.close()
        h.close()


@pytest.mark.parametrize("failure", ["bad_reply", "overflow"])
def test_wrong_party_or_signed_bound_burns_both(failure):
    h = Harness()
    t = h.transport().open(h.admission)
    setattr(h, failure, True)
    try:
        with pytest.raises(BoundaryError):
            t.linear("lm_head", np.ones((1, 3), np.float32))
        assert sum(r.url.path.endswith("/cancel") for r in h.requests) == 2
        with pytest.raises(BoundaryError):
            t.linear("lm_head", np.ones((1, 3), np.float32))
    finally:
        t.close()
        h.close()


def test_partial_stream_bytes_retained_and_coverage_marked_incomplete():
    h = Harness()
    original = h.handle
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("connection retired")
    def handler(request):
        if request.url.path.endswith("/execute"):
            h.requests.append(request)
            return httpx.Response(200, stream=BrokenStream())
        return original(request)
    h.clients[0].close()
    h.clients[0] = httpx.Client(base_url="http://127.0.0.1:8001", transport=httpx.MockTransport(handler))
    t = h.transport().open(h.admission)
    try:
        with pytest.raises(httpx.ReadError):
            t.linear("lm_head", np.ones((1, 3), np.float32))
        assert not t.body_coverage_complete
        assert t.role_link_bodies[ROLES[0]]["online_download_bytes"] == 7
        assert t.operation_counters["head"]["download_bytes"] == 7
        assert sum(r.url.path.endswith("/cancel") for r in h.requests) == 2
    finally:
        t.close()
        h.close()


def test_security_origins_scales_bias_and_distinct_credentials():
    h = Harness()
    try:
        for options in ({"worker_b": h.clients[0]}, {"api_key_b": KEYS[0]},
                        {"worker_b": "http://provider.example"},
                        {"worker_b": "https://u:p@provider.example"},
                        {"worker_b": "https://provider.example/path"},
                        {"head_scales": h.scales * 2}, {"bias": np.zeros(4, np.float32)}):
            with pytest.raises(BoundaryError):
                h.transport(**options)
        t = h.transport(worker_b="https://provider.example")
        owned = t._clients[1]
        t.close()
        assert owned.is_closed and not h.clients[0].is_closed
    finally:
        h.close()


@pytest.mark.parametrize("x", [np.ones((2, 3), np.float32), np.ones((1, 3), np.float64),
                               np.asarray([[np.nan, 0, 1]], np.float32)])
def test_invalid_head_input_burns_before_online_send(x):
    h = Harness()
    t = h.transport().open(h.admission)
    try:
        with pytest.raises(BoundaryError):
            t.linear("lm_head", x)
        assert not any(r.url.path.endswith("/execute") for r in h.requests)
        assert sum(r.url.path.endswith("/cancel") for r in h.requests) == 2
    finally:
        t.close()
        h.close()


def test_abort_signals_then_drains_blocking_operation_without_late_logits():
    h = Harness()
    entered, release = threading.Event(), threading.Event()
    original = h.handle
    def blocking(request):
        if request.url.path.endswith("/execute"):
            entered.set()
            assert release.wait(5)
        return original(request)
    h.clients[0].close()
    h.clients[0] = httpx.Client(base_url="http://127.0.0.1:8001", transport=httpx.MockTransport(blocking))
    t = h.transport().open(h.admission)
    errors, values = [], []
    def run():
        try:
            values.append(t.linear("lm_head", np.ones((1, 3), np.float32)))
        except BoundaryError as exc:
            errors.append(exc)
    caller = threading.Thread(target=run)
    canceller = threading.Thread(target=t.abort)
    try:
        caller.start()
        assert entered.wait(5)
        canceller.start()
        # Wait for cancellation signal without releasing the HTTP response yet.
        deadline = time.monotonic() + 5
        while True:
            with t._state_lock:
                closed = t._state == "closed"
            if closed:
                break
            assert time.monotonic() < deadline
            threading.Event().wait(0.001)
        release.set()
        caller.join(5)
        canceller.join(5)
        assert not caller.is_alive() and not canceller.is_alive()
        assert errors and not values
        assert sum(r.url.path.endswith("/cancel") for r in h.requests) == 2
    finally:
        release.set()
        t.close()
        h.close()


def test_head_budget_is_monotonic_and_stage_is_exclusive():
    h = Harness()
    from dataclasses import replace
    t = h.transport().open(replace(h.admission, head_calls=1))
    try:
        t.linear("lm_head", np.ones((1, 3), np.float32))
        with pytest.raises(BoundaryError):
            t.linear("lm_head", np.ones((1, 3), np.float32))
        assert t.operation_counters["head"]["calls"] == 1
        assert sum(r.url.path.endswith("/cancel") for r in h.requests) == 2
        another = h.transport().open(h.admission)
        with pytest.raises(BoundaryError):
            another.linear("body_stage", np.ones((1, 3), np.float32))
        another.close()
    finally:
        t.close()
        h.close()


@pytest.mark.parametrize("indices", [[-1], [4], [1.0], [[1]], []])
def test_invalid_lookup_indices_send_no_private_queries(indices):
    h = Harness()
    t = h.transport().open(h.admission)
    try:
        with pytest.raises(BoundaryError):
            t.token_lookup(indices)
        assert not any(r.url.path.endswith("/execute") for r in h.requests)
    finally:
        t.close()
        h.close()


def test_strict_numeric_control_acknowledgements_and_no_credential_redirect():
    h = Harness()
    original = h.handle
    def wrong_epoch(request):
        response = original(request)
        if request.url.path.endswith("/sessions"):
            row = parse_control(response.content)
            row["epoch"] = 0.0
            return httpx.Response(200, content=canonical(row))
        return response
    h.clients[0].close()
    h.clients[0] = httpx.Client(base_url="http://127.0.0.1:8001", transport=httpx.MockTransport(wrong_epoch))
    t = h.transport()
    try:
        with pytest.raises(BoundaryError):
            t.open(h.admission)
        assert sum(r.url.path.endswith("/cancel") for r in h.requests) == 1
        t.close()
        seen = []
        def redirect(request):
            seen.append(request)
            return httpx.Response(307, headers={"location": "https://different.example/leak"})
        h.clients[0].close()
        h.clients[0] = httpx.Client(base_url="http://127.0.0.1:8001", follow_redirects=True,
                                    transport=httpx.MockTransport(redirect))
        t = h.transport()
        with pytest.raises(BoundaryError):
            t.open(h.admission)
        assert seen and all(r.url.host == "127.0.0.1" for r in seen)
    finally:
        t.close()
        h.close()

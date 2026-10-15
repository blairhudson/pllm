"""Current-step parallel head ownership, exact arithmetic and failure drainage."""

from dataclasses import replace
import threading
import time

import httpx
import msgpack
import numpy as np
import pytest

from pllm.runtime.private_boundary_protocol import (
    BoundaryError, ROLES, pack_frame, unpack_frame,
)
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from test_private_boundary_transport import Harness


class ParallelHarness(Harness):
    def __init__(self, bits=8, dispatch="parallel"):
        super().__init__(bits)
        self.d = replace(self.d, head_dispatch=dispatch)
        self.admission = replace(self.admission, descriptor_digest=self.d.digest)
        self.before = lambda request, row, party: None
        self.after = lambda response, row, party: response
        self.rows = []
        self.guard = threading.Lock()

    def handle(self, request):
        if not request.url.path.endswith("/execute"):
            return super().handle(request)
        party = int(request.url.port) - 8001
        row = unpack_frame(request.content)
        with self.guard:
            self.rows.append((party, row, threading.current_thread()))
        self.before(request, row, party)
        response = super().handle(request)
        return self.after(response, row, party)


@pytest.fixture
def harnesses():
    owners = []

    def make(*args, **kwargs):
        h = ParallelHarness(*args, **kwargs)
        owners.append(h)
        return h

    yield make
    for h in owners:
        h.close()


def cancels(h):
    return [r for r in h.requests if r.url.path.endswith("/cancel")]


def invoke(t, values, errors):
    try:
        values.append(t.linear("lm_head", np.ones((1, 3), np.float32)))
    except BaseException as error:
        errors.append(error)


def wait_closed(t):
    deadline = time.monotonic() + 5
    while True:
        with t._state_lock:
            if t._state == "closed":
                return
        assert time.monotonic() < deadline
        threading.Event().wait(0.001)


def assert_retired(t, h):
    assert len(cancels(h)) == 2
    count = len(h.requests)
    with pytest.raises(BoundaryError):
        t.linear("lm_head", np.ones((1, 3), np.float32))
    with pytest.raises(BoundaryError):
        t.open(h.admission)
    with pytest.raises(BoundaryError):
        t.complete()
    t.close()
    assert len(h.requests) == count
    assert not any(thread.is_alive() for _, _, thread in h.rows
                   if thread.name.startswith("pllm-boundary-head"))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(3,), (1, 3), (1, 1, 3)])
def test_parallel_barrier_full_integer_and_logit_parity(harnesses, bits, shape):
    h = harnesses(bits)
    barrier = threading.Barrier(2)
    h.before = lambda *_: barrier.wait(5)
    t = h.transport().open(h.admission)
    x = np.asarray([-0.25, 3.75, -1.0], np.float32).reshape(shape)
    q = quantize_activation_per_row(x, bits=bits)
    expected_integer = q.values.astype(np.int64) @ h.weights.astype(np.int64).T
    expected = dequantize_matmul(expected_integer, q.scales, h.scales,
                                output_shape=shape[:-1] + (4,))
    try:
        for sequence in range(2):
            actual = t.linear("lm_head", x)
            np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
            rows = {p: row for p, row, _ in h.rows if row["sequence"] == sequence}
            assert set(rows) == {0, 1}
            assert all(row["operation"] == "head" for row in rows.values())
            shares = [np.frombuffer(rows[p]["frames"][0], "<u4").astype(np.int64)
                      for p in (0, 1)]
            integers = [(share @ h.weights.astype(np.int64).T) % (1 << 32)
                        for share in shares]
            combined = (integers[0] + integers[1]) % (1 << 32)
            centered = np.where(combined >= 1 << 31, combined - (1 << 32), combined)
            np.testing.assert_array_equal(centered, expected_integer[0])
            assert not any(thread.is_alive() for _, _, thread in h.rows)
        counters = t.operation_counters["head"]
        assert counters["calls"] == counters["rows"] == 2
        assert counters["upload_bytes"] == sum(v["online_upload_bytes"] for v in t.role_link_bodies.values())
        assert counters["download_bytes"] == sum(v["online_download_bytes"] for v in t.role_link_bodies.values())
        assert t.body_coverage_complete
        t.complete()
    finally:
        t.close()


@pytest.mark.parametrize("dispatch", ["serial", "parallel"])
def test_lookup_remains_serial_and_head_default_remains_serial(harnesses, dispatch):
    h = harnesses(dispatch=dispatch)
    order = []

    def before(request, row, party):
        assert threading.current_thread() is threading.main_thread()
        order.append(party)
        assert order == ([0] if party == 0 else [0, 1])

    h.before = before
    t = h.transport().open(h.admission)
    try:
        if dispatch == "serial":
            # This must be the descriptor default, not a transport override.
            assert type(h.d).__dataclass_fields__["head_dispatch"].default == "serial"
            t.linear("lm_head", np.ones((1, 3), np.float32))
        else:
            # Isolate lookup scheduling from the separate native context API.
            def lookup(request, row, party):
                before(request, row, party)
                return httpx.Response(200, content=pack_frame(
                    role=ROLES[party], paired_session=row["session"], epoch=row["epoch"],
                    sequence=row["sequence"], operation="lookup", frames=[b"record"], response=True))

            def handle(request):
                if request.url.path.endswith("/execute"):
                    row = unpack_frame(request.content)
                    return lookup(request, row, int(request.url.port) - 8001)
                return h.handle(request)

            for p in (0, 1):
                h.clients[p].close()
                h.clients[p] = httpx.Client(base_url=f"http://127.0.0.1:{8001 + p}",
                                           transport=httpx.MockTransport(handle))
            t.close()
            t = h.transport().open(h.admission)
            with t._lock:
                assert t._exchange("lookup", [b"query-a"], [b"query-b"]) == [[b"record"], [b"record"]]
        assert order == [0, 1]
    finally:
        t.close()


@pytest.mark.parametrize("failed_party", [0, 1])
@pytest.mark.parametrize("error_type", [httpx.ReadError, KeyboardInterrupt])
def test_partial_stream_failure_drains_other_party_and_charges_bytes(harnesses, failed_party, error_type):
    h = harnesses()
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    failed = threading.Event()
    stream_closed = threading.Event()

    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            failed.set()
            raise error_type("partial head failed")

        def close(self):
            stream_closed.set()

    def before(request, row, party):
        entered[party].set()
        assert entered[1 - party].wait(5)
        if party != failed_party:
            assert release.wait(5)

    h.before = before
    h.after = lambda response, row, party: (httpx.Response(200, stream=BrokenStream())
                                            if party == failed_party else response)
    t = h.transport().open(h.admission)
    values, errors = [], []
    caller = threading.Thread(target=invoke, args=(t, values, errors))
    try:
        caller.start()
        assert failed.wait(5)
        assert caller.is_alive() and not cancels(h)
        release.set()
        caller.join(5)
        assert not caller.is_alive() and not values
        assert len(errors) == 1 and isinstance(errors[0], error_type)
        assert stream_closed.is_set()
        assert not t.body_coverage_complete
        bodies = t.role_link_bodies
        assert bodies[ROLES[failed_party]]["online_download_bytes"] == 7
        assert bodies[ROLES[1 - failed_party]]["online_download_bytes"] > 7
        assert all(v["online_upload_bytes"] > 0 for v in bodies.values())
        assert t.operation_counters["head"]["download_bytes"] == sum(v["online_download_bytes"] for v in bodies.values())
        assert_retired(t, h)
    finally:
        release.set()
        caller.join(5)
        t.close()


@pytest.mark.parametrize("failed_party", [0, 1])
@pytest.mark.parametrize("fault", ["malformed", "role", "session", "epoch", "sequence", "operation", "rows", "status"])
def test_bad_reply_in_either_direction_burns_whole_pair(harnesses, failed_party, fault):
    h = harnesses()
    barrier = threading.Barrier(2)
    h.before = lambda *_: barrier.wait(5)

    def after(response, row, party):
        if party != failed_party:
            return response
        if fault == "malformed":
            return httpx.Response(200, content=b"bad frame")
        if fault == "status":
            return httpx.Response(503, content=b"refused")
        reply = unpack_frame(response.content, response=True)
        changes = {"role": ROLES[1 - party], "session": "f" * 32,
                   "epoch": row["epoch"] + 1, "sequence": row["sequence"] + 1,
                   "operation": "lookup", "rows": [b"extra", b"extra"]}
        reply["frames" if fault == "rows" else fault] = changes[fault]
        # Deliberately bypass producer validation to exercise malformed peer data.
        return httpx.Response(200, content=msgpack.packb(reply, use_bin_type=True))

    h.after = after
    t = h.transport().open(h.admission)
    try:
        with pytest.raises(BoundaryError):
            t.linear("lm_head", np.ones((1, 3), np.float32))
        assert len(h.rows) == 2
        assert t.body_coverage_complete
        assert_retired(t, h)
    finally:
        t.close()


@pytest.mark.parametrize("action", ["abort", "close"])
@pytest.mark.parametrize("owned", [False, True])
def test_external_cancellation_drains_both_without_output(harnesses, monkeypatch, action, owned):
    h = harnesses()
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    left = [threading.Event(), threading.Event()]

    def before(request, row, party):
        entered[party].set()
        assert release[party].wait(5)
        left[party].set()

    h.before = before
    if owned:
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _, request: h.handle(request))
        t = h.transport(worker_a="http://127.0.0.1:8001", worker_b="http://127.0.0.1:8002")
    else:
        t = h.transport()
    t.open(h.admission)
    values, errors = [], []
    caller = threading.Thread(target=invoke, args=(t, values, errors))
    canceller = threading.Thread(target=getattr(t, action))
    try:
        caller.start()
        assert all(event.wait(5) for event in entered)
        with pytest.raises(BoundaryError, match="pending work"):
            t.complete()
        canceller.start()
        wait_closed(t)
        release[0].set()
        assert left[0].wait(5)
        assert canceller.is_alive() and caller.is_alive() and not cancels(h)
        assert not any(client.is_closed for client in t._clients)
        release[1].set()
        caller.join(5)
        canceller.join(5)
        assert not caller.is_alive() and not canceller.is_alive()
        assert not values and len(errors) == 1 and isinstance(errors[0], BoundaryError)
        assert all(client.is_closed == (owned and action == "close") for client in t._clients)
        assert_retired(t, h)
        assert not any(client.is_closed for client in h.clients)
        assert all(client.is_closed == owned for client in t._clients)
    finally:
        for event in release:
            event.set()
        caller.join(5)
        if canceller.ident is not None:
            canceller.join(5)
        t.close()


@pytest.mark.parametrize("party", [0, 1])
def test_previous_step_reply_is_rejected_as_replay(harnesses, party):
    h = harnesses()
    previous = {}

    def after(response, row, index):
        if row["sequence"] == 0:
            previous[index] = response.content
        elif index == party:
            return httpx.Response(200, content=previous[index])
        return response

    h.after = after
    t = h.transport().open(h.admission)
    try:
        t.linear("lm_head", np.ones((1, 3), np.float32))
        with pytest.raises(BoundaryError, match="ownership"):
            t.linear("lm_head", np.ones((1, 3), np.float32))
        assert len(h.rows) == 4
        assert_retired(t, h)
    finally:
        t.close()


def test_concurrent_callers_cannot_issue_future_pair_before_current_drains(harnesses):
    h = harnesses()
    barrier = threading.Barrier(2)
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()

    def before(request, row, party):
        barrier.wait(5)
        if row["sequence"] == 0:
            entered.set()
            assert release.wait(5)

    h.before = before
    t = h.transport().open(h.admission)
    values, errors = [], []
    first = threading.Thread(target=invoke, args=(t, values, errors))

    def run_second():
        second_started.set()
        invoke(t, values, errors)

    second = threading.Thread(target=run_second)
    try:
        first.start()
        assert entered.wait(5)
        second.start()
        assert second_started.wait(5)
        assert len(h.rows) == 2 and t._heads == 1
        release.set()
        first.join(5)
        second.join(5)
        assert not first.is_alive() and not second.is_alive()
        assert len(values) == 2 and not errors
        assert [row["sequence"] for _, row, _ in h.rows] == [0, 0, 1, 1]
        assert all(not thread.is_alive() for _, _, thread in h.rows)
        t.complete()
        t.complete()  # Successful completion remains idempotent.
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
        t.close()


def test_second_submission_failure_drains_first_and_retires_pair(harnesses, monkeypatch):
    import pllm.runtime.private_boundary_transport as module

    h = harnesses()
    entered, release, rejected = (threading.Event() for _ in range(3))

    def before(*_):
        entered.set()
        assert release.wait(5)

    class BrokenOwner(module.ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            if args[0] == 1:
                assert entered.wait(5)
                rejected.set()
                raise RuntimeError("second worker could not start")
            return super().submit(fn, *args, **kwargs)

    h.before = before
    monkeypatch.setattr(module, "ThreadPoolExecutor", BrokenOwner)
    t = h.transport().open(h.admission)
    values, errors = [], []
    caller = threading.Thread(target=invoke, args=(t, values, errors))
    try:
        caller.start()
        assert rejected.wait(5)
        assert caller.is_alive() and not cancels(h)
        release.set()
        caller.join(5)
        assert not caller.is_alive() and not values
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
        assert len(h.rows) == 1 and h.rows[0][0] == 0
        assert t.role_link_bodies[ROLES[0]]["online_upload_bytes"] > 0
        assert t.role_link_bodies[ROLES[1]]["online_upload_bytes"] == 0
        assert_retired(t, h)
    finally:
        release.set()
        caller.join(5)
        t.close()


def test_failed_completion_cannot_later_report_success(harnesses, monkeypatch):
    h = harnesses()
    t = h.transport().open(h.admission)
    original = t._post

    def refused(index, path, *args, **kwargs):
        status, raw = original(index, path, *args, **kwargs)
        return (503, b"refused") if path.endswith("/complete") else (status, raw)

    monkeypatch.setattr(t, "_post", refused)
    try:
        with pytest.raises(BoundaryError, match="completion failed"):
            t.complete()
        count = len(h.requests)
        with pytest.raises(BoundaryError, match="requires an active session"):
            t.complete()
        assert len(h.requests) == count
    finally:
        t.close()

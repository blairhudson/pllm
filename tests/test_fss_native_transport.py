"""Generated native bridge tests, not HTTP authentication/checkpoint qualification.

Callbacks use canonical pllm.fss.callback.v1 JSON. kernel contains context
{plan,session,operation,first_tensor_index}, plan_digest, ordinal; role is the
recipient/own holder. Material adds pc/packet_bytes_bound; peer adds frame/
frame_bytes. All digests are lowercase hex. No fabricated native outputs.
"""
import concurrent.futures
import hashlib
import json
import math
import os
import queue
import signal
import struct
import tempfile
import threading
import time

import pytest

native = pytest.importorskip("pllm._native")
pytestmark = pytest.mark.skipif(
    not all(hasattr(native, name) for name in ("FssRoleRequest", "FssClientRequest", "FssSourceBuilder")),
    reason="native role bridge not built",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture(scope="module", params=["serial", "wavefront-v1"])
def program(request):
    from pllm import Deployment, ExecutionBudget, Model
    from pllm.configuration import Experiment
    from pllm.profiles import FuseFssSourceCpu
    config = canonical({"model_type": "gpt2", "n_embd": 2, "n_head": 1, "n_layer": 1,
                        "n_positions": 2, "vocab_size": 2, "n_inner": 2})
    builder = native.FssSourceBuilder(config, 1, 1, "fusefss.gpt2-complete-ring50-q12.v1")
    shapes = json.loads(builder.metadata())["source_shapes"]
    for name, shape in sorted(shapes.items()):
        values = [0.0] * math.prod(shape)
        if len(shape) == 1 and name.endswith("weight"):
            values = [1.0] * len(values)
        elif name == "transformer.wte.weight":
            values = [1.0, 0.0, 0.0, 1.0]
        builder.append(name, "F32", 0, struct.pack("<" + "f" * len(values), *values))
    value = builder.seal_wavefront() if request.param == "wavefront-v1" else builder.seal()
    experiment = Experiment(
        name="generated-native-transport", pipeline=FuseFssSourceCpu(
            Model.path("/generated/fss-native-fixture"),
            numeric_profile="fusefss.gpt2-complete-ring50-q12.v1",
            online_schedule=request.param,
        ),
        deployment=Deployment.local(root="local://fss-native-fixture"),
        budget=ExecutionBudget(requests=1, max_input_tokens=1, max_new_tokens=1),
    )
    # Synthetic nonzero file/tokenizer pin labels. These tests exercise exact
    # typed plan/profile admission, NOT file-origin/source-lock authentication.
    contract = native.fss_source_contract(experiment.canonical_bytes(), config, 1, 1, "01" * 32, "02" * 32)
    yield value, contract
    value.close()


def request(program, role):
    program, contract = program
    manifest = program.manifest(contract)
    client = native.FssClientRequest(contract, bytes([7]) * 32, (manifest,) * 3, [0])
    return client, native.FssRoleRequest(program, contract, client.request_bytes(), role)


def delivery_contract(program, delivery):
    source, contract = program
    if delivery == "single":
        return source, contract
    # Transport policy is not a tensor-derived claim; native validates the full
    # typed contract and binds the new bytes into Manifest/Request. SDK generation
    # of the experiment/composition identity is covered by native SDK tests.
    value = json.loads(contract)
    value["material_delivery"] = "batch-v1"
    return source, canonical(value)


def test_batched_admission_prices_all_overlap_and_request_binds_selector(program):
    source, single_contract = program
    _, batch_contract = delivery_contract(program, "batch-v1")
    single = json.loads(native.fss_transport_admission(source, single_contract))
    batched = json.loads(native.fss_transport_admission(source, batch_contract))
    assert "material_delivery" not in single and "delivery_batch" not in single
    assert batched["material_delivery"] == "batch-v1"
    extra = batched["delivery_batch"]
    assert extra["records_bound"] == 64
    assert extra["body_bytes_bound"] == 1 << 20
    assert extra["envelope_bytes_bound"] == 1024
    assert extra["record_overhead_bytes"] == 8
    assert batched["combined_reservation_bytes"] == (single["combined_reservation_bytes"]
        + extra["native_reservation_bytes"] + extra["python_overlap_bytes"])
    assert batched["combined_reservation_bytes"] <= 64 << 20
    for field in ("kernels", "primitive_requests_per_party", "cumulative_packet_bytes_per_party_bound",
                  "guard_capacity", "frames_per_party"):
        assert batched[field] == single[field]
    for chosen, other in ((single_contract, batch_contract), (batch_contract, single_contract)):
        manifest = source.manifest(chosen)
        client = native.FssClientRequest(chosen, bytes([33]) * 32, (manifest,) * 3, [0])
        try:
            with pytest.raises(Exception):
                native.FssRoleRequest(source, other, client.request_bytes(), 2)
        finally:
            client.close()


@pytest.mark.parametrize("failure", ["partial", "lost", "non-none"])
def test_batched_delivery_failure_burns_complete_request_without_suffix_retry(program, failure):
    client, role = request(delivery_contract(program, "batch-v1"), 2)

    class Transport:
        cancellations = 0
        calls = 0
        accepted = [0, 0]

        def deliver(self, meta, packet):
            pytest.fail("small fixture must enter explicit batch callback")

        def deliver_batch(self, recipient, records):
            assert type(recipient) is int and recipient in (0, 1)
            assert type(records) is tuple and 1 < len(records) <= 64
            self.calls += 1
            for index, record in enumerate(records):
                assert type(record) is tuple and len(record) == 2
                meta, packet = record
                value = parse_meta(meta, "material", recipient)
                assert type(packet) is bytes and len(packet) <= value["packet_bytes_bound"]
                self.accepted[recipient] += 1
                if failure == "partial" and index == 0:
                    raise RuntimeError("PRIVATE-BATCH-PREFIX-DETAIL")
            if failure == "lost":
                raise RuntimeError("PRIVATE-BATCH-LOST-ACK-DETAIL")
            return False  # Anything except None is not an acknowledgement.

        def finish(self):
            pytest.fail("failed batch cannot seal")

        def cancel(self):
            self.cancellations += 1
            self.accepted = [0, 0]

    callback = Transport()
    try:
        with pytest.raises(Exception) as error:
            role.run(None, callback)
        assert "PRIVATE-BATCH" not in str(error.value)
        assert callback.calls == callback.cancellations == 1
        assert callback.accepted == [0, 0]
        with pytest.raises(Exception):
            role.run(None, callback)
        assert callback.calls == callback.cancellations == 1
    finally:
        role.close()
        client.close()


def parse_meta(data, kind, role):
    assert type(data) is bytes and len(data) <= 1024
    value = json.loads(data)
    assert canonical(value) == data
    assert value["schema"] == "pllm.fss.callback.v1"
    assert value["kind"] == kind and value["role"] == role
    assert set(value["kernel"]) == {"context", "plan_digest", "ordinal"}
    context = value["kernel"]["context"]
    assert set(context) == {"plan", "session", "operation", "first_tensor_index"}
    for key in ("plan", "session", "operation"):
        assert len(context[key]) == 64 and bytes.fromhex(context[key]).hex() == context[key]
    return value


def test_public_admission_and_malformed_run_retire(program):
    client, role = request(program, 0)
    admission = json.loads(role.metadata())
    assert admission["schema"] == "pllm.fss.role_admission.v1"
    assert admission["kernels"] > 0 and admission["primitive_requests_per_party"] > 0
    assert 1 <= admission["guard_capacity"] <= 4096
    assert admission["largest_packet_bytes"] <= admission["cumulative_packet_bytes_per_party_bound"]
    assert admission["combined_reservation_bytes"] <= 64 << 20
    with pytest.raises(Exception):
        role.run()  # Owner removed before signature conversion.
    with pytest.raises(Exception):
        role.metadata()
    client.close()


def test_program_admission_before_tokens_and_rejects_wrong_typed_contract(program):
    source, contract = program
    admission = json.loads(native.fss_transport_admission(source, contract))
    assert admission["schema"] == "pllm.fss.program_transport_admission.v1"
    assert admission["scope"] == "program-pricing-not-request-admission"
    assert "transcript_digest" not in admission and "context" not in admission
    assert admission["combined_reservation_bytes"] <= 64 << 20
    substituted = json.loads(contract)
    if substituted.get("online_schedule") == "wavefront-v1":
        del substituted["online_schedule"]
    else:
        substituted["online_schedule"] = "wavefront-v1"
    with pytest.raises(Exception, match="schedule differs"):
        native.fss_transport_admission(source, canonical(substituted))
    assert 0 < admission["peer_frame_bytes_bound"] < json.loads(source.metadata())["frame_bytes"]
    client, role = request(program, 0)
    try:
        assert json.loads(role.metadata())["peer_frame_bytes_bound"] == admission["peer_frame_bytes_bound"]
    finally:
        role.close()
        client.close()
    for field in ("profile", "plan", "empty"):
        if field == "profile":
            bad_contract = contract.replace(
                b'"numeric_profile":"fusefss.gpt2-complete-ring50-q12.v1"',
                b'"numeric_profile":"fusefss.dense-rope-ring50-q12.v1"',
            )
        elif field == "plan":
            # Preserve native canonical float spelling and every unrelated byte.
            bad_contract = b'"query_sequence":2'.join(contract.rsplit(b'"query_sequence":1', 1))
        else:
            bad_contract = b"{}"
        assert bad_contract != contract
        with pytest.raises(Exception):
            native.fss_transport_admission(source, bad_contract)
        # Deliberately counterfeit *public* manifests model the original raw-hash
        # bug. Client framing is not a checkpoint/contract correctness proof.
        # Local role admission must reject the mismatching typed document even
        # when its hash and every public request field are mutually consistent.
        forged = json.loads(source.manifest(contract))
        forged["contract"] = hashlib.sha256(bad_contract).hexdigest()
        forged = canonical(forged)
        fake_client = native.FssClientRequest(bad_contract, bytes([8]) * 32, (forged,) * 3, [0])
        try:
            for role in (0, 1, 2):
                with pytest.raises(Exception):
                    native.FssRoleRequest(source, bad_contract, fake_client.request_bytes(), role)
        finally:
            fake_client.close()


@pytest.mark.parametrize("method", ["close", "cancel"])
def test_close_before_run_is_terminal(program, method):
    client, role = request(program, 2)
    getattr(role, method)()
    with pytest.raises(Exception):
        role.run(None, object())
    client.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
def test_role_fork_rejects_without_calling_inherited_transport(program):
    client, role = request(program, 2)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            signal.alarm(20)
            try:
                role.metadata()
            except Exception:
                try:
                    role.run(None, object())
                except Exception:
                    os.write(write_fd, b"1")
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    try:
        result = os.read(read_fd, 1)
        status = os.waitpid(pid, 0)[1]
        assert result == b"1" and status == 0
        assert json.loads(role.metadata())["role"] == 2
    finally:
        os.close(read_fd)
        role.close()
        client.close()


def test_callback_failure_is_sanitized_and_cancellation_once(program):
    client, role = request(program, 2)

    class Transport:
        cancellations = 0

        def deliver(self, meta, packet):
            value = parse_meta(meta, "material", json.loads(meta)["role"])
            assert type(packet) is bytes and len(packet) <= value["packet_bytes_bound"]
            raise RuntimeError("PRIVATE-CALLBACK-DETAIL-MUST-NOT-ESCAPE")

        def finish(self):
            pytest.fail("failed dealer cannot finish")

        def cancel(self):
            self.cancellations += 1

    transport = Transport()
    with pytest.raises(Exception) as error:
        role.run(None, transport)
    assert "PRIVATE-CALLBACK" not in str(error.value)
    assert transport.cancellations == 1
    with pytest.raises(Exception):
        role.run(None, transport)
    assert transport.cancellations == 1
    client.close()


@pytest.mark.parametrize("material_delivery", ["single", "batch-v1"])
def test_concurrent_cancel_wakes_callback_without_exclusive_pyclass_borrow(program, material_delivery):
    client, role = request(delivery_contract(program, material_delivery), 2)

    class Transport:
        entered = threading.Event()
        cancelled = threading.Event()
        callback_returned = threading.Event()
        cancel_entered = threading.Event()
        drain_allowed = threading.Event()
        cancellations = 0

        def deliver(self, meta, packet):
            self.entered.set()
            assert self.cancelled.wait(30), "cancel must wake the bounded callback"
            self.callback_returned.set()

        def deliver_batch(self, role, records):
            self.deliver(*records[0])

        def finish(self):
            pytest.fail("cancelled request cannot finish")

        def cancel(self):
            self.cancellations += 1
            self.cancelled.set()
            self.cancel_entered.set()
            assert self.drain_allowed.wait(30), "caller retains reservations through cancel drain"

    transport = Transport()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(role.run, None, transport)
        try:
            assert transport.entered.wait(30)
            cancel = pool.submit(role.cancel)
            assert transport.cancel_entered.wait(30)
            assert transport.callback_returned.wait(30)
            with pytest.raises(concurrent.futures.TimeoutError):
                future.result(timeout=0.1)
            transport.drain_allowed.set()
            cancel.result(timeout=30)
            with pytest.raises(Exception):
                future.result(timeout=30)
        finally:
            transport.cancelled.set()
            transport.drain_allowed.set()
            role.close()
    assert transport.cancellations == 1
    client.close()


@pytest.mark.parametrize("returned", [None, bytearray(b"x"), b"", b"bad", b"x" * (1 << 20)])
def test_holder_reply_type_corruption_or_bound_failure_burns(program, returned):
    client, role = request(program, 0)

    class Transport:
        cancellations = 0

        def material(self, meta):
            parse_meta(meta, "material", 0)
            return returned

        def exchange(self, meta, frame):
            parse_meta(meta, "peer", 0)
            return returned

        def finish(self):
            pytest.fail("invalid reply cannot complete")

        def cancel(self):
            self.cancellations += 1

    transport = Transport()
    with pytest.raises(Exception):
        role.run(client.take_input(0), transport)
    assert transport.cancellations == 1
    with pytest.raises(Exception):
        role.metadata()
    client.close()


@pytest.mark.parametrize("material_delivery", ["single", "batch-v1"])
def test_generated_native_offline_dealer_then_two_holders_and_client_final_only(program, tmp_path, material_delivery):
    program, contract = delivery_contract(program, material_delivery)
    manifest = program.manifest(contract)
    client = native.FssClientRequest(contract, bytes([9]) * 32, (manifest,) * 3, [0])
    roles = [native.FssRoleRequest(program, contract, client.request_bytes(), role) for role in range(3)]
    peers = [queue.Queue(maxsize=1), queue.Queue(maxsize=1)]
    cancelled = threading.Event()
    counts = [0, 0]
    admissions = [json.loads(role.metadata()) for role in roles]

    class Tape:
        """Bounded private temporary TEST tape, NOT authenticated encryption.

        Production tape authentication/encryption, disk/CPU charges and holder
        acknowledgements belong to the parent SDK, not this bridge test.
        """
        def __init__(self, role):
            self.role, self.records, self.consumed, self.sealed = role, 0, 0, False
            self.file = tempfile.TemporaryFile(mode="w+b", buffering=0, dir=tmp_path)

        def deliver(self, meta, packet):
            assert not self.sealed
            assert len(packet) <= admissions[self.role]["largest_packet_bytes"]
            for value in (struct.pack("<II", len(meta), len(packet)), meta, packet):
                assert self.file.write(value) == len(value)
            self.records += 1
            bound = admissions[self.role]["cumulative_packet_bytes_per_party_bound"]
            bound += admissions[self.role]["primitive_requests_per_party"] * (1024 + 8)
            assert self.file.tell() <= bound

        def seal(self):
            assert self.records == admissions[self.role]["primitive_requests_per_party"]
            self.file.seek(0)
            self.sealed = True

        def material(self, meta):
            assert self.sealed and self.consumed < self.records
            header = self.file.read(8)
            assert len(header) == 8
            meta_len, packet_len = struct.unpack("<II", header)
            assert 0 < meta_len <= 1024
            assert 0 < packet_len <= admissions[self.role]["largest_packet_bytes"]
            assert self.file.read(meta_len) == meta
            packet = self.file.read(packet_len)
            assert len(packet) == packet_len
            self.consumed += 1
            return packet

        def finish(self):
            assert self.consumed == self.records and not self.file.read(1)
            self.file.close()

    tapes = [Tape(0), Tape(1)]

    def put(q, value):
        deadline = time.monotonic() + 120
        while not cancelled.is_set() and time.monotonic() < deadline:
            try:
                q.put(value, timeout=0.05)
                return
            except queue.Full:
                pass
        raise RuntimeError("bounded transport cancelled")

    def get(q):
        deadline = time.monotonic() + 120
        while not cancelled.is_set() and time.monotonic() < deadline:
            try:
                value = q.get(timeout=0.05)
                q.task_done()
                return value
            except queue.Empty:
                pass
        raise RuntimeError("bounded transport cancelled")

    class Dealer:
        batches = 0
        cross_plan = False

        def deliver(self, meta, packet):
            role = json.loads(meta)["role"]
            value = parse_meta(meta, "material", role)
            assert role in (0, 1) and len(packet) <= value["packet_bytes_bound"]
            tapes[role].deliver(meta, packet)

        def deliver_batch(self, role, records):
            assert material_delivery == "batch-v1"
            assert type(role) is int and role in (0, 1)
            assert type(records) is tuple and 1 <= len(records) <= 64
            assert 1024 + sum(8 + len(meta) + len(packet) for meta, packet in records) <= 1 << 20
            self.batches += 1
            self.cross_plan |= (json.loads(records[0][0])["kernel"]["ordinal"]
                                != json.loads(records[-1][0])["kernel"]["ordinal"])
            for record in records:
                assert type(record) is tuple and len(record) == 2
                assert json.loads(record[0])["role"] == role
                self.deliver(*record)

        def finish(self):
            for tape in tapes:
                tape.seal()

        def cancel(self):
            cancelled.set()

    class Holder:
        def __init__(self, role):
            self.role = role

        def material(self, meta):
            parse_meta(meta, "material", self.role)
            packet = tapes[self.role].material(meta)
            counts[self.role] += 1
            return packet

        def exchange(self, meta, frame):
            parse_meta(meta, "peer", self.role)
            put(peers[self.role], (meta, frame))
            other_meta, other_frame = get(peers[1 - self.role])
            expected = json.loads(meta)
            expected["role"] = 1 - self.role
            assert json.loads(other_meta) == expected
            return other_frame

        def finish(self):
            tapes[self.role].finish()

        def cancel(self):
            cancelled.set()

    try:
        # Strict offline barrier: dealer completes and BOTH recipient tapes are
        # sealed/acknowledged before any holder input or peer opening is sent.
        dealer = Dealer()
        assert roles[2].run(None, dealer) is None
        assert (dealer.batches > 0) == (material_delivery == "batch-v1")
        if material_delivery == "batch-v1":
            assert dealer.cross_plan
        assert all(tape.sealed for tape in tapes)
        inputs = [client.take_input(0), client.take_input(1)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            left = pool.submit(roles[0].run, inputs[0], Holder(0))
            right = pool.submit(roles[1].run, inputs[1], Holder(1))
            try:
                response0, response1 = left.result(timeout=180), right.result(timeout=180)
                assert client.finish(response0, response1) == [0]
                assert counts == [admissions[0]["primitive_requests_per_party"]] * 2
                assert all(q.empty() for q in peers)
            finally:
                cancelled.set()
                for role in roles:
                    role.cancel()
    finally:
        client.close()
        # Drain exported immutable Python buffers before the test's reservation
        # scope ends. Production HTTP lifecycle owns the corresponding leases.
        for q in peers:
            while not q.empty():
                q.get_nowait()
                q.task_done()
        for tape in tapes:
            tape.file.close()

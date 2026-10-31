"""Final-only FSS client; public setup completes before either input is sent."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import secrets
import time

from pllm import _native
from pllm.deployment.resources import reserve_incremental_memory

from .client import ProtocolError
from .delegated_transport import _Peers, _connections
from .fss_protocol import (
    ADMISSION_LIMIT, CLIENT_MEMORY, CONTROL_LIMIT, HOLDERS, PEER_COUNTERS, ROLES,
    TRANSPORT, clear_failure_frames, collect_peer_bytes, digest, pack, unpack,
)
from .fss_tape import (MaterialTape, TapeResources, capacity_limits, DISK_HEADROOM_BYTES,
                       MAX_METADATA_BYTES, MAX_PACKET_BYTES)
from . import fss_protocol as p


DECODER_TIMEOUT_LIMIT = 24 * 60 * 60
MAX_GUARD_DISK_BYTES = 32 << 30


def _contract(contract):
    if type(contract) is bytes:
        if not 0 < len(contract) <= 1 << 20:
            raise ValueError("FuseFSS source contract exceeds bounds")
        contract = json.loads(contract)
    if type(contract) is not dict:
        raise ValueError("FuseFSS source contract differs")
    return contract


def transport_capacity(contract):
    """Read a locally recomputed source contract; this is not authentication."""
    capacity = _contract(contract).get("capacity", "capsule")
    capacity_limits(capacity)
    return capacity


def transport_material_delivery(contract):
    return p.material_delivery(_contract(contract).get("material_delivery", "single"))


def timeout_limit(capacity):
    capacity_limits(capacity)
    return DECODER_TIMEOUT_LIMIT if capacity == "decoder" else 600


def client_memory_bytes(capacity):
    """Bounded outer owner; decoder adds the shared 64MiB tokenizer allowance.

    Native request/primitive limits remain unchanged. The extra Python/tokenizer
    ownership is charged through the ordinary process-wide memory reservation.
    """
    from .delegated_source import _DECODER_TOKENIZER_BYTES
    capacity_limits(capacity)
    return CLIENT_MEMORY + (4 * _DECODER_TOKENIZER_BYTES if capacity == "decoder" else 0)


@dataclass(frozen=True)
class TransportResources:
    """Pure price for a caller-owned reservation, never an admission certificate.

    Source files are priced separately by the caller from the native source
    contract. The guard is a separate native owner, not part of the opaque tape.
    No source payload, material, record index or per-record metadata is retained.
    """

    contract_digest: str
    tape: TapeResources
    guard_disk_bytes: int
    guard_reservation_bytes: int
    combined_reservation_bytes: int
    material_delivery: str = "single"
    python_batch_reservation_bytes: int = 0
    native_batch_reservation_bytes: int = 0
    python_receive_reservation_bytes: int = 0
    native_receive_reservation_bytes: int = 0

    @property
    def transport_disk_bytes_per_holder(self):
        return self.tape.max_file_bytes + self.guard_disk_bytes

    @property
    def required_disk_free_bytes_per_holder(self):
        return self.transport_disk_bytes_per_holder + DISK_HEADROOM_BYTES


def _tape_metadata_storage(capacity):
    """Local capacity policy shared by pricing and completion accounting."""
    capacity_limits(capacity)
    return "external_aad" if capacity == "decoder" else "inline"


def price_transport(admission, contract: bytes) -> TransportResources:
    """Price native-derived bounds without reading source or creating any file.

    Exact transport counts exist only after native import/range admission. The
    worker prices its own native role's metadata and reserves disk before tape
    or guard creation. Source storage has its earlier, separate native contract
    and reservation. A remote/public price cannot authorize execution.
    Decoder opts into v2 external-metadata AAD; Capsule keeps inline/v1. This
    local storage policy never follows a remote or native metadata-storage hint.
    """
    if type(contract) is not bytes or type(admission) is not dict:
        raise ValueError("FuseFSS transport pricing contract differs")
    capacity = transport_capacity(contract)
    if admission.get("capacity", "capsule") != capacity:
        raise ValueError("FuseFSS native transport capacity differs")
    def integer(name, maximum, default=None):
        value = admission.get(name, default)
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError("FuseFSS transport resource bound differs")
        return value
    delivery = transport_material_delivery(contract)
    if p.material_delivery(admission.get("material_delivery", "single")) != delivery:
        raise ValueError("FuseFSS native material delivery differs")
    combined = integer("combined_reservation_bytes", 64 << 20)
    native_batch = python_batch = 0
    if delivery == "batch-v1":
        batch = admission.get("delivery_batch")
        if type(batch) is not dict or set(batch) != {
                "records_bound", "body_bytes_bound", "envelope_bytes_bound",
                "record_overhead_bytes", "native_reservation_bytes", "python_overlap_bytes"}:
            raise ValueError("FuseFSS native material batch pricing differs")
        for name, expected in (("records_bound", p.MATERIAL_BATCH_RECORDS),
                               ("body_bytes_bound", p.MATERIAL_BATCH_BYTES),
                               ("envelope_bytes_bound", p.MATERIAL_BATCH_ENVELOPE_BYTES),
                               ("record_overhead_bytes", 8),
                               ("python_overlap_bytes", p.MATERIAL_BATCH_PYTHON_BYTES)):
            if type(batch[name]) is not int or batch[name] != expected:
                raise ValueError("FuseFSS native material batch bound differs")
        native_batch = batch["native_reservation_bytes"]
        if type(native_batch) is not int or not 2 * p.MATERIAL_BATCH_BYTES <= native_batch <= combined:
            raise ValueError("FuseFSS native material batch reservation differs")
        python_batch = p.MATERIAL_BATCH_PYTHON_BYTES
    elif "delivery_batch" in admission:
        raise ValueError("FuseFSS unexpected material batch pricing")
    # Online receive bytes are independent of OFFLINE delivery. Native capacity
    # pays native window only; Python copies stay inside the unchanged role owner.
    from .fss_tape import MATERIAL_RECEIVE_BYTES, MATERIAL_RECEIVE_RECORDS, MATERIAL_RECEIVE_PYTHON_BYTES
    receive = admission.get("material_receive")
    python_receive = native_receive = 0
    expects_receive = _contract(contract).get("online_schedule") in ("wavefront-v1", "tensor-v1")
    if expects_receive:
        if type(receive) is not dict or set(receive) != {"records_bound", "payload_bytes_bound",
                "native_reservation_bytes_per_holder", "python_overlap_bytes_per_holder"}:
            raise ValueError("FuseFSS native material receive pricing differs")
        for name, expected in (("records_bound", MATERIAL_RECEIVE_RECORDS),
                               ("payload_bytes_bound", MATERIAL_RECEIVE_BYTES),
                               ("python_overlap_bytes_per_holder", MATERIAL_RECEIVE_PYTHON_BYTES)):
            if type(receive[name]) is not int or receive[name] != expected:
                raise ValueError("FuseFSS native material receive bound differs")
        native_receive = receive["native_reservation_bytes_per_holder"]
        if type(native_receive) is not int or not MATERIAL_RECEIVE_BYTES <= native_receive <= combined // 2:
            raise ValueError("FuseFSS native material receive reservation differs")
        python_receive = MATERIAL_RECEIVE_PYTHON_BYTES
    elif "material_receive" in admission:
        raise ValueError("FuseFSS unexpected material receive pricing")
    metadata = integer("callback_metadata_bytes_bound", MAX_METADATA_BYTES)
    if metadata == 0:
        raise ValueError("FuseFSS callback metadata bound differs")
    tape = MaterialTape.price(records=admission.get("primitive_requests_per_party"),
        packet_bytes=admission.get("largest_packet_bytes"),
        total_packet_bytes=admission.get("cumulative_packet_bytes_per_party_bound"),
        capacity=capacity, metadata_bytes_per_record=metadata if capacity == "decoder" else MAX_METADATA_BYTES,
        metadata_storage=_tape_metadata_storage(capacity))
    default = 0 if capacity == "capsule" else None
    return TransportResources(hashlib.sha256(contract).hexdigest(), tape,
        integer("guard_disk_bytes", MAX_GUARD_DISK_BYTES, default),
        integer("guard_reservation_bytes", 64 << 20, default),
        combined, delivery, python_batch, native_batch, python_receive, native_receive)


@dataclass(frozen=True, slots=True)
class FssResult:
    selected: tuple[int, ...]


def _ack(peers, index, action, expected, payload=b"", *, phase="setup", limit=CONTROL_LIMIT):
    value = unpack(peers.post(index, action, payload, phase=phase, limit=limit), limit=limit)
    if (type(value) is not dict or set(value) != set(expected)
            or any(type(value[key]) is not type(want) or value[key] != want
                   for key, want in expected.items())):
        raise ProtocolError("FuseFSS acknowledgement differs")


def _receipt(raw, role, *, capacity="capsule"):
    from .fss_protocol import FINAL_FRAME_LIMIT

    packet_limit, file_limit = capacity_limits(capacity)
    value = unpack(raw, limit=CONTROL_LIMIT)
    if type(value) is not dict or set(value) != {"frame", "peer_bytes", "storage"}:
        raise ProtocolError("FuseFSS completion envelope differs")
    frame = value["frame"]
    if type(frame) is not bytes or (frame != b"" if role == "dealer" else not 0 < len(frame) <= FINAL_FRAME_LIMIT):
        raise ProtocolError("FuseFSS completion frame exceeds its bound")
    counts = collect_peer_bytes(value["peer_bytes"], role)
    storage = value["storage"]
    if type(storage) is not dict:
        raise ProtocolError("FuseFSS storage receipt differs")
    if role == "dealer":
        if storage:
            raise ProtocolError("FuseFSS dealer cannot own a holder tape")
    else:
        required = {"state", "encrypted_file_bytes_written", "encrypted_file_bytes_read",
                    "packet_bytes_written", "packet_bytes_read", "records_written", "records_read",
                    "sealed_file_bytes", "sealed_digest"}
        if (set(storage) != required or storage["state"] != "finished"
                or any(type(storage[k]) is not int or not 0 <= storage[k] <= file_limit
                       for k in required - {"state", "sealed_digest"})
                or storage["records_read"] != storage["records_written"]
                or storage["packet_bytes_read"] != storage["packet_bytes_written"]
                or storage["encrypted_file_bytes_read"] != storage["encrypted_file_bytes_written"]
                or storage["sealed_file_bytes"] != storage["encrypted_file_bytes_written"]):
            raise ProtocolError("FuseFSS complete tape accounting differs")
        records, packets, encrypted = (storage["records_read"], storage["packet_bytes_read"],
                                       storage["encrypted_file_bytes_read"])
        # The mode comes from the locally selected capacity, never the receipt.
        # V2 stores only the 24-byte header + 16-byte tag beyond the packet;
        # v1 additionally stores nonempty metadata. Neither mode permits empty
        # metadata INPUT; that bound and the full original AAD remain in the tape.
        metadata_min, metadata_max = ((0, 0) if _tape_metadata_storage(capacity) == "external_aad"
                                      else (1, MAX_METADATA_BYTES))
        if (records >= 1 << 32 or not records <= packets <= min(packet_limit, records * MAX_PACKET_BYTES)
                or not packets + records * (40 + metadata_min) <= encrypted <= packets + records * (40 + metadata_max)):
            raise ProtocolError("FuseFSS tape framing accounting differs")
        digest(storage["sealed_digest"])
    return frame, counts, storage


def _finish_receipts(peers, boundary, failure, **receipt_options):
    """Join both receipts; bound local drain, leaving stragglers to global close.

    Fixed ownership: two bounded CONTROL_LIMIT readers/results, one cancellation
    watcher, join/drain Futures and their fixed Task/result containers; caller
    interruption can add one drain-fence Task on this same owner. No new pool,
    thread, queue, retries or history. Validation is synchronous on that loop,
    so only one unpack runs at a time. At most four CONTROL_LIMIT collector/raw
    payload copies overlap, plus two validated small receipt packages. These
    remain in the existing HTTP-control phase envelope, not a new byte credit
    or a socket/TLS/allocator/stack/physical-memory proof. A fixed five-entry
    retirement-root set retains reads that outlast cooperative drain.
    """
    import asyncio
    from .delegated_transport import _IO_DRAIN_TIMEOUT

    # One response-local I/O owner admits exactly one finish collector. Never
    # replace the root set while a previous collector/callback still owns it.
    if hasattr(peers.io, "_fss_finish_pending"):
        raise ProtocolError("FuseFSS finish collection is one-use")
    first_error = owned_task = None
    submitted = False
    # all_tasks() is weak: explicitly retain the fixed set until completion or
    # the existing I/O owner's global shutdown cancels and joins it. Callbacks
    # capture only this set, never a native owner or an attempt observer.
    pending = set()

    def settled(future):
        pending.discard(future)
        if not future.cancelled():
            future.exception()  # Retrieve late errors without replacing first_error.

    def track(future):
        if len(pending) >= 5:
            raise ProtocolError("FuseFSS finish retirement exceeds its fixed task bound")
        pending.add(future)
        future.add_done_callback(settled)
        return future

    def failed(error, role=None):
        nonlocal first_error
        if first_error is None:
            first_error = error
            if role is not None:
                # Another outstanding call may have written the one-slot marker.
                boundary(role, "finish", "entered")
            failure()

    async def receive(index, role):
        raw = receipt = None
        try:
            if first_error is not None:
                raise asyncio.CancelledError
            peers.check()
            boundary(role, "finish", "entered")
            raw = await peers._post(index, "finish", b"", phase="online", limit=CONTROL_LIMIT,
                                    timeout=max(0.0, peers.deadline - time.monotonic()))
            if first_error is not None:
                raise asyncio.CancelledError
            receipt = _receipt(raw, role, **receipt_options)
            peers.live[index] = False
            boundary(role, "finish", "validated")
            return receipt
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            failed(error, role)
            raise
        finally:
            raw = receipt = None

    async def cancellation():
        try:
            while True:
                peers.check()
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            failed(error)
            raise

    async def owned():
        nonlocal owned_task
        # Authoritative admission runs without an await on the sole I/O loop;
        # concurrent callers cannot replace a previous collector's root set.
        if hasattr(peers.io, "_fss_finish_pending"):
            raise ProtocolError("FuseFSS finish collection is one-use")
        peers.io._fss_finish_pending = pending
        owned_task = track(asyncio.current_task())
        tasks = []
        watcher = joined = None
        try:
            async with asyncio.timeout(max(0.0, peers.deadline - time.monotonic())):
                for index, role in enumerate(HOLDERS):
                    tasks.append(track(asyncio.create_task(receive(index, role))))
                watcher = track(asyncio.create_task(cancellation()))
                joined = track(asyncio.gather(*tasks))
                await asyncio.wait((joined, watcher), return_when=asyncio.FIRST_COMPLETED)
                if first_error is not None:
                    raise first_error
                if watcher.done():
                    await watcher
                receipts = await joined
                peers.check()
                return receipts
        except TimeoutError:
            error = ProtocolError("GKM request cancelled or timed out", 499)
            failed(error)
            raise error from None
        except BaseException as error:
            failed(error)
            raise
        finally:
            # Cancel every read BEFORE awaiting any one of them. In particular,
            # a failed B must never leave us waiting for A's peer-frame barrier.
            for task in tasks:
                task.cancel()
            if watcher is not None:
                watcher.cancel()
            draining = [*tasks, *(() if watcher is None else (watcher,)),
                        *(() if joined is None else (joined,))]
            if draining:
                # Unlike timeout(gather(...)), wait does not wait for children
                # to acknowledge cancellation after its timeout. Leave any
                # resistant read strongly owned for peers.close()/io.close().
                # first_error already bars late validation and native access.
                await asyncio.wait(draining, timeout=_IO_DRAIN_TIMEOUT)

    try:
        if peers.closed:
            raise ProtocolError("GKM worker channels are retired")
        peers.check()
        submitted = True
        return peers.io.run(owned(), timeout=peers.deadline - time.monotonic())
    except BaseException as error:
        # Same failed-online accounting as _Peers.post; bypass only its serial
        # dispatch wrapper, not the bounded HTTP reader or audit counters.
        peers.calls_known = peers.server_known = False
        failed(error)
        if submitted and not peers.io.closed and (owned_task is None or not owned_task.done()):
            # _IOOwner.run cancels its Future on a caller-side interruption or
            # dispatch timeout. Fence that asynchronous cancellation before the
            # native owner is closed. Use the existing I/O drain interval, not
            # a renewed request deadline or another five-second peer lease.
            async def drained():
                await asyncio.sleep(0)
                if owned_task is not None:
                    await asyncio.wait((owned_task,), timeout=_IO_DRAIN_TIMEOUT)

            try:
                peers.io.run(drained(), timeout=_IO_DRAIN_TIMEOUT)
            except BaseException:
                # Preserve the initiating error. The existing peers.close()
                # finally remains responsible for global pool/thread retirement.
                pass
        raise


def execute_request(experiment, source, ids, outputs, connections, timeout, attempt, *, completed_program=None):
    """Return fixed-count token IDs after both complete one-use holder responses."""
    capacity = transport_capacity(source.contract)
    delivery = transport_material_delivery(source.contract)
    configured_delivery = p.material_delivery(experiment.pipeline.components["linear"].params.get("material_delivery", "single"))
    if delivery != configured_delivery:
        raise ValueError("FuseFSS native source material delivery differs")
    if (type(timeout) not in (float, int) or not math.isfinite(timeout) or not 0 < timeout <= timeout_limit(capacity)
            or type(ids) not in (tuple, list) or not 1 <= len(ids) <= 16
            or any(type(i) is not int or not 0 <= i < source.contract["vocabulary"] for i in ids)
            or type(outputs) is not int or not 1 <= outputs <= 4 or not callable(attempt)):
        raise ValueError("FuseFSS request exceeds its bounded transport contract")
    # Validate origins/credentials before creating a native input-share owner.
    connections = dict(zip(ROLES, _connections(connections, roles=ROLES), strict=True))
    deadline = time.monotonic() + timeout
    peers = native = manifests = frame = frames = raw = value = storage = payload = None
    reservation = reserve_incremental_memory(client_memory_bytes(capacity))
    with reservation:
        def boundary(role, action, status):
            try:
                record = getattr(attempt, "record_control_boundary", None)
                if record is not None:
                    record(role, action, status)
            except BaseException:
                pass

        def failure():
            try:
                observe = getattr(attempt, "fail_control_boundary", None)
                if observe is not None:
                    observe()
            except BaseException:
                pass

        try:
            if attempt():
                raise ProtocolError("FuseFSS request cancelled", 499)
            boundary("client", "contract", "entered")
            contract = source.runtime_contract(experiment, len(ids), outputs)
            if transport_capacity(contract) != capacity or transport_material_delivery(contract) != delivery:
                raise ValueError("FuseFSS runtime capacity differs from source")
            boundary("client", "contract", "validated")
            receipt_options = {} if capacity == "capsule" else {"capacity": capacity}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("FuseFSS request timed out", 499)
            boundary("client", "channels", "entered")
            peers = _Peers()
            peers.start(connections, remaining, attempt, roles=ROLES, namespace="fss")
            # Channel construction also spends the original request budget.
            peers.deadline = min(peers.deadline, deadline)
            peers.request_id = secrets.token_hex(32)
            boundary("client", "channels", "validated")
            epochs = {}
            for index, role in enumerate(ROLES):
                boundary(role, "manifest", "entered")
                value = unpack(peers.post(index, "manifest", phase="setup", limit=CONTROL_LIMIT),
                               limit=CONTROL_LIMIT)
                expected = {"schema": "pllm.fss.worker.v1", "role": role,
                            "configuration": experiment.configuration_digest(),
                            "source_lock": source.source_lock_digest, "tokenizer": source.tokenizer_digest,
                            "transport": TRANSPORT}
                if (type(value) is not dict or set(value) != set(expected) | {"epoch"}
                        or any(value[key] != expected[key] for key in expected)):
                    raise ProtocolError("FuseFSS worker source or transport differs")
                epochs[role] = digest(value["epoch"])
                boundary(role, "manifest", "validated")
            manifests = []
            for index, role in enumerate(ROLES):
                # Lost or malformed acknowledgements still require retirement.
                peers.live[index] = True
                admission = {"request": peers.request_id, "epoch": epochs[role], "epochs": epochs,
                             "rows": len(ids), "outputs": outputs, "contract": contract}
                if capacity == "decoder":
                    peers.check()
                    remaining = min(timeout, peers.deadline - time.monotonic())
                    if remaining <= 0:
                        raise ProtocolError("FuseFSS request timed out", 499)
                    # Recompute after each earlier role's source import. This is
                    # a duration on authenticated client control, not a remote
                    # monotonic timestamp or a renewable peer-traffic lease.
                    admission["remaining_seconds"] = remaining
                payload = pack(admission, limit=ADMISSION_LIMIT)
                boundary(role, "admit", "entered")
                value = unpack(peers.post(index, "admit", payload, phase="setup", limit=CONTROL_LIMIT),
                               limit=CONTROL_LIMIT)
                if (type(value) is not dict or set(value) != {"manifest"}
                        or type(value["manifest"]) is not bytes or not 0 < len(value["manifest"]) <= 1024):
                    raise ProtocolError("FuseFSS native manifest differs")
                manifests.append(value["manifest"])
                boundary(role, "admit", "validated")
            boundary("client", "client_request", "entered")
            native = _native.FssClientRequest(contract, bytes.fromhex(peers.request_id), tuple(manifests), ids)
            payload = pack({"request_frame": native.request_bytes()}, limit=CONTROL_LIMIT)
            boundary("client", "client_request", "validated")
            for index in range(3):
                boundary(ROLES[index], "bind", "entered")
                _ack(peers, index, "bind", {"accepted": True}, payload)
                boundary(ROLES[index], "bind", "validated")
            boundary("dealer", "prepare", "entered")
            _ack(peers, 2, "prepare", {"prepared": True})
            boundary("dealer", "prepare", "validated")
            # Dealer retires before input release; final-only token shares never
            # flow through it. Its direct setup counters remain charged.
            boundary("dealer", "finish", "entered")
            raw = peers.post(2, "finish", phase="setup", limit=CONTROL_LIMIT)
            _, peer_counts, _ = _receipt(raw, "dealer", **receipt_options)
            peers.live[2] = False
            boundary("dealer", "finish", "validated")
            for index in range(2):
                boundary("client", "input_share", "entered")
                frame = native.take_input(index)
                boundary("client", "input_share", "validated")
                boundary(ROLES[index], "input", "entered")
                _ack(peers, index, "input", {"accepted": True}, frame, phase="online")
                boundary(ROLES[index], "input", "validated")
                frame = None
            for index in range(2):
                boundary(ROLES[index], "start", "entered")
                _ack(peers, index, "start", {"started": True}, phase="online")
                boundary(ROLES[index], "start", "validated")
            frames = []
            for role, (frame, counts, storage) in zip(
                    HOLDERS, _finish_receipts(peers, boundary, failure, **receipt_options), strict=True):
                frames.append(frame)
                for key, count in counts.items():
                    peer_counts[key] += count
                peers.counts[f"fss_storage.{role}.write_bytes"] = storage["encrypted_file_bytes_written"]
                peers.counts[f"fss_storage.{role}.read_bytes"] = storage["encrypted_file_bytes_read"]
            boundary("client", "reconstruct", "entered")
            selected = tuple(native.finish(*frames))
            if len(selected) != outputs:
                raise ProtocolError("FuseFSS incomplete final token result")
            boundary("client", "reconstruct", "validated")
            peers.counts.update(peer_counts)
            peers.check()
            if completed_program is not None:
                # Native admission checked all three manifests; only retain their
                # common public identity after both outputs have been accepted.
                completed_program(manifests[0])
            return FssResult(selected)
        except BaseException as error:
            failure()
            frame = frames = raw = value = storage = payload = None
            clear_failure_frames(error)
            raise
        finally:
            payload = None
            # A failed attempt's frozen marker cannot be overwritten by cleanup.
            boundary("client", "retire", "entered")
            frame = frames = raw = value = storage = manifests = None
            unretired = []  # bounded2: only owners whose close actually fails
            try:
                try:
                    if native is not None:
                        native.close()
                except BaseException:
                    unretired.append(native)
                    # Observe before the existing finally continues peer cleanup.
                    failure()
                    raise
                finally:
                    try:
                        if peers is not None:
                            try:
                                peers.close()
                            except BaseException:
                                unretired.append(peers)
                                raise
                            audit = peers.audit()
                            for key in ("generation_prefill_online_body_bytes", "generation_decode_online_body_bytes",
                                        "prefill_remote_linear_calls", "decode_remote_linear_calls",
                                        "prefill_remote_linear_server_seconds", "decode_remote_linear_server_seconds",
                                        "online_steps", "inference_stage_calls", "inference_server_ns"):
                                audit.pop(key, None)
                            # Missing directed receipts do not become zero-cost links.
                            if not all(key in peers.counts for key in PEER_COUNTERS):
                                for key in PEER_COUNTERS:
                                    audit.pop(key, None)
                            record = getattr(attempt, "record_audit", None)
                            if record is not None:
                                record(audit)
                    except BaseException:
                        failure()
                        raise
            except BaseException as error:
                if unretired:
                    # Root the SAME charged token before optional client-seat
                    # handoff. Context exit cannot refund failed retirement.
                    reservation.retain_failed(tuple(unretired))
                    try:
                        attempt.owner._retain_failed_request_reservation(attempt, reservation)
                    except BaseException as handoff_error:
                        # Diagnostics are optional, unlike same-token retention.
                        # Never iterate foreign note storage or replace cleanup
                        # precedence with a failing attribute/note implementation.
                        try:
                            notes = getattr(error, "__notes__", None)
                            if notes is None or (type(notes) is list and all(type(note) is str for note in notes)):
                                error.add_note(f"FuseFSS cleanup handoff failed: {type(handoff_error).__name__}")
                        except BaseException:
                            pass
                raise
            finally:
                native = peers = None
            boundary("client", "retire", "validated")

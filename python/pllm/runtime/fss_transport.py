"""Final-only FSS client; public setup completes before either input is sent."""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class FssResult:
    selected: tuple[int, ...]


def _ack(peers, index, action, expected, payload=b"", *, phase="setup", limit=CONTROL_LIMIT):
    value = unpack(peers.post(index, action, payload, phase=phase, limit=limit), limit=limit)
    if (type(value) is not dict or set(value) != set(expected)
            or any(type(value[key]) is not type(want) or value[key] != want
                   for key, want in expected.items())):
        raise ProtocolError("FuseFSS acknowledgement differs")


def _receipt(raw, role):
    value = unpack(raw, limit=CONTROL_LIMIT)
    if type(value) is not dict or set(value) != {"frame", "peer_bytes", "storage"}:
        raise ProtocolError("FuseFSS completion envelope differs")
    frame = value["frame"]
    if type(frame) is not bytes or (frame != b"" if role == "dealer" else not 0 < len(frame) <= 176):
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
                or any(type(storage[k]) is not int or not 0 <= storage[k] <= 1 << 30
                       for k in required - {"state", "sealed_digest"})
                or storage["records_read"] != storage["records_written"]
                or storage["packet_bytes_read"] != storage["packet_bytes_written"]
                or storage["encrypted_file_bytes_read"] != storage["encrypted_file_bytes_written"]
                or storage["sealed_file_bytes"] != storage["encrypted_file_bytes_written"]):
            raise ProtocolError("FuseFSS complete tape accounting differs")
        digest(storage["sealed_digest"])
    return frame, counts, storage


def execute_request(experiment, source, ids, outputs, connections, timeout, attempt, *, completed_program=None):
    """Return fixed-count token IDs after both complete one-use holder responses."""
    if (type(timeout) not in (float, int) or not math.isfinite(timeout) or not 0 < timeout <= 600
            or type(ids) not in (tuple, list) or not 1 <= len(ids) <= 16
            or any(type(i) is not int or not 0 <= i < source.contract["vocabulary"] for i in ids)
            or type(outputs) is not int or not 1 <= outputs <= 4 or not callable(attempt)):
        raise ValueError("FuseFSS request exceeds its bounded transport contract")
    # Validate origins/credentials before creating a native input-share owner.
    connections = dict(zip(ROLES, _connections(connections, roles=ROLES), strict=True))
    deadline = time.monotonic() + timeout
    peers = native = manifests = frame = frames = raw = value = storage = None
    with reserve_incremental_memory(CLIENT_MEMORY):
        try:
            if attempt():
                raise ProtocolError("FuseFSS request cancelled", 499)
            contract = source.runtime_contract(experiment, len(ids), outputs)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("FuseFSS request timed out", 499)
            peers = _Peers(connections, remaining, attempt, roles=ROLES, namespace="fss")
            peers.request_id = secrets.token_hex(32)
            epochs = {}
            for index, role in enumerate(ROLES):
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
            manifests = []
            for index, role in enumerate(ROLES):
                # Lost or malformed acknowledgements still require retirement.
                peers.live[index] = True
                payload = pack({"request": peers.request_id, "epoch": epochs[role], "epochs": epochs,
                                "rows": len(ids), "outputs": outputs, "contract": contract},
                               limit=ADMISSION_LIMIT)
                value = unpack(peers.post(index, "admit", payload, phase="setup", limit=CONTROL_LIMIT),
                               limit=CONTROL_LIMIT)
                if (type(value) is not dict or set(value) != {"manifest"}
                        or type(value["manifest"]) is not bytes or not 0 < len(value["manifest"]) <= 1024):
                    raise ProtocolError("FuseFSS native manifest differs")
                manifests.append(value["manifest"])
            native = _native.FssClientRequest(contract, bytes.fromhex(peers.request_id), tuple(manifests), ids)
            payload = pack({"request_frame": native.request_bytes()}, limit=CONTROL_LIMIT)
            for index in range(3):
                _ack(peers, index, "bind", {"accepted": True}, payload)
            _ack(peers, 2, "prepare", {"prepared": True})
            # Dealer retires before input release; final-only token shares never
            # flow through it. Its direct setup counters remain charged.
            raw = peers.post(2, "finish", phase="setup", limit=CONTROL_LIMIT)
            _, peer_counts, _ = _receipt(raw, "dealer")
            peers.live[2] = False
            for index in range(2):
                frame = native.take_input(index)
                _ack(peers, index, "input", {"accepted": True}, frame, phase="online")
                frame = None
            for index in range(2):
                _ack(peers, index, "start", {"started": True}, phase="online")
            frames = []
            for index, role in enumerate(HOLDERS):
                raw = peers.post(index, "finish", phase="online", limit=CONTROL_LIMIT)
                frame, counts, storage = _receipt(raw, role)
                frames.append(frame)
                for key, count in counts.items():
                    peer_counts[key] += count
                peers.counts[f"fss_storage.{role}.write_bytes"] = storage["encrypted_file_bytes_written"]
                peers.counts[f"fss_storage.{role}.read_bytes"] = storage["encrypted_file_bytes_read"]
                peers.live[index] = False
            selected = tuple(native.finish(*frames))
            if len(selected) != outputs:
                raise ProtocolError("FuseFSS incomplete final token result")
            peers.counts.update(peer_counts)
            peers.check()
            if completed_program is not None:
                # Native admission checked all three manifests; only retain their
                # common public identity after both outputs have been accepted.
                completed_program(manifests[0])
            return FssResult(selected)
        except BaseException as error:
            frame = frames = raw = value = storage = None
            clear_failure_frames(error)
            raise
        finally:
            frame = frames = raw = value = storage = manifests = None
            try:
                if native is not None:
                    native.close()
            finally:
                if peers is not None:
                    peers.close()
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
                native = peers = None

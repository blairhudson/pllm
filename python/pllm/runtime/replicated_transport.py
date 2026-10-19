"""Three authenticated input owners, direct peer work, and final-only disclosure."""
from dataclasses import dataclass
import json

from pllm.deployment.resources import reserve_incremental_memory
from .client import ProtocolError
from .delegated_protocol import clear_failure_frames, digest, pack, unpack
from .delegated_transport import _Peers
from .replicated_protocol import ROLES, FRAME_LIMIT, ADMISSION_LIMIT, REQUEST_MEMORY, PEER_COUNTERS
from .replicated_protocol import execution_policy
from .replicated_peer import TRANSPORT


@dataclass(frozen=True)
class ReplicatedResult:
    selected: tuple[int, ...]


def execute_request(experiment, source, ids, outputs, connections, timeout, attempt):
    reservation = reserve_incremental_memory(REQUEST_MEMORY)
    reservation.__enter__()
    peers = publisher = result = manifest = chunk = shares = frame = frames = raw = value = None
    try:
        peers = _Peers(connections, timeout, attempt, roles=ROLES, namespace="replicated")
        publisher, manifest = source.publisher(experiment, ids, outputs, attempt)
        peers.request_id = digest(bytes(json.loads(manifest)["session"]).hex())
        streams = execution_policy(experiment)["peer_transport"] == "streams"
        epochs = {}
        for index, role in enumerate(ROLES):
            advertised = unpack(peers.post(index, "manifest", phase="setup", limit=4096))
            expected = {"schema": "pllm.replicated.worker.v1", "role": role,
                "configuration": experiment.configuration_digest(), "source_lock": source.source_lock_digest,
                "tokenizer": source.tokenizer_digest}
            if streams:
                expected["peer_transport"] = TRANSPORT
            if (set(advertised) != set(expected) | {"epoch"}
                    or any(advertised[key] != value for key, value in expected.items())):
                raise ProtocolError("replicated worker manifest differs")
            epochs[role] = digest(advertised["epoch"])
            peers.live[index] = True
            raw = peers.post(index, "admit", pack({"request": peers.request_id,
                "epoch": digest(advertised["epoch"]), "manifest": manifest}, limit=ADMISSION_LIMIT),
                phase="setup", limit=4096)
            if unpack(raw) != {"accepted": True}:
                raise ProtocolError("replicated admission acknowledgement differs")
        if streams:
            for index in range(3):
                raw = peers.post(index, "bind_peers", pack({"transport": TRANSPORT, "epochs": epochs}),
                                 phase="setup", limit=4096)
                if unpack(raw) != {"accepted": True}:
                    raise ProtocolError("replicated peer binding acknowledgement differs")
        for action in ("setup_public", "setup_dispatch", "setup_seed", "setup_dispatch", "activate"):
            for index in range(3):
                if unpack(peers.post(index, action, phase="setup", limit=4096)) != {"accepted": True}:
                    raise ProtocolError("replicated setup acknowledgement differs")
        while (chunk := publisher.next_chunk()) is not None:
            key, offset, shares = chunk
            for index, frame in enumerate(shares):
                raw = peers.post(index, "input", pack({"key": key, "offset": offset, "frame": frame}, limit=FRAME_LIMIT),
                                 phase="setup", limit=4096)
                if unpack(raw) != {"accepted": True}:
                    raise ProtocolError("replicated input acknowledgement differs")
            chunk = shares = frame = None
        if unpack(peers.post(0, "execute", phase="online", limit=4096)) != {"complete": True}:
            raise ProtocolError("replicated completion barrier differs")
        frames = []
        peer_counts = dict.fromkeys(PEER_COUNTERS, 0)
        for index, role in enumerate(ROLES):
            value = unpack(peers.post(index, "finish", phase="online", limit=FRAME_LIMIT), limit=FRAME_LIMIT)
            if set(value) != {"frame", "peer_bytes"} or set(value["peer_bytes"]) != set(ROLES):
                raise ProtocolError("replicated final-share envelope differs")
            for peer, counters in value["peer_bytes"].items():
                expected = {f"{phase}_{direction}_bytes" for phase in ("setup", "online")
                            for direction in ("upload", "download")}
                if (type(counters) is not dict or set(counters) != expected
                        or any(type(v) is not int or not 0 <= v < 1 << 63 for v in counters.values())
                        or peer == role and any(counters.values())):
                    raise ProtocolError("replicated peer accounting differs")
                if peer != role:
                    for phase in ("setup", "online"):
                        peer_counts[f"peer_link.{role}.{peer}.{phase}_bytes"] += counters[f"{phase}_upload_bytes"]
                        peer_counts[f"peer_link.{peer}.{role}.{phase}_bytes"] += counters[f"{phase}_download_bytes"]
            frames.append(value["frame"])
            peers.live[index] = False
        peers.counts.update(peer_counts)
        result = ReplicatedResult(tuple(publisher.finish(*frames)))
        return result
    except BaseException as error:
        chunk = shares = frame = frames = raw = value = None
        clear_failure_frames(error)
        raise
    finally:
        chunk = shares = frame = frames = raw = value = None
        if publisher is not None:
            publisher.close()
        if peers is not None:
            peers.close()
            # Buffered final tokens cannot identify prefill/decode body windows.
            audit = peers.audit()
            for key in ("generation_prefill_online_body_bytes", "generation_decode_online_body_bytes",
                        "prefill_remote_linear_calls", "decode_remote_linear_calls",
                        "prefill_remote_linear_server_seconds", "decode_remote_linear_server_seconds"):
                audit.pop(key, None)
            for key in ("online_steps", "inference_stage_calls", "inference_server_ns"):
                audit.pop(key, None)
            record = getattr(attempt, "record_audit", None)
            if record is not None:
                record(audit)
        publisher = manifest = result = peers = None
        reservation.__exit__(None, None, None)

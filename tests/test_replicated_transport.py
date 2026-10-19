"""SDK handshake substitutions and final-only acceptance, with bounded mocks."""
import json
from types import SimpleNamespace

import pytest

from pllm.runtime import replicated_transport as transport
from pllm.runtime.delegated_protocol import pack, unpack
from pllm.runtime.replicated_protocol import ROLES, execution_policy
from pllm.runtime.replicated_peer import TRANSPORT
from test_replicated_execution_policy import experiment_for


@pytest.mark.parametrize("choice", ["http", "streams"])
@pytest.mark.parametrize("substitution", [None, "transport", "source", "configuration", "epoch", "binding_ack", "final_acceptance"])
def test_sdk_binds_authenticated_epochs_before_setup_and_retires(monkeypatch, choice, substitution):
    experiment = experiment_for(weights="public", peer_transport=choice)
    events, owners = [], []
    epochs = {role: str(index + 1) * 64 for index, role in enumerate(ROLES)}
    source = SimpleNamespace(source_lock_digest="b" * 64, tokenizer_digest="c" * 64)

    class Reservation:
        def __init__(self, size):
            assert size == transport.REQUEST_MEMORY  # sockets belong to workers

        def __enter__(self):
            events.append("reserved")

        def __exit__(self, *_):
            events.append("refunded")

    class Publisher:
        def next_chunk(self):
            return None

        def finish(self, *frames):
            events.append("native_acceptance")
            assert frames == (b"final0", b"final1", b"final2")
            if substitution == "final_acceptance":
                raise ValueError("aggregate checked validity rejected")
            return (4,)

        def close(self):
            events.append("publisher_closed")

    class Peers:
        def __init__(self, *args, **kwargs):
            self.live = [False] * 3
            self.counts = {}
            self.request_id = None
            self.admitted = self.bound = 0
            owners.append(self)

        def post(self, index, action, raw=b"", **kwargs):
            events.append((index, action))
            if action == "manifest":
                value = {"schema": "pllm.replicated.worker.v1", "role": ROLES[index],
                         "configuration": experiment.configuration_digest(),
                         "source_lock": source.source_lock_digest, "tokenizer": source.tokenizer_digest,
                         "epoch": epochs[ROLES[index]]}
                if choice == "streams":
                    value["peer_transport"] = TRANSPORT
                if substitution == "transport":
                    if choice == "streams":
                        value.pop("peer_transport")
                    else:
                        value["peer_transport"] = TRANSPORT
                elif substitution in {"source", "configuration", "epoch"}:
                    value[{"source": "source_lock"}.get(substitution, substitution)] = "wrong"
                return pack(value)
            if action == "admit":
                assert unpack(raw)["epoch"] == epochs[ROLES[index]]
                self.admitted += 1
            elif action == "bind_peers":
                assert choice == "streams" and self.admitted == 3
                assert unpack(raw) == {"transport": TRANSPORT, "epochs": epochs}
                self.bound += 1
                if substitution == "binding_ack":
                    return pack({"accepted": False})
            elif action == "execute":
                return pack({"complete": True})
            elif action == "finish":
                counters = {role: {f"{phase}_{direction}_bytes":
                    (0 if role == ROLES[index] else 1 if direction == "upload" else 2)
                    for phase in ("setup", "online") for direction in ("upload", "download")}
                    for role in ROLES}
                return pack({"frame": f"final{index}".encode(), "peer_bytes": counters})
            else:
                assert self.admitted == 3
                assert self.bound == (3 if choice == "streams" else 0)
            return pack({"accepted": True})

        def audit(self):
            return {}

        def close(self):
            events.append("peers_closed")

    source.publisher = lambda *args: (Publisher(), json.dumps({"session": [170] * 32}).encode())
    monkeypatch.setattr(transport, "reserve_incremental_memory", Reservation)
    monkeypatch.setattr(transport, "_Peers", Peers)
    should_fail = substitution is not None and not (substitution == "binding_ack" and choice == "http")
    if should_fail:
        with pytest.raises((ValueError, transport.ProtocolError)):
            transport.execute_request(experiment, source, (3,), 1, {}, 10, SimpleNamespace())
    else:
        result = transport.execute_request(experiment, source, (3,), 1, {}, 10, SimpleNamespace())
        assert result.selected == (4,)
        assert execution_policy(experiment)["peer_transport"] == choice
        assert all(value == 3 for value in owners[0].counts.values())
    assert events[-3:] == ["publisher_closed", "peers_closed", "refunded"]
    if substitution in {"transport", "source", "configuration", "epoch"}:
        assert "native_acceptance" not in events
        assert not any(event == (0, "execute") for event in events)

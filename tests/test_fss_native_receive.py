"""Generated native frozen groups + existing encrypted tape; not trained serving.

Run only with a rebuilt matching extension during parent serial validation.
"""
import concurrent.futures
import json
import queue
import threading
import time

import pytest

from pllm.runtime.fss_tape import MaterialTape
from pllm.runtime.fss_transport import price_transport
from test_fss_native_transport import native, parse_meta, program


def test_legacy_four_argument_scalar_constructor_keeps_admission_and_optional_window_is_explicit(program):
    source, contract = program
    manifest = source.manifest(contract)
    client = native.FssClientRequest(contract, bytes([60]) * 32, (manifest,) * 3, [0])
    scalar = native.FssRoleRequest(source, contract, client.request_bytes(), 0)
    window = native.FssRoleRequest(source, contract, client.request_bytes(), 0, material_window=True)
    try:
        a, b = json.loads(scalar.metadata()), json.loads(window.metadata())
        assert "material_receive" not in a
        for name in ("transcript_digest", "kernels", "frames_per_party", "primitive_requests_per_party"):
            assert a[name] == b[name]
        assert a["combined_reservation_bytes"] <= b["combined_reservation_bytes"] <= 64 << 20
        assert ("material_receive" in b) == (json.loads(contract).get("online_schedule") == "wavefront-v1")
    finally:
        scalar.close()
        window.close()
        client.close()


def test_optional_provider_introspection_error_still_cancels_registered_callback_once(program):
    source, contract = program
    if json.loads(contract).get("online_schedule") != "wavefront-v1":
        pytest.skip("no optional provider lookup in scalar mode")
    manifest = source.manifest(contract)
    client = native.FssClientRequest(contract, bytes([59]) * 32, (manifest,) * 3, [0])
    role = native.FssRoleRequest(source, contract, client.request_bytes(), 0, material_window=True)
    class Hostile:
        cancelled = 0
        @property
        def material_window(self):
            raise RuntimeError("private attribute detail")
        def cancel(self):
            self.cancelled += 1
    callback = Hostile()
    try:
        with pytest.raises(Exception) as error:
            role.run(None, callback)
        assert "private attribute" not in str(error.value)
        assert callback.cancelled == 1
        with pytest.raises(Exception):
            role.run(None, callback)
        assert callback.cancelled == 1
    finally:
        role.close()
        client.close()


@pytest.mark.parametrize("fault", [None, "legacy", "missing", "type", "partial", "reordered", "duplicate", "cancel"])
def test_real_native_group_tape_window_reduces_bridge_calls_and_burns_failures(program, tmp_path, fault):
    source, contract = program
    is_wave = json.loads(contract).get("online_schedule") == "wavefront-v1"
    legacy = fault == "legacy"
    if fault is not None and not is_wave:
        pytest.skip("scalar callback compatibility is covered by the no-fault case")
    manifest = source.manifest(contract)
    client = native.FssClientRequest(contract, bytes([61]) * 32, (manifest,) * 3, [0])
    roles = [native.FssRoleRequest(source, contract, client.request_bytes(), r,
                                  material_window=is_wave) for r in range(3)]
    admissions = [json.loads(role.metadata()) for role in roles]
    prices = [price_transport(value, contract) for value in admissions[:2]]
    tapes = [MaterialTape("3d" * 32, f"worker_{'ab'[r]}", records=price.tape.records,
        packet_bytes=price.tape.packet_bytes, total_packet_bytes=price.tape.total_packet_bytes,
        metadata_bytes_per_record=price.tape.metadata_bytes_per_record,
        metadata_storage=price.tape.metadata_storage, directory=tmp_path) for r, price in enumerate(prices)]
    stopped, frames = threading.Event(), [queue.Queue(maxsize=1), queue.Queue(maxsize=1)]

    def get(q):
        deadline = time.monotonic() + 20
        while not stopped.is_set() and time.monotonic() < deadline:
            try:
                return q.get(timeout=.02)
            except queue.Empty:
                pass
        raise RuntimeError("bounded test peer retired")

    class Dealer:
        def deliver(self, raw, packet):
            tapes[json.loads(raw)["role"]].append(raw, packet)
        def finish(self):
            for tape in tapes:
                tape.seal()
        def cancel(self):
            stopped.set()

    class Holder:
        def __init__(self, role):
            self.role, self.scalar_calls, self.windows, self.peer_count = role, 0, [], 0
            self.cancelled, self.finished, self.injected = 0, 0, False
        def material(self, raw):
            self.scalar_calls += 1
            return tapes[self.role].take(raw)
        def material_window(self, metadata):
            assert prices[self.role].python_receive_reservation_bytes > 0
            assert type(metadata) is tuple and 1 <= len(metadata) <= 8
            parsed = [parse_meta(raw, "material", self.role) for raw in metadata]
            assert all(value["kernel"] == parsed[0]["kernel"] for value in parsed)
            self.windows.append(len(metadata))
            packets = tapes[self.role].take_window(tuple((raw, value["packet_bytes_bound"])
                for raw, value in zip(metadata, parsed)), cancelled=stopped.is_set)
            inject = self.role == 1 and fault is not None and not self.injected
            if fault in ("reordered", "duplicate") and len(packets) < 2:
                inject = False
            if inject:
                self.injected = True
                if fault == "missing":
                    return packets[:-1]
                if fault == "type":
                    return list(packets)
                if fault == "partial":
                    raise RuntimeError("private partial provider detail")
                if fault == "reordered":
                    return packets[::-1]
                if fault == "duplicate":
                    return (packets[0],) * len(packets)
                roles[self.role].cancel()
            return packets
        def exchange(self, raw, frame):
            frames[self.role].put((raw, frame), timeout=5)
            other_raw, other_frame = get(frames[1 - self.role])
            expected = json.loads(raw)
            expected["role"] = 1 - self.role
            assert json.loads(other_raw) == expected
            self.peer_count += 1
            return other_frame
        def finish(self):
            tapes[self.role].finish()
            self.finished += 1
        def cancel(self):
            self.cancelled += 1
            stopped.set()

    holders = [Holder(0), Holder(1)]
    class Legacy:
        """No optional attribute: selection must happen before consuming input."""
        def __init__(self, holder):
            self.holder = holder
        def material(self, raw):
            return self.holder.material(raw)
        def exchange(self, raw, frame):
            return self.holder.exchange(raw, frame)
        def finish(self):
            return self.holder.finish()
        def cancel(self):
            return self.holder.cancel()
    try:
        assert roles[2].run(None, Dealer()) is None
        roles[2].close()  # Existing offline isolation precedes both input releases.
        inputs = [client.take_input(r) for r in range(2)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(roles[r].run, inputs[r], Legacy(holders[r]) if legacy else holders[r]) for r in range(2)]
            if fault is None or legacy:
                outputs = [future.result(timeout=30) for future in futures]
                assert client.finish(*outputs) == [0]
                for r, holder in enumerate(holders):
                    assert holder.scalar_calls + sum(holder.windows) == admissions[r]["primitive_requests_per_party"]
                    assert holder.peer_count == admissions[r]["frames_per_party"]
                    assert holder.finished == 1 and holder.cancelled == 0
                    if is_wave and not legacy:
                        assert any(count > 1 for count in holder.windows)
                        assert holder.scalar_calls + len(holder.windows) < admissions[r]["primitive_requests_per_party"]
                    else:
                        assert not holder.windows
            else:
                for future in futures:
                    with pytest.raises(Exception) as error:
                        future.result(timeout=30)
                    assert "private partial" not in str(error.value)
                assert holders[1].injected
                assert all(holder.finished == 0 and holder.cancelled == 1 for holder in holders)
                for r in range(2):
                    with pytest.raises(Exception):
                        roles[r].run(inputs[r], holders[r])
    finally:
        stopped.set()
        for role in roles:
            role.close()
        for tape in tapes:
            tape.close()
        client.close()
        for q in frames:
            while not q.empty():
                q.get_nowait()

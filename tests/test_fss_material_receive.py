"""Local window/tape/pricing tests; no native execution or serving claims."""
import copy
from dataclasses import fields
import json
from typing import get_type_hints

import pytest

from pllm.runtime import benchmark_memory as memory, fss_protocol as p, fss_tape as tape
from pllm.runtime.fss_transport import TransportResources, price_transport
from test_fss_batch_memory import batch_quote
from test_fss_benchmark import contract, experiment


def receive_quote():
    return {"records_bound": 8, "payload_bytes_bound": 64 << 10,
            "native_reservation_bytes_per_holder": 128 << 10,
            "python_overlap_bytes_per_holder": tape.MATERIAL_RECEIVE_PYTHON_BYTES}


@pytest.mark.parametrize("schedule", [None, "wavefront-v1", "tensor-v1"])
@pytest.mark.parametrize("delivery", ["single", "batch-v1"])
def test_price_transport_constructs_complete_scalar_and_window_resource_shapes(contract, schedule, delivery):
    """Exercise the real dataclass constructor, never a price/resource stub."""
    raw = dict(contract)
    raw.pop("online_schedule", None)
    quote = batch_quote("capsule")
    if delivery == "single":
        quote.pop("delivery_batch")
        quote.pop("material_delivery")
        raw.pop("material_delivery", None)
    else:
        raw["material_delivery"] = delivery
    if schedule is not None:
        raw["online_schedule"] = schedule
        quote["material_receive"] = receive_quote()
    result = price_transport(quote, json.dumps(raw).encode())
    declared = fields(TransportResources)
    assert tuple(field.name for field in declared) == (
        "contract_digest", "tape", "guard_disk_bytes", "guard_reservation_bytes",
        "combined_reservation_bytes", "material_delivery", "python_batch_reservation_bytes",
        "native_batch_reservation_bytes", "python_receive_reservation_bytes", "native_receive_reservation_bytes")
    hints = get_type_hints(TransportResources)
    assert all(field.default == 0 and hints[field.name] is int for field in declared[-2:])
    assert type(result) is TransportResources
    assert result.material_delivery == delivery
    assert result.python_batch_reservation_bytes == (p.MATERIAL_BATCH_PYTHON_BYTES if delivery == "batch-v1" else 0)
    assert result.native_batch_reservation_bytes == (quote["delivery_batch"]["native_reservation_bytes"]
                                                     if delivery == "batch-v1" else 0)
    assert result.python_receive_reservation_bytes == (655360 if schedule else 0)
    assert result.native_receive_reservation_bytes == (128 << 10 if schedule else 0)
    # Eight-argument scalar construction stays compatible with both new defaults.
    legacy = TransportResources(*(getattr(result, field.name) for field in declared[:8]))
    assert legacy.python_receive_reservation_bytes == legacy.native_receive_reservation_bytes == 0


def filled(tmp_path, *, count=17, storage="inline", packet_bytes=32):
    owner = tape.MaterialTape("12" * 32, "worker_a", records=count,
        packet_bytes=packet_bytes, total_packet_bytes=count * packet_bytes,
        directory=tmp_path, metadata_storage=storage)
    rows = tuple((f"metadata-{i}".encode(), f"packet-{i}".encode()) for i in range(count))
    for meta, packet in rows:
        owner.append(meta, packet)
    owner.seal()
    return owner, rows


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_full_full_tail_same_records_bytes_and_scalar_cursors(tmp_path, storage):
    scalar, rows = filled(tmp_path, storage=storage)
    window, _ = filled(tmp_path, storage=storage)
    expected = tuple(scalar.take(meta) for meta, _ in rows)
    actual, calls = [], []
    for offset in range(0, len(rows), 8):
        requests = tuple((meta, 32) for meta, _ in rows[offset:offset + 8])
        actual.extend(window.take_window(requests))
        calls.append(len(requests))
        assert window.receipt()["records_read"] == min(offset + 8, len(rows))
    assert calls == [8, 8, 1] and tuple(actual) == expected
    scalar.finish()
    window.finish()
    a, b = scalar.receipt(), window.receipt()
    for key in ("state", "records_read", "records_written", "packet_bytes_read",
                "packet_bytes_written", "encrypted_file_bytes_read", "encrypted_file_bytes_written"):
        assert a[key] == b[key]
    # Separate salted ciphertext commitments are not comparable transcripts.
    assert scalar.closed and window.closed


@pytest.mark.parametrize("kind", ["duplicate", "reordered", "publicmeta", "missing", "type", "bound", "bytes"])
def test_malformed_or_partial_window_burns_without_returning_prefix(tmp_path, kind):
    owner, rows = filled(tmp_path, count=2)
    requests = tuple((meta, 32) for meta, _ in rows)
    if kind == "duplicate":
        requests = (requests[0], requests[0])
    elif kind == "reordered":
        requests = requests[::-1]
    elif kind == "publicmeta":
        requests = ((b"wrong-public-binding", 32), requests[1])
    elif kind == "missing":
        requests += ((b"missing-record", 32),)
    elif kind == "type":
        requests = list(requests)
    elif kind == "bound":
        requests = ((requests[0][0], True), requests[1])
    else:
        requests = tuple((meta, tape.MATERIAL_RECEIVE_BYTES) for meta, _ in rows)
    result = None
    with pytest.raises(tape.MaterialTapeError):
        result = owner.take_window(requests)
    assert result is None and owner.closed and owner._file is owner._aead is owner._binding is None
    with pytest.raises(tape.MaterialTapeError):
        owner.take(rows[0][0])


def test_each_header_rejected_before_oversized_ciphertext_allocation(tmp_path, monkeypatch):
    owner, rows = filled(tmp_path, count=1)
    sizes = []
    original = owner._read_exact
    def read(size):
        sizes.append(size)
        return original(size)
    monkeypatch.setattr(owner, "_read_exact", read)
    with pytest.raises(tape.MaterialTapeError, match="geometry"):
        owner.take_window(((rows[0][0], 1),))
    assert sizes == [24] and owner.closed


@pytest.mark.parametrize("failure", ["cancel", "allocation", "provider"])
def test_partial_read_cancel_or_exception_closes_before_caller_can_retry(tmp_path, monkeypatch, failure):
    owner, rows = filled(tmp_path, count=3)
    original, calls = owner._take_locked, []
    def take(meta, bound):
        calls.append(meta)
        if len(calls) == 2 and failure != "cancel":
            raise (MemoryError() if failure == "allocation" else RuntimeError("private detail"))
        return original(meta, bound)
    monkeypatch.setattr(owner, "_take_locked", take)
    with pytest.raises(tape.MaterialTapeError) as error:
        owner.take_window(tuple((meta, 32) for meta, _ in rows),
            cancelled=lambda: failure == "cancel" and len(calls) == 1)
    assert "private detail" not in str(error.value)
    assert owner.receipt()["records_read"] == 1
    assert owner.closed and owner._aead is None


def test_online_price_distinct_from_offline_and_native_role_caps_unchanged(experiment, contract):
    pipeline = experiment.pipeline.with_params(linear__online_schedule="tensor-v1")
    raw = contract | {"online_schedule": "tensor-v1"}
    quote = batch_quote("capsule")
    quote.pop("delivery_batch")
    quote.pop("material_delivery")
    quote["material_receive"] = receive_quote()
    resources = price_transport(quote, json.dumps(raw).encode())
    assert resources.python_batch_reservation_bytes == 0
    assert resources.python_receive_reservation_bytes == 655360
    assert resources.native_receive_reservation_bytes == 128 << 10
    for native in (None, quote):
        result = memory._fss_memory({}, pipeline, 2, 3, 0, None, 0,
                                  contract=raw, transport_admission=native)
        owners = result["allocation_owners"]["each_role"]
        assert owners["material_receive_python_overlap"] == resources.python_receive_reservation_bytes
        assert owners["native_kernel_owner"] == 64 << 20
        assert result["incremental_owner_bounds"]["per_role_execution"] == p.ROLE_MEMORY
        assert sum(owners.values()) <= p.ROLE_MEMORY


@pytest.mark.parametrize("name,value", [("records_bound", 9), ("records_bound", True),
    ("payload_bytes_bound", 0), ("python_overlap_bytes_per_holder", 0),
    ("native_reservation_bytes_per_holder", True), ("native_reservation_bytes_per_holder", 1),
    ("native_reservation_bytes_per_holder", 64 << 20)])
def test_receive_price_rejects_hostile_or_uncovered_owner_fields(contract, name, value):
    raw = contract | {"online_schedule": "tensor-v1", "material_delivery": "batch-v1"}
    quote = batch_quote("capsule") | {"material_receive": receive_quote()}
    quote = copy.deepcopy(quote)
    quote["material_receive"][name] = value
    with pytest.raises(ValueError, match="receive"):
        price_transport(quote, json.dumps(raw).encode())


@pytest.mark.parametrize("delivery", ["single", "batch-v1"])
def test_online_python_owner_cannot_spend_native_or_offline_credit(experiment, contract, monkeypatch, delivery):
    pipeline = experiment.pipeline.with_params(linear__online_schedule="tensor-v1", linear__material_delivery=delivery)
    raw = contract | {"online_schedule": "tensor-v1"}
    if delivery == "batch-v1":
        raw["material_delivery"] = delivery
    # Force an owner that exceeds the unchanged role envelope. This is a pure
    # pricing rejection, not a runtime cap change or native execution test.
    monkeypatch.setattr(tape, "MATERIAL_RECEIVE_PYTHON_BYTES", p.ROLE_MEMORY)
    with pytest.raises(memory.BenchmarkMemoryError, match="memory caps"):
        memory._fss_memory({}, pipeline, 2, 3, 0, None, 0, contract=raw)

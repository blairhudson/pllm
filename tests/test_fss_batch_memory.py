"""Batch memory/source-identity plumbing using explicit native pricing doubles.

The real shared transport price is exercised, not native queue or serving proof.
"""
import copy
import hashlib
import json

import pytest

from pllm.runtime import benchmark_delegated as delegated, benchmark_memory as memory
from pllm.runtime import fss_protocol as p, fss_source
from pllm.runtime.fss_transport import price_transport
from test_fss_benchmark import contract, experiment, source, imported_source, decoder_contract


def batch_quote(capacity):
    return {"material_delivery": "batch-v1", "delivery_batch": {
        "records_bound": 64, "body_bytes_bound": 1 << 20, "envelope_bytes_bound": 1024,
        "record_overhead_bytes": 8, "native_reservation_bytes": (2 << 20) + 4096,
        "python_overlap_bytes": 11_014_144}, "combined_reservation_bytes": 64 << 20,
        "primitive_requests_per_party": 4, "largest_packet_bytes": 1024,
        "cumulative_packet_bytes_per_party_bound": 4096, "callback_metadata_bytes_bound": 1024,
        **({"capacity": "decoder", "guard_disk_bytes": 8192, "guard_reservation_bytes": 4096}
           if capacity == "decoder" else {})}


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
def test_metadata_batch_price_reserves_python_overlap_without_claiming_native_queue_proof(experiment, contract, capacity):
    single = experiment.pipeline.with_params(linear__capacity=capacity)
    batch = single.with_params(linear__material_delivery="batch-v1")
    contract = decoder_contract(contract) if capacity == "decoder" else contract
    before = memory._fss_memory({}, single, 2, 3, 0, None, 0, contract=contract)
    after = memory._fss_memory({}, batch, 2, 3, 0, None, 0,
                              contract=contract | {"material_delivery": "batch-v1"})
    previous = before["allocation_owners"]["each_role"]
    owners = after["allocation_owners"]["each_role"]
    assert (sum(owners.values()) - owners["metadata_parse_copies"]
            == sum(previous.values()) - previous["metadata_parse_copies"] + p.MATERIAL_BATCH_PYTHON_BYTES)
    assert owners["metadata_parse_copies"] > previous["metadata_parse_copies"]
    assert owners["material_batch_python_overlap"] == p.MATERIAL_BATCH_PYTHON_BYTES
    assert owners["native_kernel_owner"] == previous["native_kernel_owner"] == 64 << 20
    assert sum(owners.values()) <= p.ROLE_MEMORY + p.MATERIAL_BATCH_PYTHON_BYTES
    bounds = after["incremental_owner_bounds"]
    assert bounds["per_role_execution"] == p.ROLE_MEMORY + p.MATERIAL_BATCH_PYTHON_BYTES
    assert before["incremental_owner_bounds"]["per_role_execution"] == p.ROLE_MEMORY
    assert bounds["per_role"] == max(bounds["per_role_execution"], bounds["source_metadata_preflight"])
    assert after["native_total_peak_bytes"] >= before["native_total_peak_bytes"]
    assert after["components"] == before["components"]  # No source/tape/guard disk discount.
    admission = after["fss_admission"]
    assert admission["metadata_only"] and admission["transport_admission"] is None
    assert admission["material_batch_native_reservation_bytes"] is None
    assert not admission["serving_qualified"]
    assert "before native import" in admission["material_batch_memory_scope"]
    assert not any(key.startswith("material_batch") for key in before["fss_admission"])


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
def test_imported_batch_price_separates_python_and_native_owners_and_keeps_disk(experiment, contract, capacity):
    single = experiment.pipeline.with_params(linear__capacity=capacity)
    batch = single.with_params(linear__material_delivery="batch-v1")
    contract = decoder_contract(contract) if capacity == "decoder" else contract
    quoted = batch_quote(capacity)
    single_quote = {key: value for key, value in quoted.items() if key not in {"material_delivery", "delivery_batch"}}
    before = memory._fss_memory({}, single, 2, 3, 0, None, 0, contract=contract, transport_admission=single_quote)
    raw = contract | {"material_delivery": "batch-v1"}
    shared = price_transport(quoted, json.dumps(raw).encode())
    after = memory._fss_memory({}, batch, 2, 3, 0, None, 0, contract=raw, transport_admission=quoted)
    assert after["components"] == before["components"]
    assert after["allocation_owners"]["each_role"]["material_batch_python_overlap"] == shared.python_batch_reservation_bytes
    admission = after["fss_admission"]
    assert admission["transport_admission"] == quoted
    assert admission["material_batch_native_reservation_bytes"] == shared.native_batch_reservation_bytes == (2 << 20) + 4096
    assert admission["material_batch_python_reservation_bytes"] == quoted["delivery_batch"]["python_overlap_bytes"]
    assert after["incremental_owner_bounds"]["per_role_execution"] == p.ROLE_MEMORY + 11_014_144
    assert admission["material_batch_native_in_kernel_owner"]
    assert after["allocation_owners"]["each_role"]["native_kernel_owner"] == 64 << 20
    # Four original packets per holder: batching does not replace four tape
    # records with one HTTP batch or remove the two separate replay guards.
    expected_tape = 2 * (4096 + 4 * (40 + (4096 if capacity == "capsule" else 0)))
    assert after["components"]["fss_encrypted_tape_disk_bytes"] == expected_tape
    assert after["components"]["fss_guard_disk_bytes"] == (16384 if capacity == "decoder" else 0)


@pytest.mark.parametrize("missing", ["material_delivery", "delivery_batch", "records_bound",
    "body_bytes_bound", "envelope_bytes_bound", "record_overhead_bytes", "native_reservation_bytes", "python_overlap_bytes"])
def test_batch_memory_rejects_missing_native_transport_fields(experiment, contract, missing):
    batch = experiment.pipeline.with_params(linear__material_delivery="batch-v1")
    quoted = batch_quote("capsule")
    del (quoted if missing in {"material_delivery", "delivery_batch"} else quoted["delivery_batch"])[missing]
    with pytest.raises(memory.BenchmarkMemoryError, match="native admission"):
        memory._fss_memory({}, batch, 2, 3, 0, None, 0,
            contract=contract | {"material_delivery": "batch-v1"}, transport_admission=quoted)


def test_old_flat_batch_hint_cannot_replace_native_delivery_resources(experiment, contract):
    batch = experiment.pipeline.with_params(linear__material_delivery="batch-v1")
    quoted = batch_quote("capsule")
    quoted.pop("delivery_batch")
    quoted.update(material_batch_max_records=64, material_batch_max_bytes=1 << 20,
                  material_batch_envelope_bytes=1024, material_batch_reservation_bytes=2 << 20)
    with pytest.raises(memory.BenchmarkMemoryError, match="native admission"):
        memory._fss_memory({}, batch, 2, 3, 0, None, 0,
            contract=contract | {"material_delivery": "batch-v1"}, transport_admission=quoted)


@pytest.mark.parametrize("delivery,returned", [("batch-v1", None), ("single", "batch-v1")])
def test_memory_cannot_substitute_native_material_mode(experiment, contract, delivery, returned):
    pipeline = experiment.pipeline.with_params(linear__material_delivery=delivery)
    native = contract if returned is None else contract | {"material_delivery": returned}
    with pytest.raises(memory.BenchmarkMemoryError, match="native admission"):
        memory._fss_memory({}, pipeline, 2, 3, 0, None, 0, contract=native)


def test_benchmark_batch_keeps_numeric_fingerprint_but_changes_contract_binding(experiment, source):
    single, _ = delegated.inspect_runtime(experiment)
    batch = experiment.with_params(pipeline__linear__material_delivery="batch-v1")
    source.contract = source.contract | {"material_delivery": "batch-v1"}
    source.contract_bytes = json.dumps(source.contract).encode()
    batched, _ = delegated.inspect_runtime(batch)
    assert single["numeric"] == batched["numeric"]
    assert single["model_fingerprint"] == batched["model_fingerprint"]
    assert single["source_contract_bytes_sha256"] != batched["source_contract_bytes_sha256"]
    assert "material_delivery" not in single and batched["material_delivery"] == "batch-v1"
    with pytest.raises(ValueError, match="material delivery differs"):
        delegated.inspect_runtime(experiment)


def test_ordinary_memory_api_gets_batch_price_from_live_owner_only(experiment, imported_source, monkeypatch):
    loaded, quote, resources = imported_source
    batch = experiment.with_params(pipeline__linear__material_delivery="batch-v1")
    source = loaded.source
    source.contract = source.contract | {"material_delivery": "batch-v1"}
    source.contract_bytes = loaded.contract_bytes = json.dumps(source.contract).encode()
    def runtime_contract(exp, rows, outputs):
        assert exp is batch and (rows, outputs) == (2, 3)
        return source.contract_bytes
    source.runtime_contract = runtime_contract
    manifest = json.loads(loaded.manifest_bytes)
    manifest["contract"] = hashlib.sha256(source.contract_bytes).hexdigest()
    loaded.manifest_bytes = json.dumps(manifest).encode()
    quote.update(batch_quote("capsule"), contract_digest=manifest["contract"])
    monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(16 << 30, 15 << 30, 0, 20 << 30))
    monkeypatch.setattr(fss_source, "load_role_source", lambda *a, **k: pytest.fail("source reimported"))
    report = memory.benchmark_memory(batch, fss_imported=loaded)
    assert report["admitted"]
    admission = report["estimate"]["fss_admission"]
    assert admission["transport_admission"] == quote
    assert admission["complete_program_resources"] == resources
    assert admission["numeric_admission"] == "native_imported_program" and not admission["metadata_only"]
    assert admission["material_batch_native_reservation_bytes"] == quote["delivery_batch"]["native_reservation_bytes"]
    assert not memory.benchmark_memory(batch, fss_imported=loaded, memory_budget_bytes=1 << 20)["admitted"]
    # No caller-supplied serialized price can stand in for its native owner.
    with pytest.raises(memory.BenchmarkMemoryError, match="imported source pricing") as error:
        memory.benchmark_memory(batch, fss_imported=copy.deepcopy(quote))
    assert "live imported source owner" in str(error.value.__cause__)
    quote.pop("material_delivery")
    with pytest.raises(memory.BenchmarkMemoryError, match="imported source pricing"):
        memory.benchmark_memory(batch, fss_imported=loaded)

"""Locality accounting, locked evidence, and existing seeded-ring invariants."""

from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import msgpack
import numpy as np
import pytest

from pllm.runtime.preparation_protocol import (
    CorrectionPush,
    PreparationRequest,
    expand_output_mask,
    expand_preparation_mask,
    seeded_ring_profile,
)
from pllm.runtime.stage_protocol import MaskedStageRequest
from pllm.runtime.transformer_client import (
    PreparedInventory,
    PreparedStageRows,
    TransformerClientError,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "preparation_placement_probe", ROOT / "scripts/probe_preparation_placement.py"
)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def evidence():
    return json.loads(probe.ARTIFACT.read_text())


@pytest.mark.parametrize("phase", ["warm", "cold"])
@pytest.mark.parametrize("name", probe.NAMES)
def test_locality_preserves_sums_and_does_not_remove_correction_uplink(phase, name):
    sample = evidence()["archived_ledgers"][phase][name]
    edges = sample["edges"]
    logical = sample["all_link_body_bytes"]
    prep_client = sum(
        edge["serialized_body_bytes"]
        for edge in edges
        if {edge["source"], edge["destination"]} == {"client", "preparation"}
    )
    for placement in probe.PLACEMENTS:
        result = probe.place(edges, placement)
        assert result["all_logical_link_body_bytes"] == logical
        assert (
            result["lan_body_bytes"] + result["wan_body_bytes"] + result["loopback_body_bytes"]
            == result["all_network_link_body_bytes"]
        )
        removed = prep_client if placement == "client-host-ipc-counterfactual" else 0
        assert result["all_network_link_body_bytes"] == logical - removed
        assert result["absent_network_ipc_logical_body_bytes"] == removed
        assert result["online_wan_body_bytes"] == sample["online_body_bytes"]
        correction = next(
            edge
            for edge in result["directed_links"]
            if edge["source"] == "preparation" and edge["destination"] == "inference"
        )
        assert correction["locality"] == "wan"
        assert (
            correction["serialized_body_bytes"]
            == sample["stage_expected_edges"]["preparation->inference"]
        )


def test_retained_evidence_reproduces_every_saved_field():
    saved = evidence()
    rebuilt = probe.build(saved["archived_ledgers"])
    assert rebuilt == saved
    baseline = rebuilt["resource_floors"]["prepared-request-sized"]
    assert baseline["preparation_linear_macs_per_inventory"] == 70 * 357826560
    assert baseline["correction_tensor_minimum_u16_body_bytes"] > 178970558 / 10
    assert baseline["resource_admission"].startswith("unknown")


def test_source_hash_rejects_changed_evidence(tmp_path):
    path = tmp_path / "changed.json"
    path.write_text('{"source": "different"}')
    with pytest.raises(ValueError, match="source lock mismatch"):
        probe.locked_json(path, probe.LOCKS["incremental-network-qwen25-2026-10-01.json"])


@pytest.mark.parametrize("field", ["source_lock_digest", "prompt_digest", "inventory"])
def test_workload_and_source_mutations_fail_closed(field):
    retained = evidence()["archived_ledgers"]
    sample = retained["warm"]["prepared-request-sized"]
    if field == "inventory":
        sample[field]["reused"] = 1
    else:
        sample[field] = "0" * 64
    with pytest.raises(ValueError, match="locked cohort/source"):
        probe.build(retained)


def test_directed_body_shift_cannot_hide_inside_preserved_total():
    retained = evidence()["archived_ledgers"]
    sample = retained["warm"]["client-attention"]
    # Keep total and stage attribution valid, but shift one authorization byte
    # from LAN-bound direction to the other. Canonical retained hash must reject.
    sample["edges"][0]["serialized_body_bytes"] -= 1
    sample["edges"][1]["serialized_body_bytes"] += 1
    with pytest.raises(ValueError, match="directed ledger source lock"):
        probe.build(retained)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "negative", "bool"])
def test_invalid_body_ledger_not_treated_as_zero(mutation):
    edges = copy.deepcopy(evidence()["archived_ledgers"]["warm"]["client-attention"]["edges"])
    if mutation == "duplicate":
        edges.append(edges[0].copy())
    elif mutation == "missing":
        edges.pop()
    else:
        edges[0]["serialized_body_bytes"] = -1 if mutation == "negative" else True
    with pytest.raises(ValueError):
        probe.place(edges, "trusted-lan-appliance")


def request(bits=16, rows=3):
    bound = {16: 32767, 24: 100000, 32: 10000000}[bits]
    profile = seeded_ring_profile(bound)
    return PreparationRequest(
        attempt_id="a" * 32,
        session_id="inventory-test",
        model=probe.MODEL,
        body_fingerprint=probe.FINGERPRINT,
        stage_id="layers.0.test",
        weight_digest="b" * 64,
        rows=rows,
        in_features=3,
        out_features=2,
        weight_bits=8,
        activation_bits=8,
        signed_output_bound=bound,
        ring=profile.ring,
        modulus=profile.modulus,
        wire_bits=profile.wire_bits,
        seed=bytes(range(32)),
    )


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_signed_ring_arithmetic_and_production_wire_never_send_root_to_inference(bits):
    req = request(bits)
    weights = np.array([[63, -63, 31], [-4, 7, -9]], dtype=np.int64)
    x = np.array([[-127, 127, 0], [1, -2, 3], [127, -126, -127]], dtype=np.int64)
    r = expand_preparation_mask(req).astype(np.int64)
    s = expand_output_mask(req).astype(np.int64)
    correction = (r @ weights.T - s) % req.modulus
    push = CorrectionPush(
        attempt_id=req.attempt_id,
        session_id=req.session_id,
        model=req.model,
        body_fingerprint=req.body_fingerprint,
        stage_id=req.stage_id,
        weight_digest=req.weight_digest,
        rows=req.rows,
        in_features=req.in_features,
        out_features=req.out_features,
        weight_bits=req.weight_bits,
        activation_bits=req.activation_bits,
        signed_output_bound=req.signed_output_bound,
        ring=req.ring,
        modulus=req.modulus,
        wire_bits=req.wire_bits,
        correction=correction.astype(np.uint32),
    )
    wire = push.pack()
    decoded = CorrectionPush.unpack(wire)
    np.testing.assert_array_equal(decoded.correction, correction)
    assert "seed" not in decoded.__dataclass_fields__
    assert "z" not in msgpack.unpackb(wire, raw=False)
    assert req.seed not in wire
    for index in range(req.rows):
        ref = probe.ring_reference(
            weights.tolist(), x[index].tolist(), r[index].tolist(), s[index].tolist(), bits
        )
        np.testing.assert_array_equal(ref["correction"], decoded.correction[index])
        expected = x[index] @ weights.T
        np.testing.assert_array_equal(ref["recovered"], expected % req.modulus)
        signed = np.array(ref["recovered"], dtype=np.int64)
        signed[signed >= req.modulus // 2] -= req.modulus
        np.testing.assert_array_equal(signed, expected)
        online = MaskedStageRequest(
            model=req.model,
            stage_id=req.stage_id,
            correlation_id="c" * 32,
            masked_input=np.array([ref["masked_input"]], dtype=np.uint32),
            activation_scales=np.ones(1, dtype=np.float32),
            modulus=req.modulus,
            wire_bits=bits,
            ring=req.ring,
            body_fingerprint=req.body_fingerprint,
            weight_digest=req.weight_digest,
            weight_bits=8,
            activation_bits=8,
            session_id=req.session_id,
            out_features=req.out_features,
            signed_output_bound=req.signed_output_bound,
        ).pack()
        assert req.seed not in online
        assert req.seed not in [
            value
            for value in msgpack.unpackb(online, raw=False).values()
            if isinstance(value, bytes)
        ]


def test_mask_domains_and_inventory_reservations_keep_fresh_rows_one_use():
    req = request(rows=3)
    r, s = expand_preparation_mask(req), expand_output_mask(req)
    altered = replace(req, attempt_id="d" * 32)
    assert not np.array_equal(r, expand_preparation_mask(altered))
    assert not np.array_equal(r[:, :2], s)
    stages = {req.stage_id: PreparedStageRows(req, r, s)}
    inventory = PreparedInventory(req.session_id, 3, stages)
    first = inventory.reserve(2)
    _, _, first_ids = first.take(req.stage_id, 1)
    first.close()  # Unused reserved row burns; next lease must not reclaim it.
    second = inventory.reserve(1)
    _, _, second_ids = second.take(req.stage_id, 1)
    second.close()
    assert first_ids != second_ids
    assert inventory.status()["burned"] == 1
    assert inventory.status()["consumed"] == 2
    with pytest.raises(TransformerClientError, match="enough rows"):
        inventory.reserve(1)
    with pytest.raises(TransformerClientError, match="closed"):
        second.take(req.stage_id, 1)
    assert inventory.status()["burned"] == 1
    assert inventory.status()["consumed"] == 2

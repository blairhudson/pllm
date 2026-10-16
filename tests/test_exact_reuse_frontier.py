"""State/ledger counterexamples for the bounded frontier reference."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "exact_frontier", Path(__file__).parents[1] / "scripts/probe_exact_reuse_frontier.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.mark.parametrize("seed", [0, 7, 19, 0xFFFFFFFF])
@pytest.mark.parametrize("prompt", [[1], [1, 2, 3], [3, 2, 1], [1, 2, 1, 2]])
def test_full_state_and_rng_independent_oracle(seed, prompt):
    ledger = m.Ledger()
    store = m.Store(ledger)
    for _ in range(2):
        actual = m.run(prompt, seed, 5, ledger, store, delta=True)
        assert actual == m.reference(prompt, seed, 5)
        assert len(actual.state.ids) == len(prompt) + 4
        assert actual.pending == actual.output[-1]
    ledger.audit()


def test_installed_reuse_not_relabelled_as_new():
    for workload in ("fresh", "repeat", "restart", "unrelated"):
        a, b = m.workload(workload, False), m.workload(workload, True)
        assert a["ledger"]["all_link_bytes"] == b["ledger"]["all_link_bytes"]
    a, b = m.workload("branch", False), m.workload("branch", True)
    assert a["result_commitments"] == b["result_commitments"]
    assert a["ledger"]["claimed"] == b["ledger"]["claimed"] == 23
    assert a["ledger"]["all_link_bytes"] == b["ledger"]["all_link_bytes"]
    assert a["prefix_owner"] == b["prefix_owner"]


def test_delta_reduces_repeated_pages_not_network():
    a, b = m.workload("repeat", False), m.workload("repeat", True)
    assert a["ledger"]["validation_bytes"] > b["ledger"]["validation_bytes"]
    assert a["ledger"]["all_link_bytes"] == b["ledger"]["all_link_bytes"]


@pytest.mark.parametrize("field,value", [
    ("ids", (2, 1)), ("kv", ((1, 1), (1, 1))),
    ("context", ("changed-source",)), ("logits", (1,) * m.VOCAB)])
def test_delta_parent_cannot_be_selected_by_path_only(field, value):
    ledger = m.Ledger()
    store = m.Store(ledger)
    parent = m.oracle([1, 2])
    store.publish(m.advance(parent, 3))
    with pytest.raises(ValueError, match="parent seal"):
        store.lookup([1, 2, 3], parent=replace(parent, **{field: value}), delta=True)


def test_pending_selection_never_published_as_executed():
    store = m.Store(m.Ledger())
    result = m.run([1, 2], 7, 1, store.ledger, store)
    assert result.state.ids == (1, 2)
    assert (*result.state.ids, result.pending) not in store.nodes


def test_failed_retirement_and_wrong_head_reject():
    store = m.Store(m.Ledger())
    with pytest.raises(ValueError, match="unretired"):
        store.publish(m.oracle([1]), acknowledged=False)
    assert not store.nodes
    store.publish(m.oracle([1]))
    with pytest.raises(ValueError, match="trusted head"):
        store.lookup([1], trusted_head=0)
    with pytest.raises(ValueError, match="trusted head"):
        store.reopen(0)
    assert store.reopen(store.head).lookup([1]) == m.oracle([1])


def test_corruption_charged_before_reject():
    ledger = m.Ledger()
    store = m.Store(ledger)
    state = m.oracle([1])
    store.publish(state)
    saved, raw, tag = store.nodes[(1,)]
    store.nodes[(1,)] = (saved, raw + b"x", tag)
    with pytest.raises(ValueError, match="authentication"):
        store.lookup([1])
    assert ledger.disk_read_bytes > 0
    assert ledger.hits == 0


def test_nonce_or_material_savings_never_refunded():
    ledger = m.Ledger()
    ledger.execute(2, burn=3)
    assert ledger.issued == 5 and ledger.claimed == 2 and ledger.burned == 3
    ledger.audit()
    ledger.burned = 0
    with pytest.raises(ValueError, match="conservation"):
        ledger.audit()


def test_receipt_retry_is_not_new_stochastic_invocation():
    ledger = m.Ledger()
    receipts = m.Receipts(ledger)
    calls = []

    def execute():
        calls.append(1)
        return m.run([1, 2], 7, 3, ledger)

    request = {"tokens": [1, 2], "seed": 7}
    a = receipts.call("operation", request, execute)
    assert receipts.call("operation", request, execute) == a
    assert len(calls) == 1
    with pytest.raises(ValueError, match="conflict"):
        receipts.call("operation", request, execute, fresh=True)
    with pytest.raises(ValueError, match="conflict"):
        receipts.call("operation", {"tokens": [2, 1]}, execute)
    receipts.call("new-operation", request, execute, fresh=True)
    assert len(calls) == 2


def test_capacity_and_no_semantic_or_reordered_match():
    store = m.Store(m.Ledger(), max_nodes=1)
    store.publish(m.oracle([1, 2]))
    assert store.lookup([2, 1]) is None
    with pytest.raises(ValueError, match="bound"):
        store.publish(m.oracle([1, 3]))
    with pytest.raises(ValueError, match="bound"):
        m.run([1] * 128, 7, 2, m.Ledger())


def test_report_unknowns_stay_unknown():
    data = m.report()
    assert data["real_geometry"]["optimized_checkpoint_bytes_saved"] is None
    assert not data["installed_evidence_separate_not_pooled"]["new_candidate_256_mib_qualification"]
    assert data["workloads"]["repeat"]["incremental_all_link_bytes_saved"] == 0
    assert data["application_receipt_retry"]["incremental_all_link_bytes_saved"] == 0


@pytest.mark.parametrize("changed", ["source", "numeric", "KV", "verification", "policy"])
def test_authenticated_changed_contract_is_not_authority(changed):
    store = m.Store(m.Ledger())
    state = m.oracle([1, 2], context=(changed,))
    store.publish(state)
    with pytest.raises(ValueError, match="binding"):
        store.lookup([1, 2])


def test_frozen_evidence_exact_read_only_replay():
    path = Path(__file__).parents[1] / "docs/evidence/exact-reuse-frontier-reference-v2.json"
    assert m.report() == json.loads(path.read_text())


def test_directed_correction_conservation_including_burned_rows():
    ledger = m.Ledger()
    ledger.execute(2, burn=3)
    ledger.execute(3, burn=1)
    assert (ledger.issued, ledger.claimed, ledger.burned) == (9, 5, 4)
    assert ledger.links["preparation->inference"] == 24 * 9
    assert ledger.links["preparation->client"] == 32 * 2  # Fixture controls only.
    assert ledger.links["client->inference"] == 32 * 2 + 16 * 5
    assert ledger.links["inference->client"] == 32 * 2 + 24 * 5
    total = ledger.report()["all_link_bytes"]
    assert total == 672
    assert ledger.report()["unmodeled_runtime_directed_bytes"] == {
        "inference->preparation_acknowledgement": None}
    # Aggregate conservation cannot detect corrections misrouted to the client.
    ledger.links["preparation->inference"] -= 24
    ledger.links["preparation->client"] += 24
    assert sum(ledger.links.values()) == total
    with pytest.raises(ValueError, match="directed fixture conservation"):
        ledger.audit()


def test_checkpoint_geometry_correction_direction_and_unknown_controls():
    data = m.report()
    geometry = data["real_geometry"]["all_stage_u32_row_ceiling_by_link"]
    assert "preparation->client" not in geometry
    assert geometry["preparation->inference"] == geometry["inference->client"] > 0
    assert data["synthetic_topology"]["actual_runtime_control_and_framing_bytes"] is None
    for workload in data["workloads"].values():
        assert not workload["candidate"]["ledger"]["complete_runtime_ledger"]


@pytest.mark.parametrize("maximum", [True, 0, -1, 129, 1.5])
def test_store_node_limit_cannot_be_increased(maximum):
    with pytest.raises(ValueError, match="node bound"):
        m.Store(m.Ledger(), max_nodes=maximum)


@pytest.mark.parametrize("ids", [[True], [-1], [m.VOCAB], [1.0], [1] * 129, iter([1])])
def test_token_bounds_checked_before_lookup_or_publication(ids):
    store = m.Store(m.Ledger())
    with pytest.raises(ValueError, match="token sequence"):
        store.lookup(ids)
    assert not store.nodes and store.ledger.lookups == 0


def test_direct_publication_cannot_bypass_token_limit():
    store = m.Store(m.Ledger())
    with pytest.raises(ValueError, match="token sequence"):
        store.publish(m.State(ids=(1,) * 129))
    assert not store.nodes and store.ledger.disk_written_bytes == 0


@pytest.mark.parametrize("kwargs", [
    {"outputs": True}, {"outputs": 1.5}, {"outputs": 0},
    {"seed": -1}, {"seed": True}, {"seed": 2 ** 32},
    {"cut": True}, {"cut": 0}, {"cut": 3},
])
def test_unsupported_lifecycle_inputs_reject_before_issuance(kwargs):
    ledger = m.Ledger()
    options = {"seed": 7, "outputs": 2, **kwargs}
    with pytest.raises(ValueError, match="lifecycle bound"):
        m.run([1, 2], ledger=ledger, **options)
    assert ledger.issued == 0 and not any(ledger.links.values())


@pytest.mark.parametrize("rows,burn", [(True, 0), (0, 0), (1, -1), (1, True), (1.5, 0)])
def test_invalid_row_counts_do_not_change_ledger(rows, burn):
    ledger = m.Ledger()
    with pytest.raises(ValueError, match="row counts"):
        ledger.execute(rows, burn)
    assert ledger.issued == 0 and ledger.sessions == 0


def test_v1_is_rejected_and_preserved_not_rewritten_as_v2():
    data = m.report()
    assert data["schema"].endswith("v2")
    old = data["superseded_v1"]
    assert old["sha256"] == m.V1_SHA256
    assert "scripts/probe_exact_reuse_frontier.py" in old["current_source_mismatches"]
    assert "tests/test_exact_reuse_frontier.py" in old["current_source_mismatches"]
    assert "docs/research/exact-reuse-frontier-reference.md" in old["current_source_mismatches"]
    assert all(row["bytes_saved"] == 0 for row in data["conditional_sibling_scaling_not_measurements"])


def test_installed_scheduled_and_budget_limited_positions():
    assert m.installed_positions(10, 4096)[0] == (1, 2, 4, 8, 9)
    assert 48 in m.installed_positions(50, 4096)[0]
    assert m.installed_positions(10, 7)[0] == (1, 2, 4, 7)
    data = m.report()["boundary_screen"]
    assert [row["gross_additional_rows_per_sibling_ceiling"] for row in data] == [0, 0, 5]
    assert data[-1]["installed_nearest_retained_rows"] == 40


def test_unscheduled_boundary_compares_nearest_cut_not_zero_and_prices_publication():
    arms = []
    for extra in (None, 45):
        ledger, prefixes = m.Ledger(), m.ResidentPrefixes()
        for suffix in ([1, 1], [2, 2]):
            prompt = [3] * 45 + suffix
            actual = m.run(prompt, 7, 4, ledger, ordinary=True, prefixes=prefixes, cut=extra)
            assert actual == m.reference(prompt, 7, 4)
        arms.append((ledger, prefixes))
    assert arms[0][0].claimed - arms[1][0].claimed == 5  # Not 45.
    assert arms[1][1].publications > arms[0][1].publications
    assert arms[1][1].capture_bytes == arms[0][1].capture_bytes


@pytest.mark.parametrize("budget,entry_limit", [(32768, 4096), (64, 4096), (100, 4096), (512, 3)])
def test_prefix_retention_matches_actual_source_cache_on_synthetic_tensors(budget, entry_limit):
    # Load only the source Python cache and generated numeric state. No model,
    # checkpoint loading, native decoder execution or runtime mutation occurs.
    import numpy as np
    from pllm.runtime.transformer_client import LayerCache, RuntimeSnapshot, _snapshot_state_basis

    path = Path(__file__).parents[1] / "python/pllm/runtime/prefill_cache.py"
    source_spec = importlib.util.spec_from_file_location("pllm.runtime._frontier_reference_cache", path)
    source = importlib.util.module_from_spec(source_spec)
    sys.modules[source_spec.name] = source
    source_spec.loader.exec_module(source)
    actual = source.ExactPrefillCache(budget, causal_reduction="prefix_f32")
    actual._MAX_ENTRIES = entry_limit
    fixture = m.ResidentPrefixes(budget, entry_limit)
    binding, bundle = "a" * 64, "b" * 64
    for prompt in ([1, 2, 3, 4, 5, 6, 7, 8, 1, 1],
                   [1, 2, 3, 4, 5, 6, 7, 8, 2, 2], [1, 2, 3, 4, 5], [7, 8, 9]):
        state = m.oracle(prompt)
        key = source.prefill_key(binding, bundle, prompt, causal_reduction="prefix_f32")
        arrays = np.asarray(state.kv, dtype=np.float32)
        basis = _snapshot_state_basis("completed_prefill", len(prompt), binding, "c" * 64,
                                      "d" * 64, None, len(prompt))
        snapshot = RuntimeSnapshot(len(prompt), [LayerCache(arrays[:, 0:1, None],
                                   arrays[:, 1:2, None], len(prompt))], {}, binding, basis)
        actual.put_prefixes(binding, bundle, prompt, snapshot)
        actual.put(key, snapshot, np.asarray(state.logits, dtype=np.float32))
        fixture.publish_prefill(state)
        assert actual._bytes == fixture.bytes <= budget
        assert len(actual._entries) == len(fixture.entries) <= entry_limit
        expected_keys = [source.prefill_key(binding, bundle, ids, causal_reduction="prefix_f32")
                         for ids in fixture.entries]
        assert list(actual._entries) == expected_keys
        for query in (prompt, [1, 2, 3, 4, 5, 6, 7, 8, 3, 3],
                      [1, 2, 3, 4, 5, 6, 7, 8], [7, 8, 2]):
            query_key = source.prefill_key(binding, bundle, query, causal_reduction="prefix_f32")
            hit = actual.get(query_key, position=len(query), layers=1)
            if hit is not None:
                position = hit[0].position
            else:
                hit = actual.longest_prefix(binding, bundle, query, layers=1)
                position = None if hit is None else hit[0]
            toy = fixture.lookup(query)
            assert (None if toy is None else len(toy.ids)) == position
            assert actual._bytes == fixture.bytes
            assert len(actual._entries) == len(fixture.entries)

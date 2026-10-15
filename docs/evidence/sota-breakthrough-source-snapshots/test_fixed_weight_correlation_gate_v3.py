"""Independent exact-ring oracle, role separation, leakage, renewal and costing."""
import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/probe_fixed_weight_correlation_gate.py"
SPEC = importlib.util.spec_from_file_location("fixed_weight_gate", PATH)
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


@pytest.mark.parametrize("mode", gate.MODES)
def test_arbitrary_ring32_chosen_inputs(mode):
    rng = random.Random(11)
    for m, n in ((3, 5), (5, 3), (1, 1)):
        w = tuple(tuple(rng.randrange(-128, 128) for _ in range(n)) for _ in range(m))
        client, inference = gate.setup(w, 9, mode, rng)
        authority = gate.Authority(client.context)
        for j in range(9):
            x = tuple(rng.choice([-2**32, -128, 0, 127, 2**32-1, rng.randrange(2**32)])
                      for _ in range(n))
            result, transcript = gate.query(client, inference, authority, j, x)
            expected = tuple(sum(w[a][b]*x[b] for b in range(n)) & 0xffffffff for a in range(m))
            assert result == expected
            assert len(transcript["client_to_inference"]) == n
            assert len(transcript["inference_to_client"]) == m
            assert not hasattr(inference, "r_root")
            assert not hasattr(inference, "output_tape")
            assert client.r_root not in inference.__dict__.values()
            r, s = client.row(j)
            q = inference.row(j)
            assert tuple((s[a]+q[a]) & 0xffffffff for a in range(m)) == tuple(
                sum(w[a][b]*r[b] for b in range(n)) & 0xffffffff for a in range(m))


@pytest.mark.parametrize("mode", gate.MODES)
def test_actual_role_material_not_hidden(mode):
    c, i = gate.setup(gate.W, 4, mode, random.Random(1))
    assert bool(c.weights) == (mode in ("direct_weight", "zero_local"))
    assert len(c.output_tape) == (4 if mode == "zero_tape" else 0)
    assert len(i.correction_tape) == (4 if mode == "prepared" else 0)
    if mode.startswith("zero"):
        assert i.row(0) == (0, 0, 0)
    if mode == "direct_weight":
        assert c.pad_root == i.q_root


@pytest.mark.parametrize("outcome", ["expired", "cancelled", "malformed", "bad-reply"])
def test_failure_burn_survives_client_restart_and_clone(outcome):
    c, _ = gate.setup(gate.W, 3, "zero_tape", random.Random(4))
    external = gate.Authority(c.context)
    with pytest.raises(ValueError, match="burned"):
        external.reserve(0, c.context.digest(), outcome)
    restored_client = gate.ClientKey(**c.__dict__)
    with pytest.raises(ValueError, match="replayed"):
        external.reserve(0, restored_client.context.digest())
    external.reserve(1, c.context.digest())
    external.reserve(2, c.context.digest())
    with pytest.raises(ValueError):
        external.reserve(3, c.context.digest())


def test_binding_failure_burns_and_epoch_changes_masks():
    c, i = gate.setup(gate.W, 2, "prepared", random.Random(4))
    c2, i2 = gate.setup(gate.W, 2, "prepared", random.Random(4), "renewal-1")
    assert c.r_root == c2.r_root  # Even this unsafe root reuse differs by domain.
    assert c.row(0) != c2.row(0)
    external = gate.Authority(c.context)
    with pytest.raises(ValueError, match="binding"):
        gate.query(c, i2, external, 0, (0,)*5)
    with pytest.raises(ValueError, match="replayed"):
        gate.query(c, i, external, 0, (0,)*5)
    with pytest.raises(ValueError, match="burned"):
        external.reserve(1, "wrong-stage-digest")


def test_authority_rollback_is_explicitly_not_solved():
    c, i = gate.setup(gate.W, 1, "zero_local", random.Random(3))
    # Recreating authority is an attack, not a safe restoration method.
    x, y = (0,)*5, (1,)*5
    _, first = gate.query(c, i, gate.Authority(c.context), 0, x)
    _, second = gate.query(c, i, gate.Authority(c.context), 0, y)
    assert tuple((b-a) % gate.MOD for a, b in zip(
        first["client_to_inference"], second["client_to_inference"], strict=True)) == (1,)*5


def test_exhaustive_full_view_and_shortcut_attacks():
    result = gate.leakage_controls()
    assert result["ideal_full_view_tv"] == result["zero_correction_full_view_tv"] == "0"
    for name in ("kernel_mask_tv", "even_multiplier_tv", "public_low_rank_tv", "cross_column_relation_tv"):
        assert result[name] == "1"
    assert result["q0_is_ideal_independent_s"] is False
    assert result["reused_r_reveals_delta"]
    assert result["fresh_two_column_zero_correction_tv"] == "0"


def test_public_linear_masks_require_ring_surjectivity():
    assert gate.linear_mask_gate(((2, 0), (0, 2)))["rank_mod2"] == 0
    assert gate.linear_mask_gate(((1,), (3,)))["surjective_ring32"] is False
    assert gate.linear_mask_gate(((1, 2, 3), (0, 1, 4)))["surjective_ring32"] is True


def test_no_malicious_output_integrity_is_claimed():
    c, i = gate.setup(gate.W, 1, "prepared", random.Random(7))
    r, s = c.row(0)
    reply = i.evaluate(0, tuple(-v % gate.MOD for v in r))
    bad = gate.combine(gate.combine(reply, (1, 0, 0)), s)
    assert bad == (1, 0, 0)  # Expected zero; no check can detect tampering here.


@pytest.mark.parametrize("group", [1, 2, 3, 4])
def test_lookup_not_dense_mac_disguised_as_prg(group):
    rng = random.Random(9)
    for _ in range(5):
        x = tuple(rng.randrange(-gate.MOD, gate.MOD) for _ in range(5))
        expect = tuple(sum(gate.W[a][b]*x[b] for b in range(5)) & 0xffffffff for a in range(3))
        assert gate.lookup_mv(gate.W, x, group) == expect
    cost = gate.lookup_ledger(3, 5, group)
    table = gate.lookup_tables(gate.W, group)
    assert cost["table_bytes"] == 4*sum(len(t)*len(t[0]) for t in table)
    assert cost["logical_table_read_bytes_per_row"] == 32*len(table)*3*4


def test_independent_payload_ledger_and_renewal():
    # 7 successful + 2 burned, capacity4 -> THREE full prepaid epochs.
    cost = gate.ledger(3, 5, 7, 4, "prepared", discarded=2)
    assert cost["epochs"] == 3 and cost["issued_columns"] == 12
    assert cost["directed_bytes"] == {
        "preparation_to_client": 3*(64+64),
        "preparation_to_inference": 3*64+12*3*4,
        "client_to_inference": 7*(40+5*4),
        "inference_to_client": 7*(40+3*4),
    }
    assert cost["all_link_bytes"] == 384+336+420+364
    assert cost["preparation_dense_MACs"] == 12*3*5
    for mode in gate.MODES:
        candidate = gate.ledger(3, 5, 7, 4, mode, discarded=2)
        assert sum(candidate["directed_bytes"].values()) == candidate["all_link_bytes"]
        assert candidate["client_process_RSS_bytes"] is None
        assert gate.admission(candidate, cost)["scale_admitted"] is False


@pytest.mark.parametrize("mode,setup_words,client_words,inference_words", [
    ("prepared", {"r": 20, "s": 12}, {"r": 15, "s": 9}, {}),
    ("zero_tape", {"r": 20}, {"r": 15}, {}),
    ("zero_local", {}, {"r": 15}, {}),
    ("direct_weight", {}, {"r": 15, "q": 9}, {"q": 9}),
])
def test_intercepted_prg_words_match_every_role_ledger(
        monkeypatch, mode, setup_words, client_words, inference_words):
    # Literal independent counts: Q4, m3, n5; three successful expansions.
    # Observe actual expand invocations, not formulas reconstructed from ledger.
    original = gate.expand
    observed = {}

    def intercepted(seed, context, index, label, count):
        observed[label] = observed.get(label, 0)+count
        return original(seed, context, index, label, count)

    monkeypatch.setattr(gate, "expand", intercepted)
    client, inference = gate.setup(gate.W, 4, mode, random.Random(51))
    cost = gate.ledger(3, 5, 3, 4, mode)
    assert observed == setup_words
    assert sum(observed.values())*4 == cost["preparation_PRG_output_bytes"]
    observed.clear()
    for index in range(3):
        client.row(index)
    assert observed == client_words
    assert sum(observed.values()) == cost["client_PRG_output_words"]
    assert sum(observed.values())*4 == cost["client_PRG_output_bytes"]
    observed.clear()
    for index in range(3):
        inference.row(index)
    assert observed == inference_words
    assert sum(observed.values())*4 == cost["inference_PRG_output_bytes"]


def test_discarded_capacity_does_not_claim_failed_online_bytes_are_paid():
    for mode in gate.MODES:
        cost = gate.ledger(3, 5, 3, 4, mode, discarded=2)
        assert cost["issued_columns"] == 8
        assert cost["directed_bytes"]["client_to_inference"] == 3*(40+5*4)
        assert cost["directed_bytes"]["inference_to_client"] == 3*(40+3*4)
        assert cost["directed_totals_exclude_failed_attempt_online_traffic"] is True
        assert cost["failed_attempt_online_bytes"] == {
            "client_to_inference": None, "inference_to_client": None}
        assert "failed-attempt online traffic" in cost["scope"]


def test_zero_tape_moves_material_and_direct_weight_adds_work():
    baseline = gate.ledger(9728, 896, 288, 128, "prepared", discarded=1)
    tape = gate.ledger(9728, 896, 288, 128, "zero_tape", discarded=1)
    direct = gate.ledger(9728, 896, 288, 128, "direct_weight", discarded=1)
    assert tape["client_external_tape_write_bytes"] == 384*9728*4
    assert tape["all_link_bytes"] == baseline["all_link_bytes"]-3*32
    assert direct["client_dense_MACs"] == 288*9728*896
    assert direct["client_W_bytes"] == 9728*896
    assert direct["client_paged_W_read_bytes"] == direct["client_dense_MACs"]
    assert not gate.admission(tape, baseline)["not_a_client_output_tape"]
    assert not gate.admission(direct, baseline)["no_added_dense_client_product"]
    lookup = gate.lookup_stage_ledger(9728, 896, 288, 128, discarded=1)
    assert lookup["public_subset_table_bytes"] == 139460608
    assert lookup["client_external_public_table_read_bytes"] == 288*278921216
    assert lookup["client_dense_MACs"] == 0
    assert not gate.admission(lookup, baseline)["traffic_strictly_beats_baseline"]
    assert not gate.admission(lookup, baseline)["client_ring_additions_no_greater_than_baseline"]


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_bad_capacity_rejected(bad):
    with pytest.raises(ValueError):
        gate.setup(gate.W, bad, "prepared", random.Random(1))
    with pytest.raises(ValueError):
        gate.ledger(3, 5, 7, bad, "prepared")


def test_source_provenance_has_exact_page_and_file_hashes():
    sources = gate.provenance()
    assert len(sources["papers"]) == 4
    for paper in sources["papers"]:
        assert len(paper["pdf_sha256"]) == len(paper["text_sha256"]) == 64
        assert all(len(e["page_text_sha256"]) == 64 and e["excerpt"] for e in paper["reviewed_excerpts"])


def test_source_text_hash_drift_rejected(monkeypatch):
    monkeypatch.setitem(gate.TEXT_HASHES, "2025/1223", "0"*64)
    with pytest.raises(ValueError, match="text hash drift"):
        gate.provenance()


def test_source_pdf_hash_drift_rejected(tmp_path):
    evidence = tmp_path/"docs/evidence"
    evidence.mkdir(parents=True)
    manifest = json.loads((gate.ROOT/"docs/evidence/compact-setup-fulltext-review.json").read_text())
    (evidence/"compact-setup-fulltext-review.json").write_text(json.dumps(manifest))
    (tmp_path/"papers").mkdir()
    (tmp_path/"papers/2025-1223.pdf").write_bytes(b"changed primary source")
    with pytest.raises(ValueError, match="PDF hash drift"):
        gate.provenance(tmp_path)

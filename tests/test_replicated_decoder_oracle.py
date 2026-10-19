"""Independent arithmetic and full-prefix controls; no wheel/checkpoint needed."""

import importlib.util
from itertools import repeat
from pathlib import Path
import struct

import numpy as np
import pytest

from replicated_decoder_oracle import (
    CHECKED_RANGE_LIMIT, GAP, Q, CheckedRangeSchedule, OracleError, PublicMatrix,
    RangeCheck, Tape, _assert_fixture_gqa_coverage, adapt_tapes, array, compare_trace, linear,
    generated_native_layout, mul, reciprocal, reconstruct_test_shares, ring, rmsnorm, rotary_half,
    rsqrt, run_decoder, silu, softmax, source_f32, stable_argmax,
)


class FixtureTape(Tape):
    """Test-data authoring ONLY. Replay always uses strict Tape, never this class.

    Named deterministic nonzero low residues provide fully explicit captured
    batches after authoring, without coupling to production PRSS or arithmetic.
    """

    def __init__(self, mode="mixed"):
        super().__init__({})
        self.mode = mode

    def truncate(self, name, raw, shift=18):
        raw = np.asarray(raw, dtype=object)
        if name in self.batches:
            raise OracleError("duplicate fixture operation")
        unit = 1 << shift
        start = sum((i+1)*ord(c) for i, c in enumerate(name))
        residues = [0 if self.mode == "zero" else unit-1 if self.mode == "ceil"
                    else (start + i*104729) % unit for i in range(raw.size)]
        self.batches[name] = {"shift": shift, "residues": np.array(residues, dtype=object).reshape(raw.shape)}
        return super().truncate(name, raw, shift)


def offline_module(relative):
    path = Path(__file__).resolve().parents[1] / "paper/reproduction-contracts" / relative
    spec = importlib.util.spec_from_file_location("independent_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("raw", [-GAP, -Q-1, -Q, -1, 0, 1, Q-1, Q, GAP-1])
@pytest.mark.parametrize("residue", [0, 1, Q//2, Q-1])
def test_signed_rounding_matches_independent_source_high_bit_formula(raw, residue):
    shared = offline_module("shared_profile_oracle.py")
    tape = Tape({"op": {"shift": 18, "residues": residue}})
    assert tape.truncate("op", raw).item() == shared.puma_trunc(raw, 64, 18, residue + (12345 << 18))
    tape.finish()


def test_rounding_exact_distribution_and_negative_floor():
    # Exhaust all low residues for a smaller shift, including negative values.
    for value in (-33, -16, -1, 0, 7, 15, 31):
        results = []
        for residue in range(16):
            results.append(Tape({"x": {"shift": 4, "residues": residue}}).truncate("x", value, 4).item())
        floor, rem = divmod(value, 16)
        assert results.count(floor+1) == rem
        assert set(results) <= {floor, floor+1}


def test_truncation_rejects_wrap_missing_excess_reuse_shape_and_noninteger():
    for raw in (GAP, -GAP-1, 1 << 64):
        with pytest.raises(OracleError, match="quarter-gap"):
            Tape({"x": {"shift": 18, "residues": 0}}).truncate("x", raw)
    with pytest.raises(OracleError, match="missing"):
        Tape({}).truncate("x", 0)
    tape = Tape({"x": {"shift": 18, "residues": 0}})
    with pytest.raises(OracleError, match="excess"):
        tape.finish()
    tape.truncate("x", 0)
    with pytest.raises(OracleError, match="repeated"):
        tape.truncate("x", 0)
    for residues in ([0], -1, Q):
        with pytest.raises(OracleError, match="shape/residue"):
            Tape({"x": {"shift": 18, "residues": residues}}).truncate("x", 0)
    with pytest.raises(OracleError, match="integer"):
        Tape({"x": {"shift": 18, "residues": 0}}).truncate("x", 0.5)


def test_explicit_physical_adapter_reorders_tiles_and_rejects_uncounted_draws():
    physical = {"tile.a": {"shift": 18, "masks": [2**64-1, 2**40+7]},
                "tile.b": {"shift": 18, "masks": [9]}}
    layout = {"dot": {"shift": 18, "shape": [3], "lanes": [["tile.b", 0], ["tile.a", 1], ["tile.a", 0]]}}
    tape = adapt_tapes(physical, layout)
    assert tape.batches["dot"]["residues"].tolist() == [9, 7, Q-1]
    tape.truncate("dot", array([0, 0, 0]))
    tape.finish()
    with pytest.raises(OracleError, match="excess"):
        adapt_tapes(physical, {})
    repeated = {"dot": {"shift": 18, "shape": [3], "lanes": [["tile.a", 0]]*3}}
    with pytest.raises(OracleError, match="reused"):
        adapt_tapes(physical, repeated)
    mismatch = {"dot": {"shift": 5, "shape": [3], "lanes": layout["dot"]["lanes"]}}
    with pytest.raises(OracleError, match="shift mismatch"):
        adapt_tapes(physical, mismatch)


def test_complete_dot_truncates_once_not_per_element_or_tile():
    # Individually both terms round down to zero; full contraction rounds to one.
    tape = Tape({"dot": {"shift": 18, "residues": [[0]]}})
    assert linear([[1, 1]], [[Q-1, Q-1]], tape, "dot").tolist() == [[1]]
    tape.finish()
    # A long contraction exercises exact object accumulation beyond native tiles.
    n = 2051
    tape = Tape({"dot": {"shift": 18, "residues": [[Q-1]]}})
    assert linear([[1]*n], [[Q-1]*n], tape, "dot").item() == n
    # Cancelling unwrapped dot is legal; a wrapped *final* dot is not.
    tape = Tape({"dot": {"shift": 18, "residues": [0]}})
    assert linear([1 << 32, 1 << 32], [[1 << 32, -(1 << 32)]], tape, "dot").item() == 0
    with pytest.raises(OracleError, match="quarter-gap"):
        linear([1 << 32], [[1 << 32]], FixtureTape(), "dot")


def test_source_conversion_binary32_toward_zero_and_range():
    assert source_f32(struct.pack("<4f", 0.1, -0.1, 1/Q, -1/Q)).tolist() == [26214, -26214, 1, -1]
    assert source_f32(struct.pack("<f", 1 << 22)).item() == 1 << 40
    for value in (float("nan"), float("inf"), 1e38, (1 << 22)+1):
        with pytest.raises(OracleError, match="source Q18 range"):
            source_f32(struct.pack("<f", value))
    with pytest.raises(OracleError):
        source_f32(b"123")


@pytest.mark.parametrize("value", [1, Q//4, Q//2, Q, 2*Q, 4*Q, (1 << 34)-1])
def test_hal_reciprocal_and_rsqrt_match_preexisting_independent_fixture(value):
    shared = offline_module("shared_profile_oracle.py")
    t = FixtureTape("zero")
    assert rsqrt(value, t, "rs").item() == shared.puma_rsqrt(value, repeat(0))
    assert reciprocal(value, t, "rec").item() == shared.puma_reciprocal(value, repeat(0))
    assert len([name for name in t.batches if name.startswith("rs/")]) == 6
    assert len([name for name in t.batches if name.startswith("rec/")]) == 6


def test_mixed_tapes_match_independent_scalar_hal_not_only_zero_rounding():
    shared = offline_module("shared_profile_oracle.py")
    for fn, reference in ((rsqrt, shared.puma_rsqrt), (reciprocal, shared.puma_reciprocal)):
        for value in (37, Q-1, Q+1, 9*Q+7):
            t = FixtureTape()
            actual = fn(value, t, "op").item()
            residues = [int(batch["residues"]) for batch in t.batches.values()]
            assert actual == reference(value, iter(residues))


@pytest.mark.parametrize("partitioned", [False, True])
def test_rmsnorm_epsilon_one_separate_rounding_and_public_partitions(partitioned):
    shared = offline_module("shared_profile_oracle.py")
    row, gamma = [Q//4, -Q//2, Q//8], [Q, Q//2, 2*Q]
    t = FixtureTape("zero")
    actual = rmsnorm(row, gamma, t, "norm", partitioned)
    assert actual.tolist() == shared.puma_rmsnorm(row, gamma, repeat(0))
    replay = Tape(t.batches)
    assert rmsnorm(row, gamma, replay, "norm", not partitioned).tolist() == actual.tolist()
    replay.finish()
    zero = rmsnorm([0, 0], [Q, Q], FixtureTape(), "zero", partitioned)
    assert zero.tolist() == [0, 0]


def test_rotary_separate_rounding_not_accumulated_product():
    # Each half sums two separately rounded signed products, not a joint dot.
    t = FixtureTape("zero")
    actual = rotary_half([1, 1], [Q//2], [Q//2], t, "rope")
    assert actual.tolist() == [-1, 0]  # combined-dot wrong result would [0,1]
    assert len(t.batches) == 2


def test_silu_guard_cutoffs_identity_zero_and_per_coefficient_truncation():
    xs = [-1000000*Q, -8*Q-1, -8*Q, -4*Q-1, -4*Q, 0, 4*Q, 4*Q+1, 1000000*Q]
    t = FixtureTape("zero")
    out = silu(xs, t, "act").tolist()
    assert out[0:2] == [0, 0]
    assert out[5] == 2229  # pinned approximation, NOT exact analytic SiLU(0)
    assert out[-2:] == xs[-2:]
    assert len(t.batches) == 9  # 3 powers, 2 low terms, 4 high terms
    # Independently expand floor equations at a nonintegral coefficient point.
    x = -4*Q-1
    x2 = x*x//Q
    assert out[3] == -80413 + (-21489*x//Q) + (-1453*x2//Q)
    replay = Tape(t.batches)
    assert silu(xs, replay, "act").tolist() == out
    replay.finish()


def test_softmax_strict_minus14_cutoff_guard_and_all_draws():
    t = FixtureTape("ceil")
    values = softmax([0, -14*Q, -14*Q+1, -1000000*Q], t, "sm")
    assert values[0] > 0
    assert values[1] == values[3] == 0
    assert values[2] == 1  # ceil tape distinguishes strict cutoff by one Q18 unit
    assert t.batches["sm/reduce"]["shift"] == 5
    assert len(t.batches) == 13
    assert all(b["residues"].size == (1 if "/reciprocal/" in name else 4)
               for name, b in t.batches.items())
    replay = Tape(t.batches)
    assert softmax([0, -14*Q, -14*Q+1, -1000000*Q], replay, "sm").tolist() == values.tolist()
    replay.finish()


def test_streamed_public_weight_blocks_keep_one_complete_contraction_round():
    reads = []
    shape = (10000, 257)  # source exceeds per-array oracle bound

    def reader(start, stop):
        reads.append((start, stop))
        return np.array([[Q-1]*257 for _ in range(stop-start)], dtype=object)

    weight = PublicMatrix(shape, reader)
    t = FixtureTape("zero")
    out = linear([[1]*257], weight, t, "stream")
    assert out.shape == (1, 10000)
    assert set(out.flat) == {256}
    assert len(t.batches) == 1
    assert len(reads) > 1 and all((b-a)*shape[1] <= 65536 for a, b in reads)
    assert reads[0][0] == 0 and reads[-1][1] == shape[0]
    with pytest.raises(OracleError, match="reader shape"):
        PublicMatrix((2, 3), lambda a, b: [[0]]).rows(0, 1)
    with pytest.raises(OracleError, match="bounded public"):
        weight.rows(0, 129)


def generated_fixture(family, layers=2, *, wide=False, head_dim=18):
    """Nonconstant source-derived-compatible geometry, no production imports."""
    hidden, mid, heads, kv, dim = (72, 131, 4, 2, 18) if wide else (4, 6, 2, 1, 2)
    if wide:
        dim, hidden = head_dim, 4*head_dim
    cfg = dict(model_type=family, hidden_size=hidden, intermediate_size=mid,
               num_hidden_layers=layers, num_attention_heads=heads,
               num_key_value_heads=kv, head_dim=dim, vocab_size=5,
               attention_bias=family == "qwen2", hidden_act="silu",
               tie_word_embeddings=False)
    weights = {}

    def put(name, shape, salt=0):
        count = int(np.prod(shape))
        weights[name] = np.array([((i*7+salt)%13-6)*(Q//(1024 if wide else 128)) for i in range(count)], dtype=object).reshape(shape)

    weights["model.embed_tokens.weight"] = array([
        [Q//2, Q//4, -Q//8, Q//16], [-Q//3, Q//5, Q//7, -Q//9],
        [Q//6, -Q//4, Q//3, Q//8], [Q//8, Q//9, -Q//7, Q//6],
        [-Q//5, -Q//6, Q//7, Q//8]])
    if wide:
        weights["model.embed_tokens.weight"] = array([
            [(row*11+col*3)%17*(Q//128)-Q//16 for col in range(hidden)] for row in range(5)])
        weights["model.embed_tokens.weight"][0, 0] = Q//4
        weights["model.embed_tokens.weight"][1, 0] = -Q//4
    weights["model.norm.weight"] = array([Q]*hidden)
    put("lm_head.weight", (5, hidden), 4)
    for layer in range(layers):
        src = f"model.layers.{layer}."
        weights[src + "input_layernorm.weight"] = array([Q]*hidden)
        weights[src + "post_attention_layernorm.weight"] = array([Q]*hidden)
        for kind, count in (("q", heads), ("k", kv), ("v", kv)):
            put(src + f"self_attn.{kind}_proj.weight", (count*dim, hidden), ord(kind)+layer)
            if family == "qwen2":
                put(src + f"self_attn.{kind}_proj.bias", (count*dim,), ord(kind))
        put(src + "self_attn.o_proj.weight", (hidden, heads*dim), layer)
        for kind in ("gate", "up"):
            put(src + f"mlp.{kind}_proj.weight", (mid, hidden), len(kind)+layer)
        put(src + "mlp.down_proj.weight", (hidden, mid), layer+2)
        if family == "qwen3":
            weights[src + "self_attn.q_norm.weight"] = array([Q]*dim)
            weights[src + "self_attn.k_norm.weight"] = array([Q]*dim)
    # Frozen public rows for width=2 (frequency=1); no host libm at replay.
    rotary = {0: ([Q], [0]), 1: ([141637], [220586]),
              2: ([-109090], [238366]), 3: ([-259520], [36994])}
    if wide:
        rotary = {position: (cos*(dim//2), sin*(dim//2)) for position, (cos, sin) in rotary.items()}
    return cfg, weights, rotary


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_wide_multilayer_gqa_has_distinct_groups_and_rejects_wrong_value_group(family):
    cfg, weights, rotary = generated_fixture(family, wide=True)
    authored = FixtureTape()
    expected = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    _assert_fixture_gqa_coverage(cfg, expected.trace, 2, authored.batches)
    replay = run_decoder(cfg, weights, [0, 1], 2, Tape(authored.batches), rotary)
    compare_trace(expected.trace, replay.trace)
    for selection in range(2):
        for layer in range(2):
            p = f"s{selection}/layer.{layer}/"
            assert expected.trace[p+"full_prefix_k"].shape == (2, 2+selection, 18)
            values = expected.trace[p+"full_prefix_v"]
            assert not np.array_equal(values[0], values[1]), "periodic fixture must not hide a GQA routing error"
            name = p+"context/h2/r1"
            wrong = linear(expected.trace[p+"probabilities"][2, 1, :2], values[0, :2].T,
                           Tape({name: authored.batches[name]}), name)
            assert not np.array_equal(wrong, expected.trace[p+"context"][2, 1])
            assert np.all(expected.trace[p+"probabilities"][:, 0, 1:] == 0)
            assert expected.trace[p+"gate_up"].shape == (2+selection, 131)
    assert expected.trace["s1/logits"].shape == (5,)
    degenerate = dict(expected.trace)
    repeated = expected.trace["s0/layer.0/full_prefix_v"].copy()
    repeated[1] = repeated[0]
    degenerate["s0/layer.0/full_prefix_v"] = repeated
    with pytest.raises(OracleError, match="degenerate"):
        _assert_fixture_gqa_coverage(cfg, degenerate, 2)


@pytest.mark.parametrize("head_dim", [18, 32])
def test_complete_and_tail_guard_tiles_preserve_values_and_sticky_late_failure(head_dim):
    cfg, weights, rotary = generated_fixture("qwen3", wide=True, head_dim=head_dim)
    hidden = cfg["hidden_size"]
    weights["lm_head.weight"] = np.zeros((5, hidden), dtype=object)
    weights["lm_head.weight"][0, 0], weights["lm_head.weight"][2, 0] = Q//32, -Q//32
    names = [f"s{s}/layer.{layer}/{kind}/input" for s in range(2)
             for layer in range(2) for kind in ("input_norm", "post_attention_norm", "q_norm", "k_norm")]
    names += [f"s{s}/final_norm/input" for s in range(2)]
    schedule = CheckedRangeSchedule.from_mapping({name: RangeCheck() for name in names})
    authored = FixtureTape()
    ordinary = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    accepted = run_decoder(cfg, weights, [0, 1], 2, Tape(authored.batches), rotary, checked_ranges=schedule)
    assert ordinary.tokens[0] == accepted.tokens[0] == 2
    compare_trace(ordinary.trace, {name: accepted.trace[name] for name in ordinary.trace})
    first = accepted.trace["s0/layer.0/input_norm/input/checked_range/all_valid"]
    assert first.tolist() == [1, 1]  # 144 lanes with tail, or 256 complete lanes
    weights["model.embed_tokens.weight"][2, 0] = CHECKED_RANGE_LIMIT+1
    failed_tape = FixtureTape()
    rejected = run_decoder(cfg, weights, [0, 1], 2, failed_tape, rotary, checked_ranges=schedule)
    assert not rejected.accepted and rejected.tokens == []
    assert rejected.trace["s0/selected"].item() == 2
    at = "s1/layer.0/input_norm/input/checked_range/"
    assert rejected.trace[at+"all_valid"].tolist() == ([1, 0] if head_dim == 18 else [1, 1, 0])
    assert rejected.trace[at+"tile_sticky"].tolist() == rejected.trace[at+"all_valid"].tolist()
    assert rejected.trace["s1/layer.1/q_norm/input/checked_range/sticky"].item() == 0
    strict = run_decoder(cfg, weights, [0, 1], 2, Tape(failed_tape.batches), rotary, checked_ranges=schedule)
    compare_trace(rejected.trace, strict.trace)
    for name in (at+"valid", at+"all_valid", at+"tile_sticky"):
        incomplete = dict(strict.trace)
        incomplete.pop(name)
        with pytest.raises(OracleError, match="missing"):
            compare_trace(rejected.trace, incomplete)


def _public_native_events(stage_lanes, *, node=10, offset=0):
    """Synthetic PUBLIC schedule controls, not native execution evidence."""
    return [dict(binding=f"b{offset+i}", node=node, lanes=lanes, loop_offset=0,
                 invocation=0, tensor_step=i) for i, lanes in enumerate(stage_lanes)]


def test_public_map_multilayer_gqa_norm_rows_are_exact_not_just_same_total():
    cfg, _, _ = generated_fixture("qwen3", wide=True)
    stages = [18, 1, 1, 1, 1, 1, 1, 1, 18, 18]
    events = _public_native_events(stages*8)
    spans = [dict(selection=0, id="layer.1.q_norm", start=10, end=11)]
    layout = generated_native_layout(cfg, 2, 2, events, spans)
    assert layout["s0/layer.1/q_norm/normalize"]["shape"] == [4, 2, 18]
    assert layout["s0/layer.1/q_norm/mean"]["shape"] == [4, 2]
    # Keep identical total vector lanes while corrupting individual row widths.
    broken = [dict(event) for event in events]
    broken[0]["lanes"], broken[10]["lanes"] = 17, 19
    with pytest.raises(OracleError, match="row/tile widths"):
        generated_native_layout(cfg, 2, 2, broken, spans)
    with pytest.raises(OracleError, match="overlapping"):
        generated_native_layout(cfg, 2, 2, events, spans*2)


def test_public_map_full_prefix_private_contractions_retain_every_actual_tail():
    cfg, _, _ = generated_fixture("qwen3", wide=True)
    # Actual compiler matrix rule: min(MAX_LANES/inner, 128) output lanes.
    # At visible key count 71, 18-wide context contracts in 14+4 lane tiles.
    events, count = [], 0
    for head in range(4):
        for row in range(71):
            tile = min(1024//(row+1), 128)
            for start in range(0, 18, tile):
                events.extend(_public_native_events([min(tile, 18-start)], node=10+count, offset=count))
                count += 1
    spans = [dict(selection=7, id="layer.1.attention_values", start=10, end=10+count)]
    layout = generated_native_layout(cfg, 64, 8, events, spans)
    tail = layout["s7/layer.1/context/h3/r70"]
    assert tail["shape"] == [18]
    assert len({binding for binding, _ in tail["lanes"]}) == 2
    assert sum(len(spec["lanes"]) for spec in layout.values()) == 4*71*18
    with pytest.raises(OracleError, match="tile crossing"):
        generated_native_layout(cfg, 64, 8, events[:-1], spans)


def test_public_map_activation_tails_account_for_each_term_and_missing_excess_draw():
    cfg, _, _ = generated_fixture("llama", wide=True)
    events = []
    for start in range(0, 71*131, 128):
        chunk = _public_native_events([min(128, 71*131-start)]*9, offset=len(events))
        for event in chunk:
            event["loop_offset"] = start
        events.extend(chunk)
    spans = [dict(selection=7, id="layer.1.silu", start=10, end=11)]
    layout = generated_native_layout(cfg, 64, 8, events, spans)
    physical = {event["binding"]: {"shift": 18, "masks": [Q-1]*event["lanes"]} for event in events}
    tape = adapt_tapes(physical, layout)
    assert len(tape.batches) == 9
    assert all(batch["residues"].shape == (71, 131) for batch in tape.batches.values())
    for name in layout:
        tape.truncate(name, np.zeros((71, 131), dtype=object))
    tape.finish()
    physical["unused"] = {"shift": 18, "masks": [0]}
    with pytest.raises(OracleError, match="excess"):
        adapt_tapes(physical, layout)
    physical.pop("unused")
    physical.pop(events[-1]["binding"])
    with pytest.raises(OracleError, match="missing"):
        adapt_tapes(physical, layout)


@pytest.mark.parametrize("field,value", [("num_attention_heads", 0), ("num_key_value_heads", 0),
                                        ("head_dim", 129), ("hidden_size", 0),
                                        ("intermediate_size", False), ("num_hidden_layers", 0)])
def test_public_geometry_rejects_before_division_or_any_tape_consumption(field, value):
    cfg, weights, rotary = generated_fixture("qwen3", wide=True)
    cfg[field] = value
    with pytest.raises(OracleError, match="geometry"):
        generated_native_layout(cfg, 2, 2, [], [])
    tape = FixtureTape()
    with pytest.raises(OracleError, match="geometry"):
        run_decoder(cfg, weights, [0, 1], 2, tape, rotary)
    assert tape.batches == {}


def test_complete_full_vocabulary_dense_public_head_bounded_reader_and_scalar_recipe():
    """Complete head-only numeric control; NOT a checkpoint/model conformance claim."""
    vocab, calls = 151936, []

    def coefficients(row):
        if row == 0:
            return Q//32, 0
        if row == vocab-1:
            return -Q//16, 0
        return ((row*7)%13-6)*(Q//8192), ((row*11)%17-8)*(Q//8192)

    def read(start, stop):
        calls.append((start, stop))
        return array([coefficients(row) for row in range(start, stop)])

    matrix = PublicMatrix((vocab, 2), read)
    residues = np.array([(row*104729+7) % Q for row in range(vocab)], dtype=object)
    tape = Tape({"head": {"shift": 18, "residues": residues}})
    logits = linear([-Q//4, Q//16], matrix, tape, "head")
    tape.finish()
    # Separately pinned unwrapped scalar dot + one final stochastic truncation.
    expected = np.array([((-Q//4)*coefficients(row)[0]+(Q//16)*coefficients(row)[1]
                          + int(residues[row]))//Q for row in range(vocab)], dtype=object)
    compare_trace({"full_head": expected}, {"full_head": logits})
    assert stable_argmax(logits) == vocab-1
    assert calls[0][0] == 0 and calls[-1][1] == vocab
    assert all(stop-start <= 128 for start, stop in calls)
    assert all(left[1] == right[0] for left, right in zip(calls, calls[1:]))
    logits[-1] += 1
    with pytest.raises(OracleError, match="mismatch"):
        compare_trace({"full_head": expected}, {"full_head": logits})


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_complete_decoder_gqa_full_prefix_tapes_logits_and_private_feedback(family):
    cfg, weights, rotary = generated_fixture(family)
    authored = FixtureTape()
    expected = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    replay = Tape(authored.batches)
    actual = run_decoder(cfg, weights, [0, 1], 2, replay, rotary)
    compare_trace(expected.trace, actual.trace)
    assert len(actual.tokens) == 2
    assert actual.trace["s0/logits"].shape == (5,)
    assert actual.trace["s1/logits"].shape == (5,)
    assert actual.trace["s1/last_hidden"].shape == (4,)
    assert actual.trace["s1/input.tokens"].tolist() == [0, 1, actual.tokens[0]]
    assert actual.trace["s1/feedback"].tolist() == [0, 1, *actual.tokens]
    for selection in range(2):
        prefix = 2+selection
        for layer in range(2):
            p = f"s{selection}/layer.{layer}/"
            assert actual.trace[p + "full_prefix_k"].shape == (1, prefix, 2)
            assert actual.trace[p + "full_prefix_v"].shape == (1, prefix, 2)
            probs = actual.trace[p + "probabilities"]
            assert probs.shape == (2, prefix, prefix)
            assert all(probs[h, r, k] == 0 for h in range(2) for r in range(prefix) for k in range(r+1, prefix))
            assert any(v != 0 for v in actual.trace[p + "gate_up"].flat)
    assert any(v != 0 for v in actual.trace["s1/logits"].flat)
    assert not np.array_equal(actual.trace["s1/final_norm"][0], actual.trace["s1/final_norm"][1])
    # Single hidden corruption fails even with unchanged final selected IDs.
    corrupted = dict(actual.trace)
    corrupted["s0/layer.0/full_prefix_k"] = corrupted["s0/layer.0/full_prefix_k"].copy()
    corrupted["s0/layer.0/full_prefix_k"][0, 0, 0] += 1
    with pytest.raises(OracleError, match="trace mismatch"):
        compare_trace(expected.trace, corrupted)


def test_causal_prefix_independent_of_future_inputs_and_gqa_head_mapping():
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    first = run_decoder(cfg, weights, [0, 1], 1, FixtureTape("zero"), rotary)
    changed = run_decoder(cfg, weights, [0, 4], 1, FixtureTape("zero"), rotary)
    assert np.array_equal(first.trace["s0/final_norm"][0], changed.trace["s0/final_norm"][0])
    assert not np.array_equal(first.trace["s0/logits"], changed.trace["s0/logits"])
    # Both query heads attend same single KV head at first causal position.
    context = first.trace["s0/layer.0/context"]
    assert np.array_equal(context[0, 0], context[1, 0])


def test_full_decoder_streamed_public_sources_equal_array_fixture():
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    authored = FixtureTape()
    expected = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    streamed = {name: PublicMatrix(value.shape, lambda a, b, value=value: value[a:b])
                if value.ndim == 2 else value for name, value in weights.items()}
    actual = run_decoder(cfg, streamed, [0, 1], 2, Tape(authored.batches), rotary)
    compare_trace(expected.trace, actual.trace)


def test_stable_full_vocabulary_argmax_and_feedback_not_public_input_reuse():
    assert stable_argmax([1, 9, 9, -1]) == 1
    assert stable_argmax([-9, -9, -10]) == 0
    assert stable_argmax([0]*127 + [1]) == 127  # no sampled/head subset
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    weights["lm_head.weight"] = array([[0]*4]*5)
    out = run_decoder(cfg, weights, [3, 4], 2, FixtureTape(), rotary)
    assert out.tokens == [0, 0]
    assert out.trace["s1/input.tokens"].tolist() == [3, 4, 0]
    assert out.trace["s1/feedback"].tolist() == [3, 4, 0, 0]


def test_trace_missing_private_evidence_cannot_fabricate_equality():
    with pytest.raises(OracleError, match="nonempty"):
        compare_trace({}, {})
    for actual in ({}, {"x": None}, {"x": [0], "extra": [1]}, {"x": [[0]]}):
        with pytest.raises(OracleError):
            compare_trace({"x": [0]}, actual)
    assert reconstruct_test_shares([[2**64-1], [7], [2**64-8]]).tolist() == [-2]
    assert ring([2**64, 2**63, -2**63-1]).tolist() == [0, -2**63, 2**63-1]
    with pytest.raises(OracleError):
        reconstruct_test_shares([[1], [2]])


def test_decoder_unsupported_geometry_and_missing_public_inputs_fail_closed():
    cfg, weights, rotary = generated_fixture("llama")
    with pytest.raises(OracleError, match="rotary"):
        run_decoder(cfg, weights, [0, 1], 1, FixtureTape(), {})
    with pytest.raises(OracleError, match="unsupported"):
        run_decoder(dict(cfg, rope_scaling={"type": "linear", "factor": 2}), weights, [0], 1, FixtureTape(), rotary)
    with pytest.raises(OracleError, match="token domain"):
        run_decoder(cfg, weights, [5], 1, FixtureTape(), rotary)
    with pytest.raises(OracleError, match="source"):
        run_decoder(cfg, {}, [0], 1, FixtureTape(), rotary)


def guard_placements(cfg, outputs):
    names = []
    for selection in range(outputs):
        root = f"s{selection}/"
        for layer in range(cfg["num_hidden_layers"]):
            p = root + f"layer.{layer}/"
            names += [p + "input_norm/input", p + "post_attention_norm/input"]
            if cfg["model_type"] == "qwen3":
                names += [p + "q_norm/input", p + "k_norm/input"]
        names += [root + "final_norm/input"]
    return {name: RangeCheck() for name in names}


def test_checked_range_inclusive_endpoints_full_ring_and_exact_schema():
    bound = CHECKED_RANGE_LIMIT
    incoming = [-2**63, -bound-1, -bound, -1, 0, bound, bound+1, 2**63-1]
    safe, valid = RangeCheck().apply(incoming)
    assert valid.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert safe.tolist() == [0, 0, -bound, -1, 0, bound, 0, 0]
    # Core guard semantics are safe even for full signed interval endpoints,
    # without a subtraction that could wrap at MIN/MAX.
    for rule in (RangeCheck(-2**63, 2**63-1), RangeCheck(-2**63, -2**63, -2**63),
                 RangeCheck(2**63-1, 2**63-1, 2**63-1)):
        safe, valid = rule.apply(incoming)
        for original, selected, bit in zip(incoming, safe, valid):
            expected = int(rule.lower <= original <= rule.upper)
            assert bit == expected
            assert selected == (original if expected else rule.fallback)
    for args in ((-1.0, 1, 0), (False, 1, 0), (-2**63-1, 1, 0),
                 (-1, 2**63, 0), (1, -1, 0), (-1, 1, 2)):
        with pytest.raises(OracleError):
            RangeCheck(*args)


def test_checked_range_schedule_is_copied_frozen_and_no_private_auto_placement():
    from dataclasses import FrozenInstanceError

    placements = {"s0/final_norm/input": RangeCheck()}
    schedule = CheckedRangeSchedule.from_mapping(placements)
    placements.clear()
    assert len(schedule.entries) == 1
    with pytest.raises(FrozenInstanceError):
        schedule.entries = ()
    with pytest.raises(FrozenInstanceError):
        schedule.entries[0][1].fallback = 1
    for entries in ([('x/input', RangeCheck())],
                    (("x/input", RangeCheck()), ("x/input", RangeCheck())),
                    (("x", RangeCheck()),)):
        with pytest.raises(OracleError):
            CheckedRangeSchedule(entries)
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    with pytest.raises(OracleError, match="unused guard"):
        run_decoder(cfg, weights, [0, 1], 1, FixtureTape(), rotary,
                    checked_ranges=CheckedRangeSchedule((("s0/unknown/input", RangeCheck()),)))
    with pytest.raises(OracleError, match="immutable"):
        run_decoder(cfg, weights, [0, 1], 1, FixtureTape(), rotary, checked_ranges={})


@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3"])
def test_checked_range_accepted_unchanged_and_no_extra_truncations(family):
    cfg, weights, rotary = generated_fixture(family, layers=1)
    authored = FixtureTape()
    original = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    schedule = CheckedRangeSchedule.from_mapping(guard_placements(cfg, 2))
    replay = Tape(authored.batches)
    checked = run_decoder(cfg, weights, [0, 1], 2, replay, rotary, checked_ranges=schedule)
    assert checked.accepted and checked.tokens == original.tokens
    compare_trace(original.trace, {name: checked.trace[name] for name in original.trace})
    assert checked.trace["acceptance.valid"] == 1
    for name, _ in schedule.entries:
        assert np.array_equal(checked.trace[name], checked.trace[name + "/checked_range/safe"])
        assert checked.trace[name + "/checked_range/sticky"] == 1
    # Explicit empty schedule retains EXACT old trace as well as values/tapes.
    empty = run_decoder(cfg, weights, [0, 1], 2, Tape(authored.batches), rotary,
                        checked_ranges=CheckedRangeSchedule())
    compare_trace(original.trace, empty.trace)


@pytest.mark.parametrize("failure_site", ["early", "late"])
def test_checked_range_failure_sticky_full_request_no_returned_ids(failure_site):
    cfg, weights, rotary = generated_fixture("qwen3", layers=2)
    placements = guard_placements(cfg, 2)
    failed = "s0/layer.0/input_norm/input" if failure_site == "early" else "s1/final_norm/input"
    placements[failed] = RangeCheck(0, 0)
    schedule = CheckedRangeSchedule.from_mapping(placements)
    authored = FixtureTape()
    rejected = run_decoder(cfg, weights, [0, 1], 2, authored, rotary, checked_ranges=schedule)
    assert rejected.accepted is False and rejected.tokens == []
    assert rejected.trace["acceptance.valid"] == 0
    assert rejected.trace[failed + "/checked_range/all_valid"] == 0
    assert set(rejected.trace[failed + "/checked_range/safe"].flat) == {0}
    # Continued execution including internal selection/feedback is diagnostic
    # ONLY. No IDs are returned even if the failure occurs after first selection.
    assert "s1/logits" in rejected.trace and "s1/feedback" in rejected.trace
    assert len(rejected.trace["s1/feedback"]) == 4
    guard_keys = [name for name in rejected.trace if name.endswith("/checked_range/sticky")]
    sticky_values = [int(rejected.trace[name]) for name in guard_keys]
    first_failure = sticky_values.index(0)
    assert all(value == 0 for value in sticky_values[first_failure:])
    if failure_site == "early":
        assert rejected.trace["s1/final_norm/input/checked_range/all_valid"] == 1
        assert rejected.trace["s1/final_norm/input/checked_range/sticky"] == 0
    else:
        assert all(value == 1 for value in sticky_values[:first_failure])
    replay = run_decoder(cfg, weights, [0, 1], 2, Tape(authored.batches), rotary,
                         checked_ranges=schedule)
    assert replay.tokens == [] and not replay.accepted
    compare_trace(rejected.trace, replay.trace)


def test_checked_range_guard_precedes_unsafe_square_and_consumes_no_tape():
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    weights["model.embed_tokens.weight"][0, 0] = 2**63-1
    # Residual retains original invalid lane; every subsequent RMS input must
    # have its own public guard. Sticky failure does not authorize unsafe work.
    schedule = CheckedRangeSchedule.from_mapping(guard_placements(cfg, 1))
    t = FixtureTape()
    rejected = run_decoder(cfg, weights, [0], 1, t, rotary, checked_ranges=schedule)
    assert not rejected.accepted and rejected.tokens == []
    first = "s0/layer.0/input_norm/input/checked_range/"
    assert rejected.trace[first + "safe"][0, 0] == 0
    assert rejected.trace[first + "valid"][0, 0] == 0
    assert all("checked_range" not in name for name in t.batches)
    with pytest.raises(OracleError, match="quarter-gap"):
        run_decoder(cfg, weights, [0], 1, FixtureTape(), rotary)


def test_silu_operation_order_against_independent_pinned_scalar_recipe():
    # Pinned source coefficients and literal signed rounding, not oracle mul,
    # activation helpers, production arithmetic, or analytic floating SiLU.
    values = [-8*Q-1, -8*Q, -4*Q-1, -4*Q, -17, 0, 37, 4*Q, 4*Q+1]
    t = FixtureTape()
    actual = silu(values, t, "pinned")
    for lane, original in enumerate(values):
        def tr(raw, suffix):
            residue = int(t.batches["pinned/" + suffix]["residues"][lane])
            return (raw + residue)//Q

        x = original if -8*Q <= original <= 4*Q else 0
        x2 = tr(x*x, "x2")
        x4 = tr(x2*x2, "x4")
        x6 = tr(x2*x4, "x6")
        low = -80413 + tr(-21489*x, "low.x") + tr(-1453*x2, "low.x2")
        high = (2229 + tr(131072*x, "high.x") + tr(59806*x2, "high.x2")
                + tr(-2913*x4, "high.x4") + tr(71*x6, "high.x6"))
        expected = 0 if original < -8*Q else low if original < -4*Q else high if original <= 4*Q else original
        assert actual[lane] == expected


def test_softmax_operation_order_against_independent_pinned_scalar_recipe():
    values = [123, -Q+9, -14*Q+123, -14*Q+124, -100*Q]
    t = FixtureTape()
    actual = softmax(values, t, "pinned")

    def tr(raw, suffix, lane=None, shift=18):
        residues = t.batches["pinned/" + suffix]["residues"]
        residue = int(residues if lane is None else residues[lane])
        return (raw+residue)//(1 << shift)

    exponentials = []
    for lane, value in enumerate(values):
        z = value-max(values)
        keep = z > -14*Q
        e = Q + tr(z if keep else 0, "reduce", lane, 5)
        for square in range(5):
            e = tr(e*e, f"square.{square}", lane)
        exponentials.append(e if keep else 0)
    denominator = sum(exponentials)
    factor = 1 << (36-denominator.bit_length())
    c = tr(denominator*factor, "reciprocal/c")
    w = tr((763940-2*c)*factor, "reciprocal/w")
    error = Q-tr(denominator*w, "reciprocal/xw")
    r = tr(w*(Q+error), "reciprocal/r")
    error2 = tr(error*error, "reciprocal/e2")
    inverse = tr(r*(Q+error2), "reciprocal/out")
    assert actual.tolist() == [tr(e*inverse, "probabilities", lane) for lane, e in enumerate(exponentials)]


def test_rmsnorm_mixed_rows_order_against_independent_pinned_scalar_recipe():
    shared = offline_module("shared_profile_oracle.py")
    values = array([[Q//3, -Q//7, 17], [Q, 23, -Q//2]])
    gamma = [Q//2, Q, 2*Q]
    t = FixtureTape()
    actual = rmsnorm(values, gamma, t, "pinned", partitioned=True)
    for row in range(2):
        # Original independent scalar fixture consumes row lanes followed by
        # scalar denominator/rsqrt, row normalization, then row public gamma.
        masks = list(t.batches["pinned/squares"]["residues"][row])
        masks += [t.batches["pinned/mean"]["residues"][row]]
        for suffix in ("u", "u2", "u3", "u4", "polynomial", "out"):
            masks += [t.batches["pinned/rsqrt/" + suffix]["residues"][row]]
        masks += list(t.batches["pinned/normalize"]["residues"][row])
        masks += list(t.batches["pinned/gamma"]["residues"][row])
        assert actual[row].tolist() == shared.puma_rmsnorm(values[row], gamma, iter(masks))

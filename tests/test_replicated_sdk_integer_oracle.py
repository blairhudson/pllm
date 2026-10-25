"""D51 V5 mask-conditioned integer observations, NOT fixed SDK expectations.

Uses the pre-existing independent decoder and FixtureTape, never a publisher,
runtime source, native implementation, model loader or second evaluator. Source
recipe checks read Python/Rust text only. Both original prompts remain intact.

Q/K projections AND biases are identically zero. Their rotary products are zero
for every valid coefficient and every residue (TR(0,r)=0). Identity rows below
therefore stand for arbitrary valid rotary rows in this fixture ONLY; they are
not a native coefficient-generation/conformance claim.

Checked policy guards every RMSNorm input before any row evaluation, not merely
sites whose sampled values look unsafe. Source references: graph_driver.rs
153–192, compact_norm.rs::NormSink::{checked,checked_rows}, policy.rs::checked_bound.
SiLU(0)=2229 in the source polynomial, not zero. A zero gate-up must be established
from the up operand and actual truncation, not assumed from analytic SiLU.
"""

import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from replicated_decoder_oracle import (
    Q, CheckedRangeSchedule, RangeCheck, Tape, compare_trace, run_decoder,
)
from test_replicated_decoder_oracle import FixtureTape


ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "hello hello": (2, 2, 3, 3, 2, 2),
    "world hello": (2, 2, 4, 3, 2, 2),
}
MODES = ("zero", "ceil", "mixed")
PREFIX = "model.layers.0."
GUARDS = tuple(
    f"s{s}/{site}/input" for s in range(2) for site in (
        "layer.0/input_norm", "layer.0/post_attention_norm", "final_norm"))
SCHEDULE = CheckedRangeSchedule.from_mapping({
    name: RangeCheck(-256 * Q, 256 * Q, 0) for name in GUARDS})


class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
                "pllm", "torch", "transformers", "safetensors", "poulpy"}:
            raise AssertionError(f"runtime/model/native import forbidden: {fullname}")


@pytest.fixture(scope="module", autouse=True)
def forbid_runtime_factories():
    # Run with --noconftest so unrelated production/config fixtures never load.
    assert not any(name == "pllm._native" for name in sys.modules)
    trap = NoRuntime()
    sys.meta_path.insert(0, trap)
    try:
        yield
    finally:
        sys.meta_path.remove(trap)


def _same_ast(node, text):
    assert ast.dump(node) == ast.dump(ast.parse(text, mode="eval").body)


def v5_fixture():
    """Verify actual recipe constants without importing/calling its factory."""
    source = (ROOT / "tests/test_replicated_sdk.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "experiment_fixture")
    assignments = {node.targets[0].id: node.value for node in function.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    entries = dict(zip((ast.literal_eval(key) for key in assignments["config"].keys),
                       assignments["config"].values, strict=True))
    config = {key: ast.literal_eval(value) for key, value in entries.items()
              if key not in {"model_type", "intermediate_size", "attention_bias"}}
    _same_ast(entries["model_type"], "family")
    _same_ast(entries["intermediate_size"], "intermediate_size")
    _same_ast(entries["attention_bias"], 'family == "qwen2"')
    defaults = dict(zip((arg.arg for arg in function.args.kwonlyargs),
                        function.args.kw_defaults, strict=True))
    assert ast.literal_eval(defaults["intermediate_size"]) == 2
    config.update(model_type="qwen2", intermediate_size=2, attention_bias=True)
    assert (config["hidden_size"], config["num_hidden_layers"], config["vocab_size"]) == (2, 1, 5)
    assert ast.literal_eval(assignments["vocabulary"]) == {
        "<bos>": 0, "<eos>": 1, "<unk>": 2, "hello": 3, "world": 4}
    chain = next(node for node in ast.walk(function)
                 if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "active and key == 'model.embed_tokens.weight'")
    clauses = []
    while isinstance(chain, ast.If):
        assignment = next(node for node in chain.body if isinstance(node, ast.Assign))
        clauses.append((chain.test, assignment.value))
        if len(chain.orelse) == 1 and isinstance(chain.orelse[0], ast.If):
            chain = chain.orelse[0]
        else:
            final = next(node for node in chain.orelse if isinstance(node, ast.Assign))
            _same_ast(final.value, "np.zeros(shape)")
            break
    expected = (
        ('active and key == "model.embed_tokens.weight"',
         '[[0,0],[5/32,5/32],[0,0],[5/32,5/32],[-5/32,-5/32]]'),
        ('key == "model.embed_tokens.weight"', '[[1,0],[-1,0],[1,0],[-1,0],[-1,0]]'),
        ('active and key == "lm_head.weight"', '[[0,0],[1,0],[0,0],[0,0],[-1,0]]'),
        ('key == "lm_head.weight"', '[[0,0],[1,0],[-1,0],[0,0],[0,0]]'),
        ('"norm" in key', 'np.ones(shape)'),
        ('active and key.endswith("v_proj.weight")', 'np.eye(2)'),
        ('active and key.endswith("o_proj.weight")', '-np.eye(2)/2'),
        ('active and "mlp." in key', 'np.full(shape,1/64)'),
    )
    assert len(clauses) == len(expected)
    for (test, value), (expected_test, expected_value) in zip(clauses, expected, strict=True):
        _same_ast(test, expected_test)
        _same_ast(value, expected_value)
    assert Q == 262144
    assert Fraction(5, 32) * Q == 40960
    assert Fraction(1, 64) * Q == 4096
    weights = {
        "model.embed_tokens.weight": [[0, 0], [40960, 40960], [0, 0],
                                     [40960, 40960], [-40960, -40960]],
        "model.norm.weight": [Q, Q],
        "lm_head.weight": [[0, 0], [Q, 0], [0, 0], [0, 0], [-Q, 0]],
        PREFIX + "input_layernorm.weight": [Q, Q],
        PREFIX + "post_attention_layernorm.weight": [Q, Q],
    }
    for name in ("q", "k", "v"):
        weights[PREFIX + f"self_attn.{name}_proj.weight"] = (
            [[Q, 0], [0, Q]] if name == "v" else [[0, 0], [0, 0]])
        weights[PREFIX + f"self_attn.{name}_proj.bias"] = [0, 0]
    weights[PREFIX + "self_attn.o_proj.weight"] = [[-131072, 0], [0, -131072]]
    for name in ("gate", "up", "down"):
        weights[PREFIX + f"mlp.{name}_proj.weight"] = [[4096, 4096], [4096, 4096]]
    return config, weights


class ObservedFixtureTape(FixtureTape):
    """Instrumentation only: all arithmetic remains in existing FixtureTape/Tape."""
    def __init__(self, mode):
        super().__init__(mode)
        self.raw = {}

    def truncate(self, name, raw, shift=18):
        self.raw[name] = np.array(raw, dtype=object, copy=True)
        return super().truncate(name, raw, shift)


@pytest.fixture(scope="module")
def observations():
    config, weights = v5_fixture()
    rotary = {position: ([Q], [0]) for position in range(7)}
    rows = []
    for prompt, ids in COHORTS.items():
        for mode in MODES:
            authored = ObservedFixtureTape(mode)
            observed = run_decoder(config, weights, ids, 2, authored, rotary,
                                   checked_ranges=SCHEDULE)
            frozen = deepcopy(authored.batches)
            replay = Tape(frozen)
            repeated = run_decoder(config, weights, ids, 2, replay, rotary,
                                   checked_ranges=SCHEDULE)
            compare_trace(observed.trace, repeated.trace)
            assert observed.tokens == repeated.tokens and observed.accepted == repeated.accepted
            assert replay.used == set(frozen) == authored.used
            authored.finish()
            replay.finish()
            rows.append((prompt, mode, authored, observed))
    assert len(rows) == 6  # Exactly six authorings + six strict replays; no sweep.
    return rows


def test_source_guard_placement_and_fixture_recipe():
    v5_fixture()
    base = ROOT / "crates/pllm-compiler/src/replicated_decoder"
    driver = (base / "graph_driver.rs").read_text()
    body = driver[driver.index("pub(in crate::replicated_decoder) fn rms_norm"):]
    assert body.index("sink.checked_rows(original, width)?") < body.index("for row in S::rows")
    compact = (base / "compact_norm.rs").read_text()
    assert "self.graph.imports.checked_bound.is_some()" in compact
    policy = (base / "policy.rs").read_text()
    assert "i64::from(self.max_abs_activation) * super::ONE" in policy
    assert len(SCHEDULE.entries) == 6


def test_complete_mask_conditioned_runs_and_strict_replay(observations):
    for prompt, mode, authored, result in observations:
        assert result.accepted == bool(result.trace["acceptance.valid"].item())
        assert len(result.tokens) == 2 if result.accepted else result.tokens == []
        for selection in range(2):
            assert result.trace[f"s{selection}/input.tokens"].shape == (6 + selection,)
            assert result.trace[f"s{selection}/logits"].shape == (5,)
            assert result.trace[f"s{selection}/feedback"].shape == (7 + selection,)
        for site in GUARDS:
            assert site + "/checked_range/sticky" in result.trace
            assert result.trace[site].shape == result.trace[site + "/checked_range/valid"].shape
        print(json.dumps({"prompt": prompt, "mode": mode, "accepted": result.accepted,
            "accepted_ids": result.tokens,
            "diagnostic_selected": [int(result.trace[f"s{s}/selected"]) for s in range(2)],
            "guard_validity": {name: result.trace[name + "/checked_range/all_valid"].tolist()
                               for name in GUARDS},
            "batches": len(authored.batches),
            "residue_words": sum(b["residues"].size for b in authored.batches.values()),
            "scope": "mask-conditioned independent oracle; NOT universal SDK IDs"}, sort_keys=True))


def test_rotary_zero_and_silu_up_branch_are_not_idealized(observations):
    for _, _, tape, result in observations:
        for selection in range(2):
            p = f"s{selection}/layer.0/"
            for suffix in ("q_proj", "k_proj", "q_rotary", "k_rotary", "scores", "scaled_scores"):
                assert np.all(result.trace[p + suffix] == 0)
            for name, raw in tape.raw.items():
                if name.startswith(p + "q_rotary/") or name.startswith(p + "k_rotary/"):
                    assert np.all(raw == 0)
            gate = result.trace[p + "gate_proj"]
            up = result.trace[p + "up_proj"]
            activation = result.trace[p + "silu"]
            assert np.all(activation[gate == 0] == 2229)
            assert np.all(result.trace[p + "gate_up"][up == 0] == 0)


def test_local_interval_attempt_and_exact_head_invariant(observations):
    """Valid local enclosure, explicitly NOT a composition across changed raw inputs."""
    ambiguous = []
    for prompt, mode, tape, result in observations:
        local_non_singletons = 0
        for name, raw in tape.raw.items():
            batch = tape.batches[name]
            unit = 1 << batch["shift"]
            lo, hi = raw // unit, (raw + unit - 1) // unit
            actual = (raw + batch["residues"]) // unit
            assert np.all(lo <= actual) and np.all(actual <= hi)
            local_non_singletons += int(np.any(lo != hi))
        for selection in range(2):
            x = int(result.trace[f"s{selection}/last_hidden"][0])
            # All head raw terms are multiples of Q: this identity holds for
            # EVERY head residue, conditional on this supplied hidden word.
            assert result.trace[f"s{selection}/logits"].tolist() == [0, x, 0, 0, -x]
        assert local_non_singletons > 0
        ambiguous.append((prompt, mode, local_non_singletons))
    print("UNRESOLVED universal interval composition: later raw operands depend on prior "
          "rounding, RMS denominator/rsqrt buckets and SiLU/up correlations; local "
          "endpoints for one executed trace do not enclose the whole decoder.", ambiguous)


def test_full_residue_witnesses_are_available_without_new_capture_hooks(observations):
    # Pytest's ordinary -s log can carry an explicit finite counterexample.
    # Never output production PRSS seeds, owners, captures or model source data.
    for prompt in COHORTS:
        rows = [row for row in observations if row[0] == prompt]
        groups = {(tuple(row[3].tokens), row[3].accepted) for row in rows}
        if len(groups) <= 1:
            print(f"{prompt!r}: sampled schedules agree; UNIVERSAL outcome remains unproved")
            continue
        left = rows[0]
        right = next(row for row in rows[1:]
                     if (row[3].tokens, row[3].accepted) != (left[3].tokens, left[3].accepted))
        for _, mode, tape, result in (left, right):
            witness = {name: {"shift": batch["shift"], "residues": batch["residues"].tolist()}
                       for name, batch in tape.batches.items()}
            digest = hashlib.sha256(json.dumps(witness, sort_keys=True).encode()).hexdigest()
            print("COMPLETE INDEPENDENT RESIDUE WITNESS", json.dumps({
                "prompt": prompt, "input_ids": COHORTS[prompt], "outputs": 2,
                "mode": mode, "accepted_ids": result.tokens, "accepted": result.accepted,
                "diagnostic_selected": [int(result.trace[f"s{s}/selected"]) for s in range(2)],
                "guard_schedule": list(GUARDS), "tape_sha256": digest,
                "tape": witness}, sort_keys=True))
        print(f"{prompt!r}: fixed accepted-ID/validity assertion refuted by explicit tapes")

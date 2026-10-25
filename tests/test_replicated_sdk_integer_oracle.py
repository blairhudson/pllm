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
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from replicated_decoder_oracle import (
    GAP, Q, CheckedRangeSchedule, RangeCheck, Tape, compare_trace, reciprocal,
    rsqrt, run_decoder, silu,
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


# Bounds only: no point-valued decoder, native output or new rounding policy.
# Each pair encloses integers, not floats. Hulls deliberately lose correlations
# (never narrow them). Per-row denominator/inverse is formed ONCE and shared by
# both coordinates; a complete dot is accumulated before its single truncation.
MAX_PROOF_CASES = 128000
MAX_PROOF_WORK = 1000000


class ProofGap(ValueError):
    """An overapproximation/premise/budget failed; not an observed invalid run."""


class IntegerBounds:
    def __init__(self, case_limit=MAX_PROOF_CASES, work_limit=MAX_PROOF_WORK):
        assert 0 <= case_limit <= MAX_PROOF_CASES and 0 <= work_limit <= MAX_PROOF_WORK
        self.case_limit = case_limit
        self.work_limit = work_limit
        self.cases = 0
        self.work = 0
        self.cache = {}
        self.guards = {}
        self.rows = {}
        self.branch_guards = {}
        self.branch_rows = {}

    def charge(self, work=1, cases=0):
        if self.cases + cases > self.case_limit or self.work + work > self.work_limit:
            raise ProofGap(f"proof budget exhausted at cases={self.cases}, work={self.work}")
        self.cases += cases
        self.work += work

    def add(self, a, b):
        self.charge()
        out = (a[0] + b[0], a[1] + b[1])
        if not -(1 << 63) <= out[0] <= out[1] < 1 << 63:
            raise ProofGap(f"unwrapped affine/ring premise: {out}")
        return out

    def mul(self, a, b):
        self.charge()
        terms = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
        return min(terms), max(terms)

    def square(self, a):
        self.charge()
        return (0 if a[0] <= 0 <= a[1] else min(a[0]**2, a[1]**2),
                max(a[0]**2, a[1]**2))

    def trunc(self, raw, shift=18):
        self.charge()
        if not -GAP <= raw[0] <= raw[1] < GAP:
            raise ProofGap(f"unwrapped quarter-gap premise: {raw}")
        unit = 1 << shift
        return raw[0] // unit, (raw[1] + unit - 1) // unit

    def product(self, a, b):
        return self.trunc(self.mul(a, b))

    def scaled(self, a, coefficient):
        return self.mul(a, (coefficient, coefficient))

    def dot(self, row, coefficients):
        raw = (0, 0)
        for value, coefficient in zip(row, coefficients, strict=True):
            raw = self.add(raw, self.scaled(value, coefficient))
        return self.trunc(raw)

    def point_scalar(self, fn, value):
        """Exactly six source truncations: <=64 endpoint branches, not a sweep."""
        key = fn.__name__, value
        if key in self.cache:
            return self.cache[key]
        values = []
        for choices in range(64):
            # Conservative scalar arithmetic/guard-step allowance, not Python
            # bytecodes or a physical RAM/runtime claim. Limits never increase.
            self.charge(512, cases=1)
            tape = EndpointTape(choices)
            values.append(int(fn(value, tape, "proof")))
            assert len(tape.used) == 6
            tape.finish()
        self.cache[key] = min(values), max(values)
        return self.cache[key]

    def inverse_root(self, incoming):
        if not 0 < incoming[0] <= incoming[1] < (1 << 35):
            raise ProofGap(f"rsqrt positive bit-reversal domain: {incoming}")
        if incoming[0] == incoming[1]:
            return self.point_scalar(rsqrt, incoming[0])
        # No polynomial monotonicity assumption. Split all possible bit lengths;
        # separately enclose every source multiply/TR and the accumulated poly.
        outputs = []
        for exponent in range(incoming[0].bit_length(), incoming[1].bit_length()+1):
            self.charge(cases=1)
            x = (max(incoming[0], 1 << (exponent-1)),
                 min(incoming[1], (1 << exponent)-1))
            factor = 1 << (35-exponent)
            u = self.product(x, (factor, factor))
            u2 = self.trunc(self.square(u))
            u3 = self.product(u2, u)
            u4 = self.product(u3, u)
            raw = (0, 0)
            for power, coefficient in zip((u, u2, u3, u4),
                                           (-4057974, 10085068, -13072088, 6823457), strict=True):
                raw = self.add(raw, self.scaled(power, coefficient))
            poly = self.add(self.trunc(raw), (1086023, 1086023))
            comp = (724 if exponent % 2 == 0 else 512) * (1 << (17-exponent//2))
            outputs.append(self.product(poly, (comp, comp)))
        return min(x[0] for x in outputs), max(x[1] for x in outputs)

    def norm(self, row, site, row_index):
        if any(not -256*Q <= a <= b <= 256*Q for a, b in row):
            raise ProofGap(f"inclusive input guard {site} row {row_index}: {row}")
        self.guards[site, row_index] = tuple(row)
        total = (0, 0)
        for value in row:
            total = self.add(total, self.trunc(self.square(value)))
        denominator = self.add(self.trunc(self.scaled(total, Q//2)), (1, 1))
        inv = self.inverse_root(denominator)
        # Gamma Q is its own, exact, separate product/truncation.
        result = tuple(self.product(self.product(value, inv), (Q, Q)) for value in row)
        self.rows[site, row_index] = (denominator, inv, result)
        return result

    def silu(self, x):
        # This fixture must stay in this branch; failing enclosure refuses the
        # proof rather than silently dropping another source polynomial branch.
        if not -4*Q <= x[0] <= x[1] <= 4*Q:
            raise ProofGap(f"SiLU branch enclosure needs partition: {x}")
        x2 = self.trunc(self.square(x))
        x4 = self.trunc(self.square(x2))
        x6 = self.product(x2, x4)
        # Low branch is computed even though not selected: validate its raw/TR
        # premises too, matching the existing helper's instruction schedule.
        self.add(self.add((-80413, -80413), self.trunc(self.scaled(x, -21489))),
                 self.trunc(self.scaled(x2, -1453)))
        out = (2229, 2229)
        for power, coefficient in zip((x, x2, x4, x6),
                                       (131072, 59806, -2913, 71), strict=True):
            out = self.add(out, self.trunc(self.scaled(power, coefficient)))
        return out

    def prefix(self, ids, epoch):
        """Fixture-specific inequality chain, not a second numerical decoder."""
        embeddings = ((0, 0), (40960, 40960), (0, 0),
                      (40960, 40960), (-40960, -40960))
        rows = [tuple((v, v) for v in embeddings[token]) for token in ids]
        normalized = [self.norm(row, f"s{epoch}/layer.0/input_norm/input", i)
                      for i, row in enumerate(rows)]
        outputs = []
        for i, row in enumerate(rows):
            n = i+1
            # Q=K=0 -> all causal scores zero, exp(0)=Q exactly under every
            # source TR. Still count/validate exponent reduction and 5 squares.
            self.trunc((0, 0), 5)
            exp = (Q, Q)
            for _ in range(5):
                exp = self.product(exp, exp)
            inverse = self.point_scalar(reciprocal, n*Q)
            probability = self.product(exp, inverse)  # NOT ideal 1/n, no renorm.
            attention = []
            for col in range(2):
                raw = (0, 0)
                for prior in normalized[:n]:
                    # V identity is a full dot/TR: exact Q coefficient, including
                    # zero products, not an assumption of exact negation.
                    v = self.dot(prior, (Q, 0) if col == 0 else (0, Q))
                    raw = self.add(raw, self.mul(probability, v))
                context = self.trunc(raw)  # Sum BEFORE the one complete dot TR.
                projected = self.dot((context, (0, 0)), (-Q//2, 0))
                attention.append(self.add(row[col], projected))
            post = self.norm(tuple(attention), f"s{epoch}/layer.0/post_attention_norm/input", i)
            gate = self.dot(post, (4096, 4096))
            up = self.dot(post, (4096, 4096))
            activation = self.silu(gate)  # Keeps source nonzero SiLU(0).
            gated = self.product(activation, up)
            down = self.dot((gated, gated), (4096, 4096))
            outputs.append(tuple(self.add(a, down) for a in attention))
        final = [self.norm(row, f"s{epoch}/final_norm/input", i)
                 for i, row in enumerate(outputs)]
        x = final[-1][0]
        # Exact head lemma: every full dot is Q*0, Q*x or -Q*x. All residues
        # therefore give [0,x,0,0,-x]; ties choose 0, never 2/3.
        for coefficient in (0, Q, -Q):
            self.trunc(self.scaled(x, coefficient))
        choices = ({1} if x[0] > 0 else {4} if x[1] < 0 else
                   ({0} | ({1} if x[1] > 0 else set()) | ({4} if x[0] < 0 else set())))
        return x, choices


class EndpointTape(Tape):
    """Six scalar-helper floor/ceil choices; all equations still existing Tape."""
    def __init__(self, choices):
        super().__init__({})
        self.choices = choices

    def truncate(self, name, raw, shift=18):
        index = len(self.batches)
        assert index < 6
        residue = (1 << shift)-1 if (self.choices >> index) & 1 else 0
        self.batches[name] = {"shift": shift, "residues": np.full(np.shape(raw), residue, dtype=object)}
        return super().truncate(name, raw, shift)


@pytest.fixture(scope="module")
def universal_attempt():
    results = {}
    total_cases = total_work = 0
    for prompt, ids in COHORTS.items():
        proof = IntegerBounds(MAX_PROOF_CASES-total_cases, MAX_PROOF_WORK-total_work)
        try:
            first_x, first = proof.prefix(ids, 0)
            if prompt == "hello hello" and first != {4}:
                raise ProofGap(f"first head sign insufficient for (4,4): {first_x}, {sorted(first)}")
            branches = {}
            for token in sorted(first):
                # Recompute exactly the selected embedding history for this
                # branch, not an independently chosen second input cohort.
                second_x, second = proof.prefix((*ids, token), 1)
                proof.branch_guards[token] = dict(proof.guards)
                proof.branch_rows[token] = dict(proof.rows)
                if prompt == "hello hello" and second != {4}:
                    raise ProofGap(f"second head sign insufficient for (4,4): {second_x}, {sorted(second)}")
                branches[token] = {"last_hidden0": second_x, "possible_ids": sorted(second)}
            results[prompt] = {"status": "enclosed", "first_x": first_x,
                               "first_ids": sorted(first), "branches": branches}
        except ProofGap as error:
            results[prompt] = {"status": "unresolved", "first_gap": str(error)}
        results[prompt].update(cases=proof.cases, work=proof.work, proof=proof)
        total_cases += proof.cases
        total_work += proof.work
        print("UNIVERSAL INTEGER ATTEMPT", json.dumps({
            k: v for k, v in results[prompt].items() if k != "proof"}, sort_keys=True))
    assert total_cases <= MAX_PROOF_CASES and total_work <= MAX_PROOF_WORK
    print("AGGREGATE PROOF BUDGET", total_cases, total_work)
    return results


def test_interval_rounding_negative_exact_and_domain_boundary_controls():
    b = IntegerBounds()
    assert b.trunc((-Q-1, -Q-1)) == (-2, -1)
    assert b.trunc((-Q, -Q)) == (-1, -1)
    assert b.trunc((Q+1, Q+1)) == (1, 2)
    assert b.square((-3, 2)) == (0, 9)
    assert b.mul((-3, 2), (-4, 5)) == (-15, 12)
    assert b.dot(((1, 1), (1, 1)), (Q-1, Q-1)) == (1, 2)
    assert b.trunc((-GAP, -GAP)) == (-GAP//Q, -GAP//Q)
    with pytest.raises(ProofGap, match="quarter-gap"):
        b.trunc((GAP, GAP))
    with pytest.raises(ProofGap, match="quarter-gap"):
        b.trunc((-GAP-1, -GAP-1))
    with pytest.raises(ProofGap, match="positive bit-reversal"):
        b.inverse_root((0, 1))
    with pytest.raises(ProofGap, match="positive bit-reversal"):
        b.inverse_root((1 << 35, 1 << 35))
    b.norm(((-256*Q, -256*Q), (0, 0)), "endpoint", 0)
    with pytest.raises(ProofGap, match="inclusive input guard"):
        b.norm(((-256*Q-1, -256*Q-1), (0, 0)), "short", 0)
    b.norm(((256*Q, 256*Q), (0, 0)), "positive_endpoint", 0)
    with pytest.raises(ProofGap, match="inclusive input guard"):
        b.norm(((256*Q+1, 256*Q+1), (0, 0)), "positive_short", 0)
    exhausted = IntegerBounds()
    exhausted.work = MAX_PROOF_WORK
    with pytest.raises(ProofGap, match="budget exhausted"):
        exhausted.charge()
    exhausted = IntegerBounds()
    exhausted.cases = MAX_PROOF_CASES
    with pytest.raises(ProofGap, match="budget exhausted"):
        exhausted.charge(cases=1)


def test_proof_inequalities_are_bound_to_existing_scalar_coefficients_and_head():
    # Keep bound derivations coupled to the existing independent helpers, not
    # copied constants that can silently drift from their arithmetic contract.
    tree = ast.parse(inspect.getsource(rsqrt))
    polynomial = next(node.value for node in ast.walk(tree)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "p" for t in node.targets))
    _same_ast(polynomial, 'tape.truncate(name + "/polynomial", '
              '-4057974*u + 10085068*u2 - 13072088*u3 + 6823457*u4) + 1086023')
    tree = ast.parse(inspect.getsource(silu))
    high = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "high" for t in node.targets)]
    expected = ('2229 + tape.truncate(name + "/high.x", 131072*safe)',
                'high + tape.truncate(name + "/high.x2", 59806*x2)',
                'high + tape.truncate(name + "/high.x4", -2913*x4)',
                'high + tape.truncate(name + "/high.x6", 71*x6)')
    for node, expression in zip(high, expected, strict=True):
        _same_ast(node, expression)
    # Signed and one-unit controls for the algebraic exact-head lemma. These
    # inequalities cover every residue, not just sampled endpoint evaluations.
    for x in (-Q-1, -1, 0, 1, Q+1):
        assert Q*x <= Q*x + (Q-1) < Q*(x+1)
        assert Q*(-x) <= Q*(-x) + (Q-1) < Q*(-x+1)
    b = IntegerBounds()
    enclosure = b.inverse_root((Q-1, Q+1))
    for x in (Q-1, Q, Q+1):
        lo, hi = b.point_scalar(rsqrt, x)
        assert enclosure[0] <= lo <= hi <= enclosure[1]


def test_interval_attempt_preserves_all_prefix_guards_and_sample_comparisons(
        universal_attempt, observations):
    # Conditional integer-model outer bounds, not native SDK qualification or
    # a claim that every member of the world outer set is reachable.
    expected_branches = {
        "hello hello": {4: [4]},
        "world hello": {0: [0, 1, 4], 1: [1], 4: [4]},
    }
    assert set(universal_attempt) == set(expected_branches)
    for prompt, entry in universal_attempt.items():
        assert entry["status"] == "enclosed", entry
        assert entry["first_ids"] == sorted(expected_branches[prompt])
        assert {token: branch["possible_ids"] for token, branch in entry["branches"].items()} == (
            expected_branches[prompt])
        assert entry["cases"] <= MAX_PROOF_CASES and entry["work"] <= MAX_PROOF_WORK
        proof = entry["proof"]
        for observed_prompt, _, _, observed in observations:
            if prompt != observed_prompt:
                continue
            first = int(observed.trace["s0/selected"])
            guards = proof.branch_guards.get(first, proof.guards)
            rows = proof.branch_rows.get(first, proof.rows)
            for (site, row), enclosure in guards.items():
                if site.startswith("s1/") and first not in proof.branch_guards:
                    continue
                incoming = observed.trace[site][row]
                assert all(lo <= int(v) <= hi for v, (lo, hi) in zip(incoming, enclosure, strict=True))
            for (site, row), (_, _, enclosure) in rows.items():
                if site.startswith("s1/") and first not in proof.branch_rows:
                    continue
                outgoing = observed.trace[site.removesuffix("/input")][row]
                assert all(lo <= int(v) <= hi for v, (lo, hi) in zip(outgoing, enclosure, strict=True))
        assert all((f"s0/{site}/input", row) in proof.guards
                   for site in ("layer.0/input_norm", "layer.0/post_attention_norm", "final_norm")
                   for row in range(6))
        assert all((f"s1/{site}/input", row) in proof.guards
                   for site in ("layer.0/input_norm", "layer.0/post_attention_norm", "final_norm")
                   for row in range(7))
        for observed_prompt, _, _, observed in observations:
            if prompt == observed_prompt:
                assert observed.accepted
                first, second = observed.tokens
                assert first in entry["first_ids"]
                assert second in entry["branches"][first]["possible_ids"]
        if prompt == "hello hello":
            assert entry["first_ids"] == [4]
            assert entry["branches"][4]["possible_ids"] == [4]

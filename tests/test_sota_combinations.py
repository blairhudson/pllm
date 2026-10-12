"""Lightweight frozen-factory and evidence gates; no serving or checkpoint loads."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import pytest

from benchmarks.research import sota_combinations as sota
from pllm import Experiment, Model
from pllm.metrics import BuildRef
from pllm.tokenization import IndexedTokenizer


def production_study():
    pairs, _ = sota.placement_pairs()
    return sota.study(model=Model.hf(sota.MODEL, revision=sota.REVISION),
                      tokenizer=IndexedTokenizer("/public/frozen-tokenizer", digest="a" * 64),
                      build=BuildRef("/installed/python", "b" * 64), placement=pairs)


def test_five_way_factories_and_exact_roundtrip():
    study = production_study()
    work = study.plan(max_trials=256)
    assert len(work.suite.cases) == 10
    assert work.suite.sampling.processes == 3
    assert work.suite.sampling.order == "balanced"
    assert work.suite.sampling.warmups == 0
    assert work.suite.sampling.repetitions == 1
    for case in work.suite.cases:
        params = case.target.to_spec()["parameters"]
        assert "client_mlp" not in params and "client_stage" not in params
        assert not params.get("client_metadata")
        assert params["backend"] == "native"
        assert params["wan"] is not None
        assert params["scenario"]["kind"] == "repeat"
        assert params["scenario"]["count"] == 2
        assert params["scenario"]["max_output_tokens"] == 8
        exp = Experiment.from_spec(params["model"]["experiment"])
        assert Experiment.from_spec(exp.to_spec()).canonical_bytes() == exp.canonical_bytes()
        assert exp.budget.max_input_tokens == 384 and exp.budget.max_new_tokens == 8
        assert exp.budget.requests == 2
        c = exp.pipeline.components
        assert "public_prefix" not in c
        assert c["quantization"].params["causal_reduction"] == "prefix_f32"
        assert c["linear"].params["prefill_chunk_rows"] == 16
        assert c["inventory"].params["allocation"] == "demand"
        assert c["inventory"].params["refill"] == "on-demand"
        serial = case.name.split("/")[0] == "288"
        assert c["inventory"].params.get("stage_window", 1) == (1 if serial else 4)
        assert c["inventory"].params.get("stage_window_bytes", 16 << 20) == ((16 if serial else 64) << 20)
        assert c["delivery"].params["storage"] == "paged"
        assert c["delivery"].params["compression"] == "zlib"
        assert c["tokenizer"].component == "pllm/indexed-tokenizer/v1"
        assert c["cache"].params["fixed_input_tokens"] == 384


def test_only_declared_axes_change_and_configs_are_immutable():
    work = production_study().plan(max_trials=256)
    components = {}
    for case in work.suite.cases:
        workload, _, variant = case.name.partition("/")
        variant = variant or "control"
        components[workload, variant] = case.target.to_spec()["parameters"]["model"]["experiment"]["pipeline"]["components"]
    for workload in ("150", "288"):
        control = components[workload, "control"]
        for name in sota.VARIANTS[1:]:
            candidate = components[workload, name]
            changed = {key for key in set(control) | set(candidate) if control.get(key) != candidate.get(key)}
            expected = {"tiling": {"scheduler"}, "digit8": {"preparation"},
                        "both": {"scheduler", "preparation"},
                        "both-placement": {"scheduler", "preparation", "placement"}}[name]
            assert changed == expected
            if "scheduler" in changed:
                assert candidate["scheduler"]["params"] == {"tile_rows": 64}
            if "preparation" in changed:
                assert candidate["preparation"]["params"] == {"storage": "paged", "kernel": "digit8", "min_rows": 64}
        assert len(components[workload, "both-placement"]["placement"]["params"]["layer_roles"]) == 38
    with pytest.raises(FrozenInstanceError):
        work.suite.sampling.processes = 4
    original = work.suite.digest
    exported = work.suite.to_spec()
    exported["cases"].clear()
    assert work.suite.digest == original


@pytest.mark.parametrize("tiny", [False, True])
def test_workload_specific_window_overrides_recipe_for_every_variant(monkeypatch, tiny):
    original = sota.build_experiment
    def legacy_recipe(*args, **kwargs):
        return original(*args, **kwargs).with_params(pipeline__inventory__stage_window_bytes=16 << 20)
    monkeypatch.setattr(sota, "build_experiment", legacy_recipe)
    study = (sota.study(model=Model.tiny(), tokenizer=None,
        build=BuildRef("/installed/python", "b" * 64), workloads=("tiny",), tiny=True)
        if tiny else production_study())
    for case in study.suite.cases:
        exp = Experiment.from_spec(case.target.to_spec()["parameters"]["model"]["experiment"])
        inventory = exp.pipeline.components["inventory"].params
        serial = case.name.split("/")[0] == "288"
        assert inventory.get("stage_window", 1) == (1 if serial else 4)
        assert inventory.get("stage_window_bytes", 16 << 20) == ((16 if serial else 64) << 20)
        assert inventory["allocation"] == "demand"
        assert inventory["refill"] == "on-demand"
        assert exp.budget.max_new_tokens == (2 if tiny else 8)
        assert exp.budget.requests == 2
    assert study.suite.sampling.memory_bytes == 8 << 30
    assert study.suite.sampling.timeout_seconds == 1200


@pytest.mark.parametrize("suffix", [".json", ".plan.json", ".progress.json", ".summary.json"])
def test_rerun_requires_fresh_output_and_every_companion(tmp_path, suffix):
    from scripts.benchmark_sota_combinations import fresh_output_paths
    output = tmp_path / "rerun.json"
    existing = output.with_suffix(suffix)
    existing.write_text("immutable initial evidence")
    with pytest.raises(FileExistsError, match="companion evidence already exists"):
        fresh_output_paths(output)
    assert existing.read_text() == "immutable initial evidence"


def test_fresh_rerun_companions_do_not_reuse_initial_evidence(tmp_path):
    from scripts.benchmark_sota_combinations import fresh_output_paths
    initial = tmp_path / "initial.json"
    initial.write_text("immutable initial evidence")
    rerun = tmp_path / "rerun.json"
    assert fresh_output_paths(rerun) == tuple(rerun.with_suffix(suffix) for suffix in
        (".plan.json", ".progress.json", ".summary.json"))
    assert initial.read_text() == "immutable initial evidence"
    assert not rerun.exists()


def test_selected_window_still_rejects_overlarge_individual_job_before_issuance():
    from pllm.runtime.preparation_window import run_preparation_window
    issued = []
    with pytest.raises(ValueError, match="job exceeds the declared work window"):
        run_preparation_window([object()], width=4, max_bytes=sota.PREPARATION_STAGE_WINDOW_BYTES,
            cost=lambda _: sota.PREPARATION_STAGE_WINDOW_BYTES + 1,
            prepare=lambda item: issued.append(item), accept=lambda _: None,
            abort=lambda: None, discard=lambda _: None, cancelled=lambda: False)
    assert issued == []


def test_real_shape_header_demand_rejects_original_long_window_without_tensor_loading():
    from pllm import lower_model
    config = {
        "model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864,
        "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
        "vocab_size": 151936, "max_position_embeddings": 32768, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": True,
    }
    plan = lower_model(config, batch=1, max_input_tokens=384, max_new_tokens=8)
    for case in production_study().suite.cases:
        rows = int(case.name.split("/")[0])
        exp = Experiment.from_spec(case.target.to_spec()["parameters"]["model"]["experiment"])
        geometry = sota.preparation_window_check(plan, exp, rows=rows)
        assert geometry["requests"]["cold"]["largest_job_bytes"] == (
            38_909_952 if rows == 150 else 73_107_456)
        assert geometry["requests"]["repeat"]["largest_job_bytes"] == 1_738_752
        assert geometry["concurrent_cap_applies"] == (rows == 150)
        assert set(geometry["requests"]["repeat"]["phase_rows"]["prefill"].values()) == {0}
        if rows == 288:
            from pllm import Pipeline
            from pllm.preparation import PreparedInventory
            components = dict(exp.pipeline.components)
            components["inventory"] = PreparedInventory(**{
                **components["inventory"].params,
                "stage_window": 4, "stage_window_bytes": 64 << 20,
            })
            original = exp.with_params(pipeline=Pipeline(model=exp.pipeline.model, components=components))
            with pytest.raises(sota.Rejected, match="preparation_work_window_limit"):
                sota.preparation_window_check(plan, original, rows=rows)


def test_frozen_placement_matches_selected_case(tmp_path):
    pairs, record = sota.placement_pairs()
    assert len(pairs) == len(set(pairs)) == 38
    bad = deepcopy(record)
    bad["selection"]["layer_roles"][0][0] = 23
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(sota.Rejected, match="placement_record_mismatch"):
        sota.placement_pairs(path)


def test_atomic_progress_and_source_hashes(tmp_path):
    path = tmp_path / "nested/progress.json"
    sota.atomic_json(path, {"complete": False, "results": []})
    before = sota.hashes([path])
    sota.atomic_json(path, {"complete": False, "results": [{"status": "blocked"}]})
    assert before != sota.hashes([path])
    assert json.loads(path.read_text())["results"][0]["status"] == "blocked"
    assert list(path.parent.glob(".progress.json-*")) == []


class FakeRun:
    def __init__(self, document):
        self.document = document

    def to_dict(self):
        return deepcopy(self.document)

    def compare(self, control, candidate, metric):
        return {"status": "comparable", "control": control, "candidate": candidate, "metric": metric}


def evidence(*, output_tokens=8):
    rows = []
    for name in ("150", "150/tiling"):
        requests = [{"tokens": {"input_tokens": 150, "output_tokens": output_tokens, "authoritative": True},
                     "cold": index == 0,
                     "model_fingerprint": "body", "generation": {"output_text_digest": "matched"},
                     "privacy": {"prefill_cache_hits": index, "prefill_cache_misses": 1 - index,
                                 "prefill_prefix_tokens_reused": index * 150}}
                    for index in (0, 1)]
        rows.append({"status": "completed", "execution": {"case": {"name": name}, "replicate": 0,
                     "details": {"report": {"checks": {"passed": True}, "runs": requests,
                                            "configuration": {"source_lock_digest": "source"}}}}})
    return {"complete": True, "results": rows, "suite": {"cases": [{"name": "150"}, {"name": "150/tiling"}]}}


PLACEMENT_RECORD = {"source_lock_digest": "source", "prior_body_fingerprint": "body"}


@pytest.mark.parametrize("status", ["blocked", "failed", "timeout", "cancelled"])
def test_failed_or_blocked_evidence_never_ranked(status):
    document = evidence()
    document["results"][1]["status"] = status
    with pytest.raises(sota.Rejected, match="incomplete_or_failed_cohort"):
        sota.audit(FakeRun(document), placement_record=PLACEMENT_RECORD)


@pytest.mark.parametrize("mutation,code", [
    ("body", "measured_workload_or_body_mismatch"),
    ("tokens", "measured_workload_or_body_mismatch"),
    ("output", "matched_outputs_differ"),
])
def test_mismatched_identity_rejected(mutation, code):
    document = evidence()
    request = document["results"][1]["execution"]["details"]["report"]["runs"][0]
    if mutation == "body":
        request["model_fingerprint"] = "different"
    elif mutation == "tokens":
        request["tokens"]["input_tokens"] = 149
    else:
        request["generation"]["output_text_digest"] = "different"
    with pytest.raises(sota.Rejected, match=code):
        sota.audit(FakeRun(document), placement_record=PLACEMENT_RECORD)


def test_complete_matching_report_retains_raw_output_counts():
    report = sota.audit(FakeRun(evidence()), placement_record=PLACEMENT_RECORD)
    assert report["actual_outputs"][0]["output_tokens"] == [8, 8]
    assert report["comparisons"]


def test_natural_eos_is_retained_without_forcing_the_cap():
    report = sota.audit(FakeRun(evidence(output_tokens=3)), placement_record=PLACEMENT_RECORD)
    assert report["actual_outputs"][0]["output_tokens"] == [3, 3]


def test_repeat_cache_miss_and_unstable_replicates_reject():
    document = evidence()
    document["results"][0]["execution"]["details"]["report"]["runs"][1]["privacy"]["prefill_cache_hits"] = 0
    with pytest.raises(sota.Rejected, match="cold_repeat_cache_contract_mismatch"):
        sota.audit(FakeRun(document), placement_record=PLACEMENT_RECORD)
    document = evidence()
    extra = deepcopy(document["results"][0])
    extra["execution"]["replicate"] = 1
    for request in extra["execution"]["details"]["report"]["runs"]:
        request["generation"]["output_text_digest"] = "changed-between-replicates"
    document["results"].append(extra)
    with pytest.raises(sota.Rejected, match="matched_outputs_differ"):
        sota.audit(FakeRun(document), placement_record=PLACEMENT_RECORD)


def test_failed_admission_preserves_the_rejected_case(monkeypatch):
    from pllm.sources import _tiny_model_config
    tiny = sota.study(model=Model.tiny(), tokenizer=None,
        build=BuildRef("/installed/python", "b" * 64), workloads=("tiny",), tiny=True)
    saved = []
    monkeypatch.setattr(sota, "benchmark_memory", lambda *a, **k: {"admitted": False, "cause": "capacity"})
    with pytest.raises(sota.Rejected, match="topology_memory_limit"):
        sota.preflight(tiny.suite, tiny=True, checkpoint=lambda rows: saved.append(deepcopy(rows)))
    assert saved[-1][0]["case"] == "tiny"
    assert saved[-1][0]["admission"] == {"admitted": False, "cause": "capacity"}
    assert _tiny_model_config()["num_hidden_layers"] == 1

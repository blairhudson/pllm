"""Matched five-way SDK screen; workload-specific issuance, no runtime adapters."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from pllm import Experiment, Pipeline, lower_model
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario, benchmark_memory
from pllm.preparation import ModelAwareCorrections
from pllm.roles import ClientLinearRoles
from pllm.schedulers import MlpPrefillTiling
from pllm.search import FrontierStudy

from examples.prepared_sdk import build_experiment

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PLACEMENT_PLAN = Path("docs/evidence/research-all-link-placement-sdk-confirmation.plan.json")
VARIANTS = ("control", "tiling", "digit8", "both", "both-placement")
PREPARATION_STAGE_WINDOW_BYTES = 64 << 20
# Traffic decomposition remains unknown until the generic runtime attribution
# supplies it. Never divide scenario totals by tokens to invent these objectives.
OBJECTIVES = ("client.peak_rss", "client.cold_cpu", "preparation.cold_cpu",
              "preparation.peak_rss", "response.ttft", "response.decode_tps",
              "network.base_body_bytes_per_request",
              "network.output_dependent_body_bytes_per_generated_token")
DIAGNOSTICS = ("network.all_link_stream_bytes", "network.covered_body_bytes")


class Rejected(ValueError):
    """Public rejection code, deliberately independent of private input text."""


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(document, out, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def hashes(paths):
    return {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}


def public_input_counts(model, tokenizer, inputs):
    """Use the SDK's renderer and selected tokenizer, without model tensor loads."""
    from pllm.deployment.execution import _TokenizerView
    from pllm.model_loader import resolve_model
    from pllm.runtime.indexed_tokenizer import IndexedClientTokenizer
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    source_lock = None
    if model.kind == "tiny":
        from types import SimpleNamespace
        from pllm.sources import _tiny_model_config
        config = _tiny_model_config()
        # Same public byte-tokenizer fixture used by _materialize_tiny_model;
        # header-only preflight does not materialize its random tensors.
        descriptor = {"kind": "byte", "chat_template":
            "{% for message in messages %}{{ message['role'] }}: "
            "{{ message['content'] }}\n{% endfor %}assistant: "}
        manifest = SimpleNamespace(chat_template=descriptor["chat_template"])
    else:
        source = resolve_model(model)
        source_lock = source.source_lock_digest
        config = json.loads((source.path / "config.json").read_bytes())
        config = config.get("text_config", config)
        descriptor = MaskedTransformerEngine._load_tokenizer_descriptor(source.path, source.manifest, config)
        manifest = source.manifest
    view = _TokenizerView(descriptor, config, manifest)
    encoder = IndexedClientTokenizer(view, tokenizer) if tokenizer is not None else view.tokenizer()
    try:
        counts = {}
        for workload, value in inputs.items():
            rendered = view.render_prompt([{"role": "user", "content": value["prompt"]}])
            count = len(encoder.encode(rendered, add_bos=bool(descriptor.get("add_bos_token", True))))
            if (workload != "tiny" and count != int(workload)) or not 64 < count <= 384:
                raise Rejected("public_fixture_token_count_mismatch")
            counts[workload] = count
        return counts, source_lock
    finally:
        if hasattr(encoder, "close"):
            encoder.close()


def placement_pairs(path=PLACEMENT_PLAN):
    record = json.loads(Path(path).read_bytes())
    pairs = tuple((layer, role) for layer, role in record["selection"]["layer_roles"])
    selected = record["suite"]["cases"][1]["target"]["parameters"]["model"]["experiment"]
    if (len(pairs) != 38 or len(set(pairs)) != 38
            or selected["pipeline"]["model"] != {"source": MODEL, "revision": REVISION}
            or pairs != tuple(map(tuple, selected["pipeline"]["components"]["placement"]["params"]["layer_roles"]))):
        raise Rejected("placement_record_mismatch")
    return pairs, record


def study(*, model, tokenizer, build, workloads=("150", "288"), processes=3,
          placement=None, tiny=False):
    if not workloads or len(set(workloads)) != len(workloads) or set(workloads) - {"150", "288", "tiny"}:
        raise Rejected("workload_selection")
    if tiny != (set(workloads) == {"tiny"}):
        raise Rejected("tiny_workload_mismatch")
    if not tiny and model.to_spec() != {"source": MODEL, "revision": REVISION}:
        raise Rejected("pinned_model_required")
    if not tiny and tokenizer is None:
        raise Rejected("indexed_tokenizer_required")
    pairs = ((0, "attention_output"),) if tiny else placement
    if not pairs or (not tiny and len(pairs) != 38):
        raise Rejected("placement_required")
    cases = []
    for name in workloads:
        bound, outputs = (128, 2) if tiny else (384, 8)
        # The 288-token initial job exceeds the supported concurrent work cap.
        # Use the existing serial contract, identically across its five variants;
        # never relax workspace pricing or compare issuance across workloads.
        stage_window, stage_window_bytes = ((1, 16 << 20) if name == "288" else
                                            (4, PREPARATION_STAGE_WINDOW_BYTES))
        experiment = build_experiment(model, tokenizer=tokenizer).with_params(
            budget__max_input_tokens=bound, budget__max_new_tokens=outputs,
            pipeline__cache__fixed_input_tokens=bound,
            pipeline__inventory__stage_window=stage_window,
            pipeline__inventory__stage_window_bytes=stage_window_bytes)
        if tiny:
            # One-layer terminal pruning removes the only row-independent MLP
            # region. Disable only in every transport control/candidate alike.
            experiment = experiment.with_params(pipeline__linear__prefill_pruning="none")
        for variant in VARIANTS:
            components = dict(experiment.pipeline.components)
            if variant in {"tiling", "both", "both-placement"}:
                components["scheduler"] = MlpPrefillTiling(tile_rows=64)
            if variant in {"digit8", "both", "both-placement"}:
                components["preparation"] = ModelAwareCorrections(storage="paged", kernel="digit8", min_rows=64)
            if variant == "both-placement":
                components["placement"] = ClientLinearRoles(layer_roles=pairs)
            candidate = experiment.with_params(pipeline=Pipeline.from_profile(
                experiment.pipeline.profile, model=model, components=components))
            target = BenchmarkTarget.response(candidate, backend="native",
                scenario=Scenario.repeat(max_output_tokens=outputs), wan=WanConditions())
            case_name = name if variant == "control" else f"{name}/{variant}"
            cases.append(BenchmarkCase(case_name, target, build))
    # Explicit public Experiment cases avoid adding absent slots through
    # serialized Variant.with_params paths (unsupported by current SDK).
    return FrontierStudy(BenchmarkSuite(tuple(cases), Sampling(processes=processes,
        repetitions=1, warmups=0, order="balanced", threads=1,
        timeout_seconds=1200, memory_bytes=8 << 30), "sota-combinations-v1"))


def native_placement_check(experiment, config, *, rows, transport_fixture=False):
    """Compile source shapes, check exact ownership and independent phase limits."""
    from pllm.runtime.semantic_stages import scheduled_stage_specs, semantic_stage_role
    plan = lower_model(config, batch=1, max_input_tokens=experiment.budget.max_input_tokens,
                       max_new_tokens=experiment.budget.max_new_tokens)
    pairs = set(map(tuple, experiment.pipeline.components["placement"].params["layer_roles"]))
    stages = {(s.layer_index, s.role): s for s in scheduled_stage_specs(plan, experiment.pipeline)
              if s.layer_index is not None}
    if not pairs <= stages.keys():
        raise Rejected("placement_source_shapes")
    schedule = plan.runtime_schedule(experiment.pipeline).to_dict()
    evidence = {}
    for phase in ("prefill", "decode"):
        operations = {op["id"]: op for op in getattr(plan, phase)["operations"]}
        steps = {(s["layer"], semantic_stage_role(s, operations)): s
                 for s in schedule[phase]["steps"] if s["weight_ids"] and s["layer"] is not None}
        owned = {key for key, step in steps.items() if step["executor"] == "client_linear"}
        if owned != pairs or set(steps) != set(stages):
            raise Rejected("native_ownership_mismatch")
        costs = {key: s.in_features * s.out_features * (
            rows if phase == "prefill" and not steps[key].get("terminal_row_only", False) else 1)
            for key, s in stages.items()}
        total, local = sum(costs.values()), sum(costs[key] for key in pairs)
        if total <= 0 or (not transport_fixture and local * 10 > total):
            raise Rejected("remote_body_mac_limit")
        evidence[phase] = {"body_macs": total, "client_macs": local,
                           "remote_fraction": 1 - local / total,
                           "production_mac_gate": not transport_fixture}
    return evidence


def preparation_window_check(plan, experiment, *, rows):
    """Exact header-only issuance geometry; topology admission stays separate."""
    from pllm.runtime.preparation_window import preparation_job_work_bytes, validate_preparation_window
    from pllm.runtime.semantic_stages import scheduled_prepared_phase_rows, scheduled_stage_specs

    components = experiment.pipeline.components
    if components["inventory"].params["allocation"] != "demand" or "verification" in components:
        raise Rejected("preparation_geometry_contract")
    inventory = components["inventory"].params
    width = inventory.get("stage_window", 1)
    max_bytes = inventory.get("stage_window_bytes", 16 << 20)
    stages = {stage.id: stage for stage in scheduled_stage_specs(plan, experiment.pipeline)}
    record = {"stage_window": width, "stage_window_bytes": max_bytes,
              "concurrent_cap_applies": width > 1, "requests": {}}
    for request, prefill_rows in (("cold", rows), ("repeat", 0)):
        demand = scheduled_prepared_phase_rows(plan, experiment.pipeline, prefill_rows,
            max(0, experiment.budget.max_new_tokens - 1), continuation=request == "repeat")
        costs = {stage_id: preparation_job_work_bytes(
            rows=max(1, demand["prefill"].get(stage_id, 0) + demand["decode"].get(stage_id, 0)),
            in_features=stages[stage_id].in_features, out_features=stages[stage_id].out_features,
        ) for stage_id in demand["prefill"].keys() | demand["decode"].keys()}
        record["requests"][request] = {"phase_rows": demand, "job_bytes": costs,
                                     "largest_job_bytes": max(costs.values(), default=0)}
        try:
            validate_preparation_window(width=width, max_bytes=max_bytes, job_bytes=costs.values())
        except ValueError as exc:
            raise Rejected("preparation_work_window_limit") from exc
    return record


def preflight(suite, *, tiny=False, placement_record=None, input_counts=None, checkpoint=None):
    """Header/config-only admission; no tensor loads, providers or measured clients."""
    from pllm.model_loader import resolve_model
    from pllm.sources import _tiny_model_config
    records = []
    for case in suite.cases:
        exp = Experiment.from_spec(case.target.to_spec()["parameters"]["model"]["experiment"])
        # Exact public round trip is part of replay admission.
        if Experiment.from_spec(exp.to_spec()).canonical_bytes() != exp.canonical_bytes():
            raise Rejected("experiment_replay_mismatch")
        source_lock = None
        config = _tiny_model_config() if tiny else None
        if not tiny:
            source = resolve_model(exp.pipeline.model)
            source_lock = source.source_lock_digest
            if source_lock != placement_record["source_lock_digest"]:
                raise Rejected("placement_source_lock_mismatch")
            config = json.loads((source.path / "config.json").read_bytes())
        record = {"case": case.name, "source_lock_digest": source_lock}
        records.append(record)
        try:
            admitted = benchmark_memory(exp, backend="native", enforced_wan=True,
                                        memory_budget_bytes=suite.sampling.memory_bytes)
            record["admission"] = admitted
            plan = lower_model(config, batch=1, max_input_tokens=exp.budget.max_input_tokens,
                                max_new_tokens=exp.budget.max_new_tokens)
            schedule = plan.runtime_schedule(exp.pipeline).to_dict()
            workload = case.name.split("/")[0]
            rows = (input_counts or {}).get(workload, 8 if tiny else int(workload))
            record["preparation_work_window"] = preparation_window_check(plan, exp, rows=rows)
            if "scheduler" in exp.pipeline.components:
                tiling = schedule["mlp_tiling"]
                if tiling["tile_rows"] != 64 or len(tiling["regions"]) != (1 if tiny else 23):
                    raise Rejected("unexpected_tiled_regions")
                record["tiling"] = {"tile_rows": 64, "regions": len(tiling["regions"]),
                    "layers": [region["layer"] for region in tiling["regions"]]}
            if "placement" in exp.pipeline.components:
                record["placement"] = native_placement_check(exp, config,
                    rows=rows, transport_fixture=tiny)
        except BaseException as exc:
            record["failure_code"] = str(exc) if isinstance(exc, Rejected) else "native_preflight_error"
            if checkpoint is not None:
                checkpoint(records)
            raise
        if checkpoint is not None:
            checkpoint(records)
        if not admitted["admitted"]:
            raise Rejected("topology_memory_limit")
    return records


def audit(run, *, tiny=False, placement_record=None):
    """Keep natural EOS counts; compare every request and every matched replicate."""
    document = run.to_dict()
    if not document["complete"] or any(r["status"] != "completed" for r in document["results"]):
        raise Rejected("incomplete_or_failed_cohort")
    groups = {}
    actual = []
    for row in document["results"]:
        execution = row["execution"]
        name = execution["case"]["name"]
        workload = name.split("/")[0]
        report = execution["details"]["report"]
        runs = report["runs"]
        if (report["checks"]["passed"] is not True or len(runs) != 2
                or any(r["tokens"]["authoritative"] is not True for r in runs)):
            raise Rejected("response_checks_failed")
        for warm, request in enumerate(runs):
            privacy = request["privacy"]
            if (request.get("cold") is not (warm == 0)
                    or privacy.get("prefill_cache_hits") != warm
                    or privacy.get("prefill_cache_misses") != 1 - warm):
                raise Rejected("cold_repeat_cache_contract_mismatch")
        if not tiny and (report["configuration"]["source_lock_digest"] != placement_record["source_lock_digest"]
                or any(r["model_fingerprint"] != placement_record["prior_body_fingerprint"]
                       or r["tokens"]["input_tokens"] != int(workload) for r in runs)):
            raise Rejected("measured_workload_or_body_mismatch")
        outputs = tuple((r["tokens"]["output_tokens"], r["generation"]["output_text_digest"]) for r in runs)
        cap = 2 if tiny else 8
        if any(not 1 <= count <= cap or not commitment for count, commitment in outputs):
            raise Rejected("output_commitment_missing")
        if outputs[0] != outputs[1] or workload in groups and groups[workload] != outputs:
            raise Rejected("matched_outputs_differ")
        groups[workload] = outputs
        actual.append({"case": name, "replicate": execution["replicate"],
                       "output_tokens": [count for count, _ in outputs]})
    comparisons = []
    for workload in dict.fromkeys(c["name"].split("/")[0] for c in document["suite"]["cases"]):
        for variant in VARIANTS[1:]:
            comparisons.append({"workload": workload, "variant": variant,
                "metrics": {metric: run.compare(workload, f"{workload}/{variant}", metric)
                            for metric in OBJECTIVES + DIAGNOSTICS}})
    return {"actual_outputs": actual, "comparisons": comparisons,
            "ranking": {"eligible_metric_comparisons": sum(
                result["status"] == "comparable" for row in comparisons for result in row["metrics"].values()),
                "unranked": [{"workload": row["workload"], "variant": row["variant"],
                              "metric": metric, "reason": result.get("reason")}
                             for row in comparisons for metric, result in row["metrics"].items()
                             if result["status"] != "comparable"]},
            "limitations": ["No full-logit/KV parity oracle in this response screen.",
                 "Base/per-token traffic unknown unless generic attribution is installed.",
                 "Cold/repeat windows remain separate in raw reports; exact repeat bypasses initial MLP tiling.",
                 "Provider cleanup is enforced; cleanup CPU/time is not separately measured by this target.",
                 "Full/tiled continuation and topology pricing remain conservative.",
                "Tiny placement is one transport-fixture group, not the production 38 groups."]}

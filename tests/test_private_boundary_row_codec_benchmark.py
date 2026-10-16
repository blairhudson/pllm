"""Synthetic v2 admission/tamper tests. No checkpoint, provider or timing cohorts."""
from copy import deepcopy
from functools import lru_cache
import base64
import hashlib
import json
from pathlib import Path
import struct
import zlib

import pytest

from pllm import Experiment
from pllm.metrics import BenchmarkRun, BuildRef
from benchmarks.research import private_boundary_row_codec as study


@pytest.fixture(autouse=True)
def source_workload_overlay(monkeypatch):
    """Test owned uninstalled SDK sources; never pretend these fixtures ran SDK roles."""
    import pllm
    from tests.test_benchmark_workload_binding import load_source_worker
    worker = load_source_worker(monkeypatch)
    def sources():
        sdk = Path(pllm.__file__).parent
        paths = {name: sdk / name[4:] if name.startswith("sdk/") else study.ROOT / name for name in study.VERIFIER_SOURCES}
        for name in ("_benchmark_worker.py", "_benchmark_workload.py"):
            paths["sdk/" + name] = study.ROOT / "python" / "pllm" / name
        return {name: study.hashes([path])[str(path)] for name, path in paths.items()}
    monkeypatch.setattr(study, "verifier_sources", sources)
    return worker


def bind_fixture(execution, worker):
    """Run real Measurement finalizer, not a fake unsalted final digest."""
    from pllm.metrics import Sampling, BenchmarkTarget
    from pllm._benchmark_contract import digest
    from pllm.runtime.topology_accounting import ACCOUNTING_SCHEMA, accounting_coverage_identity
    salt, prompt = b"t" * 32, "private synthetic test input; copied observations are NOT measurements"
    m = worker.Measurement({"sampling": Sampling(processes=1).to_spec(), "salt": salt.hex(),
                            "inputs": {"prompt": prompt}, "directory": "/unused-synthetic"})
    report = execution["details"]["report"]
    report["configuration"]["prompt_digest"] = hashlib.sha256(b"pllm.benchmark.prompt.v1\0" + salt + prompt.encode()).hexdigest()
    params = execution["case"]["target"]["parameters"]
    request, = report["runs"]
    m.comparison = deepcopy(execution["comparison"])
    m.comparison["workload_digest"] = digest({"tokens": [request["tokens"]], "scenario": params["scenario"],
        "backend": report["configuration"]["provider_backend"], "wan": report["configuration"]["wan_emulation_digest"],
        "sampling": report["configuration"]["sampling"]})
    base = digest({"schema": report["schema_version"], "scenario": params["scenario"], "cached_checkpoint": True})
    coverage = accounting_coverage_identity(report.get("topology_accounting"))
    if coverage not in {ACCOUNTING_SCHEMA, "historical_coverage_not_recorded"}:
        base = digest({"response": base, "body_counter_coverage": coverage})
    m.comparison["measurement_digest"] = digest({"response": base, "state_audit": "pllm.response_state_audit.v1"})
    m.details, m.fixture = execution["details"], execution["fixture"]
    m.finalize_workload(BenchmarkTarget.from_spec(execution["case"]["target"]))
    execution["comparison"] = m.comparison
    return m.comparison["workload_digest"]


@lru_cache
def public_certificate():
    source = {"source_digest": "383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29",
        "checkpoint_digest": "1" * 64, "model_config_digest": "2" * 64, "head_weight_digest": "3" * 64,
        "head_scales_digest": "4" * 64, "lookup_digest": "5" * 64, "vocabulary": 151936,
        "hidden": 896, "activation_bits": 8, "weight_bits": 8, "signed_output_bound": 896 * 128 * 127}
    widths = bytes([25]) * source["vocabulary"]
    metadata = {"schema": "pllm.private_boundary.row_certificate.v1", "source": source,
        "numeric": study.CERT_NUMERIC, "layout": "zlib-u8-minimal-signed-widths-v1",
        "widths_digest": hashlib.sha256(widths).hexdigest(), "bounds_digest": "c" * 64,
        "implementation": {name: "d" * 64 for name in ("pllm._native", "pllm.runtime.private_boundary_loader",
            "pllm.runtime.quantization", "pllm.runtime.private_boundary_certificate")}}
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    payload = struct.pack("<8sI", b"PLLMBRC1", len(raw)) + raw + zlib.compress(widths)
    pin = {"path": "/synthetic/certificate.bin", "digest": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    return pin, payload, metadata


@lru_cache
def descriptors():
    from pllm.runtime.private_boundary_transport import BoundaryDescriptor
    pin, _, metadata = public_certificate()
    result = {}
    for name, spec in zip(study.NAMES, pair()):
        params = Experiment.from_spec(spec).pipeline.components["boundary"].get_params()
        result[name] = BoundaryDescriptor(**metadata["source"],
            composition_digest=Experiment.from_spec(spec).pipeline.digest(), schedule_digest="7" * 64,
            head_dispatch="parallel", head_output_encoding=params["head_output_encoding"],
            head_certificate_digest=pin["digest"] if name == study.CANDIDATE else "")
    return result


@lru_cache
def native_pins():
    return {name: {"pipeline_digest": Experiment.from_spec(spec).pipeline.digest(),
        "composition_digest": desc.composition_digest, "schedule_digest": desc.schedule_digest,
        "descriptor_digest": desc.digest, "numeric_body_fingerprint": "b" * 64 if name == study.CANDIDATE
            else "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974"}
        for (name, desc), spec in zip(descriptors().items(), pair())}


def archive():
    pin, payload, _ = public_certificate()
    return {"artifact_base64": base64.b64encode(payload).decode(), "artifact": pin}


def audit(document, **kwargs):
    return study.audit(document, certificate=archive(), native_commitments=native_pins(), **kwargs)


@lru_cache
def _pair_template(cap=8):
    # Configurations only; no archived measurement informs the matched recipes.
    base = study.screen.recipes()[study.confirmation.WINNER].with_params(budget__max_new_tokens=cap).to_spec()
    candidate = deepcopy(base)
    candidate["pipeline"]["components"]["boundary"]["params"].update(
        head_output_encoding="row_residues", head_certificate=public_certificate()[0])
    return base, candidate


def pair(cap=8):
    return deepcopy(_pair_template(cap))


def suite_for(build, cap=8):
    a, b = pair(cap)
    return study.matched_suite(control=Experiment.from_spec(a), candidate=Experiment.from_spec(b),
        build=build, max_output_tokens=cap)


@pytest.fixture
def cohort(source_workload_overlay):
    """Copied test observations, resealed synthetic records; never publish as evidence."""
    from pllm._benchmark_contract import digest
    from pllm.metrics.benchmark_runner import _result

    original = json.loads(Path("docs/evidence/private-boundary-decode-confirmation-150x8.json").read_bytes())
    templates = {r["execution"]["replicate"]: r for r in original["results"]
        if r["execution"]["case"]["name"] == study.confirmation.WINNER}
    build = deepcopy(templates[0]["execution"]["build"])
    manifest = study.verifier_sources()
    build["attributes"]["sdk_sources"].update({path: manifest["sdk/" + path] for path in study.SDK_SOURCES})
    build["digest"] = digest(build["attributes"])
    suite = suite_for(BuildRef(expected_digest=build["digest"]))
    document = {"schema": "pllm.benchmark_run.v1", "suite": suite.to_spec(), "complete": True,
        "cohort": "e" * 64, "results": []}
    for rep, names in enumerate((study.NAMES, tuple(reversed(study.NAMES)), tuple(reversed(study.NAMES)))):
        for name in names:
            case = next(c for c in suite.cases if c.name == name)
            exp = Experiment.from_spec(case.target.to_spec()["parameters"]["model"]["experiment"])
            row = deepcopy(templates[rep])
            execution, detail = row["execution"], row["execution"]["details"]
            detail["report"]["experiment"]["pipeline_digest"] = exp.pipeline.digest()
            comparison = execution["comparison"]
            request = detail["report"]["runs"][0]
            if name == study.CANDIDATE:
                body = "b" * 64  # Explicitly exercise permitted numeric fingerprint change.
                detail["report"]["runs"][0]["model_fingerprint"] = body
                detail["response_state_audit"]["bindings"][0]["numeric_body_fingerprint"] = body
                numeric = {n: c.to_spec() for n, c in exp.pipeline.components.items() if n in {"quantization", "nonlinear"}}
                comparison["numeric_digest"] = digest({"bodies": [body], "contract": numeric})
            desc = descriptors()[name]
            pin, payload, metadata = public_certificate()
            layout = {"widths_digest": metadata["widths_digest"], "width_counts": {"25": 151936},
                "total_bits": 25 * 151936, "payload_bytes_per_role": 474800, "subframe_header_bytes": 74,
                "certificate": None if name == study.CONTROL else {"digest": public_certificate()[0]["digest"],
                    "size_bytes": len(payload), "metadata": {**metadata,
                        "artifact_digest": pin["digest"], "artifact_size_bytes": len(payload)}}}
            head = {"schema": "pllm.private_boundary_head_accounting.v1", "descriptor": desc.to_dict(),
                "descriptor_digest": desc.digest, "layout": layout, "roles": {role: {
                    "replies": 8, "outer_body_bytes": 3800128, "subframe_bytes": 8 * (474800 + 74),
                    "header_bytes": 8 * 74, "payload_bytes": 8 * 474800} for role in ("boundary_a", "boundary_b")}}
            detail["private_boundary_head"] = head
            output = detail["report"]["runs"][0]["generation"]["output_text_digest"]
            comparison["output_digest"] = digest({"output": digest([output]),
                "state": detail["response_state_audit"]["bindings"]})
            execution["case"] = case.to_spec()
            bind_fixture(execution, source_workload_overlay)
            comparison = execution["comparison"]
            payload = {"status": row["status"], "failure_code": execution["failure_code"],
                "comparison": comparison, "details": detail, "environment": row["environment"]["attributes"],
                "metrics": row["metrics"], "samples": row["samples"], "fixture": execution["fixture"],
                "oracle": execution["oracle"], "admission": execution["admission"],
                "limitations": ["Synthetic copied test observations; not row-codec execution evidence."]}
            document["results"].append(_result(case, suite, rep, document["cohort"], build, payload).to_dict())
    BenchmarkRun(document)
    return document


def test_complete_audit_read_only_and_numeric_difference(cohort):
    before = deepcopy(cohort)
    result = audit(cohort)
    assert before == cohort
    assert result["clients"] == 6 and result["replicates"] == 3
    assert result["all_logits_active_kv_positions_selections_rng_and_outputs_match"]
    assert result["maximum_observed_swap_growth_bytes"] == 0
    assert not result["cold_network_complete"] and not result["sota_claim_admissible"]
    assert result["codec_source_certification_verified"]
    assert set(result["comparisons"][study.CANDIDATE][study.CONTROL]) == set(study.METRICS)
    for group in result["groups"].values():
        assert group["maximum_client_peak_bytes"] == max(r["process.peak_rss"] for r in group["samples"])
        assert len(group["identities"]) == 3


@pytest.mark.parametrize("cap", (8, 64))
def test_two_explicit_frozen_inputs(cap):
    a, b = pair(cap)
    original = deepcopy((a, b))
    study.validate_pair_specs(a, b, max_output_tokens=cap)
    assert (a, b) == original
    suite = suite_for(BuildRef(expected_digest="a" * 64), cap)
    assert suite.sampling.processes == 3 and suite.sampling.order == "balanced"
    assert all(c.target.to_spec()["parameters"]["state_audit"] for c in suite.cases)


@pytest.mark.parametrize("part", ("quantization", "linear", "kernels", "delivery", "inventory", "topology", "tokenizer"))
def test_structural_allowlist_rejects_other_components(part):
    a, b = pair()
    b["pipeline"]["components"][part]["params"]["unapproved"] = True
    with pytest.raises(study.Rejected, match="unapproved_experiment_difference"):
        study.validate_pair_specs(a, b, max_output_tokens=8)


@pytest.mark.parametrize("mutation", (
    "missing", "incomplete", "duplicate", "order", "cohort", "build", "case_digest", "target_digest",
    "state", "rng", "history", "selected_token", "output", "numeric_body", "source", "pipeline", "roles",
    "swap", "guard", "owner", "cpu_owner", "cpu_sum", "body_total", "body_edge", "tcp", "stage",
    "material", "executed", "metric_window", "raw_duration", "sample_duration", "final_peak", "cache",
))
def test_independent_tamper_rejection(cohort, mutation):
    row = cohort["results"][-1]
    execution = row["execution"]
    detail = execution["details"]
    report = detail["report"]
    request = report["runs"][0]
    selection = detail["response_state_audit"]["bindings"][0]["selections"][0]
    if mutation == "missing":
        cohort["results"].pop()
    elif mutation == "incomplete":
        cohort["complete"] = False
    elif mutation == "duplicate":
        cohort["results"][-1] = deepcopy(cohort["results"][-2])
    elif mutation == "order":
        cohort["results"][0], cohort["results"][1] = cohort["results"][1], cohort["results"][0]
    elif mutation == "cohort":
        execution["cohort"] = "d" * 64
    elif mutation == "build":
        execution["build"]["attributes"] = {"forged": True}
    elif mutation == "case_digest":
        execution["case_digest"] = "0" * 64
    elif mutation == "target_digest":
        row["configuration_digest"] = "0" * 64
    elif mutation in ("state", "rng", "selected_token"):
        selection[{"state": "logits_and_active_kv", "rng": "rng_after", "selected_token": "selected_token"}[mutation]] = "f" * 64
    elif mutation == "history":
        selection["cache_lengths"][0] += 1
    elif mutation == "output":
        request["generation"]["output_text_digest"] = "f" * 64
    elif mutation == "numeric_body":
        detail["response_state_audit"]["bindings"][0]["numeric_body_fingerprint"] = "f" * 64
    elif mutation == "source":
        report["configuration"]["source_lock_digest"] = "f" * 64
    elif mutation == "pipeline":
        report["experiment"]["pipeline_digest"] = "f" * 64
    elif mutation == "roles":
        report["configuration"]["roles"].remove("boundary_b")
    elif mutation == "swap":
        report["memory_guard"]["maximum_host_swap_growth_bytes"] = 1
    elif mutation == "guard":
        report["memory_guard"]["tripped"] = True
    elif mutation == "owner":
        detail["response_state_audit"]["client_memory_owners"].remove("role_supervision")
    elif mutation == "cpu_owner":
        del report["process_cpu_accounting"]["cold_first_response_cpu_seconds_by_role"]["boundary_b"]
    elif mutation == "cpu_sum":
        report["process_cpu_accounting"]["aggregate_cold_first_response_cpu_seconds"] += 1
    elif mutation == "body_total":
        report["topology_accounting"]["runs"][0]["all_link_serialized_body_bytes"] += 1
    elif mutation == "body_edge":
        report["topology_accounting"]["runs"][0]["body_bytes_by_edge"][0]["serialized_body_bytes"] += 1
    elif mutation == "tcp":
        sample = report["native_network_accounting"]["samples"]["runs"][0]["after"]["wan_emulation"]
        sample["directed_stream_bytes"]["boundary_a->client"] = 0
    elif mutation == "stage":
        stages = report["topology_accounting"]["stages"]["runs"][0]["body_bytes_by_stage_and_edge"]
        next(iter(stages.values()))["client->inference"] += 1
    elif mutation == "material":
        report["prepared_material_accounting"]["conserved"] = False
    elif mutation == "executed":
        request["privacy"]["generation_executed_decode_steps"] += 1
    elif mutation == "metric_window":
        for record in cohort["results"]:
            for metric in record["metrics"]:
                if metric["id"] in ("response.ttft", "request.0.ttft"):
                    metric["window"] = "after_first_output"
    elif mutation == "raw_duration":
        request["durations"]["full_seconds"] += 1
    elif mutation == "sample_duration":
        row["samples"][0]["duration_seconds"] += 1
    elif mutation == "final_peak":
        next(m for m in row["metrics"] if m["id"] == "process.peak_rss")["value"] = 1
    elif mutation == "cache":
        execution["fixture"]["metadata"]["owned_client_cache"] = "warm"
    with pytest.raises(study.Rejected):
        audit(cohort)


def test_actual_certificate_bytes_fixed_budget_not_measured_network(tmp_path):
    artifact = tmp_path / "certificate.bin"
    payload = b"synthetic artifact bytes only"
    artifact.write_bytes(payload)
    pin = {"path": str(artifact), "digest": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    declaration = {"missing_owners": ["client", "boundary_a", "boundary_b"],
        "delivery_locations": {name: str(artifact) for name in ("client", "boundary_a", "boundary_b")},
        "derivation": "separate offline publisher; no timing observations"}
    result = study.certificate_accounting(pin, declaration)
    assert result["artifact_bytes"] == len(payload)
    assert result["extra_setup_fixed_byte_budget"] == 3 * len(payload)
    assert result["publisher_distribution_network_bytes"] is None
    assert not result["cold_network_complete"]
    declaration["missing_owners"].append("client")
    with pytest.raises(study.Rejected):
        study.certificate_accounting(pin, declaration)
    declaration["missing_owners"].pop()
    artifact.write_bytes(b"x" * len(payload))
    with pytest.raises(study.Rejected, match="digest_mismatch"):
        study.certificate_accounting(pin, declaration)


def test_maximum_not_median_peak_gates_tier(cohort):
    for row in cohort["results"]:
        metric = next(m for m in row["metrics"] if m["id"] == "process.peak_rss")
        metric["value"] = 3 << 30 if row["execution"]["replicate"] == 2 else 1 << 30
    result = audit(cohort)
    for group in result["groups"].values():
        assert group["maximum_client_peak_bytes"] == 3 << 30
        assert group["medians"]["process.peak_rss"] == 1 << 30
        assert not group["observed_tiers"]["medium"]


def test_natural_eos_and_unused_material_not_erased(cohort, source_workload_overlay):
    from pllm.runtime.prepared_accounting import material_accounting
    from pllm.runtime.topology_accounting import private_boundary_body_accounting

    for row in cohort["results"]:
        detail = row["execution"]["details"]
        report = detail["report"]
        request = report["runs"][0]
        # Seven emitted outputs plus final EOS selection: still eight selections.
        request["tokens"]["output_tokens"] = 7
        request["generation"]["response_status"] = "completed"
        privacy = request["privacy"]
        privacy["generation_output_tokens"] = 7
        privacy["prepared_stage_rows_claimed"] -= 1
        privacy["prepared_stage_rows_burned"] += 1
        report["topology_accounting"]["runs"][0] = private_boundary_body_accounting({"privacy": privacy})
        startup = report["topology_accounting"]["startup"]["traffic_attribution"]["audit_counters"]
        keys = ["prepared_stage_rows_" + k for k in ("issued", "reserved", "claimed", "burned", "discarded")]
        report["prepared_material_accounting"] = material_accounting({k: startup[k] + privacy[k] for k in keys})
        rate = next(m["value"] for m in row["metrics"] if m["id"] == "response.decode_tps")
        request["durations"]["generation_seconds"] = 6 / rate
        for metric in row["metrics"]:
            if metric["id"] in ("request.0.decode_tps", "response.decode_tps"):
                metric["denominator"]["value"] = 6
        row["workload_digest"] = bind_fixture(row["execution"], source_workload_overlay)
    result = audit(cohort)
    for group in result["groups"].values():
        for work in group["executed_work"]:
            assert work["output_tokens"] == 7 and work["sampled_selections"] == 8
            assert work["executed_decode_steps"] == 7 and work["terminated_before_cap"]
            assert work["prepared_stage_rows_burned"] == 1


def plan_for(document):
    a, b = pair()
    pin = b["pipeline"]["components"]["boundary"]["params"]["head_certificate"]
    declaration = {"missing_owners": ["client"],
        "delivery_locations": {name: pin["path"] for name in ("client", "boundary_a", "boundary_b")},
        "derivation": "synthetic test, unmeasured"}
    certificate = study.certificate_accounting(pin, declaration, read_artifact=False)
    certificate["artifact_bytes_verified"] = True  # Synthetic original plan assertion, not actual measurement.
    certificate["artifact_base64"] = archive()["artifact_base64"]
    config = document["results"][0]["execution"]["details"]["report"]["configuration"]
    verifier = study.verifier_sources()
    prompt = b"private synthetic test input; copied observations are NOT measurements"
    public = {"path": "/synthetic/prompt.txt", "sha256": hashlib.sha256(prompt).hexdigest(),
        "size_bytes": len(prompt)}
    return {"schema": "pllm.private_boundary_row_codec_plan.v1", "suite": document["suite"],
        "control": a, "candidate": b, "max_output_tokens": 8, "tiny": False,
        "verifier_sources": verifier, "native_head_commitments": native_pins(),
        "frozen_sources": {pin["path"]: pin["digest"], public["path"]: public["sha256"]} | verifier,
        "certificate_declaration": declaration,
        "certificate": certificate, "source_lock_digest": config["source_lock_digest"],
        "prompt_digest": public["sha256"], "public_prompt": public, "public_input_counts": {"150": 150},
        "pending": "synthetic pending integration"}


def test_summary_read_only_without_local_artifact(cohort):
    from scripts.summarize_private_boundary_row_codec import summarize
    plan = plan_for(cohort)
    before = deepcopy((cohort, plan))
    result = summarize(cohort, plan, receipt=study.receipt_record(cohort, plan))
    assert (cohort, plan) == before
    assert result["certificate"]["publisher_distribution_network_bytes"] is None
    assert result["public_prompt"]["sha256"] == plan["prompt_digest"]
    assert result["public_prompt"]["sha256"] != result["identity"]["prompt_digest"]


@pytest.mark.parametrize("mutation", ("fixed_budget", "digest", "bytes_verified", "prompt", "source", "experiment",
    "public_prompt_missing", "public_prompt_path", "public_prompt_sha256", "public_prompt_size_bytes", "input_count"))
def test_replay_plan_tampering_rejected(cohort, mutation):
    from scripts.summarize_private_boundary_row_codec import summarize
    plan = plan_for(cohort)
    if mutation == "fixed_budget":
        plan["certificate"]["extra_setup_fixed_byte_budget"] += 1
    elif mutation == "digest":
        plan["frozen_sources"]["/synthetic/certificate.bin"] = "f" * 64
    elif mutation == "bytes_verified":
        plan["certificate"]["artifact_bytes_verified"] = False
    elif mutation in ("prompt", "source"):
        plan[mutation + "_digest" if mutation == "prompt" else "source_lock_digest"] = "f" * 64
    elif mutation == "public_prompt_missing":
        del plan["public_prompt"]
    elif mutation.startswith("public_prompt_"):
        key = mutation.removeprefix("public_prompt_")
        plan["public_prompt"][key] = 4097 if key == "size_bytes" else "f" * 64
    elif mutation == "input_count":
        plan["public_input_counts"]["150"] += 1
    else:
        plan["candidate"]["pipeline"]["components"]["boundary"]["params"]["head_kernel"] = "byte_dot"
    with pytest.raises(study.Rejected):
        summarize(cohort, plan, receipt=study.receipt_record(cohort, plan))


def test_exclusive_output_no_sdk_call(tmp_path, monkeypatch):
    from scripts import benchmark_private_boundary_row_codec as driver
    output = tmp_path / "run.json"
    output.with_suffix(".progress.json").write_text("existing evidence")
    monkeypatch.setattr(driver, "benchmark_suite", lambda *a, **k: pytest.fail("must not execute"))
    with pytest.raises(FileExistsError):
        driver.main(["--output", str(output), "--control", "unused", "--candidate", "unused",
            "--certificate-accounting", "unused", "--max-output-tokens", "8"])
    assert output.with_suffix(".progress.json").read_text() == "existing evidence"


def test_fabricated_workload_digest_rejected(cohort):
    for row in cohort["results"]:
        row["workload_digest"] = "f" * 64
        row["execution"]["comparison"]["workload_digest"] = "f" * 64
    with pytest.raises(study.Rejected, match="raw_workload_digest_mismatch"):
        audit(cohort)


@pytest.mark.parametrize("mutation", ("missing_binding", "missing_public", "extra_private_field", "version",
    "public_resealed", "private_changed", "private_resealed", "prompt_resealed", "fixture_resealed",
    "final", "missing_final", "missing_measurement", "measurement_version", "measurement_target_resealed"))
def test_public_workload_witness_tampering_rejected(cohort, mutation):
    from pllm._benchmark_contract import digest
    row = cohort["results"][-1]
    execution = row["execution"]
    detail = execution["details"]
    binding = detail["workload_binding"]
    if mutation == "missing_binding":
        del detail["workload_binding"]
    elif mutation == "missing_public":
        del binding["public_workload_digest"]
    elif mutation == "extra_private_field":
        binding["inputs"] = {"prompt": "forbidden witness field"}
    elif mutation == "version":
        binding["schema"] = "unknown-v9"
    elif mutation == "public_resealed":
        binding["public_workload_digest"] = "f"*64
    elif mutation in ("private_changed", "private_resealed"):
        binding["private_inputs_commitment"] = "f"*64
    elif mutation == "prompt_resealed":
        binding["prompt_commitment"] = "f"*64
    elif mutation == "fixture_resealed":
        binding["fixture_digest"] = "f"*64
    elif mutation in ("final", "missing_final"):
        row["workload_digest"] = execution["comparison"]["workload_digest"] = None if mutation == "missing_final" else "f"*64
    elif mutation == "missing_measurement":
        del detail["workload_measurement"]
    elif mutation == "measurement_version":
        detail["workload_measurement"]["workload_binding_schema"] = "unknown-v9"
    elif mutation == "measurement_target_resealed":
        detail["workload_measurement"]["target_measurement_digest"] = "f"*64
        execution["comparison"]["measurement_digest"] = digest(detail["workload_measurement"])
    if mutation.endswith("_resealed") and not mutation.startswith("measurement"):
        row["workload_digest"] = execution["comparison"]["workload_digest"] = digest(binding)
    with pytest.raises(study.Rejected):
        audit(cohort)


def test_mixed_opaque_inputs_rejected_even_when_every_changed_cell_is_self_consistent(cohort):
    from pllm._benchmark_contract import digest
    for row in cohort["results"]:
        if row["execution"]["case"]["name"] == study.CANDIDATE:
            binding = row["execution"]["details"]["workload_binding"]
            binding["private_inputs_commitment"] = "f"*64
            row["workload_digest"] = row["execution"]["comparison"]["workload_digest"] = digest(binding)
    with pytest.raises(study.Rejected, match="private_inputs_commitment_changed"):
        audit(cohort)


def test_legacy_typed_reports_readable_but_no_witness_can_be_invented(cohort):
    from pllm._benchmark_contract import digest
    # Remove new witnesses and recreate the old opaque-HMAC-shaped record. Typed
    # v2 parsing remains legal; strict new row-codec admission must reject it.
    for row in cohort["results"]:
        detail = row["execution"]["details"]
        del detail["workload_binding"]
        old_measurement = detail.pop("workload_measurement")["target_measurement_digest"]
        row["execution"]["comparison"]["measurement_digest"] = old_measurement
        row["workload_digest"] = row["execution"]["comparison"]["workload_digest"] = digest({"opaque_old_identity": True})
    before = deepcopy(cohort)
    BenchmarkRun(cohort)
    with pytest.raises(study.Rejected, match="workload_binding_missing"):
        audit(cohort)
    assert cohort == before


def test_failed_tiny_v2_historical_artifact_remains_typed_readable_and_unchanged():
    path = Path("target/certified-head-tiny-cohort-v2.json")
    if not path.exists():
        pytest.skip("optional historical parent attempt; no generation/download")
    before = path.read_bytes()
    document = json.loads(before)
    BenchmarkRun(document)
    assert all("workload_binding" not in row["execution"]["details"] for row in document["results"])
    # Exercise the exact missing-witness gate without substituting the historic
    # build/native pins or constructing a repaired cohort.
    row = document["results"][0]
    case = study.BenchmarkSuite.from_spec(document["suite"]).cases[0]
    with pytest.raises(study.Rejected, match="workload_binding_missing"):
        study._workload_witness(row, case, row["execution"]["details"]["report"])
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ("schema", "descriptor_digest", "source_digest", "checkpoint_digest",
    "model_config_digest", "composition_digest", "schedule_digest", "head_weight_digest", "head_scales_digest",
    "lookup_digest", "vocabulary", "hidden", "signed_output_bound", "activation_bits", "weight_bits",
    "head_stage_id", "ring", "wire_bits", "modulus", "leaf_bits", "head_dispatch", "head_output_encoding",
    "head_certificate_digest", "widths_digest", "width_counts", "total_bits", "payload_bytes_per_role",
    "subframe_header_bytes", "certificate_digest", "certificate_size", "certificate_source", "bounds_digest",
    "artifact_digest", "artifact_size_bytes", "publisher", "missing_role", "replies", "outer_body_bytes",
    "subframe_bytes", "header_bytes", "payload_bytes"))
def test_raw_head_telemetry_tamper_rejected(cohort, field):
    raw = cohort["results"][1]["execution"]["details"]["private_boundary_head"]
    if field in ("schema", "descriptor_digest"):
        raw[field] = "f" * 64
    elif field in raw["descriptor"]:
        value = raw["descriptor"][field]
        raw["descriptor"][field] = value + 1 if type(value) is int else "f" * 64
    elif field in raw["layout"]:
        value = raw["layout"][field]
        raw["layout"][field] = value + 1 if type(value) is int else {"24": 151936} if type(value) is dict else "f" * 64
    elif field.startswith("certificate_"):
        cert = raw["layout"]["certificate"]
        if field == "certificate_source":
            cert["metadata"]["source"]["lookup_digest"] = "f" * 64
        elif field == "certificate_size":
            cert["size_bytes"] += 1
        else:
            cert["digest"] = "f" * 64
    elif field in ("bounds_digest", "artifact_digest", "artifact_size_bytes"):
        metadata = raw["layout"]["certificate"]["metadata"]
        metadata[field] = metadata[field] + 1 if type(metadata[field]) is int else "f" * 64
    elif field == "publisher":
        raw["layout"]["certificate"]["metadata"]["implementation"]["pllm._native"] = "f" * 64
    elif field == "missing_role":
        del raw["roles"]["boundary_b"]
    else:
        raw["roles"]["boundary_b"][field] += 1
    with pytest.raises(study.Rejected):
        audit(cohort)


def test_head_telemetry_does_not_change_state_or_output_identity(cohort):
    for row in cohort["results"]:
        assert "private_boundary_head" in row["execution"]["details"]
        assert all("private_boundary_head" not in binding for binding in
            row["execution"]["details"]["response_state_audit"]["bindings"])
    result = audit(cohort)
    for pair in result["body_reconciliation"][study.CANDIDATE]:
        assert sum(pair["head_codec_partition"].values()) == pair["head_reply_delta_bytes"]


@pytest.mark.parametrize("missing", ("raw", "certificate", "native", "sdk_source"))
def test_certification_missing_evidence_rejected(cohort, missing):
    if missing == "raw":
        del cohort["results"][1]["execution"]["details"]["private_boundary_head"]
    if missing == "sdk_source":
        # Reseal the synthetic build so this exercises provenance, not a broken build hash.
        from pllm._benchmark_contract import digest
        build = deepcopy(cohort["results"][0]["execution"]["build"])
        del build["attributes"]["sdk_sources"]["runtime/private_boundary_head_codec.py"]
        build["digest"] = digest(build["attributes"])
        for case in cohort["suite"]["cases"]:
            case["build"]["expected_digest"] = build["digest"]
        suite = study.BenchmarkSuite.from_spec(cohort["suite"])
        for row in cohort["results"]:
            execution = row["execution"]
            case = next(c for c in suite.cases if c.name == execution["case"]["name"])
            execution.update(build=deepcopy(build), case=case.to_spec(), case_digest=digest(case.to_spec()),
                suite_digest=suite.digest)
    with pytest.raises(study.Rejected):
        study.audit(cohort, certificate=None if missing == "certificate" else archive(),
            native_commitments=None if missing == "native" else native_pins())


def test_nonzero_head_payload_savings_keep_envelopes_and_controls_separate(cohort):
    from pllm.runtime.private_boundary_transport import BoundaryDescriptor
    _, _, metadata = public_certificate()
    metadata = deepcopy(metadata)
    widths = bytes([24]) * 151936
    metadata["widths_digest"] = hashlib.sha256(widths).hexdigest()
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    payload = struct.pack("<8sI", b"PLLMBRC1", len(encoded)) + encoded + zlib.compress(widths)
    pin = {"path": "/synthetic/certificate.bin", "digest": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    spec = pair()[1]
    spec["pipeline"]["components"]["boundary"]["params"]["head_certificate"] = pin
    exp = Experiment.from_spec(spec)
    descriptor = BoundaryDescriptor(**metadata["source"], composition_digest=exp.pipeline.digest(),
        schedule_digest="7" * 64, head_dispatch="parallel", head_output_encoding="row_residues",
        head_certificate_digest=pin["digest"])
    row = cohort["results"][1]
    detail, report = row["execution"]["details"], row["execution"]["details"]["report"]
    raw = deepcopy(detail["private_boundary_head"])
    raw.update(descriptor=descriptor.to_dict(), descriptor_digest=descriptor.digest)
    raw["layout"].update(widths_digest=metadata["widths_digest"], width_counts={"24": 151936},
        total_bits=24 * 151936, payload_bytes_per_role=3 * 151936,
        certificate={"digest": pin["digest"], "size_bytes": len(payload), "metadata": {
            **metadata, "artifact_digest": pin["digest"], "artifact_size_bytes": len(payload)}})
    for role, counters in raw["roles"].items():
        counters["payload_bytes"] = 8 * 3 * 151936
        counters["subframe_bytes"] = counters["payload_bytes"] + counters["header_bytes"]
        counters["outer_body_bytes"] = counters["subframe_bytes"] + 8 * 142
        report["runs"][0]["privacy"][f"role_link.{role}.online_download_bytes"] = counters["outer_body_bytes"] + 153372
    report["runs"][0]["privacy"]["private_boundary_head_download_bytes"] = sum(
        c["outer_body_bytes"] for c in raw["roles"].values())
    native = deepcopy(native_pins()[study.CANDIDATE])
    native.update(pipeline_digest=exp.pipeline.digest(), composition_digest=descriptor.composition_digest,
        descriptor_digest=descriptor.digest)
    result = study._head_evidence(raw, detail["response_state_audit"]["bindings"][0], exp,
        report, (metadata, widths), pin, native)
    assert result["header_bytes"] == 16 * 74 and result["envelope_bytes"] == 16 * 142
    old = detail["private_boundary_head"]["roles"]["boundary_a"]["outer_body_bytes"] * 2
    assert result["outer_body_bytes"] - old == 16 * (3 * 151936 - 474800)


@pytest.mark.parametrize("name", study.NAMES)
def test_actual_sdk_public_layout_matches_independent_auditor(cohort, tmp_path, name):
    from pllm.runtime.private_boundary_head_codec import HeadOutputCodec
    from pllm.runtime.private_boundary_transport import BoundaryDescriptor

    original_pin, payload, metadata = public_certificate()
    path = tmp_path / "head.cert"
    path.write_bytes(payload)
    pin = {**original_pin, "path": str(path)}
    index = study.NAMES.index(name)
    spec = pair()[index]
    if name == study.CANDIDATE:
        spec["pipeline"]["components"]["boundary"]["params"]["head_certificate"] = pin
    exp = Experiment.from_spec(spec)
    descriptor = BoundaryDescriptor(**metadata["source"], composition_digest=exp.pipeline.digest(),
        schedule_digest="7" * 64, head_dispatch="parallel",
        head_output_encoding="row_residues" if name == study.CANDIDATE else "uniform_residues",
        head_certificate_digest=pin["digest"] if name == study.CANDIDATE else "")
    codec = HeadOutputCodec(descriptor, certificate=pin if name == study.CANDIDATE else None)
    details = cohort["results"][index]["execution"]["details"]
    raw = deepcopy(details["private_boundary_head"])
    raw.update(descriptor=descriptor.to_dict(), descriptor_digest=descriptor.digest, layout=codec.public_layout())
    native = {**native_pins()[name], "pipeline_digest": exp.pipeline.digest(),
        "composition_digest": descriptor.composition_digest, "descriptor_digest": descriptor.digest}
    result = study._head_evidence(raw, details["response_state_audit"]["bindings"][0], exp,
        details["report"], (metadata, bytes([25]) * 151936), pin, native)
    assert result["layout"] == codec.public_layout()
    if name == study.CANDIDATE:
        for key in ("artifact_digest", "artifact_size_bytes"):
            damaged = deepcopy(raw)
            del damaged["layout"]["certificate"]["metadata"][key]
            with pytest.raises(study.Rejected, match="head_layout_mismatch"):
                study._head_evidence(damaged, details["response_state_audit"]["bindings"][0], exp,
                    details["report"], (metadata, bytes([25]) * 151936), pin, native)


@pytest.mark.parametrize("field", ("pipeline_digest", "composition_digest", "schedule_digest",
    "descriptor_digest", "numeric_body_fingerprint"))
def test_independent_native_pin_tamper_rejected(cohort, field):
    pins = deepcopy(native_pins())
    pins[study.CANDIDATE][field] = "f" * 64
    with pytest.raises(study.Rejected):
        study.audit(cohort, certificate=archive(), native_commitments=pins)


@pytest.mark.parametrize("mutation", ("bytes", "missing", "extra", "oversized"))
def test_frozen_certificate_archive_tamper_rejected(cohort, mutation):
    certificate = archive()
    if mutation == "missing":
        del certificate["artifact_base64"]
    elif mutation == "oversized":
        certificate["artifact_base64"] = "A" * ((1 << 21) + 1)
    else:
        payload = bytearray(public_certificate()[1])
        if mutation == "bytes":
            payload[-1] ^= 1
        else:
            payload.extend(b"x")
        certificate["artifact_base64"] = base64.b64encode(payload).decode()
    with pytest.raises(study.Rejected):
        study.audit(cohort, certificate=certificate, native_commitments=native_pins())


@pytest.mark.parametrize("mutation", ("missing", "extra", "malformed", "changed", "frozen", "oversized"))
def test_verifier_provenance_rejected(cohort, mutation):
    from scripts.summarize_private_boundary_row_codec import summarize
    plan = plan_for(cohort)
    path = "scripts/summarize_private_boundary_row_codec.py"
    if mutation == "missing":
        del plan["verifier_sources"][path]
    elif mutation == "extra":
        plan["verifier_sources"]["untrusted.py"] = "a" * 64
    elif mutation == "malformed":
        plan["verifier_sources"][path] = "not-a-digest"
    elif mutation == "changed":
        plan["verifier_sources"][path] = "f" * 64
    elif mutation == "frozen":
        plan["frozen_sources"] = {"/synthetic/certificate.bin": "a" * 64}
    else:
        plan["frozen_sources"].update({str(i): "a" * 64 for i in range(65)})
    with pytest.raises(study.Rejected):
        summarize(cohort, plan, receipt=study.receipt_record(cohort, plan))


def test_current_verifier_bytes_checked(cohort, monkeypatch):
    from scripts.summarize_private_boundary_row_codec import summarize
    plan = plan_for(cohort)
    observed = dict(plan["verifier_sources"])
    observed["scripts/summarize_private_boundary_row_codec.py"] = "f" * 64
    monkeypatch.setattr(study, "verifier_sources", lambda: observed)
    with pytest.raises(study.Rejected, match="replay_verifier_changed"):
        summarize(cohort, plan, receipt=study.receipt_record(cohort, plan))


@pytest.mark.parametrize("limit", (4096, 64 << 10, 1 << 20))
def test_bounded_read_never_uses_whole_file_read(tmp_path, monkeypatch, limit):
    path = tmp_path / "input"
    path.write_bytes(b"x" * (limit + 1))
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("unbounded input read"))
    with pytest.raises(study.Rejected, match="input_size_invalid"):
        study.bounded_read(path, limit)


def test_streaming_hash_no_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"public source")
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("unbounded hash read"))
    assert study.hashes([path]) == {str(path): hashlib.sha256(b"public source").hexdigest()}


def test_certificate_oversize_bounded(tmp_path, monkeypatch):
    path = tmp_path / "certificate"
    path.write_bytes(b"x" * 101)
    pin = {"path": str(path), "digest": "a" * 64, "size_bytes": 100}
    declaration = {"missing_owners": [], "delivery_locations": {
        owner: str(path) for owner in ("client", "boundary_a", "boundary_b")}, "derivation": "test"}
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("unbounded certificate read"))
    with pytest.raises(study.Rejected, match="input_size_invalid"):
        study.certificate_accounting(pin, declaration)


@pytest.mark.parametrize("index,limit", ((0, 1 << 20), (1, 1 << 20), (2, 64 << 10), (3, 4096)))
def test_oversized_driver_inputs_rejected_before_hash_or_admission(tmp_path, monkeypatch, index, limit):
    from contextlib import nullcontext
    from scripts import benchmark_private_boundary_row_codec as driver
    paths = [tmp_path / name for name in ("control", "candidate", "declaration", "prompt")]
    for path in paths:
        path.write_bytes(b"{}")
    paths[index].write_bytes(b"x" * (limit + 1))
    monkeypatch.setattr(driver, "FileLock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(driver, "hashes", lambda *a: pytest.fail("hash before bounded validation"))
    monkeypatch.setattr(study, "preflight", lambda *a: pytest.fail("admission before bounded validation"))
    monkeypatch.setattr(driver.BuildRef, "capture", classmethod(lambda cls: pytest.fail("build before validation")))
    monkeypatch.setattr(driver, "benchmark_suite", lambda *a, **k: pytest.fail("must not execute"))
    output = tmp_path / "run.json"
    assert driver.main(["--output", str(output), "--control", str(paths[0]), "--candidate", str(paths[1]),
        "--certificate-accounting", str(paths[2]), "--prompt", str(paths[3]), "--max-output-tokens", "8"]) == 1
    progress = json.loads(output.with_suffix(".progress.json").read_bytes())
    assert progress["failure_code"] == "row_codec_input_size_invalid"
    assert not progress["measurement_complete"] and not output.exists()


@pytest.mark.parametrize("mutation", ("no_manifest", "bad_frozen_digest", "too_long_path", "wrong_tiny"))
def test_replay_manifest_schema_rejected(cohort, mutation):
    from scripts.summarize_private_boundary_row_codec import summarize
    plan = plan_for(cohort)
    if mutation == "no_manifest":
        del plan["verifier_sources"]
    elif mutation == "bad_frozen_digest":
        plan["frozen_sources"]["unused"] = "not-a-digest"
    elif mutation == "too_long_path":
        plan["frozen_sources"]["x" * 4097] = "a" * 64
    else:
        plan["tiny"] = "false"
    with pytest.raises(study.Rejected):
        summarize(cohort, plan, receipt=study.receipt_record(cohort, plan))


@pytest.mark.parametrize("mutation", ("missing", "in_range_size", "plan", "raw", "cohort", "status", "checks",
    "prompt_binding", "schema", "extra"))
def test_terminal_receipt_binds_original_records(cohort, mutation):
    from scripts.summarize_private_boundary_row_codec import summarize
    plan = plan_for(cohort)
    receipt = study.receipt_record(cohort, plan)
    if mutation == "missing":
        receipt = None
    elif mutation == "in_range_size":
        plan["public_prompt"]["size_bytes"] += 1
    elif mutation == "plan":
        plan["pending"] = "changed"
    elif mutation == "raw":
        cohort["results"][0]["execution"]["details"]["new_field"] = True
    elif mutation == "cohort":
        receipt["cohort"] = "f" * 64
    elif mutation == "status":
        receipt["status"] = "rejected"
    elif mutation == "checks":
        receipt["post_run_checks"] = "pending"
    elif mutation == "prompt_binding":
        receipt["public_prompt_digest"] = "f" * 64
    elif mutation == "schema":
        receipt["schema"] = "unknown-v2"
    else:
        receipt["unapproved"] = True
    with pytest.raises(study.Rejected, match="terminal_receipt"):
        summarize(cohort, plan, receipt=receipt)


def test_terminal_receipt_issuance_source_drift_and_interruption(cohort, monkeypatch):
    plan = plan_for(cohort)
    monkeypatch.setattr(study, "verifier_sources", lambda: plan["verifier_sources"])
    inputs = {key: value for key, value in plan["frozen_sources"].items()
        if key not in plan["verifier_sources"]}
    monkeypatch.setattr(study, "hashes", lambda paths: inputs)
    assert study.terminal_receipt(cohort, plan, []) == study.receipt_record(cohort, plan)
    inputs[plan["public_prompt"]["path"]] = "f" * 64
    with pytest.raises(study.Rejected, match="sources_changed_during_run"):
        study.terminal_receipt(cohort, plan, [])

    def interrupted(paths):
        raise KeyboardInterrupt()

    monkeypatch.setattr(study, "hashes", interrupted)
    with pytest.raises(KeyboardInterrupt):
        study.terminal_receipt(cohort, plan, [])


@pytest.mark.parametrize("padding", ("\n", "\r\n", " \t\n", ""))
def test_real_dashboard_preparation_and_tiny_tokenizer(tmp_path, monkeypatch, padding):
    """Real SDK route/parser/renderer/tokenizer; no lifespan, roles or tensors."""
    import asyncio
    from types import SimpleNamespace
    from starlette.requests import Request
    from pllm import Model
    from pllm.deployment.execution import _TokenizerView
    from pllm.runtime import dashboard
    from pllm.runtime.responses import normalize_input
    from pllm.sources import _tiny_model_config
    from benchmarks.research.sota_combinations import public_input_counts

    # Public tiny-v5 fixture text; no historical target artifact dependency.
    text = "Explain private inference and why two workers cannot individually see private input."
    raw = text + padding
    received = []
    runtime = SimpleNamespace(begin=lambda prompt, *a, **k: received.append(prompt) or "test-run")
    monkeypatch.setattr(dashboard, "DashboardRuntime", lambda *a: runtime)
    config = dashboard.DashboardConfig("127.0.0.1", 8999, None, "test", 2,
        history_path=tmp_path / "history.sqlite")
    app = dashboard.create_dashboard_app(config)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/api/run")

    async def receive():
        return {"type": "http.request", "body": json.dumps({"prompt": raw}).encode()}

    request = Request({"type": "http", "headers": [(b"content-type", b"application/json")]}, receive)
    try:
        asyncio.run(endpoint(request))
    finally:
        app.state.benchmark_history.close()
    assert received == [study.public_prompt_context(raw)] == [text]
    descriptor = {"kind": "byte", "chat_template":
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant: "}
    view = _TokenizerView(descriptor, _tiny_model_config(), SimpleNamespace(chat_template=descriptor["chat_template"]))
    rows = [row.to_prompt_dict() for row in normalize_input(received[0])]
    actual = len(view.tokenizer().encode(view.render_prompt(rows), add_bos=True))
    counts, _ = public_input_counts(Model.tiny(), None, {"tiny": {"prompt": study.public_prompt_context(raw)}})
    assert counts == {"tiny": actual}
    unnormalized, _ = public_input_counts(Model.tiny(), None, {"tiny": {"prompt": raw}})
    assert unnormalized["tiny"] == actual + len(padding.encode())
    # create_tiny_llama_checkpoint's tokenizer template emits a literal \\n.
    # This generated-source path is tiny-v5: 105 raw versus 104 prepared.
    generated = {**descriptor, "chat_template": descriptor["chat_template"].replace("\n", "\\n")}
    generated_view = _TokenizerView(generated, _tiny_model_config(),
        SimpleNamespace(chat_template=generated["chat_template"]))
    prepared = generated_view.render_prompt(rows)
    assert len(generated_view.tokenizer().encode(prepared, add_bos=True)) == 104
    unprepared = generated_view.render_prompt([{"role": "user", "content": raw}])
    assert len(generated_view.tokenizer().encode(unprepared, add_bos=True)) == 104 + len(padding.encode())


@pytest.mark.parametrize("outcome", ("completed", "interrupted", "source_drift", "check_interrupted", "raw_drift",
    "audit_rejected"))
def test_driver_terminal_receipt_only_after_successful_post_checks(tmp_path, monkeypatch, outcome):
    """Synthetic SDK boundary only; exercise real driver persistence/check ordering."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from scripts import benchmark_private_boundary_row_codec as driver
    from scripts import summarize_private_boundary_row_codec as replay
    from benchmarks.research.sota_combinations import atomic_json

    artifact = tmp_path / "head.cert"
    artifact.write_bytes(public_certificate()[1])
    a, b = pair()
    b["pipeline"]["components"]["boundary"]["params"]["head_certificate"] = {
        **public_certificate()[0], "path": str(artifact)}
    paths = {name: tmp_path / (name + ".json") for name in ("control", "candidate", "accounting", "native")}
    declaration = {"missing_owners": ["client"], "delivery_locations": {
        owner: str(artifact) for owner in ("client", "boundary_a", "boundary_b")}, "derivation": "synthetic"}
    for name, value in (("control", a), ("candidate", b), ("accounting", declaration), ("native", native_pins())):
        atomic_json(paths[name], value)
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"public fixture\n")
    output = tmp_path / "run.json"
    document = {"complete": True, "cohort": "e" * 64}
    monkeypatch.setattr(driver, "FileLock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(driver.BuildRef, "capture", classmethod(lambda cls: BuildRef(expected_digest="a" * 64)))
    monkeypatch.setattr(study, "preflight", lambda suite: [{"admission": {"admitted": True}}])

    def counts(model, tokenizer, inputs):
        assert inputs == {"150": {"prompt": "public fixture"}}
        return {"150": 150}, "f" * 64

    def execute(suite, *, inputs, output):
        assert all(value == {"prompt": "public fixture\n"} for value in inputs.values())
        atomic_json(output, document)
        if outcome == "interrupted":
            raise KeyboardInterrupt()
        if outcome == "source_drift":
            prompt.write_bytes(b"changed public fixture\n")
        if outcome == "raw_drift":
            atomic_json(output, {**document, "changed": True})
        return SimpleNamespace(to_dict=lambda: document)

    monkeypatch.setattr(driver, "public_input_counts", counts)
    monkeypatch.setattr(driver, "benchmark_suite", execute)
    def summarize(*a, **k):
        if outcome == "audit_rejected":
            raise study.Rejected("synthetic_audit_rejected")
        return {"clients": 6, "maximum_observed_swap_growth_bytes": 0}

    monkeypatch.setattr(replay, "summarize", summarize)
    if outcome == "check_interrupted":
        def interrupt(*a):
            raise KeyboardInterrupt()
        monkeypatch.setattr(study, "terminal_receipt", interrupt)
    code = driver.main(["--output", str(output), "--control", str(paths["control"]),
        "--candidate", str(paths["candidate"]), "--certificate-accounting", str(paths["accounting"]),
        "--native-head-commitments", str(paths["native"]), "--prompt", str(prompt), "--max-output-tokens", "8"])
    assert code == (0 if outcome == "completed" else 1)
    progress = json.loads(output.with_suffix(".progress.json").read_bytes())
    assert progress["measurement_complete"] is (outcome == "completed")
    assert output.with_suffix(".receipt.json").exists() is (outcome == "completed")
    assert output.with_suffix(".summary.json").exists() is (outcome == "completed")
    assert json.loads(output.read_bytes())["complete"] is True  # Preserve raw complete attempts on rejection.
    if outcome == "completed":
        plan = json.loads(output.with_suffix(".plan.json").read_bytes())
        receipt = json.loads(output.with_suffix(".receipt.json").read_bytes())
        study.verify_receipt(document, plan, receipt)
        assert plan["public_prompt"]["size_bytes"] == len(b"public fixture\n")
    else:
        assert progress["failure_code"] == {"interrupted": "interrupted", "check_interrupted": "interrupted",
            "source_drift": "sources_changed_during_run", "raw_drift": "row_codec_persisted_report_mismatch",
            "audit_rejected": "synthetic_audit_rejected"}[outcome]


def test_existing_receipt_reserves_output(tmp_path):
    from scripts import benchmark_private_boundary_row_codec as driver
    output = tmp_path / "run.json"
    receipt = output.with_suffix(".receipt.json")
    receipt.write_text("existing terminal provenance")
    with pytest.raises(FileExistsError, match="terminal receipt"):
        driver.main(["--output", str(output), "--control", "unused", "--candidate", "unused",
            "--certificate-accounting", "unused", "--max-output-tokens", "8"])
    assert receipt.read_text() == "existing terminal provenance"
    assert not output.with_suffix(".progress.json").exists()


@pytest.mark.parametrize("admitted", (False, True))
def test_metadata_preflight_before_tokenizer_or_execution(tmp_path, monkeypatch, admitted):
    from contextlib import nullcontext
    from scripts import benchmark_private_boundary_row_codec as driver

    artifact = tmp_path / "certificate.bin"
    payload = public_certificate()[1]
    artifact.write_bytes(payload)
    a, b = pair()
    b["pipeline"]["components"]["boundary"]["params"]["head_certificate"] = {
        "path": str(artifact), "digest": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    control, candidate, accounting, prompt = (tmp_path / name for name in
        ("control.json", "candidate.json", "accounting.json", "prompt.txt"))
    control.write_text(json.dumps(a))
    candidate.write_text(json.dumps(b))
    accounting.write_text(json.dumps({"missing_owners": ["client"],
        "delivery_locations": {owner: str(artifact) for owner in ("client", "boundary_a", "boundary_b")},
        "derivation": "synthetic test only"}))
    prompt.write_text("public prompt")
    monkeypatch.setattr(driver, "FileLock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(driver.BuildRef, "capture", classmethod(lambda cls: BuildRef(expected_digest="a" * 64)))
    monkeypatch.setattr(driver.study, "preflight", lambda suite: [
        {"case": c.name, "admission": {"admitted": admitted}} for c in suite.cases])
    monkeypatch.setattr(driver, "benchmark_suite", lambda *a, **k: pytest.fail("preflight must not execute"))

    def token_metadata(*args):
        assert admitted, "denied admission must precede tokenizer/source resolution"
        return {"150": 150}, "f" * 64

    monkeypatch.setattr(driver, "public_input_counts", token_metadata)
    output = tmp_path / "run.json"
    code = driver.main(["--output", str(output), "--control", str(control), "--candidate", str(candidate),
        "--certificate-accounting", str(accounting), "--prompt", str(prompt), "--max-output-tokens", "8",
        "--preflight-only"])
    assert code == (0 if admitted else 1)
    assert not output.exists() and not output.with_suffix(".summary.json").exists()
    progress = json.loads(output.with_suffix(".progress.json").read_bytes())
    assert progress["status"] == ("admitted" if admitted else "rejected")
    assert not progress["measurement_complete"]

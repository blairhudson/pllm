from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from pllm._cli.app import build_parser
from pllm.configuration import Experiment
from pllm.runtime.benchmark_cli import (
    accounted_benchmark_body_totals,
    _run_once,
    _wait_for_ready,
    build_comparison_report,
    build_loopback_report,
    run_loopback_benchmark,
)


def test_accounted_totals_include_prewarm_once_and_preserve_run_window_scope() -> None:
    def ledger(count):
        return {"all_link_serialized_body_bytes": count, "tracked_body_counter_set_present": True}
    assert accounted_benchmark_body_totals({"startup": ledger(100), "warmups": [ledger(20)], "runs": [ledger(30), ledger(40)]}) == {
        "accounted_setup_through_first_response_body_bytes": 120,
        "total_accounted_benchmark_body_bytes": 190,
    }
    assert all(value is None for value in accounted_benchmark_body_totals({"startup": None, "runs": [ledger(30)]}).values())
from pllm.runtime.topology_accounting import (
    prepared_body_accounting,
    prepared_stage_body_attribution,
)


def _record(
    run_id: str, *, ttft: float, throughput: float, full: float = 4.0
) -> dict[str, object]:
    return {
        "schema_version": "pllm.benchmark_run.v3",
        "run_id": run_id,
        "status": "completed",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_fingerprint": "sha256:model",
        "durations": {
            "full_seconds": full,
            "online_seconds": 3.0,
            "ttft_seconds": ttft,
            "tokens_per_second": throughput,
        },
        "tokens": {
            "input_tokens": 39,
            "output_tokens": 24,
            "total_tokens": 63,
            "authoritative": True,
        },
        "privacy": {
            "plaintext_prompt_bytes_sent": 0,
            "plaintext_token_ids_sent": 0,
            "preparation_requests_during_online": 0,
        },
    }


def _report(*, full: float = 4.0) -> dict[str, object]:
    return build_loopback_report(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tiny=False,
        max_output_tokens=24,
        warmup_runs=[],
        runs=[_record("run-1", ttft=0.98, throughput=8.46, full=full)],
    )


def _experiment(name: str, threads: int) -> Experiment:
    return Experiment.from_spec(
        {
            "schema": "pllm.experiment.v2",
            "name": name,
            "pipeline": {
                "model": {"source": "Qwen/Qwen2.5-0.5B-Instruct"},
                "components": {
                    "inference": {"component": "pllm/inference", "params": {}},
                    "kernels": {"component": "pllm/cpu", "params": {"threads": threads}},
                    "linear": {"component": "pllm/masked-linear", "params": {}},
                    "preparation": {
                        "component": "pllm/model-aware-corrections",
                        "params": {},
                    },
                },
            },
            "deployment": {"kind": "local", "root": "local://benchmark"},
            "budget": {"requests": 4, "max_input_tokens": 256, "max_new_tokens": 24},
        }
    )


def test_benchmark_parser_defaults_to_real_qwen() -> None:
    args = build_parser().parse_args(["benchmark", "run"])
    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args.tiny is False
    assert args.warmups == 0
    assert args.repetitions == 1
    assert args.show_dashboard is False
    assert args.inventory_policy is None
    assert args.bundle_compression is None
    assert args.prefill_cache_mib == 0


def test_stage_attribution_reconciles_only_exact_protocol_bodies() -> None:
    before = {("client", "inference", "layers.0.qkv"): 100}
    after = {
        ("client", "inference", "layers.0.qkv"): 230,
        ("inference", "client", "layers.0.qkv"): 270,
        ("client", "preparation", "layers.0.qkv"): 30,
    }
    privacy = {
        "preparation_upload_bytes": 30,
        "preparation_download_bytes": 0,
        "correction_push_bytes": 0,
        "inference_upload_bytes": 130,
        "inference_download_bytes": 270,
        "bundle_network_bytes": 20_000,
        "session_authorization_upload_bytes": 400,
    }
    result = prepared_stage_body_attribution(before, after, privacy)
    assert result["reconciled_with_protocol_bodies"]
    assert result["body_bytes_by_stage_and_edge"] == {
        "layers.0.qkv": {
            "client->inference": 130,
            "inference->client": 270,
            "client->preparation": 30,
        }
    }
    assert sum(result["measured_body_bytes_by_edge"].values()) == 430
    incomplete = prepared_stage_body_attribution(
        before, after, {**privacy, "inference_download_bytes": 271}
    )
    assert incomplete["reconciled_with_protocol_bodies"] is False
    assert incomplete["body_bytes_by_stage_and_edge"] is None
    with pytest.raises(ValueError, match="invalid or decreasing stage telemetry"):
        prepared_stage_body_attribution({("client", "inference", 42): 0}, {}, privacy)


def test_benchmark_parser_accepts_request_sized_inventory() -> None:
    args = build_parser().parse_args([
        "benchmark", "run", "--inventory-policy", "request-sized",
        "--bundle-compression", "zlib",
        "--prefill-cache-mib", "64",
    ])
    assert args.inventory_policy == "request-sized"
    assert args.bundle_compression == "zlib"
    assert args.prefill_cache_mib == 64


def test_locked_bundle_compression_evidence_preserves_exact_matched_costs() -> None:
    evidence = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/evidence"
         / "client-bundle-compression-qwen25-2026-09-27.json").read_text()
    )
    assert evidence["model_fingerprint"] == (
        "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974"
    )
    cold = evidence["cold_one_response"]
    warm = evidence["after_one_warmup"]
    assert cold["body_bytes_saved"] == (
        cold["uncompressed"]["client_bundle_body_bytes"]
        - cold["compressed"]["client_bundle_body_bytes"]
    )
    assert cold["body_bytes_saved"] == (
        cold["uncompressed"]["all_link_covered_body_bytes"]
        - cold["compressed"]["all_link_covered_body_bytes"]
    )
    assert {
        cold["uncompressed"]["online_all_link_body_bytes"],
        cold["compressed"]["online_all_link_body_bytes"],
        warm["uncompressed"]["online_all_link_body_bytes"],
        warm["compressed"]["online_all_link_body_bytes"],
    } == {62248704}
    assert cold["compressed"]["aggregate_cold_cpu_seconds"] > (
        cold["uncompressed"]["aggregate_cold_cpu_seconds"]
    )
    assert warm["uncompressed"]["client_bundle_body_bytes"] == 0
    assert warm["compressed"]["client_bundle_body_bytes"] == 0
    assert warm["uncompressed"]["all_link_covered_body_bytes"] == (
        warm["compressed"]["all_link_covered_body_bytes"]
    )


def test_locked_exact_prefill_cache_evidence_preserves_cohort_and_savings() -> None:
    evidence = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/evidence"
         / "exact-prefill-cache-qwen25-2026-09-27.json").read_text()
    )
    assert evidence["model_fingerprint"] == (
        "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974"
    )
    assert evidence["cohort"]["input_tokens"] == 39
    assert evidence["cohort"]["output_tokens"] == 1
    assert len(set(evidence["first_cold_response"].values())) == 1
    warm = evidence["second_warm_response"]
    assert warm["covered_body_bytes_saved"] == (
        warm["without_cache"]["all_link_covered_body_bytes"]
        - warm["with_64_mib_cache"]["all_link_covered_body_bytes"]
    )
    assert warm["without_cache"]["online_all_link_body_bytes"] == 62248704
    assert warm["with_64_mib_cache"]["online_all_link_body_bytes"] == 0
    assert warm["with_64_mib_cache"]["inventory_required_rows_per_stage"] == 1


def test_locked_semantic_stage_attribution_excludes_mlp_only_tenfold_claim() -> None:
    evidence = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/evidence"
         / "prepared-stage-attribution-qwen25-2026-09-29.json").read_text()
    )
    assert evidence["source"]["body_fingerprint"] == (
        "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974"
    )
    for output in ("8", "32"):
        cohort = evidence["cohorts"][output]
        roles = cohort["stage_body_bytes_by_semantic_role"]
        total = cohort["covered_all_link_body_bytes"]
        mlp = roles["mlp_gate_up"] + roles["mlp_down"]
        assert cohort["stage_protocol_edges_reconciled"]
        assert sum(roles.values()) == cohort["attributed_stage_body_bytes"]
        assert sum(roles.values()) + cohort["other_setup_control_and_bundle_body_bytes"] == total
        assert total - mlp == cohort["optimistic_remaining_if_all_mlp_stages_free_bytes"]
        assert total - mlp > cohort["tenfold_all_link_budget_bytes"]


def test_benchmark_dashboard_display_is_explicit() -> None:
    args = build_parser().parse_args(["benchmark", "run", "--show-dashboard"])
    assert args.show_dashboard is True


@pytest.mark.integration
def test_tiny_benchmark_runs_in_process_over_shared_role_topology() -> None:
    report = run_loopback_benchmark(
        model="unused",
        model_id=None,
        tiny=True,
        prompt="private",
        max_output_tokens=1,
        warmups=0,
        repetitions=1,
        timeout_seconds=120,
    )
    assert report["checks"]["passed"] is True
    assert report["summary"]["completed_runs"] == 1
    material = report["prepared_material_accounting"]
    assert material["conserved"] is True
    assert material["remaining_stage_rows"] == 0
    assert material["counts"]["issued"] > 0
    compute = report["process_cpu_accounting"]
    assert set(compute["startup_cpu_seconds_by_role"]) == {
        "client", "inference", "preparation",
    }
    assert compute["aggregate_startup_cpu_seconds"] > 0
    assert compute["aggregate_cold_first_response_cpu_seconds"] >= (
        compute["aggregate_startup_cpu_seconds"]
    )
    assert compute["full_response_compute_cap_checked"] is False
    processes = report["runs"][0]["processes"]
    assert set(processes) == {"client", "inference", "preparation"}
    assert processes["client"]["cpu_seconds"] is not None
    startup = report["topology_accounting"]["startup"]
    assert startup is not None and startup["tracked_body_counter_set_present"]
    assert startup["all_link_serialized_body_bytes"] > 0
    assert any(
        edge["source"] == "preparation" and edge["destination"] == "inference"
        and edge["serialized_body_bytes"] > 0
        for edge in startup["body_bytes_by_edge"]
    )
    topology = report["topology_accounting"]["runs"][0]
    assert topology["tracked_body_counter_set_present"] is True
    assert topology["total_wire_bytes"] is None
    assert topology["full_response_compute_cap_checked"] is False
    assert sum(edge["serialized_body_bytes"] for edge in topology["body_bytes_by_edge"]) == (
        topology["all_link_serialized_body_bytes"]
    )
    stage = report["topology_accounting"]["stages"]
    assert stage["startup"]["reconciled_with_protocol_bodies"]
    assert stage["runs"][0]["reconciled_with_protocol_bodies"]
    assert len(stage["runs"][0]["body_bytes_by_stage_and_edge"]) == 4
    for edge, total in stage["runs"][0]["expected_stage_body_bytes_by_edge"].items():
        assert sum(
            stage_links.get(edge, 0)
            for stage_links in stage["runs"][0]["body_bytes_by_stage_and_edge"].values()
        ) == total


@pytest.mark.integration
def test_request_sized_benchmark_prepares_only_offline_rows_needed_by_response() -> None:
    report = run_loopback_benchmark(
        model="unused", model_id=None, tiny=True, prompt="private",
        max_output_tokens=1, warmups=0, repetitions=2, timeout_seconds=120,
        inventory_policy="request-sized",
    )
    assert report["checks"]["passed"]
    assert report["configuration"]["inventory_policy"] == "request-sized"
    startup = report["topology_accounting"]["startup"]
    assert startup is not None and startup["all_link_serialized_body_bytes"] == 0
    for run, accounting in zip(
        report["runs"], report["topology_accounting"]["runs"], strict=True,
    ):
        assert run["inventory"]["generated"] == run["inventory"]["required"]
        assert run["inventory"]["required"] == run["tokens"]["input_tokens"]
        assert run["inventory"]["reused"] == 0
        assert run["privacy"]["preparation_requests_during_online"] == 0
        assert accounting["all_link_serialized_body_bytes"] > 0
    assert all(
        item["reconciled_with_protocol_bodies"]
        for item in report["topology_accounting"]["stages"]["runs"]
    )


@pytest.mark.integration
def test_opt_in_compressed_bundle_runs_with_prepared_roles() -> None:
    report = run_loopback_benchmark(
        model="unused", model_id=None, tiny=True, prompt="private",
        max_output_tokens=1, warmups=0, repetitions=1, timeout_seconds=120,
        inventory_policy="request-sized", bundle_compression="zlib",
    )
    assert report["checks"]["passed"]
    assert report["configuration"]["bundle_compression"] == "zlib"
    run = report["runs"][0]
    assert run["privacy"]["bundle_network_bytes"] > 0
    assert run["privacy"]["plaintext_prompt_bytes_sent"] == 0
    assert run["privacy"]["plaintext_token_ids_sent"] == 0
    assert run["privacy"]["preparation_requests_during_online"] == 0


@pytest.mark.integration
def test_repeated_prefill_benchmark_burns_only_decode_reservation() -> None:
    report = run_loopback_benchmark(
        model="unused", model_id=None, tiny=True, prompt="private",
        max_output_tokens=1, warmups=1, repetitions=1, timeout_seconds=120,
        inventory_policy="request-sized", prefill_cache_mib=64,
    )
    assert report["checks"]["passed"]
    assert report["configuration"]["prefill_cache_mib"] == 64
    cold, warm = report["warmup_runs"][0], report["runs"][0]
    assert cold["privacy"]["prefill_cache_misses"] == 1
    assert cold["privacy"]["masked_online_upload_bytes"] > 0
    assert warm["privacy"]["prefill_cache_hits"] == 1
    assert warm["privacy"]["masked_online_upload_bytes"] == 0
    assert warm["privacy"]["masked_online_download_bytes"] == 0
    assert warm["privacy"]["preparation_requests_during_online"] == 0
    assert warm["inventory"]["required"] == 1
    assert warm["tokens"]["input_tokens"] == cold["tokens"]["input_tokens"]
    assert warm["tokens"]["output_tokens"] == cold["tokens"]["output_tokens"]


@pytest.mark.integration
@pytest.mark.parametrize("combined", [False, True])
def test_two_worker_experiment_runs_through_standard_benchmark(tmp_path: Path, combined: bool) -> None:
    from pllm import Deployment, ExecutionBudget, Model
    from pllm.profiles import TwoOnlineOffsetCpu
    from pllm.protocols import ClientBundleTransport, TwoOnlineOffsetLinear
    from pllm.quantization import SymmetricPerRow
    from pllm.state import ClientPrefixReuse
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    checkpoint = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    experiment = Experiment(
        name="offset-benchmark",
        pipeline=TwoOnlineOffsetCpu(Model.path(str(checkpoint), model_id="offset-benchmark"), **({
            "linear": TwoOnlineOffsetLinear(input_encoding="seeded", output_encoding="row_residues"),
            "quantization": SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
            "delivery": ClientBundleTransport("artifacts", compression="zlib"),
            "cache": ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=64),
        } if combined else {})),
        deployment=Deployment.local(root=str(tmp_path)),
        budget=ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=3 if combined else 1),
    )
    report = run_loopback_benchmark(
        model=str(checkpoint), model_id="offset-benchmark", tiny=False,
        prompt="A", max_output_tokens=2, warmups=0,
        repetitions=1, timeout_seconds=120, experiment=experiment,
        prompt_sequence=("A" * 24, "A" * 24, "A" * 24 + " B") if combined else None, temperature=0,
    )
    assert report["checks"]["passed"]
    assert report["summary"]["completed_runs"] == (3 if combined else 1)
    assert set(report["runs"][0]["processes"]) == {"client", "worker_a", "worker_b"}
    assert report["runs"][0]["privacy"]["role_link.worker_a.online_upload_bytes"] > 0
    assert report["runs"][0]["privacy"]["role_link.worker_b.online_download_bytes"] > 0
    assert report["privacy_admission"]["independent_operators_verified"] is False
    accounting = report["topology_accounting"]["runs"][0]
    assert accounting["tracked_body_counter_set_present"]
    assert accounting["all_link_serialized_body_bytes"] > 0
    assert accounting["aggregate_run_window_cpu_seconds"] is not None
    assert accounting["full_response_compute_cap_checked"] is False
    startup = report["topology_accounting"]["startup"]
    assert startup["tracked_body_counter_set_present"]
    assert report["summary"]["total_accounted_benchmark_body_bytes"] == (
        startup["all_link_serialized_body_bytes"] + sum(
            row["all_link_serialized_body_bytes"] for row in report["topology_accounting"]["runs"]))
    assert report["communication_per_token"]["summary"]["setup_inclusive_mb_per_output_token"] > 0
    if combined:
        assert report["runs"][1]["privacy"]["prefill_cache_hits"] == 1
        assert report["runs"][2]["privacy"]["prefill_prefix_tokens_reused"] > 0


def test_two_worker_experiment_uses_existing_benchmark_cli(tmp_path: Path) -> None:
    from pllm import Deployment, ExecutionBudget, Model
    from pllm.profiles import TwoOnlineOffsetCpu
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    checkpoint = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    experiment = Experiment(
        name="offset-cli",
        pipeline=TwoOnlineOffsetCpu(Model.path(str(checkpoint), model_id="offset-cli")),
        deployment=Deployment.local(root=str(tmp_path)),
        budget=ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=1),
    )
    target = tmp_path / "offset-experiment.json"
    target.write_bytes(experiment.canonical_bytes())
    result = subprocess.run(
        [sys.executable, "-m", "pllm", "benchmark", "run", "--experiment", str(target),
         "--prompt", "A", "--max-output-tokens", "2", "--warmups", "0",
         "--repetitions", "1", "--format", "json"],
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["command"] == "benchmark.run"
    report = output["data"]["report"]
    assert report["checks"]["passed"]
    assert report["configuration"]["roles"] == ["client", "worker_a", "worker_b"]
    assert report["topology_accounting"]["runs"][0]["online_all_link_serialized_body_bytes"] > 0
    assert report["process_cpu_accounting"]["aggregate_cold_first_response_cpu_seconds"] > 0
    assert "user: A" not in result.stdout
    assert report["runs"][0]["privacy"]["plaintext_token_ids_sent"] == 0
    assert all("token_ids" not in run for run in report["runs"])


@pytest.mark.integration
def test_client_offset_prepared_topologies_share_one_w8a8_benchmark_cohort(
    tmp_path: Path,
) -> None:
    from pllm import Deployment, ExecutionBudget, Model
    from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu, TwoOnlineOffsetCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    checkpoint = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    model_id = "matched-three-topologies"
    source = Model.path(str(checkpoint), model_id=model_id)
    experiments = [
        Experiment(
            name=name,
            pipeline=kind(source, quantization=SymmetricPerRow(weight_bits=8, activation_bits=8)),
            deployment=Deployment.local(root=str(tmp_path)),
            budget=ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=1),
        )
        for name, kind in (
            ("client-owned", ClientOnlyCpu), ("two-online-offset", TwoOnlineOffsetCpu),
            ("prepared", MaskedLinearCpu),
        )
    ]
    runs = [
        (
            experiment,
            run_loopback_benchmark(
                model=str(checkpoint), model_id=model_id, tiny=False,
                prompt="A", max_output_tokens=2, warmups=0,
                repetitions=1, timeout_seconds=120, experiment=experiment,
                _cohort_salt=b"matched-ephemeral-cohort".ljust(32, b"\0"),
            ),
        )
        for experiment in experiments
    ]
    assert all(report["checks"]["passed"] for _, report in runs)
    assert len({report["runs"][0]["model_fingerprint"] for _, report in runs}) == 1
    assert len({(
        report["runs"][0]["tokens"]["input_tokens"],
        report["runs"][0]["tokens"]["output_tokens"],
    ) for _, report in runs}) == 1
    comparison = build_comparison_report(runs)
    assert comparison["checks"]["matched_workload"]
    assert comparison["comparison_key"]["model_fingerprint"]
    diagnostic = comparison["compute_cap_diagnostic"]
    assert diagnostic["reference_configuration_digest"] == experiments[1].configuration_digest()
    assert len(diagnostic["observations"]) == 3
    assert diagnostic["full_response_compute_cap_admitted"] is False
    assert runs[0][1]["topology_accounting"]["runs"][0]["online_client_serialized_body_bytes"] == 0
    ownership = runs[0][1]["topology_accounting"]["startup"]
    assert ownership["schema"] == "pllm.topology_model_ownership.v1"
    assert ownership["checkpoint_artifact_bytes"] > 0
    assert ownership["loaded_weight_and_local_tensor_bytes"] > 0
    assert ownership["cold_checkpoint_transfer_bytes"] is None
    assert ownership["client_peak_memory_bytes"] is None
    assert runs[1][1]["privacy_admission"]["independent_operators_verified"] is False
    assert runs[1][1]["topology_accounting"]["runs"][0]["online_client_serialized_body_bytes"] > 0
    assert runs[2][1]["topology_accounting"]["startup"]["all_link_serialized_body_bytes"] > 0
    assert all(
        report["process_cpu_accounting"]["aggregate_cold_first_response_cpu_seconds"] > 0
        for _, report in runs
    )


def test_cold_cpu_accounting_rejects_missing_role_or_nonmonotonic_samples() -> None:
    from pllm.runtime.topology_accounting import cold_process_cpu_accounting

    roles = ("client", "worker_a", "worker_b")
    sample = {
        "startup": {"client": 2.0, "worker_a": 4.0, "worker_b": 3.0},
        "first_response": {"client": 5.0, "worker_a": 5.0, "worker_b": 7.0},
    }
    accounted = cold_process_cpu_accounting(sample, roles=roles, first_measurement_is_cold=True)
    assert accounted["aggregate_startup_cpu_seconds"] == 9.0
    assert accounted["aggregate_cold_first_response_cpu_seconds"] == 17.0
    assert accounted["full_response_compute_cap_checked"] is False

    sample["first_response"]["worker_a"] = 3.0
    assert cold_process_cpu_accounting(
        sample, roles=roles, first_measurement_is_cold=True,
    )["aggregate_cold_first_response_cpu_seconds"] is None
    del sample["first_response"]["worker_a"]
    assert cold_process_cpu_accounting(
        sample, roles=roles, first_measurement_is_cold=True,
    )["aggregate_cold_first_response_cpu_seconds"] is None
    assert cold_process_cpu_accounting(
        sample, roles=roles, first_measurement_is_cold=False,
    )["aggregate_cold_first_response_cpu_seconds"] is None


def test_offset_cpu_diagnostic_flags_excess_without_admitting_a_compute_cap(tmp_path: Path) -> None:
    from pllm import Deployment, ExecutionBudget, Model
    from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu

    model = Model("Qwen/Qwen2.5-0.5B-Instruct")
    budget = ExecutionBudget(max_input_tokens=128, max_new_tokens=24, requests=1)
    baseline = Experiment(
        name="offset", pipeline=TwoOnlineOffsetCpu(model),
        deployment=Deployment.local(root=str(tmp_path)), budget=budget,
    )
    candidate = Experiment(
        name="verified", pipeline=MaskedLinearCpu(model),
        deployment=Deployment.local(root=str(tmp_path)), budget=budget,
    )
    offset_report, candidate_report = _report(), _report()
    offset_report["process_cpu_accounting"] = {
        "aggregate_cold_first_response_cpu_seconds": 28.0,
    }
    candidate_report["process_cpu_accounting"] = {
        "aggregate_cold_first_response_cpu_seconds": 142.0,
    }
    matched = build_comparison_report([(baseline, offset_report), (candidate, candidate_report)])
    diagnostic = matched["compute_cap_diagnostic"]
    assert diagnostic["observations"][1]["ratio_to_offset"] > 5.0
    assert diagnostic["observations"][1]["measured_cpu_not_above_offset"] is False
    assert diagnostic["full_response_compute_cap_admitted"] is False
    del candidate_report["process_cpu_accounting"]
    assert build_comparison_report([
        (baseline, offset_report), (candidate, candidate_report),
    ])["compute_cap_diagnostic"] is None


def test_prepared_link_ledger_charges_serialized_body_once_and_never_invents_wire() -> None:
    record = _record("accounted", ttft=1.0, throughput=1.0)
    privacy = record["privacy"]
    assert isinstance(privacy, dict)
    record["privacy"] = {
        **privacy,
        "session_authorization_upload_bytes": 7,
        "preparation_upload_bytes": 11,
        "session_authorization_download_bytes": 5,
        "preparation_download_bytes": 13,
        "correction_push_bytes": 17,
        "bundle_network_bytes": 19,
        "inference_upload_bytes": 23,
        "inference_download_bytes": 29,
        "masked_online_upload_bytes": 1_000,  # duplicate view, never charged twice
    }
    record["processes"] = {
        "client": {"cpu_seconds": 1.0},
        "preparation": {"cpu_seconds": 2.0},
        "inference": {"cpu_seconds": 3.0},
    }
    accounted = prepared_body_accounting(record)
    assert accounted["all_link_serialized_body_bytes"] == 124
    assert accounted["client_serialized_body_bytes"] == 107
    assert accounted["online_client_serialized_body_bytes"] == 52
    assert accounted["online_all_link_serialized_body_bytes"] == 52
    assert accounted["aggregate_run_window_cpu_seconds"] == 6.0
    assert accounted["body_bytes_by_edge"] == [
        {"source": "client", "destination": "preparation", "phase": "offline", "serialized_body_bytes": 18},
        {"source": "preparation", "destination": "client", "phase": "offline", "serialized_body_bytes": 18},
        {"source": "preparation", "destination": "inference", "phase": "offline", "serialized_body_bytes": 17},
        {"source": "inference", "destination": "client", "phase": "cold", "serialized_body_bytes": 19},
        {"source": "client", "destination": "inference", "phase": "online", "serialized_body_bytes": 23},
        {"source": "inference", "destination": "client", "phase": "online", "serialized_body_bytes": 29},
    ]
    assert accounted["total_wire_bytes"] is None
    assert accounted["full_response_compute_cap_checked"] is False


def test_link_ledger_marks_partial_counters_and_missing_cpu_unavailable() -> None:
    partial = prepared_body_accounting(_record("partial", ttft=1.0, throughput=1.0))
    assert partial["tracked_body_counter_set_present"] is False
    assert partial["all_link_serialized_body_bytes"] is None
    assert partial["body_bytes_by_edge"] is None
    assert partial["aggregate_run_window_cpu_seconds"] is None
    invalid = _record("invalid", ttft=1.0, throughput=1.0)
    counters = {key: 0 for key in (
        "session_authorization_upload_bytes", "preparation_upload_bytes",
        "session_authorization_download_bytes", "preparation_download_bytes",
        "correction_push_bytes", "bundle_network_bytes", "inference_upload_bytes",
        "inference_download_bytes",
    )}
    counters["correction_push_bytes"] = True
    invalid["privacy"] = counters
    assert prepared_body_accounting(invalid)["tracked_body_counter_set_present"] is False


class _RunningProcess:
    def poll(self) -> None:
        return None


@pytest.mark.integration
def test_guarded_benchmark_rejects_unpriced_crypto_memory() -> None:
    import pllm
    from pllm.profiles import ProprietaryGuarded
    from pllm.runtime.benchmark_cli import LoopbackBenchmarkError
    from pllm.sources import TinyModel

    experiment = pllm.Experiment(
        name="guarded-benchmark",
        pipeline=ProprietaryGuarded(TinyModel(model_id="guarded-benchmark")),
        deployment=pllm.Deployment.local(root="local://guarded-benchmark"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=1),
    )
    with pytest.raises(LoopbackBenchmarkError, match="memory estimation requires"):
        run_loopback_benchmark(
            model="unused", model_id=None, tiny=False, prompt="private",
            max_output_tokens=1, warmups=0, repetitions=1, timeout_seconds=30,
            experiment=experiment,
        )


def test_benchmark_polling_reads_nested_dashboard_run_state(monkeypatch) -> None:
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/snapshot":
            return httpx.Response(
                200,
                json={
                    "run": {
                        "phase": "ready",
                        "startup_step": "ready",
                        "run_id": "bench-fixed",
                        "active_run": None,
                    },
                    "otel": {},
                },
            )
        if request.url.path == "/api/run":
            return httpx.Response(200, json={"status": "started"})
        if request.url.path == "/api/runs/bench-fixed":
            return httpx.Response(200, json={"status": "completed"})
        raise AssertionError(request.url.path)

    monkeypatch.setattr("pllm.runtime.benchmark_cli.secrets.token_hex", lambda _: "fixed")
    with httpx.Client(
        transport=httpx.MockTransport(respond), base_url="http://dashboard"
    ) as client:
        process: Any = _RunningProcess()
        _wait_for_ready(client, process, float("inf"), None)
        record = _run_once(
            client,
            process,
            prompt="private",
            max_output_tokens=2,
            timeout_seconds=1,
        )

    assert record == {"status": "completed"}
    assert paths == [
        "/api/snapshot",
        "/api/run",
        "/api/snapshot",
        "/api/runs/bench-fixed",
    ]


def test_loopback_report_summarizes_records_and_has_no_text_payloads() -> None:
    report = build_loopback_report(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tiny=False,
        max_output_tokens=24,
        warmup_runs=[_record("warmup-1", ttft=1.5, throughput=7.0)],
        runs=[
            _record("run-1", ttft=1.0, throughput=8.0),
            _record("run-2", ttft=2.0, throughput=10.0),
        ],
    )

    assert report["schema_version"] == "pllm.loopback_benchmark.v1"
    assert report["checks"]["passed"] is True
    assert report["summary"]["median_ttft_seconds"] == 1.5
    assert report["summary"]["median_tokens_per_second"] == 9.0
    assert report["summary"]["total_output_tokens"] == 48
    serialized = json.dumps(report)
    assert '"prompt"' not in serialized
    assert '"generated_text"' not in serialized
    assert '"token_ids"' not in serialized


def test_comparison_report_ranks_only_matched_pipeline_runs() -> None:
    slow, fast = _experiment("slow", 1), _experiment("fast", 4)
    report = build_comparison_report([(slow, _report(full=5.0)), (fast, _report(full=3.0))])

    assert report["schema_version"] == "pllm.loopback_benchmark_comparison.v1"
    assert report["checks"]["matched_workload"] is True
    assert report["rankings"]["full_seconds"][0]["name"] == "fast"
    assert report["winners"]["full_seconds"] == fast.configuration_digest()

    mismatched: Any = _report(full=2.0)
    mismatched["runs"][0]["tokens"]["output_tokens"] = 12
    report = build_comparison_report([(slow, _report()), (fast, mismatched)])
    assert report["checks"]["matched_workload"] is False
    assert report["rankings"]["full_seconds"] == []
    assert report["winners"]["full_seconds"] is None

    cached: Any = _report(full=2.0)
    cached["runs"][0]["privacy"]["prefill_cache_hits"] = 1
    report = build_comparison_report([(slow, _report()), (fast, cached)])
    assert report["checks"]["matched_workload"] is True
    assert report["winners"]["full_seconds"] == fast.configuration_digest()

    cached["configuration"]["prompt_digest"] = "different prompt"
    report = build_comparison_report([(slow, _report()), (fast, cached)])
    assert report["checks"]["matched_workload"] is False
    assert report["winners"]["full_seconds"] is None


@pytest.mark.parametrize("field,value", [("provider_backend", "docker"), ("link_conditions_digest", "shaped"),
                                       ("wan_emulation", True), ("wan_emulation_digest", "consumer")])
def test_transport_conditions_cannot_enter_unmatched_ranking(field, value):
    first, second = _experiment("first", 1), _experiment("second", 4)
    control, changed = _report(), _report()
    changed["configuration"][field] = value
    report = build_comparison_report([(first, control), (second, changed)])
    assert not report["checks"]["matched_workload"]
    assert report["rankings"]["full_seconds"] == []
    control["configuration"][field] = value
    assert build_comparison_report([(first, control), (second, changed)])["checks"]["matched_workload"]


@pytest.mark.parametrize("options,enforced", [([], False), (["--wan"], True),
    (["--wan-upload-mbps", "8", "--wan-party", "client:20:4"], True),
    (["--wan-estimate", "--wan-upload-mbps", "8"], False)])
def test_wan_cli_dispatches_enforcement_explicitly(monkeypatch, capsys, options, enforced):
    from pllm.cli import main
    from pllm.runtime import benchmark_cli
    captured = {}
    def run(**kwargs):
        captured.update(kwargs)
        return _report()
    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", run)
    main(["benchmark", "run", "--tiny", "--format", "json", *options])
    assert captured["emulate_wan"] is enforced
    assert captured["wan"].download_mbps == 100
    if "--wan-party" in options:
        assert captured["wan"].access("client").upload_mbps == 4
    assert json.loads(capsys.readouterr().out)["data"]["report"]["checks"]["passed"]


def test_benchmark_command_writes_sanitized_report(monkeypatch, capsys, tmp_path: Path) -> None:
    from pllm.cli import main
    from pllm.runtime import benchmark_cli

    captured: dict[str, object] = {}

    def run(**kwargs):
        captured.update(kwargs)
        return _report()

    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", run)
    output = tmp_path / "benchmark.json"
    main(
        [
            "benchmark",
            "run",
            "--prompt",
            "private prompt",
            "--output",
            str(output),
            "--format",
            "json",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert captured["prompt"] == "private prompt"
    assert captured["show_dashboard"] is False
    assert captured["inventory_policy"] == "prewarm"
    assert callable(captured["progress"])
    assert result["command"] == "benchmark.run"
    assert report["checks"]["passed"] is True
    assert "private prompt" not in output.read_text(encoding="utf-8")


def test_benchmark_command_compares_experiments_and_saves_lowest_latency(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    from pllm.cli import main
    from pllm.runtime import benchmark_cli

    slow, fast = _experiment("slow", 1), _experiment("fast", 4)
    slow_path, fast_path = tmp_path / "slow.json", tmp_path / "fast.json"
    slow_path.write_bytes(slow.canonical_bytes())
    fast_path.write_bytes(fast.canonical_bytes())

    def run(**kwargs):
        threads = kwargs["experiment"].pipeline.components["kernels"].params["threads"]
        return _report(full=5.0 if threads == 1 else 3.0)

    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", run)
    output, best = tmp_path / "comparison.json", tmp_path / "best.json"
    main(
        [
            "benchmark",
            "run",
            "--experiment",
            str(slow_path),
            "--experiment",
            str(fast_path),
            "--output",
            str(output),
            "--save-best",
            str(best),
            "--format",
            "json",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    saved = Experiment.from_spec(json.loads(best.read_text(encoding="utf-8")))
    assert report["rankings"]["full_seconds"][0]["name"] == "fast"
    assert saved.configuration_digest() == fast.configuration_digest()
    assert "slow" not in capsys.readouterr().err

"""Read-only replay of one source-present, full-share Qwen functionality sample."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "qwen-response-v1"
CONFORMANCE = HERE / "decoder-capacity-v1"
SPEC = importlib.util.spec_from_file_location("gkm_sdk_archive", HERE / "verify.py")
SDK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SDK)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_payloads(archive):
    index = json.loads(SDK.common.read_regular(archive / "index.json"))
    rows = index["files"]
    require(1 <= len(rows) <= 256 and sum(row["bytes"] for row in rows) <= 32 << 20,
            "archive budget exceeded")
    payloads = {}
    for row in rows:
        name = row["path"]
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() == name and name not in payloads,
                "invalid archive member")
        data = SDK.common.read_regular(archive / relative, row["bytes"])
        require(hashlib.sha256(data).hexdigest() == row["sha256"], "archive member changed")
        payloads[name] = data
    actual = {path.relative_to(archive).as_posix() for path in archive.rglob("*") if path.is_file()}
    require(actual == set(payloads) | {"index.json"}, "archive manifest incomplete")
    return index, payloads


def verify_response(payloads, conformance):
    def record(name):
        return json.loads(payloads[name])

    installed = record("installed-precheck.json")
    require(installed == record("installed-postcheck.json")
            == json.loads(conformance["build/installed-postcheck.json"]),
            "installed bytes differ from conformance")
    require(record("source-lock.json") == record("source-lock-postcheck.json")
            == json.loads(conformance["qwen-smoke-v2/source-lock.json"]),
            "checkpoint source changed")
    decoder = record("native-contract.json")
    runtime = record("runtime-contract.json")
    original_runtime = json.loads(conformance["qwen-smoke-v2/runtime-contract.json"])
    require(decoder == json.loads(conformance["qwen-smoke-v2/native-contract.json"])
            and runtime["decoder"] == decoder
            and {k: v for k, v in runtime.items() if k != "configuration"}
            == {k: v for k, v in original_runtime.items() if k != "configuration"},
            "native source or numerical contract changed")
    inputs = record("inputs.json")
    require(hashlib.sha256(payloads["runner.py"]).hexdigest() == inputs["script_sha256"],
            "response runner changed")
    require(inputs["source_revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
            and inputs["source_repository"] == "Qwen/Qwen2.5-0.5B-Instruct"
            and inputs["public_prompt"] == "Hello."
            and inputs["max_input_tokens"] == decoder["input_rows"] == 64
            and inputs["max_output_tokens"] == decoder["decode_steps"] + 1 == 2,
            "source or workload differs")
    capture = record("capture.json")
    require(hashlib.sha256(payloads["archive.py"]).hexdigest() == capture["script_sha256"]
            and capture["python"] == inputs["python"] and capture["live_worker_pids"] == [],
            "post-run capture differs")
    require(record("execution.log") == {
        "status": "completed", "output": str(Path(capture["original_directory"]) / "benchmark.json"),
    }, "execution log differs")

    document = record("benchmark.json")
    require(document["complete"] is True and len(document["results"]) == 1,
            "response sample incomplete")
    result, = document["results"]
    execution = result["execution"]
    details = execution["details"]
    report = details["report"]
    require(result["status"] == "completed" and execution["failure_code"] is None
            and len(report["runs"]) == 1 and not report["warmup_runs"]
            and result["repetitions"] == 1 and result["warmups"] == 0
            and execution["cohort"] == document["cohort"], "response sample incomplete")
    experiment = record("experiment.json")
    require(execution["case"]["target"]["parameters"]["model"]["experiment"] == experiment
            and experiment["pipeline"]["components"]["linear"]["params"]["capacity"] == "decoder"
            and report["experiment"]["pipeline_digest"] == runtime["composition"],
            "response experiment differs")
    build = execution["build"]["attributes"]
    expected_sources = {
        name.removeprefix("pllm/"): row["sha256"]
        for name, row in installed["files"].items() if name.endswith((".py", ".json"))
    }
    require(build["sdk_sources"] == expected_sources
            and build["native_sha256"] == installed["files"]["pllm/_native.abi3.so"]["sha256"],
            "executed build differs from installed wheel")

    def admission(value):
        estimate = value["estimate"]
        host = value["host"]
        candidate = value["candidates"]["native"]
        required = estimate["client_peak_bytes"] + sum(estimate["provider_peak_bytes"].values())
        require(set(estimate["provider_peak_bytes"]) == {"worker_a", "worker_b"}
                and value["admitted"] is True and value["selected_backend"] == "native"
                and candidate["admitted"] is True and not candidate["reasons"]
                and required == estimate["native_total_peak_bytes"] == candidate["required_host_bytes"]
                and host["admission_budget_bytes"] == max(0, host["available"] - host["reserve_bytes"])
                and 0 < required <= host["admission_budget_bytes"], "host admission differs")
        return estimate

    frozen_estimate = admission(record("preflight.json"))
    estimate = admission(details["topology_admission"])
    # The ordinary response target redirects deployment.root to its private
    # directory. Only the configuration-bound runtime digest may change here.
    for key, value in frozen_estimate.items():
        if key == "gkm_admission":
            require({k: v for k, v in value.items() if k != "runtime_contract_digest"}
                    == {k: v for k, v in estimate[key].items() if k != "runtime_contract_digest"},
                    "execution source admission differs")
        else:
            require(estimate[key] == value, "execution admission differs")
    require(set(estimate) == set(frozen_estimate), "execution admission differs")
    require(admission(report["memory_preflight"]) == estimate, "report admission differs")
    contract = details["runtime_contract"]
    require(contract == report["runtime_contract"]
            and contract["source_lock_digest"] == runtime["source_lock"]
            and contract["tokenizer_digest"] == runtime["tokenizer"]
            and contract["checkpoint_digest"] == record("source-lock.json")["checkpoint_digest"]
            and contract["native_runtime_contract_digest"] == estimate["gkm_admission"]["runtime_contract_digest"]
            and contract["verification_window"] == "request_end"
            and contract["output_delivery"] == "buffered_until_verified"
            and contract["sharing"] == "full" and contract["generation_policy"] == "fixed_length_greedy"
            and contract["vocabulary"] == 151936 and contract["width"] == 896,
            "executed runtime contract differs")

    run, = report["runs"]
    required_checks = {
        "passed", "all_runs_completed", "authoritative_token_usage", "fixed_count_verified_delivery",
        "memory_guard_passed", "model_fingerprint_present", "no_online_preparation_requests",
        "no_plaintext_prompt_bytes_sent", "no_plaintext_token_ids_sent", "sampling_matches_request",
        "two_worker_bodies_tracked",
    }
    require(set(report["checks"]) == required_checks
            and all(value is True for value in report["checks"].values())
            and run["status"] == "completed" and run["failure"] is None and run["cold"] is True
            and run["tokens"] == {"authoritative": True, "input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            "verified response checks failed")
    require(execution["oracle"] == {"kind": "ordinary_role_runtime_checks", "outcome": "passed", "checked_values": 2}
            and run["generation"]["response_status"] == "incomplete"
            and report["privacy_admission"]["independent_operators_verified"] is False,
            "functionality sample promoted beyond its oracle")
    guard = report["memory_guard"]
    require(guard["tripped"] is False and guard["maximum_host_swap_growth_bytes"] == 0,
            "host pressure check failed")
    memory = report["native_process_memory"]
    require(memory["complete"] is True and memory["samples"] > 0
            and memory["sampled_total_peak_rss_bytes"] > 0, "simultaneous memory incomplete")

    metrics = {row["id"]: row for row in result["metrics"]}
    require(len(metrics) == len(result["metrics"]), "duplicate metric")

    def metric(name, value, unit, window):
        row = metrics[name]
        require(isinstance(row["value"], (int, float)) and math.isfinite(row["value"])
                and math.isclose(row["value"], value, rel_tol=1e-12, abs_tol=1e-9)
                and row["unit"] == unit and row["window"] == window
                and row["origin"] == "measured" and row["sample_count"] == 1,
                "metric differs: " + name)

    client_peak = report["client_process_memory"]["after"]["lifetime_peak_rss_bytes"]
    metric("client.peak_rss", client_peak, "bytes", "process_birth_through_scenario")
    metric("process.peak_rss", client_peak, "bytes", "process_birth_through_target")
    metric("request.0.full", run["durations"]["full_seconds"], "seconds", "request_setup_through_completion")
    metric("response.ttft", run["durations"]["ttft_seconds"], "seconds", "online_to_first_accepted_text")
    metric("target.latency", details["timings"]["wall_seconds"][0], "seconds", "execution")
    for role in ("worker_a", "worker_b"):
        metric(role + ".peak_rss", run["processes"][role]["rss_peak_bytes"], "bytes", "role_birth_through_scenario")
    cpu = report["process_cpu_accounting"]
    require(set(cpu["cold_first_response_cpu_seconds_by_role"]) == {"client", "worker_a", "worker_b"}
            and math.isclose(sum(cpu["cold_first_response_cpu_seconds_by_role"].values()),
                             cpu["aggregate_cold_first_response_cpu_seconds"], rel_tol=1e-12),
            "role CPU accounting differs")
    for role, value in cpu["cold_first_response_cpu_seconds_by_role"].items():
        metric(role + ".cold_cpu", value, "seconds", "benchmark_start_through_first_response"
               if role == "client" else "role_birth_through_first_response")
    for name in ("response.decode_tps", "request.0.decode_tps"):
        require(metrics[name]["value"] is None and metrics[name]["origin"] == "not_available"
                and metrics[name]["unavailable_reason"] == "request_end_verification_buffers_all_output",
                "buffered emission promoted to decode throughput")
    require(contract["decode_tps_supported"] is False
            and run["measurement"]["decode_tps_supported"] is False
            and run["durations"]["tokens_per_second"] is None
            and metrics["network.all_link_stream_bytes"]["value"] is None
            and report["wan_readiness"]["emulation"] is None, "measurement scope changed")

    ledger, = report["topology_accounting"]["runs"]
    bodies = ledger["all_link_serialized_body_bytes"]
    online = ledger["online_all_link_serialized_body_bytes"]
    require(sum(row["serialized_body_bytes"] for row in ledger["body_bytes_by_edge"]) == bodies
            and sum(row["serialized_body_bytes"] for row in ledger["body_bytes_by_edge"] if row["phase"] == "online") == online
            and report["summary"]["total_accounted_benchmark_body_bytes"] == bodies
            and ledger["total_wire_bytes"] is None, "directed bodies do not reconcile")
    audit = run["privacy"]
    require(audit["generation_prefill_online_body_bytes"] + audit["generation_decode_online_body_bytes"] == online
            and audit["generation_completed_responses"] == 1 and audit["generation_output_tokens"] == 2
            and audit["generation_executed_decode_steps"] == 1
            and audit["inference_stage_calls"] == 2 * audit["online_steps"] == 974
            and audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0,
            "runtime accounting differs")
    for edge in ledger["body_bytes_by_edge"]:
        if edge["phase"] == "bundle":
            require(edge["serialized_body_bytes"] == 0, "unexpected client bundle")
            continue
        upload = edge["source"] == "client"
        role = edge["destination"] if upload else edge["source"]
        key = f"role_link.{role}.{edge['phase']}_{'upload' if upload else 'download'}_bytes"
        require(edge["serialized_body_bytes"] == audit[key], "directed role counter differs")
    metric("network.covered_body_bytes", bodies, "bytes", "setup_through_scenario")
    return {
        "status": "completed", "input_tokens": 4, "output_tokens": 2,
        "installed_files": len(installed["files"]), "wheel_sha256": installed["wheel_sha256"],
        "request_seconds": run["durations"]["full_seconds"],
        "target_seconds": details["timings"]["wall_seconds"][0],
        "client_peak_rss_bytes": client_peak,
        "worker_peak_rss_bytes": {role: run["processes"][role]["rss_peak_bytes"] for role in ("worker_a", "worker_b")},
        "sampled_simultaneous_total_peak_rss_bytes": memory["sampled_total_peak_rss_bytes"],
        "cold_first_aggregate_cpu_seconds": cpu["aggregate_cold_first_response_cpu_seconds"],
        "covered_all_link_body_bytes": bodies, "online_all_link_body_bytes": online,
        "maximum_observed_host_swap_growth_bytes": 0, "decode_tps": None,
        "full_logits_kv_oracle": False, "quality_or_comparative_result": False,
        "scope": "one source-present local full-share functionality sample; runtime acceptance only",
    }


def verify(archive=ARCHIVE):
    index, payloads = load_payloads(archive)
    require(index["schema"] == "pllm.gkm_qwen_response_archive.v1"
            and index["conformance_archive"] == CONFORMANCE.name
            and hashlib.sha256(SDK.common.read_regular(CONFORMANCE / "index.json")).hexdigest()
            == index["conformance_index_sha256"], "conformance archive changed")
    SDK.verify(CONFORMANCE)
    _, conformance = load_payloads(CONFORMANCE)
    return {"verified_files": len(payloads), **verify_response(payloads, conformance)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))

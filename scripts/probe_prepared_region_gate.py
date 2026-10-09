"""Screen one-evaluator gated operators against historical prepared body budgets.

Only bounded SDK operators execute. Compiler shapes price a hypothetical fresh
one-use circuit for each MLP element; no protected decoder or role transport is
created. The retained W8A8 control supplies a budget, not numeric equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pllm import Model, lower_model
from pllm.metrics import OperatorBenchmarkReport, benchmark_operator
from pllm.nonlinear import CompactGatedMultiplyQ12, LogRowGatedMultiplyQ9
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.shared_resources import resident_mlp_resource_gate

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "docs/evidence/latent-response-network-qwen25-2026-09-28.json"
METHODS = {
    "logrow-silu-gated-q18-q9": "pllm/logrow-gated-multiply-q9/v1",
    "compact-silu-gated-q18-q12": "pllm/compact-gated-multiply-q12/v1",
}
CHOICES = ("logrow", "compact-quadratic", "compact-cubic")


def operator(choice: str):
    if choice == "logrow":
        return LogRowGatedMultiplyQ9()
    if choice == "compact-quadratic":
        return CompactGatedMultiplyQ12(max_pieces=1, degree=2)
    if choice == "compact-cubic":
        return CompactGatedMultiplyQ12(max_pieces=8, degree=3)
    raise ValueError("unknown bounded operator")


def project(plan, composition, report: OperatorBenchmarkReport, *, output_tokens: int,
            covered_body_budget: int) -> dict:
    """Project serial scalar material without admitting a new numeric contract."""
    if not isinstance(report, OperatorBenchmarkReport):
        raise TypeError("projection needs an immutable SDK operator report")
    if type(covered_body_budget) is not int or not 0 < covered_body_budget <= 1 << 40:
        raise ValueError("covered-body budget must be a bounded positive integer")
    config = report.configuration
    resources = report.resources
    spec = config.get("component", {})
    if (config.get("schema") != "pllm.operator_benchmark.v1"
            or type(config.get("lanes")) is not int or config["lanes"] != 1
            or report.method not in METHODS
            or spec.get("component") != METHODS[report.method]
            or spec.get("params", {}).get("backend", "half_gates") != "half_gates"
            or resources.role_names != ("garbler_client", "evaluator")
            or resources.peer_frame_bytes != 0):
        raise ValueError("screen requires a serial one-lane LogRow or Compact half-gates report")
    sample_count = config.get("samples")
    if (type(sample_count) is not int or not 1 <= sample_count <= 16
            or len(report.samples) != sample_count
            or report.quality.checked_outputs != sample_count
            or report.quality.exact_integer_outputs != sample_count):
        raise ValueError("operator report lacks complete integer parity")
    expected_digest = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if expected_digest != report.configuration_digest:
        raise ValueError("operator configuration digest differs")
    parts = resources.key_payload_bytes_by_operation
    if (not parts or any(type(value) is not int or value < 0
                         for row in parts for value in row[1:])
            or tuple(sum(row[i] for row in parts) for i in (1, 2))
            != resources.raw_key_payload_bytes_per_party
            or any(type(value) is not int or value <= 0
                    for value in resources.raw_key_payload_bytes_per_party)):
        raise ValueError("operator material accounting does not conserve role ownership")
    if any(type(value) is not int or value < 0 for value in (
        resources.client_entry_exit_payload_estimate_bytes,
        resources.public_metadata_bytes_estimate,
        resources.public_profile_bytes_estimate,
    )):
        raise ValueError("operator boundary and public metadata estimates are incomplete")
    # Reuse the existing native-checked, model-neutral geometry/coverage gate.
    # None of its FSS/Beaver cost fields contributes to this different layout.
    geometry = resident_mlp_resource_gate(
        plan, composition, response_new_tokens=output_tokens, fixed_scale_bits=7,
        maximum_material_bytes_per_party=1 << 40,
        maximum_online_all_link_body_bytes=1 << 40)
    elements = geometry["prefill_elements"] + geometry["decode_elements"]
    client, evaluator = (elements * value for value in resources.raw_key_payload_bytes_per_party)
    budgets = {"control": covered_body_budget, "10x": covered_body_budget // 10}

    def delivery(edges):
        floor = sum(edge["payload_bytes"] for edge in edges)
        return {
            "hypothetical_material_links": edges,
            "required_material_payload_bytes": floor,
            "budget_decisions": {
                name: "veto_material_alone" if floor > limit else "inconclusive_missing_work"
                for name, limit in budgets.items()
            },
            "complete_cost_known": False,
            "executable": False,
        }

    return {
        "plan_digest": geometry["plan_digest"],
        "schedule_digest": geometry["schedule_digest"],
        "composition_digest": geometry["composition_digest"],
        "operator_configuration_digest": report.configuration_digest,
        "operator_numeric_digest": report.numeric_contract_digest,
        "prefill_elements": geometry["prefill_elements"],
        "decode_elements": geometry["decode_elements"],
        "gated_elements": elements,
        "output_tokens": output_tokens,
        "covered_body_budgets_bytes": budgets,
        "client_owned_encoding_bytes": client,
        "evaluator_owned_material_bytes": evaluator,
        "online_entry_exit_payload_estimate_bytes":
            elements * resources.client_entry_exit_payload_estimate_bytes,
        "public_metadata_bytes_estimate": elements * resources.public_metadata_bytes_estimate,
        "public_profile_bytes_estimate_once": resources.public_profile_bytes_estimate,
        "material_by_operation_and_owner": [
            [name, elements * left, elements * right] for name, left, right in parts],
        "client_issued_hypothesis": delivery([
            {"source": "client", "destination": "inference", "payload_bytes": evaluator}]),
        "offline_preparation_issued_hypothesis": delivery([
            {"source": "preparation", "destination": "inference", "payload_bytes": evaluator},
            {"source": "preparation", "destination": "client", "payload_bytes": client}]),
        "one_and_per_element_sensitivity": {
            "assumption": "one fresh uncompressed half-gate AND; two 16-byte ciphertexts per element",
            "payload_bytes": 32 * elements,
            "exceeds_10x_budget": 32 * elements > budgets["10x"],
            "universal_protocol_lower_bound": False,
        },
        "unpriced_requirements": [
            "protected dynamic quantization, private scales and exact float conversion",
            "linear-to-label and label-to-linear composition without client reconstruction",
            "attention, normalization, KV, head, framing and checkpoint distribution",
            "offline garbler separation, authenticated role transport and one-use lifecycle",
            "whole-client and Preparation CPU/RSS, whole-response TTFT and decode throughput",
        ],
        "same_numeric_identity_as_control": False,
        "checkpoint_quality_measured": False,
        "offline_preparation_implemented": False,
        "whole_decoder_executable": False,
        "full_wire_bytes": None,
    }


def run(samples: int, *, choice: str | None = None) -> dict:
    if choice is None:
        workers = []
        for selected in CHOICES:
            worker = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", selected,
                 "--samples", str(samples)], check=True, capture_output=True, text=True, timeout=180)
            workers.append(json.loads(worker.stdout))
        result = workers[0]
        for worker in workers[1:]:
            for key in ("source", "config_sha256", "control_sha256", "code_sha256"):
                if worker[key] != result[key]:
                    raise ValueError("source or implementation changed between fresh processes")
            result["cases"].extend(worker["cases"])
            result["checked_outputs"] += worker["checked_outputs"]
        result["fresh_operator_processes"] = len(workers)
        return result

    from huggingface_hub import hf_hub_download

    control_bytes = CONTROL.read_bytes()
    control = json.loads(control_bytes)
    source = control["source"]
    cache = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    path = Path(hf_hub_download(source["model"], "config.json", revision=source["revision"],
                               cache_dir=cache, local_files_only=True))
    config_bytes = path.read_bytes()
    config = json.loads(config_bytes)
    composition = MaskedLinearCpu(
        Model.hf(source["model"], revision=source["revision"]),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
    report = benchmark_operator(operator(choice), lanes=1, samples=samples)
    projections = []
    for cohort in control["cohorts"]:
        outputs = cohort["output_tokens"]
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=outputs)
        projected = project(plan, composition, report, output_tokens=outputs,
                            covered_body_budget=cohort["prepared_control"]["covered_all_link_body_bytes"])
        if (projected["plan_digest"] != cohort["official_plan_digest"]
                or projected["schedule_digest"] != cohort["official_schedule_digest"]
                or projected["composition_digest"] != source["pipeline_digest"]):
            raise ValueError("compiler geometry differs from historical control")
        projections.append(projected)
    return {
        "schema": "pllm.prepared_region_gate.v1",
        "scope": "fresh bounded operators; hypothetical one-evaluator material; historical body budgets",
        "source": source,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "control": str(CONTROL.relative_to(ROOT)),
        "control_sha256": hashlib.sha256(control_bytes).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fresh_operator_processes": 1,
        "checked_outputs": report.quality.checked_outputs,
        "whole_decoder_executable": False,
        "cases": [{"choice": choice, "operator_report": report.to_dict(), "projections": projections}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=CHOICES, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, choices=range(1, 17), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run(args.samples, choice=args.worker), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required")
    result = run(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}; protected decoder remains unadmitted")


if __name__ == "__main__":
    main()

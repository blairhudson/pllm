"""Measure a pinned smaller pretrained decoder under the ordinary prepared roles.

The model/prompt is a separate quality cohort from the Qwen control; matching
input/output counts and a working stage protocol do not establish utility or
full-wire savings. No prompt, token IDs, or response text is saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from pllm import Deployment, ExecutionBudget, Experiment, Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.benchmark_cli import run_loopback_benchmark
from pllm.runtime.semantic_stages import scheduled_stage_specs

_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
_INPUT = "Explain why neither server sees the prompt."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-quality", action="store_true")
    args = parser.parse_args()
    source = Path(
        hf_hub_download(
            _MODEL,
            "config.json",
            revision=_REVISION,
            local_files_only=True,
        )
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=32)
    experiment = Experiment(
        name="smollm-semantic-candidate-control",
        pipeline=MaskedLinearCpu(
            Model.hf(_MODEL, revision=_REVISION),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        deployment=Deployment.local(root="local://smollm-prepared-control"),
        budget=ExecutionBudget(requests=2, max_input_tokens=128, max_new_tokens=32),
    )
    report = run_loopback_benchmark(
        model=_MODEL,
        model_id=None,
        tiny=False,
        prompt=_INPUT,
        max_output_tokens=32,
        warmups=1,
        repetitions=1,
        timeout_seconds=600,
        experiment=experiment,
        inventory_policy="request-sized",
    )
    run = report["runs"][0]
    ledger = report["topology_accounting"]["runs"][0]
    stages = report["topology_accounting"]["stages"]["runs"][0]
    remote = {
        stage.id
        for stage in scheduled_stage_specs(plan, experiment.pipeline)
        if stage.role not in {"token_lookup", "lm_head"}
    }
    if (
        run["status"] != "completed"
        or run["tokens"]["input_tokens"] != 39
        or run["tokens"]["output_tokens"] != 32
        or run["privacy"]["plaintext_prompt_bytes_sent"] != 0
        or run["privacy"]["plaintext_token_ids_sent"] != 0
        or not stages["reconciled_with_protocol_bodies"]
        or set(stages["body_bytes_by_stage_and_edge"]) != remote
    ):
        raise RuntimeError("real prepared candidate did not satisfy its protocol cohort")
    warmup = report["topology_accounting"]["warmups"][0]
    warmup_bundle_bytes = (
        next(
            (
                edge["serialized_body_bytes"]
                for edge in warmup["body_bytes_by_edge"]
                if edge["phase"] == "cold" and edge["source"] == "inference"
            ),
            None,
        )
        if warmup["body_bytes_by_edge"] is not None
        else None
    )
    result = {
        "schema": "pllm.prepared_candidate_response_cost.v1",
        "source": f"{_MODEL}@{_REVISION}",
        "plan_digest": plan.digest,
        "schedule_digest": plan.runtime_schedule(experiment.pipeline).digest,
        "body_fingerprint": run["model_fingerprint"],
        "input_tokens": 39,
        "output_tokens": 32,
        "remote_stages": len(remote),
        "stage_body_bytes_reconciled": True,
        "attributed_stage_body_bytes": sum(
            sum(edges.values()) for edges in stages["body_bytes_by_stage_and_edge"].values()
        ),
        "covered_all_link_body_bytes": ledger["all_link_serialized_body_bytes"],
        "covered_online_body_bytes": ledger["online_all_link_serialized_body_bytes"],
        "warmup_bundle_delivery_body_bytes_this_run": warmup_bundle_bytes,
        "cold_checkpoint_distribution_bytes": None,
        "total_wire_bytes": None,
        "representative_generation_quality_established": False,
    }
    if args.reference_quality:
        from pllm.runtime.reference_benchmark import run_reference_benchmark

        public_prompts = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "examples/benchmarks/selective_subspace_confirmation.json"
            ).read_text(encoding="utf-8")
        )
        quality = Experiment(
            name="smollm-public-reference-quality",
            pipeline=experiment.pipeline,
            deployment=experiment.deployment,
            budget=ExecutionBudget(requests=12, max_input_tokens=128, max_new_tokens=32),
        )
        reference = run_reference_benchmark([quality], public_prompts)
        scores = reference["candidates"][0]
        result["reference_quality"] = {
            "source_lock_digest": reference["model"]["source_lock_digest"],
            "dataset_digest": reference["cohort"]["dataset_digest"],
            "token_cohort_digest": reference["cohort"]["token_cohort_digest"],
            "sample_count": scores["sample_count"],
            "top1_agreement": scores["top1_agreement"],
            "top5_recall": scores["top_k_recall"],
            "max_abs_logit_error": scores["max_abs_logit_error"],
            "scope": reference["scope"],
        }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

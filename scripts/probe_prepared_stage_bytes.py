"""Attribute one pinned prepared Qwen response to compiled semantic stage roles.

Run with --max-output-tokens 8 or 32. This starts the ordinary two-child
benchmark, uses request-sized inventory, and reconciles each recorded stage
against the protocol-body audit. No prompts, tokens, or activation data are
saved; cold checkpoint distribution and complete wire are not measured.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from pllm import Experiment, lower_model
from pllm._cli.targets import resolve_target
from pllm.runtime.benchmark_cli import run_loopback_benchmark
from pllm.runtime.semantic_stages import scheduled_stage_specs

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_PROMPT = "Explain why neither server can see the prompt."
_CONTROL = {8: 116_843_966, 32: 178_970_558}


def stage_roles(
    output_tokens: int,
    experiment: Experiment,
) -> tuple[str, str, dict[str, str]]:
    source = Path(
        hf_hub_download(
            _MODEL,
            "config.json",
            revision=_REVISION,
            local_files_only=True,
        )
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=output_tokens)
    schedule = plan.runtime_schedule(experiment.pipeline)
    if not schedule.complete or schedule.protected_execution:
        raise ValueError("selected native decoder is not the complete public baseline")
    roles = {stage.id: stage.role for stage in scheduled_stage_specs(plan, experiment.pipeline)}
    return plan.digest, schedule.digest, roles


def summarize(
    report: dict[str, Any],
    output_tokens: int,
    experiment: Experiment,
) -> dict[str, Any]:
    plan_digest, schedule_digest, roles = stage_roles(output_tokens, experiment)
    record = report["runs"][0]
    if (
        record["tokens"]["input_tokens"] != 39
        or record["tokens"]["output_tokens"] != output_tokens
        or record["status"] != "completed"
        or record["privacy"]["plaintext_prompt_bytes_sent"] != 0
        or record["privacy"]["plaintext_token_ids_sent"] != 0
    ):
        raise RuntimeError("pinned public response did not complete its privacy/token cohort")
    stage = report["topology_accounting"]["stages"]["runs"][0]
    if not stage["reconciled_with_protocol_bodies"]:
        raise RuntimeError("one or more role telemetry samples were not reconciled")
    rows = stage["body_bytes_by_stage_and_edge"]
    expected_remote = {
        stage_id
        for stage_id, role in roles.items()
        if role
        not in {
            "lm_head",
            "token_lookup",
        }
    }
    if set(rows) != expected_remote:
        raise RuntimeError("compiled remote stages differ from measured body stages")
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for stage_id, links in rows.items():
        for edge, amount in links.items():
            groups[roles[stage_id]][edge] += amount
    ledger = report["topology_accounting"]["runs"][0]
    all_body_bytes = ledger["all_link_serialized_body_bytes"]
    attributed_bytes = sum(sum(links.values()) for links in rows.values())
    if all_body_bytes != _CONTROL[output_tokens] or attributed_bytes > all_body_bytes:
        raise RuntimeError("body or cohort fingerprint differs from locked prepared control")
    return {
        "schema": "pllm.prepared_compiled_stage_body_gate.v1",
        "plan_digest": plan_digest,
        "schedule_digest": schedule_digest,
        "body_fingerprint": record["model_fingerprint"],
        "input_tokens": 39,
        "output_tokens": output_tokens,
        "covered_all_link_body_bytes": all_body_bytes,
        "covered_online_body_bytes": ledger["online_all_link_serialized_body_bytes"],
        "attributed_stage_body_bytes": attributed_bytes,
        "other_setup_control_and_bundle_body_bytes": all_body_bytes - attributed_bytes,
        "reconciled_with_protocol_bodies": True,
        "remote_stage_count": len(rows),
        "body_bytes_by_compiled_role_and_edge": {
            role: dict(sorted(edges.items())) for role, edges in sorted(groups.items())
        },
        "tenfold_all_link_budget_bytes": all_body_bytes // 10,
        "hundredfold_all_link_budget_bytes": all_body_bytes // 100,
        "total_wire_bytes": None,
        "cold_checkpoint_distribution_bytes": None,
        "operator_independence_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-output-tokens", type=int, choices=(8, 32), default=32)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "examples/benchmarks/latent_response_cohort.py"
    prepared = resolve_target(
        f"{target}:prepared",
        factory=False,
        no_input=True,
        trust_python=True,
        output_format="json",
    ).configuration
    report = run_loopback_benchmark(
        model=_MODEL,
        model_id=None,
        tiny=False,
        prompt=_PROMPT,
        max_output_tokens=args.max_output_tokens,
        warmups=1,
        repetitions=1,
        timeout_seconds=600,
        experiment=prepared,
        inventory_policy="request-sized",
    )
    print(json.dumps(summarize(report, args.max_output_tokens, prepared), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

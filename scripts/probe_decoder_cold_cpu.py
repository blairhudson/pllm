"""Cold client-bundle-cache CPU/body diagnostic for pinned decoder role graphs.

The source snapshots already reside in the shared Hugging Face cache; this
diagnostic cannot count acquiring them, complete wire framing or GPU work.
Runs one response without warmups under a fresh disposable client bundle cache.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.benchmark_cli import run_loopback_benchmark

_SOURCE = {
    "qwen": (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        "Explain why neither server can see the prompt.",
        113_545_024,
    ),
    "smol": (
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "Explain why neither server sees the prompt.",
        55_676_860,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(_SOURCE))
    parser.add_argument("--offset", action="store_true", help="two-worker Qwen CPU comparator")
    args = parser.parse_args()
    if args.offset and args.model != "qwen":
        parser.error("only the pinned Qwen two-worker offset comparator is available")
    model, revision, prompt, matched_online = _SOURCE[args.model]
    checkpoint = Path(
        hf_hub_download(
            model,
            "config.json",
            revision=revision,
            local_files_only=True,
        )
    )
    if not checkpoint.is_file():
        raise ValueError("missing pinned source configuration")
    profile = TwoOnlineOffsetCpu if args.offset else MaskedLinearCpu
    experiment = Experiment(
        name=f"{args.model}-{'offset' if args.offset else 'prepared'}-cold-control",
        pipeline=profile(
            Model.hf(model, revision=revision),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        deployment=Deployment.local(
            root=f"local://{args.model}-{'offset' if args.offset else 'prepared'}-cold-control"
        ),
        budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=32),
    )
    original_xdg = os.environ.get("XDG_CACHE_HOME")
    original_hf = os.environ.get("HF_HUB_CACHE")
    with tempfile.TemporaryDirectory(
        prefix="pllm-cold-control-",
        dir=os.environ.get("PLLM_PROBE_TMPDIR"),
    ) as cache:
        # Keep the pinned shared Hub snapshot while isolating the client bundle.
        os.environ["HF_HUB_CACHE"] = str(checkpoint.parents[3])
        os.environ["XDG_CACHE_HOME"] = cache
        try:
            report = run_loopback_benchmark(
                model=model,
                model_id=None,
                tiny=False,
                prompt=prompt,
                max_output_tokens=32,
                warmups=0,
                repetitions=1,
                timeout_seconds=600,
                experiment=experiment,
                inventory_policy="request-sized",
            )
        finally:
            if original_xdg is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = original_xdg
            if original_hf is None:
                os.environ.pop("HF_HUB_CACHE", None)
            else:
                os.environ["HF_HUB_CACHE"] = original_hf
    run = report["runs"][0]
    ledger = report["topology_accounting"]["runs"][0]
    stage = report["topology_accounting"]["stages"]["runs"][0] if not args.offset else None
    cpu = report["process_cpu_accounting"]
    if (
        run["status"] != "completed"
        or (run["tokens"]["input_tokens"], run["tokens"]["output_tokens"]) != (39, 32)
        or (not args.offset and ledger["online_all_link_serialized_body_bytes"] != matched_online)
        or (stage is not None and not stage["reconciled_with_protocol_bodies"])
        or run["privacy"]["plaintext_prompt_bytes_sent"] != 0
        or run["privacy"]["plaintext_token_ids_sent"] != 0
        or cpu["aggregate_cold_first_response_cpu_seconds"] is None
    ):
        raise RuntimeError("cold cohort did not satisfy the matched-control admission")
    cold_bundle_bytes = (
        run["privacy"]["bundle_network_bytes"]
        if args.offset
        else next(
            edge["serialized_body_bytes"]
            for edge in ledger["body_bytes_by_edge"]
            if edge["phase"] == "cold" and edge["source"] == "inference"
        )
    )
    print(
        json.dumps(
            {
                "schema": "pllm.decoder_cold_cpu_body_gate.v1",
                "model": f"{model}@{revision}",
                "topology": "two_online_offset" if args.offset else "prepared",
                "body_fingerprint": run["model_fingerprint"],
                "input_tokens": 39,
                "output_tokens": 32,
                "covered_online_body_bytes": ledger["online_all_link_serialized_body_bytes"],
                "covered_cold_all_link_body_bytes": (
                    None if args.offset else ledger["all_link_serialized_body_bytes"]
                ),
                "cold_client_bundle_delivery_body_bytes": cold_bundle_bytes,
                "stage_body_bytes_reconciled": stage is not None,
                "cold_cpu_seconds_by_role": cpu["cold_first_response_cpu_seconds_by_role"],
                "aggregate_cold_cpu_seconds": cpu["aggregate_cold_first_response_cpu_seconds"],
                "upstream_source_download_cpu_and_bytes": None,
                "full_wire_bytes": None,
                "accelerator_energy_and_memory": None,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

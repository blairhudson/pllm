"""Cost-veto two protected region placements against a pinned Qwen decoder.

Requires the official pinned config.json in the shared Hugging Face cache.
Reports compiler-bound public shapes and unpriced protocol work, no input or
token values. It neither issues material nor executes a protected decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.region_contract_cost import compiler_region_contract_cost

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_CONTROL = _ROOT / "docs/evidence/latent-response-network-qwen25-2026-09-28.json"
_SHARED_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or str(
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
)


def run(output_tokens: int) -> dict:
    if type(output_tokens) is not int or output_tokens not in (8, 32):
        raise ValueError("the pinned cohorts have only 8 or 32 output tokens")
    source = Path(
        hf_hub_download(
            _MODEL,
            "config.json",
            revision=_REVISION,
            local_files_only=True,
            cache_dir=_SHARED_HUB_CACHE,
        )
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=output_tokens)
    composition = MaskedLinearCpu(
        Model.hf(_MODEL, revision=_REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    evidence = json.loads(_CONTROL.read_text(encoding="utf-8"))
    cohort = next(row for row in evidence["cohorts"] if row["output_tokens"] == output_tokens)
    if plan.digest != cohort["official_plan_digest"]:
        raise ValueError("pinned semantic plan differs from the matched control")
    control = cohort["prepared_control"]
    report = compiler_region_contract_cost(
        plan,
        composition,
        response_new_tokens=output_tokens,
        maximum_online_all_link_body_bytes=control["tenfold_online_budget_bytes"],
        maximum_total_all_link_body_bytes=control["tenfold_covered_budget_bytes"],
    )
    if (
        report["schedule_digest"] != cohort["official_schedule_digest"]
        or report["composition_digest"] != evidence["source"]["pipeline_digest"]
    ):
        raise ValueError("pinned composition or runtime schedule differs from the control")
    report["source"] = f"{_MODEL}@{_REVISION}"
    report["body_fingerprint_from_control"] = evidence["source"]["body_fingerprint"]
    report["matched_prepared_covered_all_link_body_bytes"] = control["covered_all_link_body_bytes"]
    report["matched_prepared_covered_online_body_bytes"] = control["online_all_link_body_bytes"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-output-tokens", type=int, choices=(8, 32), default=32)
    parser.add_argument("--summary", action="store_true", help="omit per-layer records")
    args = parser.parse_args()
    result = run(args.max_output_tokens)
    if args.summary:
        result["layer_provenance_sha256"] = hashlib.sha256(
            json.dumps(
                result.pop("layer_provenance"), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        placements = [
            result["client_attention_remote_mlp"],
            *result["two_worker_resident"].values(),
        ]
        for placement in placements:
            placement["known_online_body_bytes_by_directed_link"] = {
                f"{row['source']}→{row['destination']}:{row['tensor']}": row[
                    "optimistic_body_bytes"
                ]
                for row in placement.pop("directed_known_online_links")
            }
            placement["unknown_required_link_labels"] = sorted(
                {row["link"] for row in placement.pop("unknown_required_links_and_work")}
            )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

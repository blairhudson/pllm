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
from pllm.metrics import ProjectedPolynomialCostProbe
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.region_contract_cost import (
    compiler_region_contract_cost,
    compiler_region_reduction_gates,
)

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_CONTROL = _ROOT / "docs/evidence/latent-response-network-qwen25-2026-09-28.json"
_OFFSET_CONTROL = _ROOT / "docs/evidence/prepared-cold-compare-qwen-smol-2026-09-29.json"
_SHARED_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or str(
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
)


def run(
    output_tokens: int, *, reduction_gates: bool = False, projected_polynomial: bool = False
) -> dict:
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
    if projected_polynomial:
        offset_control = None
        if output_tokens == 32:
            recorded = json.loads(_OFFSET_CONTROL.read_text(encoding="utf-8"))[
                "qwen25_two_online_offset"
            ]
            if recorded["body_fingerprint"] != evidence["source"]["body_fingerprint"]:
                raise ValueError("offset comparator body differs from projected source")
            offset_control = {
                "evidence": str(_OFFSET_CONTROL.relative_to(_ROOT)),
                "covered_online_body_bytes": recorded["covered_online_body_bytes"],
                "covered_cold_all_link_body_bytes": recorded["covered_cold_all_link_body_bytes"],
                "scope": "same source/body and 39+32 counts; separate 64-token-bound cohort, online bodies only",
            }
        cases = []
        for bits in (24, 32, 64):
            for mode in ("dense", "derived", "contracted", "seeded"):
                row = ProjectedPolynomialCostProbe(mode=mode, ring_bits=bits).project(
                    plan,
                    composition,
                    response_new_tokens=output_tokens,
                )
                if (
                    row["schedule_digest"] != cohort["official_schedule_digest"]
                    or row["composition_digest"] != evidence["source"]["pipeline_digest"]
                ):
                    raise ValueError(
                        "polynomial projection differs from the matched compiler control"
                    )
                for placement in row["placements"].values():
                    if offset_control is not None:
                        budget = offset_control["covered_online_body_bytes"] // 10
                        placement["offset_online_tenfold_gate"] = {
                            "scope": "online-only incomplete body floor; no all-link claim",
                            "budget_bytes": budget,
                            "decision": "veto"
                            if placement["known_online_body_bytes"] > budget
                            else "inconclusive",
                        }
                    placement["prepared_reduction_gates"] = {}
                    for name, numerator, denominator in (
                        ("25_percent", 3, 4),
                        ("50_percent", 1, 2),
                        ("tenfold", 1, 10),
                    ):
                        online_budget = (
                            control["online_all_link_body_bytes"] * numerator // denominator
                        )
                        total_budget = (
                            control["covered_all_link_body_bytes"] * numerator // denominator
                        )
                        placement["prepared_reduction_gates"][name] = {
                            "online_budget_bytes": online_budget,
                            "all_link_budget_bytes": total_budget,
                            "decision": "veto"
                            if (
                                placement["known_online_body_bytes"] > online_budget
                                or placement["known_all_link_body_bytes"] > total_budget
                            )
                            else "inconclusive",
                        }
                row["layer_provenance_sha256"] = hashlib.sha256(
                    json.dumps(row.pop("layers"), sort_keys=True).encode()
                ).hexdigest()
                cases.append(row)
        return {
            "schema": "pllm.projected_polynomial_decoder_gate.v1",
            "cases": cases,
            "source": f"{_MODEL}@{_REVISION}",
            "plan_digest": plan.digest,
            "schedule_digest": cohort["official_schedule_digest"],
            "composition_digest": evidence["source"]["pipeline_digest"],
            "body_fingerprint_from_control": evidence["source"]["body_fingerprint"],
            "input_tokens": 39,
            "response_new_tokens": output_tokens,
            "related_two_offset_control": offset_control,
            "matched_prepared_covered_all_link_body_bytes": control["covered_all_link_body_bytes"],
            "matched_prepared_covered_online_body_bytes": control["online_all_link_body_bytes"],
            "whole_decoder_executable": False,
            "code_sha256": {
                str(path): hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
                for path in (
                    Path("crates/pllm-garble/src/projected_polynomial.rs"),
                    Path("python/pllm/runtime/projected_polynomial_cost.py"),
                    Path("scripts/probe_region_contract_cost.py"),
                )
            },
        }
    report = (
        compiler_region_reduction_gates(
            plan,
            composition,
            response_new_tokens=output_tokens,
            baseline_online_body_bytes=control["online_all_link_body_bytes"],
            baseline_all_link_body_bytes=control["covered_all_link_body_bytes"],
        )
        if reduction_gates
        else compiler_region_contract_cost(
            plan,
            composition,
            response_new_tokens=output_tokens,
            maximum_online_all_link_body_bytes=control["tenfold_online_budget_bytes"],
            maximum_total_all_link_body_bytes=control["tenfold_covered_budget_bytes"],
        )
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
    parser.add_argument("--reduction-gates", action="store_true", help="screen 25%/50%/10x budgets")
    parser.add_argument(
        "--projected-polynomial",
        action="store_true",
        help="cost native correlation ablations and a client-masked cut hypothesis",
    )
    parser.add_argument("--output", type=Path, help="save public cost report as JSON")
    args = parser.parse_args()
    if args.projected_polynomial and args.reduction_gates:
        parser.error("select one research cost screen")
    result = run(
        args.max_output_tokens,
        reduction_gates=args.reduction_gates,
        projected_polynomial=args.projected_polynomial,
    )
    if args.summary and not args.projected_polynomial:
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
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Saved {args.output}; executable={result['whole_decoder_executable']}")
        if args.projected_polynomial:
            for row in result["cases"]:
                if row["mode"] == "seeded":
                    cut = row["placements"]["client_masked_mlp_cut_hypothesis"]
                    print(
                        f"{row['ring_bits']} bits: client-masked floor "
                        f"{cut['known_online_body_bytes'] / 1e6:.2f} MB online / "
                        f"{cut['known_all_link_body_bytes'] / 1e6:.2f} MB all-link; "
                        f"matrix MAC ratio={row['known_matrix_mac_ratio_to_offset']:.3f}"
                    )
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

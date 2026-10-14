"""Audit historical first-request budgets without adding candidate measurements."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/research-sota-combinations-qwen25.json"
SOURCE_SHA256 = "5fb73b6883a0e42ac42b65b33a84cdfe0c6646d1e459f0df72639c38756ad54d"
MAX_SOURCE_BYTES = 8 * 1024 * 1024
VARIANTS = ("", "/tiling", "/digit8", "/both", "/both-placement")
CASES = {f"{rows}{variant}" for rows in (150, 288) for variant in VARIANTS}
COMPONENTS = {
    "prefill_material", "prefill_online", "decode_material", "decode_online",
    "setup_and_delivery",
}


def natural(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected nonnegative integer byte/count field")
    return value


def load_source(path: Path = SOURCE) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("historical source exceeds bounded input")
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("historical evidence digest differs")
    return json.loads(raw)


def body_edges(ledger: dict) -> dict[tuple[str, str, str], int]:
    if ledger["schema"] != "pllm.topology_body_accounting.v3":
        raise ValueError("expected original directed body coverage")
    result = {}
    for edge in ledger["body_bytes_by_edge"]:
        key = (edge["source"], edge["destination"], edge["phase"])
        if key in result:
            raise ValueError("duplicate directed body edge")
        result[key] = natural(edge["serialized_body_bytes"])
    if sum(result.values()) != natural(ledger["all_link_serialized_body_bytes"]):
        raise ValueError("directed body counters do not conserve")
    return result


def stream_edges(sample: dict) -> dict[str, int]:
    if sample["schema"] != "pllm.native_wan.v1":
        raise ValueError("expected original native TCP scope")
    values = {key: natural(value) for key, value in sample["directed_stream_bytes"].items()}
    if sum(values.values()) != natural(sample["total_stream_bytes"]):
        raise ValueError("directed stream counters do not conserve")
    return values


def free_removal_budget(total: int, removed: int, target: Fraction) -> dict:
    """Conditional deletion arithmetic for one measured body window, not a protocol."""
    natural(total)
    natural(removed)
    if total == 0 or removed > total or not isinstance(target, Fraction) or not 0 < target < 1:
        raise ValueError("invalid complete-window deletion sensitivity")
    allowance = Fraction(removed) - target * total
    return {
        "removed_known_body_bytes": removed,
        "zero_replacement_ceiling_fraction": float(Fraction(removed, total)),
        "target_fraction": str(target),
        "maximum_additional_body_bytes": allowance.numerator // allowance.denominator,
        "scope": "hypothetical free deletion within original covered bodies only",
    }


def first_request(result: dict) -> dict:
    execution = result["execution"]
    name = execution["case"]["name"]
    replicate = natural(execution["replicate"])
    report = execution["details"]["report"]
    if name not in CASES or replicate not in (0, 1, 2) or result["status"] != "completed":
        raise ValueError("unexpected historical case or incomplete replicate")
    if report["checks"]["passed"] is not True or report["warmup_runs"]:
        raise ValueError("historical checks or no-warmup contract differs")
    if len(report["runs"]) != 2:
        raise ValueError("expected original cold/repeat scenario")
    cold, repeat = report["runs"]
    if cold["cold"] is not True or cold["warm"] is not False or repeat["warm"] is not True:
        raise ValueError("first request is not the original cold execution")
    tokens = cold["tokens"]
    if (tokens["authoritative"] is not True or tokens["input_tokens"] != int(name.split("/")[0])
            or tokens["output_tokens"] != 8 or cold["max_output_tokens"] != 8):
        raise ValueError("historical workload or actual output count differs")

    topology = report["topology_accounting"]
    if topology["warmups"] or len(topology["runs"]) != 2:
        raise ValueError("historical accounting windows differ")
    startup = body_edges(topology["startup"])
    run = body_edges(topology["runs"][0])
    directed = {key: startup.get(key, 0) + run.get(key, 0) for key in startup.keys() | run.keys()}
    total = sum(directed.values())
    decomposition = report["traffic_decomposition"]
    saved = decomposition["cold_first_request"]
    if (decomposition["windows_aligned"] is not True
            or saved["conservation_checked"] is not True or saved["attribution_valid"] is not True
            or total != saved["total_all_link_body_bytes"]
            or total != report["summary"]["accounted_setup_through_first_response_body_bytes"]):
        raise ValueError("cold first-request windows do not reconcile")
    if (saved["generated_output_tokens"] != tokens["output_tokens"]
            or saved["input_tokens"] != tokens["input_tokens"]):
        raise ValueError("cold partition denominator differs from actual request")
    components = {name: 0 for name in COMPONENTS}
    residual = 0
    for ledger in (topology["startup"], topology["runs"][0]):
        attribution = ledger["traffic_attribution"]
        if (attribution["conservation_checked"] is not True
                or attribution["attribution_valid"] is not True
                or set(attribution["components_body_bytes"]) != COMPONENTS):
            raise ValueError("missing complete component accounting")
        current = {key: natural(value) for key, value in attribution["components_body_bytes"].items()}
        other = natural(attribution["unattributed_all_link_body_bytes"])
        if sum(current.values()) + other != ledger["all_link_serialized_body_bytes"]:
            raise ValueError("known components and residual do not conserve")
        for key, value in current.items():
            components[key] += value
        residual += other
    known_prefill = components["prefill_material"] + components["prefill_online"]
    known_decode = components["decode_material"] + components["decode_online"]
    if (known_prefill + components["setup_and_delivery"] != saved["known_base_all_link_body_bytes"]
            or known_decode != saved["known_output_dependent_all_link_body_bytes"]
            or residual != saved["unattributed_all_link_body_bytes"]):
        raise ValueError("component and cached partition attribution disagree")
    delivery = directed[("inference", "client", "cold")]
    if delivery > components["setup_and_delivery"]:
        raise ValueError("public delivery exceeds known setup owner")

    samples = report["native_network_accounting"]["samples"]
    if samples["warmups"] or len(samples["runs"]) != 2:
        raise ValueError("native counter windows differ")
    initial = stream_edges(samples["startup"]["wan_emulation"])
    before = stream_edges(samples["runs"][0]["before"]["wan_emulation"])
    after = stream_edges(samples["runs"][0]["after"]["wan_emulation"])
    if initial != before or any(after.get(edge, 0) < value for edge, value in before.items()):
        raise ValueError("startup and first-request TCP windows are not continuous")
    stream_bytes = sum(after.values())  # Startup is already inside these cumulative counters.
    if stream_bytes < total:
        raise ValueError("TCP counters undercount covered application bodies")
    return {
        "case": name, "replicate": replicate, "input_tokens": tokens["input_tokens"],
        "generated_output_tokens": tokens["output_tokens"],
        "comparison_identity": execution["comparison"],
        "covered_body_bytes": total, "native_tcp_stream_bytes": stream_bytes,
        "public_delivery_body_bytes": delivery,
        "known_prefill_body_bytes": known_prefill, "known_decode_body_bytes": known_decode,
        "other_known_setup_body_bytes": components["setup_and_delivery"] - delivery,
        "unattributed_body_bytes": residual,
        "components_body_bytes": components,
        "directed_body_bytes": {"/".join(key): value for key, value in sorted(directed.items())},
        "zero_cost_sensitivities": {
            "entire_public_delivery": free_removal_budget(total, delivery, Fraction(1, 5)),
            "known_prefill": free_removal_budget(total, known_prefill, Fraction(1, 5)),
            "known_decode_only": free_removal_budget(total, known_decode, Fraction(1, 5)),
        },
        "complete_cold_wire_bytes": None,
        "missing_checkpoint_distribution_bytes": None,
        "missing_tokenizer_distribution_bytes": None,
        "first_request_only_client_peak_rss_bytes": None,
    }


def audit(document: dict) -> dict:
    if (document.get("schema") != "pllm.benchmark_run.v1" or document.get("complete") is not True
            or len(document["results"]) != 30):
        raise ValueError("expected frozen complete thirty-client cohort")
    rows = [first_request(result) for result in document["results"]]
    identities = {(row["case"], row["replicate"]) for row in rows}
    if identities != {(case, replicate) for case in CASES for replicate in range(3)}:
        raise ValueError("missing or duplicate historical replicate")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    medians = {}
    metrics = (
        "covered_body_bytes", "native_tcp_stream_bytes", "public_delivery_body_bytes",
        "known_prefill_body_bytes", "known_decode_body_bytes", "other_known_setup_body_bytes",
        "unattributed_body_bytes",
    )
    for case, items in sorted(grouped.items()):
        if any(row["comparison_identity"] != items[0]["comparison_identity"] for row in items):
            raise ValueError("historical comparison identities differ within one case")
        medians[case] = {key: median(row[key] for row in items) for key in metrics}
    return {
        "schema": "pllm.cold_request_accounting_screen.v1", "complete": True,
        "scope": "read-only historical first-request extraction and conditional body sensitivities",
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": SOURCE_SHA256},
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": sorted(rows, key=lambda row: (row["case"], row["replicate"])),
        "case_medians": medians,
        "limitations": [
            "Historical source is cached and client tokenizer is pre-positioned; distribution is unknown.",
            "Only first-request traffic is extracted; original process peak includes its later repeat.",
            "Known prefill/decode subsets exclude residual attribution; complete marginal rates stay unknown.",
            "Free deletion is an optimistic budget, not a candidate implementation or achieved saving.",
            "Raw vocabulary representations cannot replace compressed public delivery counters byte-for-byte.",
            "No candidate quality, process RSS, privacy proof, serving timing or physical wire is measured.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(load_source())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

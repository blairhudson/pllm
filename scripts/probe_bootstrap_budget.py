"""Exact lifecycle sensitivity budgets for one-time-bootstrap research.

This is application-body accounting, not an implementation or performance probe.
Every horizon unit is one new-prefix/exact-repeat request pair. Both candidates
share the same public-artifact reuse assumption.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "docs/evidence/research-all-link-frontier.json"
NARROW = ROOT / "docs/evidence/research-narrow-ingress.json"
CARTRIDGE = ROOT / "docs/evidence/research-bootstrap-cartridge.json"
ROW_DEMAND = ROOT / "docs/evidence/research-attested-linear-bootstrap.json"
MAX_HORIZON = 4096


def natural(value: int, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True)
class LifecycleBudget:
    common_bootstrap_bytes: int
    control_pair_bytes: int
    candidate_bootstrap_bytes: int
    removed_pair_bytes: int
    helper_pair_bytes: int = 0
    renewal_bytes: int = 0
    renewal_pairs: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "common_bootstrap_bytes", "control_pair_bytes", "candidate_bootstrap_bytes",
            "removed_pair_bytes", "helper_pair_bytes", "renewal_bytes",
        ):
            natural(getattr(self, name), name)
        if self.control_pair_bytes == 0 or self.removed_pair_bytes > self.control_pair_bytes:
            raise ValueError("invalid recurring body partition")
        if self.renewal_pairs is not None:
            natural(self.renewal_pairs, "renewal_pairs", positive=True)
        elif self.renewal_bytes:
            raise ValueError("renewal cost requires an explicit renewal interval")

    def totals(self, pairs: int) -> tuple[int, int]:
        natural(pairs, "pairs", positive=True)
        if pairs > MAX_HORIZON:
            raise ValueError("horizon exceeds bounded screen")
        # Initial key issuance is charged too; it is not an unpriced epoch zero.
        epochs = (pairs + self.renewal_pairs - 1) // self.renewal_pairs if self.renewal_pairs else 0
        control = self.common_bootstrap_bytes + pairs * self.control_pair_bytes
        candidate = (
            self.common_bootstrap_bytes + self.candidate_bootstrap_bytes
            + pairs * (self.control_pair_bytes - self.removed_pair_bytes + self.helper_pair_bytes)
            + epochs * self.renewal_bytes
        )
        return control, candidate

    def saving(self, pairs: int) -> Fraction:
        control, candidate = self.totals(pairs)
        return Fraction(control - candidate, control)

    def first_horizon(self, target: Fraction, maximum: int = MAX_HORIZON) -> int | None:
        if not isinstance(target, Fraction) or not 0 < target < 1:
            raise ValueError("target must be an exact fraction between zero and one")
        natural(maximum, "maximum", positive=True)
        if maximum > MAX_HORIZON:
            raise ValueError("horizon exceeds bounded screen")
        # Renewal boundaries can decrease savings: binary search is invalid.
        return next((n for n in range(1, maximum + 1) if self.saving(n) >= target), None)

    def additional_helper_budget(self, pairs: int, target: Fraction) -> Fraction:
        if not isinstance(target, Fraction) or not 0 < target < 1:
            raise ValueError("target must be an exact fraction between zero and one")
        control, candidate = self.totals(pairs)
        return ((1 - target) * control - candidate) / pairs


def load_control(path: Path = CONTROL) -> dict:
    document = json.loads(path.read_text())
    if document.get("schema") != "pllm.all_link_frontier_screen.v1" or document.get("complete") is not True:
        raise ValueError("expected complete historical sensitivity reference")
    control = document["control"]
    classes = control["classes"]
    if set(classes) != {"corrections", "outputs", "inputs", "public_bundle", "control_and_seeds"}:
        raise ValueError("unknown historical body classification")
    natural(control["pair_bytes"], "pair_bytes", positive=True)
    for key, value in classes.items():
        natural(value, key)
    if sum(classes.values()) != control["pair_bytes"]:
        raise ValueError("historical classes do not conserve")
    directed = control["directed_counters"]
    for name, value in directed.items():
        natural(value, name)
    if sum(directed.values()) != control["pair_bytes"]:
        raise ValueError("historical directed bodies do not conserve")
    expected = {
        "corrections": directed["preparation->inference/offline"],
        "outputs": directed["inference->client/online"],
        "inputs": directed["client->inference/online"],
        "public_bundle": directed["inference->client/cold"],
        "control_and_seeds": (
            directed["client->preparation/offline"] + directed["preparation->client/offline"]
        ),
    }
    if classes != expected:
        raise ValueError("historical classes differ from directed evidence")
    return control


def prime_field_payload_screen(control: dict) -> dict:
    """Join saved row widths to the codec's proven alphabet bound, not protocols.

    Reuse the separately audited row-demand ledger. No family-name parsing or
    new checkpoint values are needed.
    Bounds are widened to the largest symmetric interval that fits each recorded
    width. This needs no private-value calibration or missing exact row norms.
    """
    narrow = json.loads(NARROW.read_text())["checkpoint_geometry"]
    if narrow["lock"]["body_fingerprint"] != control["body_fingerprint"]:
        raise ValueError("prime-field screen source mismatch")
    row_report = json.loads(ROW_DEMAND.read_text())
    if row_report.get("complete") is not True or row_report["budget"]["baseline"] != control:
        raise ValueError("prime-field screen demand mismatch")
    demand = {s["stage"]: s for s in row_report["budget"]["stages"]}
    q = 4_294_967_291  # Known prime 2^32-5; strictly greater than 2^31.
    stages = []
    widths = set()
    if not ({s["stage"] for s in narrow["stages"]} == set(control["stages"]) == set(demand)):
        raise ValueError("prime-field screen stage mismatch")
    for stage in narrow["stages"]:
        m, n = stage["shape"]
        measured = control["stages"][stage["stage"]]
        # Source-derived row demand is frozen in the original reconciled ledger.
        source = demand[stage["stage"]]
        if source["shape"] != [m, n] or source["weight_sha256"] != stage["weight_sha256"]:
            raise ValueError("prime-field screen weight mismatch")
        rows = source["pair_rows"]
        natural(rows, "stage rows", positive=True)
        histogram = stage["row_residue_width_histogram"]
        if sum(histogram.values()) != m:
            raise ValueError("invalid public row histogram")
        output_bits = 0
        for raw_width, count in histogram.items():
            width = int(raw_width)
            if not 2 <= width <= 29:
                raise ValueError("row outside proved equal-width domain")
            interval_size = (1 << width) - 1
            blocks = q // interval_size
            alphabet = (q + blocks - 1) // blocks
            if (alphabet - 1).bit_length() != width:
                raise ValueError("prime codec widens the public row")
            output_bits += count * width
            widths.add(width)
        old_input = (rows * n * stage["stage_bits"] + 7) // 8
        new_input = rows * n * 4
        output = (rows * output_bits + 7) // 8
        if (old_input > measured["client->inference"]
                or output > measured["inference->client"]
                or output > measured["preparation->inference"]):
            raise ValueError("projected payload exceeds original framed bodies")
        stages.append({"stage": stage["stage"], "rows": rows,
                       "weight_sha256": stage["weight_sha256"],
                       "old_input_payload_bytes": old_input,
                       "prime_input_payload_bytes": new_input,
                       "unchanged_output_payload_bytes": output})
    increase = sum(s["prime_input_payload_bytes"] - s["old_input_payload_bytes"] for s in stages)
    classes = control["classes"]
    common = classes["public_bundle"]
    recurring = control["pair_bytes"] - common
    saving = classes["corrections"] - increase
    # Explicit accounting sensitivities from the separate cartridge report;
    # none is a prime-field implementation, authenticated layout or RSS result.
    alternatives = []
    cartridges = json.loads(CARTRIDGE.read_text())["symbolic"]["candidates"]
    selected = {"weights_only_dense_mask_correction", "two_accumulator_checkpoint_d256",
                "two_accumulator_checkpoint_d64", "two_accumulator_checkpoint_d16", "p_only_no_verification"}
    for cartridge in cartridges:
        name, bootstrap = cartridge["name"], cartridge["bootstrap_bytes"]
        if name not in selected:
            continue
        budget = LifecycleBudget(common, recurring, bootstrap, classes["corrections"], increase)
        alternatives.append({
            "artifact": name, "bootstrap_bytes_assumption": bootstrap,
            "first_target_horizon_pairs": {
                str(t): budget.first_horizon(Fraction(t, 100)) for t in (20, 25, 30, 35, 50)
            },
        })
    return {
        "scope": "source-bound payload and ideal lifecycle sensitivity; no composed protocol",
        "prime": q, "input_bits": q.bit_length(),
        "source_body_fingerprint": control["body_fingerprint"],
        "stage_count": len(stages), "executed_stage_rows": sum(s["rows"] for s in stages),
        "recorded_row_widths": sorted(widths),
        "old_input_payload_bytes": sum(s["old_input_payload_bytes"] for s in stages),
        "prime_input_payload_bytes": sum(s["prime_input_payload_bytes"] for s in stages),
        "unchanged_output_payload_bytes": sum(s["unchanged_output_payload_bytes"] for s in stages),
        "added_input_payload_bytes": increase,
        "optimistic_removed_pair_bytes_after_input_expansion": saving,
        "warm_ceiling_before_new_costs": float(Fraction(saving, recurring)),
        "additional_helper_budget_per_pair_for_20_percent_asymptote": str(
            Fraction(saving) - Fraction(recurring, 5)),
        "alternatives": alternatives, "stages": stages,
        "unknown_not_zero": [
            "prime-field cartridge generation and parameter certification",
            "public layout/authentication and new framed bodies",
            "client reconstruction CPU, disk/cache owners and maximum process RSS",
            "secret-dependent malformed-reply handling and interactive security",
            "aggregate issuance security, renewal and failed attempts",
        ],
    }


def sensitivity_report() -> dict:
    control = load_control()
    classes = control["classes"]
    scenarios = []
    # Neither endpoint claims the unclassified public-delivery aggregate is a
    # measured immutable-only artifact. Both endpoints reproduce the first pair.
    for assumption, recurring_public in (
        ("all_recorded_public_delivery_reused_after_first_pair", 0),
        ("all_recorded_public_delivery_recurs_each_pair", classes["public_bundle"]),
    ):
        common = classes["public_bundle"] - recurring_public
        recurring = control["pair_bytes"] - common
        for bootstrap in (0, 100_000_000, 1_000_000_000, 10_000_000_000):
            budget = LifecycleBudget(common, recurring, bootstrap, classes["corrections"])
            assert budget.totals(1)[0] == control["pair_bytes"]
            scenarios.append({
                "public_reuse_assumption": assumption,
                "candidate_bootstrap_bytes": bootstrap,
                "candidate": "ideal_correction_elimination_no_replacement_traffic",
                "common_bootstrap_bytes": common,
                "control_recurring_pair_bytes": recurring,
                "removed_recurring_pair_bytes": classes["corrections"],
                "first_target_horizon_pairs": {
                    str(percent): budget.first_horizon(Fraction(percent, 100))
                    for percent in (20, 35, 50)
                },
                "horizons": [{
                    "pairs": n,
                    "control_bytes": budget.totals(n)[0],
                    "candidate_bytes": budget.totals(n)[1],
                    "fractional_saving": float(budget.saving(n)),
                } for n in (1, 2, 4, 8, 16, 32, 64, 256, 1024, MAX_HORIZON)],
            })
    return {
        "schema": "pllm.bootstrap_traffic_budget.v1",
        "complete": True,
        "scope": "exact conditional application-body accounting; no candidate protocol executed",
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), CONTROL, NARROW, CARTRIDGE, ROW_DEMAND,
                         ROOT / "scripts/probe_cyclic_mask_codec.py",
                         ROOT / "docs/evidence/research-cyclic-mask-codec.json")
        },
        "control_configuration_digest": control["configuration_digest"],
        "control_body_fingerprint": control["body_fingerprint"],
        "control_pair_bytes": control["pair_bytes"],
        "arithmetic_pair_bytes": sum(classes[k] for k in ("corrections", "outputs", "inputs")),
        "horizon_unit": "new 150-token prefix with eight outputs, then its exact repeat",
        "maximum_horizon_pairs": MAX_HORIZON,
        "null_target_meaning": "no qualifying horizon through maximum; not a protocol impossibility proof",
        "client_peak_bytes": None,
        "low_tier_qualified": None,
        "complete_base_bytes": None,
        "complete_per_generated_token_bytes": None,
        "limitations": [
            "Historical body ledger is not the latest Low-tier control or full wire.",
            "Public-delivery recurrence endpoints are assumptions, not measured cache behavior.",
            "No helper, renewal, source distribution, failed-attempt or framing costs are priced in these ideal ceilings.",
            "A finite material tape shifts traffic into bootstrap; it does not compress a matched horizon.",
            "One-time artifact reuse is independent of one-use mask/material retirement.",
            "Future measurements must include complete client CPU/RSS and separate disk/cache/device owners.",
        ],
        "scenarios": scenarios,
        "prime_field_payload_screen": prime_field_payload_screen(control),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = sensitivity_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"complete": True, "scenarios": len(report["scenarios"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()

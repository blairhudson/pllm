"""Low-RAM neighbors of the measured tiled/digit8 prepared stack.

The measured-RSS ceiling classifies complete fresh-client cohorts. Topology
admission still prices every owner independently; measured fit does not replace
conservative SDK admission.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from pllm import Experiment
from pllm.metrics import BenchmarkCase, BenchmarkSuite
from pllm.preparation import ModelAwareCorrections
from pllm.roles import ClientLinearRoles
from pllm.schedulers import MlpPrefillTiling
from pllm.search import FrontierStudy

from benchmarks.research import sota_combinations as base


_TIERS = Path(__file__).resolve().parents[2] / "python/pllm/search/client_memory_tiers.json"
LOW_BYTES = next(tier["maxBytes"] for tier in json.loads(_TIERS.read_text()) if tier["id"] == "low")
VARIANTS = ("control", "tile32", "tile16", "placement", "tile32-placement", "tile16-placement")


def components(variant="control", *, placement=()):
    if variant not in VARIANTS:
        raise ValueError("unknown Low-memory combination")
    tile = 32 if variant.startswith("tile32") else 16 if variant.startswith("tile16") else 64
    selected = {"scheduler": MlpPrefillTiling(tile_rows=tile),
                "preparation": ModelAwareCorrections(storage="paged", kernel="digit8", min_rows=64)}
    if "placement" in variant:
        if not placement:
            raise ValueError("placement needs frozen layer/role pairs")
        selected["placement"] = ClientLinearRoles(layer_roles=tuple(placement))
    return selected


def study(*, model, tokenizer, build, placement=(), processes=3,
          workloads=("150", "288"), tiny=False):
    # Identical workload, lifecycle, access caps and sampling contracts.
    original = base.study(model=model, tokenizer=tokenizer, build=build, placement=placement,
                          processes=processes, workloads=workloads, tiny=tiny)
    cases = []
    pairs = ((0, "attention_output"),) if tiny else placement
    for original_case in original.suite.cases:
        if "/" in original_case.name:
            continue
        experiment = Experiment.from_spec(
            original_case.target.to_spec()["parameters"]["model"]["experiment"])
        for variant in VARIANTS:
            selected = dict(experiment.pipeline.components) | components(variant, placement=pairs)
            name = original_case.name if variant == "control" else f"{original_case.name}/{variant}"
            cases.append(BenchmarkCase(name, original_case.target.with_params(pipeline__components=selected),
                                       build, original_case.budgets))
    return FrontierStudy(BenchmarkSuite(tuple(cases), original.suite.sampling,
                                       "low-memory-combinations"))


def memory_eligibility(document):
    """Require every planned replicate; retain unknown and over-budget results."""
    suite = document["suite"]
    expected = suite["sampling"]["processes"]
    names = [case["name"] for case in suite["cases"]]
    if not document["complete"] or len(names) != len(set(names)):
        raise base.Rejected("incomplete_memory_cohort")
    samples = defaultdict(dict)
    for result in document["results"]:
        execution = result["execution"]
        name, replicate = execution["case"]["name"], execution["replicate"]
        if (result["status"] != "completed" or name not in names
                or type(replicate) is not int or not 0 <= replicate < expected
                or replicate in samples[name]):
            raise base.Rejected("incomplete_memory_cohort")
        peak = execution["details"]["report"].get("client_process_memory", {}).get("after", {}).get(
            "lifetime_peak_rss_bytes")
        if peak is not None and (type(peak) is not int or peak <= 0):
            raise base.Rejected("invalid_client_peak")
        samples[name][replicate] = peak
    rows = []
    for name in names:
        if set(samples[name]) != set(range(expected)):
            raise base.Rejected("incomplete_memory_cohort")
        peaks = [samples[name][replicate] for replicate in range(expected)]
        maximum = None if None in peaks else max(peaks)
        rows.append({"case": name, "sample_count": expected, "peak_bytes": peaks,
                     "max_peak_bytes": maximum, "eligible": maximum is not None and maximum <= LOW_BYTES,
                     "status": "unknown" if maximum is None else "qualified" if maximum <= LOW_BYTES else "over_budget"})
    return {"tier": "low", "max_bytes": LOW_BYTES,
            "scope": "fresh_client_process_birth_through_cold_and_repeat", "cases": rows}


def audit(run, *, tiny=False, placement_record=None):
    result = base.audit(run, tiny=tiny, placement_record=placement_record, variants=VARIANTS)
    result["client_memory_budget"] = memory_eligibility(run.to_dict())
    return result

"""Maintained response ablations around the combined public prepared SDK recipe."""
from __future__ import annotations

import pllm
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario
from pllm.search import FrontierStudy, Variant

from examples.prepared_sdk import build_experiment


def study(*, model=None, tokenizer=None, build=None, sampling=None, output_tokens=8):
    experiment = build_experiment(model or pllm.Model.tiny(), tokenizer=tokenizer,
                                  root=".pllm/prepared-frontier-study")
    target = BenchmarkTarget.response(experiment, scenario=Scenario.repeat(max_output_tokens=output_tokens),
                                      wan=WanConditions())
    # Each omitted feature is an explicit neighbor of the same full-stack recipe.
    # Only exact numeric, source, privacy and lifecycle identities may rank.
    variants = [
        Variant("no-reuse", {"pipeline__cache": None}),
        Variant("whole-prefill", {"pipeline__linear__prefill_chunk_rows": 0}),
        Variant("serial-issuance", {"pipeline__inventory__stage_window": 1,
                                    "pipeline__inventory__stage_window_bytes": 16 << 20}),
        Variant("resident-client", {"pipeline__delivery__storage": "memory"}),
        Variant("resident-preparation", {"pipeline__preparation__storage": "resident"}),
        Variant("uncached-public-encoding", {"pipeline__delivery__provider_cache_bytes": 0}),
        Variant("digit8", {"pipeline__preparation__kernel": "digit8", "pipeline__preparation__min_rows": 64}),
    ]
    if tokenizer is not None:
        variants.append(Variant("original-tokenizer", {"pipeline__tokenizer": None}))
    return FrontierStudy(BenchmarkSuite((BenchmarkCase("prepared", target, build or BuildRef.capture()),),
        sampling or Sampling(processes=1, timeout_seconds=900, memory_bytes=8 << 30),
        "prepared-serving-v1"), tuple(variants), (("no-reuse", "whole-prefill"), ("serial-issuance", "digit8")))

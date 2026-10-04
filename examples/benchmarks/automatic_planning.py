"""Bounded load-time planning; cached public Qwen source, no tensor execution.

Run: python examples/benchmarks/automatic_planning.py --output /tmp/selection.json
The offers below are declared local fixtures, not authenticated remote parties.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.compiler import plan_on_load
from pllm.deployment import NetworkSnapshot, PartyOffer
from pllm.kernels import Cpu
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.search import PlanningPolicy, optimization_space


def select_on_load():
    base = Experiment(
        "automatic-public-qwen",
        MaskedLinearCpu(
            Model.hf(
                "Qwen/Qwen2.5-0.5B-Instruct",
                revision="7ae557604adf67be50417f59c2c2f167def9a775",
                local_files_only=True,
            ),
            kernels=Cpu(threads=4),
            quantization=SymmetricPerRow(
                weight_bits=8,
                activation_bits=8,
                causal_reduction="prefix_f32",
            ),
        ),
        Deployment.local(root="local://automatic-planning"),
        ExecutionBudget(requests=3, max_input_tokens=96, max_new_tokens=32),
    )
    now = time.time_ns() // 1_000_000
    offers = tuple(
        PartyOffer(
            party,
            party,
            str(index + 1) * 64,
            now + 600_000,
            (capability,),
            (2 if party == "client" else 8) << 30,
            2 << 30,
            4,
            0,
            ("*",),
            ("*",),
            devices=("cpu", "metal"),
        )
        for index, (party, capability) in enumerate(
            (
                ("client", "trusted_client"),
                ("preparation", "trusted_preparation"),
                ("inference", "masked_linear_provider"),
            )
        )
    )
    snapshot = NetworkSnapshot(
        "automatic-local", offers, (), "declared-local-fixture", now, now + 600_000
    )
    policy = PlanningPolicy(
        "client",
        now,
        64,
        128,
        ("online_all_link_body_bytes", "client_payload_bytes"),
        minimum_remote_mac_fraction=0.8,
        max_client_payload_bytes=512 << 20,
        # The planner conservatively charges six bytes per owned weight element.
        max_client_weight_bytes=1536 << 20,
        max_observation_age_ms=600_000,
    )
    space = optimization_space(
        base,
        allow_client_weights=True,
        prefix_cache_bytes=64 << 20,
        metal_min_rows=32,
    )
    started = time.perf_counter()
    decision = plan_on_load(base, snapshot=snapshot, policy=policy, space=space)
    elapsed = time.perf_counter() - started
    if decision.status != "feasible" or not decision.exhaustive:
        raise RuntimeError(
            json.dumps(
                {
                    "status": decision.status,
                    "reasons": sorted({row["reason"] for row in decision.to_spec()["rejections"]}),
                }
            )
        )
    assert decision.experiment.pipeline.components["kernels"].component == "pllm/cpu"
    assert "placement" in decision.experiment.pipeline.components
    return decision, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    decision, elapsed = select_on_load()
    args.output.write_bytes(decision.canonical_bytes())
    evidence = {
        "schema": "pllm.load_time_optimization_probe.v1",
        "scope": "cached-source resolution and bounded geometry planning; no tensor values, inference or profiling",
        "elapsed_seconds": elapsed,
        "decision_digest": decision.digest,
        "source_lock_digest": decision.request.source_lock_digest,
        "coverage": decision.to_spec()["coverage"],
        "selected_components": {
            key: value.to_spec() for key, value in decision.experiment.pipeline.components.items()
        },
        "costs": decision.to_spec()["selection"]["costs"],
        "measurements_used_for_selection": False,
        "independent_operators_verified": False,
    }
    if args.evidence:
        args.evidence.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {key: evidence[key] for key in ("elapsed_seconds", "coverage", "selected_components")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

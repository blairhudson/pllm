"""Run slice A locally: generated checkpoint, offline choice, SDK, ordinary benchmark.

Usage: python examples/benchmarks/network_planning.py
Requires development dependencies (torch for checkpoint generation). No downloads.
All exported configs/checkpoints live in one automatically cleaned temporary directory.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model, OpenAI
from pllm.compiler import plan
from pllm.deployment import NetworkSnapshot, NetworkSpec, PartyOffer, open_execution
from pllm.model_loader import resolve_model
from pllm.modeling import lower_model
from pllm.plan import PlanningResult
from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu, TwoOnlineOffsetCpu
from pllm.search import PlanningPolicy, PlanningRequest


def run_demo(*, benchmark=True):
    from pllm.runtime.benchmark_cli import run_loopback_benchmark
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    with tempfile.TemporaryDirectory(prefix="pllm-network-planning-") as temporary:
        root = Path(temporary)
        checkpoint = create_tiny_llama_checkpoint(
            root / "checkpoint",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        model = Model.path(str(checkpoint), model_id="network-planning-tiny")
        config = json.loads((checkpoint / "config.json").read_bytes())
        budget = ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2)
        model_plan = lower_model(config, batch=1, max_input_tokens=32, max_new_tokens=2)
        candidates = tuple(
            Experiment(
                name=profile.__name__,
                pipeline=profile(model),
                deployment=Deployment.local(root="local://network-planning"),
                budget=budget,
            )
            for profile in (ClientOnlyCpu, MaskedLinearCpu, TwoOnlineOffsetCpu)
        )
        now = time.time_ns() // 1_000_000
        capabilities = (
            "trusted_client",
            "trusted_preparation",
            "masked_linear_provider",
            "public_linear_provider",
        )
        offers = tuple(
            PartyOffer(
                party,
                party,
                str(index + 1) * 64,
                now + 600_000,
                ("trusted_client",) if party == "client" else capabilities[1:],
                256 << 20,
                64 << 20,
                4,
                0,
                (model_plan.to_dict()["config_digest"],),
                ("*",),
            )
            for index, party in enumerate(("client", "worker-a", "worker-b"))
        )
        snapshot = NetworkSnapshot(
            "local-demo", offers, (), "generated-fixture", now, now + 600_000
        )
        network = NetworkSpec("local-demo", snapshot)
        request = PlanningRequest(
            model_plan,
            candidates,
            PlanningPolicy(
                "client",
                now,
                8,
                1000,
                ("total_arithmetic_body_bytes", "client_weight_bytes"),
                minimum_remote_mac_fraction=0.5,
                max_observation_age_ms=600_000,
            ),
            source_lock_digest=resolve_model(model).source_lock_digest,
        )
        decision = plan(request, snapshot=snapshot)
        for name, record in (
            ("network", network),
            ("snapshot", snapshot),
            ("request", request),
            ("plan", decision),
        ):
            (root / f"{name}.json").write_bytes(record.canonical_bytes() + b"\n")
        decision = PlanningResult.from_file(root / "plan.json", request=request, snapshot=snapshot)
        with open_execution(decision, network=network) as lease:
            with OpenAI(execution=lease) as client:
                response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
                sdk = {
                    "output_tokens": response.usage.output_tokens,
                    "output_text": response.output_text,
                    "selected_placement_digest": lease.binding.to_spec()["placement"][
                        "placement_digest"
                    ],
                }
        report = None
        if benchmark:
            report = run_loopback_benchmark(
                model=str(checkpoint),
                model_id=model.model_id,
                tiny=False,
                prompt="Hi",
                max_output_tokens=2,
                warmups=0,
                repetitions=1,
                timeout_seconds=120,
                experiment=decision.experiment,
                temperature=0.0,
                capture_output_digest=True,
            )
        return {
            "status": decision.status,
            "exhaustive": decision.exhaustive,
            "selected_experiment": decision.experiment.name,
            "configuration_digest": decision.experiment.configuration_digest(),
            "sdk": sdk,
            "benchmark": report,
            "costs": decision.to_spec()["selection"]["costs"],
        }


if __name__ == "__main__":
    print(json.dumps(run_demo(), sort_keys=True, indent=2, allow_nan=False))

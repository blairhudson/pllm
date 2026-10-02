"""Immutable candidates for canonical cold/warm and context-sequence benchmarks.

Fresh prefix reuse admits only sealed completed prefills with the same actual
full-input reduction extent; fixed_input_tokens is capacity, not numeric width.
Changed-width context growth and generated decode state miss the fresh cache.
Historical pre-gate growth-context savings are not current exact-reuse claims.
Explicit previous_response_id retains separate response-owned incremental state.
"""

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles, ClientPrefixLayers
from pllm.state import ClientPrefixReuse

_SOURCE = Model.hf(
    "Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775"
)


def candidate(name, *, placement=None, policy="request-sized", encoding="none", reuse=False):
    return Experiment(
        name,
        MaskedLinearCpu(
            _SOURCE,
            kernels=Cpu(threads=4),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            placement=placement,
            inventory=PreparedInventory(policy),
            delivery=ClientBundleTransport(encoding),
            cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=512) if reuse else None,
        ),
        Deployment.local(root="local://incremental-network"),
        ExecutionBudget(requests=8, max_input_tokens=512, max_new_tokens=32),
    )


baseline = candidate("prepared-prewarm", policy="prewarm")
request_sized = candidate("prepared-request-sized")
prefix_one = candidate("client-prefix-one", placement=ClientPrefixLayers(1))
prefix_two = candidate("client-prefix-two", placement=ClientPrefixLayers(2))
attention = candidate(
    "client-attention", placement=ClientLinearRoles(["qkv_projection", "attention_output"])
)
attention_zlib = candidate(
    "client-attention-zlib", placement=attention.pipeline.placement, encoding="zlib"
)
shared_prefix = candidate("prepared-shared-prefix", reuse=True)
attention_shared_prefix = candidate(
    "client-attention-shared-prefix", placement=attention.pipeline.placement, reuse=True
)
attention_shared_prefix_zlib = candidate(
    "client-attention-shared-prefix-zlib",
    placement=attention.pipeline.placement,
    reuse=True,
    encoding="zlib",
)
assert attention.resolve().client_linear_roles == ("attention_output", "qkv_projection")


if __name__ == "__main__":
    import argparse
    import hashlib
    import json
    import platform
    from importlib.metadata import version
    from pathlib import Path
    from pllm.runtime.benchmark_cli import accounted_benchmark_body_totals

    parser = argparse.ArgumentParser(
        description="Inspect canonical benchmark reports without displaying inputs or outputs."
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--archive", type=Path, help="Export sanitized, code-bound aggregate evidence"
    )
    args = parser.parse_args()
    cohorts = []
    for path in args.reports:
        document = json.loads(path.read_text())
        rows = []
        for entry in document.get("candidates", [{"name": "single", "report": document}]):
            report = entry["report"]
            summary = {
                **report["summary"],
                **accounted_benchmark_body_totals(report.get("topology_accounting") or {}),
            }
            rows.append(
                {
                    "name": entry["name"],
                    "checks": report["checks"],
                    "summary": summary,
                    "configuration_digest": entry.get("configuration_digest"),
                    "pipeline_digest": entry.get("pipeline_digest"),
                    "client_body_placement": report.get("client_body_placement"),
                    "cold_cpu": report.get("process_cpu_accounting"),
                    "usage_sequence": [
                        {
                            "input_tokens": item["tokens"]["input_tokens"],
                            "output_tokens": item["tokens"]["output_tokens"],
                            "prefix_tokens_reused": item["privacy"].get(
                                "prefill_prefix_tokens_reused", 0
                            ),
                        }
                        for item in report["runs"]
                    ],
                }
            )
        cohorts.append(
            {
                "file": path.name,
                "comparison_key": document.get("comparison_key"),
                "candidates": rows,
            }
        )
        if args.compact:
            print(json.dumps({"file": path.name, "comparison_key": document.get("comparison_key")}))
            for row in rows:
                summary = row["summary"]
                print(
                    json.dumps(
                        {
                            "name": row["name"],
                            "passed": row["checks"]["passed"],
                            "online_bytes": summary["median_online_all_link_serialized_body_bytes"],
                            "run_bodies": summary["median_covered_all_link_serialized_body_bytes"],
                            "setup_first_bodies": summary[
                                "accounted_setup_through_first_response_body_bytes"
                            ],
                            "total_online_bytes": summary[
                                "total_run_online_all_link_serialized_body_bytes"
                            ],
                            "total_accounted_body_bytes": summary[
                                "total_accounted_benchmark_body_bytes"
                            ],
                            "total_seconds": summary["total_run_full_seconds"],
                            "full_seconds": summary["median_full_seconds"],
                            "cold_cpu": row["cold_cpu"].get(
                                "aggregate_cold_first_response_cpu_seconds"
                            ),
                            "placement": row["client_body_placement"],
                        }
                    )
                )
        else:
            print(
                json.dumps(
                    {
                        "file": path.name,
                        "comparison_key": document.get("comparison_key"),
                        "candidates": rows,
                    },
                    indent=2,
                )
            )
    if args.archive:
        root = Path(__file__).resolve().parents[2]
        paths = [
            "examples/benchmarks/incremental_network.py",
            "examples/benchmarks/conversation_contexts.json",
            "crates/pllm-compiler/src/decoder_runtime_schedule.rs",
            "crates/pllm-compiler/src/lib.rs",
            "python/pllm/roles/__init__.py",
            "python/pllm/preparation.py",
            "python/pllm/protocols/bundle_transport.py",
            "python/pllm/profiles/__init__.py",
            "python/pllm/runtime/semantic_stages.py",
            "python/pllm/runtime/model_binding.py",
            "python/pllm/runtime/transformer_client.py",
            "python/pllm/runtime/transformer_engine.py",
            "python/pllm/runtime/client.py",
            "python/pllm/runtime/servers.py",
            "python/pllm/_cli/app.py",
            "python/pllm/runtime/dashboard.py",
            "python/pllm/runtime/benchmark_cli.py",
        ]
        args.archive.write_text(
            json.dumps(
                {
                    "schema": "pllm.incremental_network_evidence.v1",
                    "date": "2026-10-01",
                    "source": {
                        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                        "weight_bits": 8,
                        "activation_bits": 8,
                    },
                    "scope": "One co-located loopback sample per candidate/cohort; serialized protocol bodies, not full wire",
                    "limitations": [
                        "Checkpoint download and TLS/headers are unmeasured",
                        "Storage counts exclude peak RAM and persistent disk",
                        "Matching body/cohort does not establish output equality or float32 quality",
                        "Timing and CPU are not representative medians; operator independence and compute-cap admission are unset",
                    ],
                    "environment": {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "packages": {
                            name: version(name)
                            for name in ("numpy", "torch", "transformers", "pllm.run")
                        },
                    },
                    "code_sha256": {
                        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                        for path in paths
                    },
                    "cohorts": cohorts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

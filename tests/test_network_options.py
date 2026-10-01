"""SDK options execute through the ordinary compiler/topology/benchmark path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport
from pllm.roles import ClientLinearRoles
from pllm.runtime.benchmark_cli import build_comparison_report, run_loopback_benchmark
from pllm.runtime.client import RuntimeClient
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientPrefixReuse


def selected(root: Path, *, cache: bool = False) -> Experiment:
    return Experiment(
        "prefix" if cache else "attention-local",
        MaskedLinearCpu(
            Model.path(str(root), model_id="network-options"),
            placement=ClientLinearRoles(["qkv_projection", "attention_output"]),
            inventory=PreparedInventory(),
            delivery=ClientBundleTransport(),
            cache=ClientPrefixReuse(fixed_input_tokens=128, max_bytes=1 << 20) if cache else None,
        ),
        Deployment.local(root=str(root.parent)),
        ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=2),
    )


def test_sdk_options_roundtrip_and_reject_conflicting_client_overrides(tmp_path: Path) -> None:
    experiment = selected(tmp_path)
    restored = Pipeline.from_spec(experiment.pipeline.to_spec())
    assert restored.digest() == experiment.pipeline.digest()
    resolved = experiment.resolve()
    assert (
        resolved.inventory_policy,
        resolved.prepared_inventory_rows,
        resolved.bundle_compression,
    ) == (
        "request-sized",
        1,
        "zlib",
    )
    common = dict(base_url="http://127.0.0.1:1", api_key="test", experiment=experiment)
    client = RuntimeClient(**common)
    try:
        assert client.prepared_inventory_rows == 1
        assert client.bundle_compression == "zlib"
    finally:
        client.close()
    with pytest.raises(ValueError, match="inventory rows conflict"):
        RuntimeClient(**common, prepared_inventory_rows=64)
    with pytest.raises(ValueError, match="bundle encoding conflicts"):
        RuntimeClient(**common, bundle_compression="none")
    for rows in (0, True, 4097):
        with pytest.raises(ValueError):
            PreparedInventory(rows=rows)
    with pytest.raises(ValueError):
        ClientBundleTransport("gzip")


@pytest.mark.integration
def test_ordered_context_benchmark_combines_placement_prefix_and_sdk_transport(
    tmp_path: Path,
) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "weights", hidden_size=128, intermediate_size=256, head_dim=32
    )
    context = "A public conversation about weather and maps. " + "Common context. " * 3
    prompts = [context, context + " Next.", context + " Next. More."]
    candidates = []
    for cache in (False, True):
        experiment = selected(root, cache=cache)
        report = run_loopback_benchmark(
            model=str(root),
            model_id="network-options",
            tiny=False,
            prompt=prompts[0],
            prompt_sequence=prompts,
            max_output_tokens=2,
            warmups=0,
            repetitions=1,
            timeout_seconds=90,
            experiment=experiment,
            _cohort_salt=b"network-options-cohort".ljust(32, b"\0"),
        )
        assert report["checks"]["passed"]
        assert report["configuration"]["inventory_policy"] == "request-sized"
        assert report["configuration"]["bundle_compression"] == "zlib"
        assert [row["context_index"] for row in report["runs"]] == [0, 1, 2]
        assert context not in json.dumps(report)
        assert all(row["privacy"]["plaintext_token_ids_sent"] == 0 for row in report["runs"])
        candidates.append((experiment, report))
    comparison = build_comparison_report(candidates)
    assert comparison["checks"]["matched_workload"]
    assert comparison["rankings"]["sequence_online_all_link_serialized_body_bytes"]
    plain, reused = [report for _, report in candidates]
    assert sum(row["privacy"]["prefill_cache_hits"] for row in reused["runs"]) == 2
    assert (
        reused["summary"]["total_run_online_all_link_serialized_body_bytes"]
        < plain["summary"]["total_run_online_all_link_serialized_body_bytes"]
    )
    reordered = json.loads(json.dumps(reused))
    reordered["runs"] = list(reversed(reordered["runs"]))
    assert not build_comparison_report([(candidates[0][0], plain), (candidates[1][0], reordered)])[
        "checks"
    ]["matched_workload"]

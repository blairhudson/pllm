"""Output-head ownership is part of the compiler-bound prepared protocol."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.roles import OutputHeadAtInference
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint


def _experiment(root: Path, *, remote: bool) -> Experiment:
    return Experiment(
        name="remote-output-head" if remote else "client-output-head",
        pipeline=MaskedLinearCpu(
            Model.path(str(root), model_id="untied-head"),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            boundary=OutputHeadAtInference() if remote else None,
        ),
        deployment=Deployment.local(root=str(root)),
        budget=ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=2),
    )


@pytest.mark.integration
def test_untied_head_remote_stage_matches_client_head_through_prepared_roles(
    tmp_path: Path,
) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "untied", num_hidden_layers=1,
        model_type="qwen2", tie_word_embeddings=False,
    )
    results = []
    for remote in (False, True):
        with build_roles(_experiment(root, remote=remote), engine_threads=1) as topology:
            with topology.client(prepared_inventory_rows=8, background_inventory_refill=False) as client:
                state = client._core._transformer_state("untied-head")
                head = state.bundle.stages["lm_head"]
                assert (head.client_weight is None) is remote
                assert (head.seeded_profile is not None) is remote
                response = client.responses.create(
                    model="untied-head", input="Testing untied head placement.",
                    max_output_tokens=2, temperature=0,
                )
                assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
                assert client.privacy_audit.plaintext_token_ids_sent == 0
                assert client.privacy_audit.preparation_requests_during_online == 0
                results.append((response.output_text, response.usage,
                                client.privacy_audit.inference_stage_calls))
    assert results[0][:2] == results[1][:2]
    assert results[1][2] > results[0][2]


def test_remote_head_component_rejects_tied_checkpoint_and_ineligible_topologies(tmp_path: Path) -> None:
    from pllm.configuration import ConfigurationError, Pipeline
    from pllm.model_loader import resolve_model
    from pllm.profiles import ClientOnlyCpu
    from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError

    tied = create_tiny_llama_checkpoint(tmp_path / "tied", tie_word_embeddings=True)
    source_manifest = resolve_model(Model.path(str(tied), model_id="tied-head")).manifest
    with pytest.raises(TransformerEngineError, match="untied"):
        asyncio.run(MaskedTransformerEngine(remote_output_head=True).load(source_manifest))

    source = Model.path(str(tmp_path), model_id="untied-head")
    pipeline = ClientOnlyCpu(source)
    with pytest.raises(ConfigurationError):
        Experiment(
            name="invalid-boundary",
            pipeline=Pipeline(model=source, components={
                **pipeline.components, "boundary": OutputHeadAtInference(),
            }),
            deployment=Deployment.local(root=str(tmp_path)),
            budget=ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=1),
        ).resolve()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("PLLM_RUN_OUTPUT_HEAD_COST_GATE"), reason="explicit placement cost gate")
def test_untied_output_head_cold_and_online_cost_by_generation_length(tmp_path: Path) -> None:
    """Use the same W8A8 model and 1/8/32-token requests for both placements."""
    from pllm.runtime.benchmark_cli import build_comparison_report, run_loopback_benchmark

    root = create_tiny_llama_checkpoint(
        tmp_path / "large-untied", num_hidden_layers=1,
        model_type="qwen2", tie_word_embeddings=False,
        vocab_size=65536, hidden_size=512, intermediate_size=1024,
        num_attention_heads=8, num_key_value_heads=2, head_dim=64,
    )
    source = Model.path(str(root), model_id="untied-large-head")
    observations = []
    for output_cap in (1, 8, 32):
        candidates = []
        for remote in (False, True):
            experiment = Experiment(
                name=f"head-{'remote' if remote else 'client'}-{output_cap}",
                pipeline=MaskedLinearCpu(
                    source, quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
                    boundary=OutputHeadAtInference() if remote else None,
                ),
                deployment=Deployment.local(root=str(tmp_path)),
                budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=output_cap),
            )
            report = run_loopback_benchmark(
                model=str(root), model_id="untied-large-head", tiny=False,
                prompt="A bounded input.",
                max_output_tokens=output_cap, warmups=0, repetitions=1,
                timeout_seconds=900, experiment=experiment,
                inventory_policy="request-sized", _cohort_salt=b"head-placement-cohort".ljust(32, b"\0"),
            )
            assert report["checks"]["passed"]
            candidates.append((experiment, report))
        comparison = build_comparison_report(candidates)
        assert comparison["checks"]["matched_workload"]
        assert comparison["comparison_key"]["model_fingerprint"]
        observations.append((output_cap, candidates, comparison))

    for output_cap, candidates, comparison in observations:
        plain, remote = [report for _, report in candidates]
        assert plain["runs"][0]["tokens"] == remote["runs"][0]["tokens"]
        assert plain["runs"][0]["privacy"]["plaintext_prompt_bytes_sent"] == 0
        assert remote["runs"][0]["privacy"]["plaintext_token_ids_sent"] == 0
        assert remote["runs"][0]["privacy"]["bundle_network_bytes"] < plain["runs"][0]["privacy"]["bundle_network_bytes"]
        assert remote["runs"][0]["privacy"]["inference_stage_calls"] > plain["runs"][0]["privacy"]["inference_stage_calls"]
        assert comparison["rankings"]["online_all_link_serialized_body_bytes"]
        if os.environ.get("PLLM_OUTPUT_HEAD_COST_GATE_REPORT"):
            print(json.dumps({
                "output_cap": output_cap,
                "model_fingerprint": comparison["comparison_key"]["model_fingerprint"],
                "placements": [
                    {
                        "name": experiment.name,
                        "input_tokens": report["runs"][0]["tokens"]["input_tokens"],
                        "output_tokens": report["runs"][0]["tokens"]["output_tokens"],
                        "bundle_bytes": report["runs"][0]["privacy"]["bundle_network_bytes"],
                        "cold_correction_bytes": report["runs"][0]["privacy"]["correction_push_bytes"],
                        "online_body_bytes": report["topology_accounting"]["runs"][0]["online_all_link_serialized_body_bytes"],
                        "stage_calls": report["runs"][0]["privacy"]["inference_stage_calls"],
                    }
                    for experiment, report in candidates
                ],
            }, sort_keys=True))

"""Semantic prefix-layer ownership preserves decoder parity and stage privacy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.roles import ClientPrefixLayers
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.model_binding import RuntimeBindingError


def _experiment(root: Path, layers: int) -> Experiment:
    return Experiment(
        f"client-prefix-{layers}",
        MaskedLinearCpu(
            Model.path(str(root), model_id="prefix-layer-checkpoint"),
            placement=ClientPrefixLayers(layers) if layers else None,
        ),
        Deployment.local(root=str(root.parent)),
        ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=2),
    )


def test_semantic_schedule_bounds_client_prefix_before_issuing_stages(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "layers", num_hidden_layers=2, model_type="qwen2", tie_word_embeddings=True,
    )
    config = json.loads((root / "config.json").read_text())
    plan = lower_model(config, max_input_tokens=64, max_new_tokens=2, batch=1)
    selected = _experiment(root, 1)
    schedule = plan.runtime_schedule(selected.pipeline).to_dict()
    assert any(step["layer"] == 0 and step["executor"] == "client_linear"
               for step in schedule["prefill"]["steps"])
    assert any(step["layer"] == 1 and step["executor"] == "remote_stage"
               for step in schedule["prefill"]["steps"])
    assert selected.resolve().client_prefix_layers == 1
    for phase in ("prefill", "decode"):
        assert all(step["executor"] == "client_linear" for step in schedule[phase]["steps"]
                   if step["layer"] == 0 and step["weight_ids"])
    with pytest.raises((ValueError, RuntimeError), match="remote suffix"):
        one_layer = create_tiny_llama_checkpoint(
            tmp_path / "one", num_hidden_layers=1, model_type="qwen2", tie_word_embeddings=True,
        )
        one_plan = lower_model(json.loads((one_layer / "config.json").read_text()),
                               max_input_tokens=32, max_new_tokens=1, batch=1)
        one_plan.runtime_schedule(_experiment(one_layer, 1).pipeline)
    with pytest.raises(ValueError, match="one to eight"):
        ClientPrefixLayers(9)


@pytest.mark.integration
def test_prepared_client_prefix_parity_and_bound_remote_stage_traffic(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "layers", num_hidden_layers=2, model_type="qwen2", tie_word_embeddings=True,
    )
    observations = []
    for layers in (0, 1):
        with build_roles(_experiment(root, layers)) as topology:
            client = topology.client()
            try:
                result = client.responses.create(
                    model="prefix-layer-checkpoint",
                    input="A short public test prompt.",
                    max_output_tokens=2,
                )
                assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
                assert client.privacy_audit.plaintext_token_ids_sent == 0
                assert client.privacy_audit.inference_stage_calls > 0
                observations.append((
                    result.output_text,
                    result.usage,
                    client.privacy_audit.inference_stage_calls,
                    client.privacy_audit.bundle_network_bytes,
                    client.privacy_audit.masked_online_upload_bytes
                    + client.privacy_audit.masked_online_download_bytes,
                ))
                if layers:
                    state = client._core._transformer_state("prefix-layer-checkpoint")
                    compiled = client._core._compiled_public_decoder(
                        state, max_input_tokens=result.usage.input_tokens, max_new_tokens=2,
                    )
                    assert compiled is not None
                    local_stage = next(
                        stage for stage in state.bundle.stages.values()
                        if stage.layer_index == 0 and stage.client_weight is not None
                    )
                    local_stage.client_weight.flat[0] ^= 1
                    with pytest.raises(RuntimeBindingError, match="client-owned prefix weight"):
                        compiled.validate()
            finally:
                client.close()
    assert observations[0][:2] == observations[1][:2]
    assert observations[1][2] < observations[0][2]
    assert observations[1][3] > observations[0][3]
    assert observations[1][4] < observations[0][4]

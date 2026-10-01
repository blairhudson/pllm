"""Role ownership binds compiler executors, exact weights and prepared inventory."""

from __future__ import annotations

import json
from pathlib import Path
import os

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.roles import ClientLinearRoles
from pllm.runtime.model_binding import RuntimeBindingError
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

ATTENTION = ("attention_output", "qkv_projection")


def experiment(root: Path, roles: tuple[str, ...] = ()) -> Experiment:
    return Experiment(
        "attention-local" if roles else "ordinary-prepared",
        MaskedLinearCpu(
            Model.path(str(root), model_id="role-placement"),
            placement=ClientLinearRoles(roles) if roles else None,
        ),
        Deployment.local(root=str(root.parent)),
        ExecutionBudget(requests=3, max_input_tokens=64, max_new_tokens=2),
    )


def test_role_selection_is_canonical_and_native_schedule_keeps_mlp_remote(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(tmp_path / "roles")
    selected = experiment(root, ATTENTION)
    assert ClientLinearRoles(list(reversed(ATTENTION))) == ClientLinearRoles(ATTENTION)
    assert selected.resolve().client_linear_roles == ATTENTION
    plan = lower_model(
        json.loads((root / "config.json").read_text()),
        batch=1,
        max_input_tokens=64,
        max_new_tokens=2,
    )
    from pllm.runtime.semantic_stages import semantic_stage_role

    for phase in ("prefill", "decode"):
        ops = {op["id"]: op for op in getattr(plan, phase)["operations"]}
        for step in plan.runtime_schedule(selected.pipeline).to_dict()[phase]["steps"]:
            if step["layer"] is not None and step["weight_ids"]:
                role = semantic_stage_role(step, ops)
                assert step["executor"] == (
                    "client_linear" if role in ATTENTION else "remote_stage"
                )
    with pytest.raises((ValueError, RuntimeError), match="remote body"):
        plan.runtime_schedule(
            experiment(
                root, ("attention_output", "mlp_down", "mlp_gate_up", "qkv_projection")
            ).pipeline
        )
    for invalid in (
        (),
        ["qkv_projection", "qkv_projection"],
        ["lm_head"],
        "qkv_projection",
        [True],
    ):
        with pytest.raises(ValueError):
            ClientLinearRoles(invalid)


@pytest.mark.integration
@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_roles_preserve_two_child_prefill_decode_and_reject_weight_drift(
    tmp_path: Path, family: str
) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / family,
        model_type=family,
        qk_norm=family == "qwen3",
        with_qkv_bias=family == "qwen2",
    )
    observations = []
    for roles in ((), ATTENTION):
        with build_roles(experiment(root, roles), engine_threads=1) as topology:
            with topology.client(
                prepared_inventory_rows=1,
                background_inventory_refill=False,
                bundle_cache_mode="off",
            ) as client:
                result = client.responses.create(
                    input="Public role placement parity.", max_output_tokens=2, temperature=0
                )
                audit = client.privacy_audit
                assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
                assert audit.preparation_requests_during_online == 0
                observations.append(
                    (
                        result.output_text,
                        result.usage,
                        audit.inference_stage_calls,
                        audit.bundle_network_bytes,
                    )
                )
                if roles:
                    state = client._core._transformer_state("role-placement")
                    compiled = client._core._compiled_public_decoder(
                        state, max_input_tokens=result.usage.input_tokens, max_new_tokens=2
                    )
                    body = [
                        stage
                        for stage in state.bundle.stages.values()
                        if stage.layer_index is not None
                    ]
                    assert all(
                        (stage.client_weight is not None) == (stage.role in ATTENTION)
                        for stage in body
                    )
                    assert all(
                        (stage.seeded_profile is None) == (stage.role in ATTENTION)
                        for stage in body
                    )
                    local = next(stage for stage in body if stage.role == "qkv_projection")
                    local.client_weight.flat[0] ^= 1
                    with pytest.raises(RuntimeBindingError, match="client-owned prefix weight"):
                        compiled.validate()
    assert observations[0][:2] == observations[1][:2]
    assert observations[1][2] * 2 == observations[0][2]
    assert observations[1][3] > observations[0][3]


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("PLLM_RUN_CLIENT_LINEAR_PARITY"), reason="explicit pinned checkpoint parity"
)
def test_pinned_qwen_attention_placement_has_exact_w8a8_prefill_decode_logits() -> None:
    import asyncio
    import numpy as np
    from pllm.model_loader import resolve_model
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.semantic_stages import client_owns_linear
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    source = resolve_model(
        Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775"),
        cache_dir=os.environ.get("HF_HUB_CACHE")
        or str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"),
    )
    config = json.loads((source.path / "config.json").read_text())
    plan = lower_model(config, batch=1, max_input_tokens=64, max_new_tokens=2)
    samples = []
    fingerprints = []
    for roles in ((), ATTENTION):
        engine = MaskedTransformerEngine(
            threads=4, weight_bits=8, activation_bits=8, client_linear_roles=roles
        )
        asyncio.run(engine.load(source.manifest))
        try:
            model = engine.models[source.manifest.id]
            bundle = ClientBundle.unpack(engine.client_bundle(source.manifest.id))
            compiled = compile_runtime_model(
                plan,
                bundle,
                composition=MaskedLinearCpu(
                    Model.hf(
                        "Qwen/Qwen2.5-0.5B-Instruct",
                        revision="7ae557604adf67be50417f59c2c2f167def9a775",
                    ),
                    placement=ClientLinearRoles(roles) if roles else None,
                ),
            )
            fingerprints.append(model.manifest.metadata["body_fingerprint"])

            def remote(stage_id, activation):
                stage = model.stages[stage_id]
                assert not client_owns_linear(stage.spec, client_linear_roles=roles)
                values = quantize_activation_per_row(activation, bits=8)
                result = dequantize_matmul(
                    stage.compiled_weight.clear(values.values),
                    values.scales,
                    stage.weight.scales,
                    output_shape=values.original_shape[:-1] + (stage.spec.out_features,),
                )
                if stage.bias is not None:
                    result += stage.bias
                return np.ascontiguousarray(result, dtype=np.float32)

            observed = []
            for text in (
                "Explain why neither server can see the prompt.",
                "What makes a river flow?",
                "Calculate seven plus nine.",
            ):
                runtime = compiled.runtime(remote)
                ids = runtime.encode_prompt(text)
                _, logits, _ = runtime.prepare_ids(ids)
                observed.append(logits.copy())
                observed.append(runtime.forward_ids([int(np.argmax(logits))])[-1].copy())
            samples.append(observed)
        finally:
            asyncio.run(engine.unload(source.manifest.id))
    assert fingerprints[0] == fingerprints[1]
    for plain, placed in zip(*samples, strict=True):
        np.testing.assert_array_equal(plain, placed)

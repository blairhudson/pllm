from __future__ import annotations

import asyncio

import numpy as np
import pytest

from pllm import Model
from pllm.model_loader import resolve_model
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError


@pytest.mark.parametrize("tied,remote_head", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("prefix", [0, 1])
def test_role_residency_preserves_metadata_bundle_and_remote_math(tmp_path, tied, remote_head, prefix):
    root = create_tiny_llama_checkpoint(tmp_path / "model", tie_word_embeddings=tied)
    manifest = resolve_model(Model.path(str(root), model_id="residency")).manifest
    options = dict(threads=1, weight_bits=8, activation_bits=8, remote_output_head=remote_head,
                   client_prefix_layers=prefix, client_linear_roles=("attention_output",))
    control = MaskedTransformerEngine(**options)
    asyncio.run(control.load(manifest))
    original = control.models[manifest.id]
    for policy in ("provider", "provider_and_bundle"):
        engine = MaskedTransformerEngine(**options, weight_residency=policy)
        compiled = []
        compile_weight = engine.kernel.compile
        def track(weight):
            compiled.append(weight.shape)
            return compile_weight(weight)
        engine.kernel.compile = track
        asyncio.run(engine.load(manifest))
        model = engine.models[manifest.id]
        for key in ("body_fingerprint", "seeded_stage_commitment"):
            assert model.manifest.metadata[key] == original.manifest.metadata[key]
        expected = {sid for sid, stage in model.stages.items()
                    if (engine._provider_owns_stage(stage.spec) if policy == "provider" else
                        not (tied and sid == "token_lookup"))}
        assert len(compiled) == len(expected)
        for sid, stage in model.stages.items():
            before = original.stages[sid]
            assert stage.metadata.pack() == before.metadata.pack()
            assert stage.public_descriptor() == before.public_descriptor()
            if sid not in expected:
                assert stage.quantized_weight is None and stage.compiled_weight is None
                assert stage.compiled_cache_entry is None
                with pytest.raises(TransformerEngineError, match="not resident"):
                    _ = stage.weight
                continue
            np.testing.assert_array_equal(stage.weight.values, before.weight.values)
            if engine._provider_owns_stage(stage.spec):
                x = np.random.default_rng(37).integers(-7, 8, (2, stage.spec.in_features), dtype=np.int8)
                np.testing.assert_array_equal(stage.compiled_weight.clear(x), before.compiled_weight.clear(x))
                assert stage.output_residue_bits == before.output_residue_bits
        if policy == "provider_and_bundle":
            assert engine.client_bundle(manifest.id) == control.client_bundle(manifest.id)
        else:
            with pytest.raises(TransformerEngineError, match="not resident"):
                engine.client_bundle(manifest.id)


def test_auxiliary_token_table_is_not_retired_as_a_tied_main_table(tmp_path):
    from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
    root = create_tiny_gemma4_checkpoint(tmp_path / "gemma", num_hidden_layers=1, ple_dim=8)
    manifest = resolve_model(Model.path(str(root), model_id="auxiliary")).manifest
    control = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8,
                                     weight_residency="provider_and_bundle")
    asyncio.run(control.load(manifest))
    asyncio.run(engine.load(manifest))
    token = engine.models[manifest.id].stages["token_lookup"]
    assert token.spec.out_features > engine.models[manifest.id].stages["lm_head"].spec.in_features
    assert token.quantized_weight is not None
    assert engine.client_bundle(manifest.id) == control.client_bundle(manifest.id)


def test_pruned_storage_is_not_an_implicit_subclass_fallback():
    class CustomEngine(MaskedTransformerEngine):
        pass
    with pytest.raises(ValueError, match="public masked-linear"):
        CustomEngine(weight_residency="provider")
    with pytest.raises(ValueError, match="public masked-linear"):
        MaskedTransformerEngine(weight_residency="unknown")

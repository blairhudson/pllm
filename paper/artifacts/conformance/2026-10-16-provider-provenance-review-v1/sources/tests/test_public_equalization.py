"""Public calibration is an immutable, checkpoint-bound numeric choice."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, lower_model
from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu
from pllm.quantization import (
    PublicPerChannelEqualized,
    SymmetricPerRow,
    fit_public_equalization_profile,
)
from pllm.runtime.model_binding import RuntimeBindingError, compile_runtime_model
from pllm.runtime.reference_benchmark import _candidate_remote
from pllm.runtime.public_equalization import (
    PublicEqualizationError,
    PublicEqualizationProfile,
    load_public_equalization_profile,
    profile_path,
)
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError


def _source(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "checkpoint", num_hidden_layers=1)
    return root, Model.path(str(root), model_id="public-calibrated-tiny")


def test_profile_is_source_bound_canonical_immutable_and_checked(tmp_path):
    root, source = _source(tmp_path)
    profile = fit_public_equalization_profile(source, ((0, 3, 5), (0, 2, 4, 6)))
    restored = PublicEqualizationProfile.unpack(profile.pack(), expected_digest=profile.digest)
    assert restored.digest == profile.digest
    assert len(profile.stage_scales) == 4
    with pytest.raises(TypeError):
        profile.stage_scales["missing"] = np.ones(1, dtype=np.float32)
    with pytest.raises(ValueError):
        next(iter(profile.stage_scales.values()))[0] = 0
    with pytest.raises(PublicEqualizationError, match="digest"):
        PublicEqualizationProfile.unpack(profile.pack(), expected_digest="0" * 64)
    payload = bytearray(profile.pack())
    payload[-1] ^= 1
    with pytest.raises(PublicEqualizationError):
        PublicEqualizationProfile.unpack(bytes(payload), expected_digest=profile.digest)
    path = profile_path(root, profile.digest)
    path.write_bytes(profile.pack())
    assert path.stat().st_size < 1 << 20
    with pytest.raises(PublicEqualizationError, match="checkpoint"):
        load_public_equalization_profile(root, profile.digest, "0" * 64)


def test_calibrated_client_executes_compiled_decoder_and_rejects_downgrade(tmp_path):
    root, source = _source(tmp_path)
    profile = fit_public_equalization_profile(source, ((0, 3, 5), (0, 2, 4, 6)))
    profile_path(root, profile.digest).write_bytes(profile.pack())
    component = PublicPerChannelEqualized(profile.digest)
    client = ClientOnlyCpu(source, quantization=component)
    prepared = MaskedLinearCpu(source, quantization=component)
    for pipeline in (client, prepared):
        experiment = Experiment(
            name="Public equalized tiny decoder",
            pipeline=pipeline, deployment=Deployment.local(root="local://public-calibrated"),
            budget=ExecutionBudget(requests=1, max_input_tokens=4, max_new_tokens=1),
        )
        assert experiment.resolve().public_equalization_digest == profile.digest
    plan = lower_model((root / "config.json").read_bytes(), batch=1, max_input_tokens=4, max_new_tokens=1)
    assert plan.runtime_schedule(client).complete
    assert plan.runtime_schedule(prepared).complete

    from pllm import load_model

    manifest = load_model(source)
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, public_equalization_digest=profile.digest)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(manifest.id, placement="client"))
    assert bundle.privacy["public_equalization_digest"] == profile.digest
    assert all(
        stage.equalization_profile_digest == profile.digest
        for name, stage in bundle.stages.items() if name == stage.id and stage.role not in {"token_lookup", "lm_head"}
    )
    compiled = compile_runtime_model(plan, bundle, composition=client)
    runtime = compiled.runtime(compiled.client_linear_executor(engine))
    _, logits, _ = runtime.prepare_ids([0, 3, 5])
    assert logits.shape == (bundle.cfg["vocab_size"],)
    assert np.isfinite(logits).all()
    benchmark_remote = _candidate_remote(engine, manifest.id)
    local_remote = compiled.client_linear_executor(engine)
    for row in compiled.stage_bindings:
        if row.stage_id in {"token_lookup", "lm_head"}:
            continue
        input_values = np.linspace(-0.5, 1.25, row.in_features, dtype=np.float32).reshape(1, -1)
        np.testing.assert_array_equal(
            benchmark_remote(row.stage_id, input_values),
            local_remote(row.stage_id, input_values),
        )
    with pytest.raises(RuntimeBindingError, match="equalization|profile|fingerprint"):
        compile_runtime_model(plan, bundle, composition=ClientOnlyCpu(source))
    altered = replace(bundle.stages[next(name for name in bundle.stages if name not in {"token_lookup", "lm_head"})],
                      input_equalization=None)
    name = altered.id
    tampered = replace(bundle, stages={**bundle.stages, name: altered})
    with pytest.raises((RuntimeBindingError, TransformerEngineError, ValueError)):
        compile_runtime_model(plan, tampered, composition=client)


def test_profile_missing_or_wrong_source_fails_before_stage_import(tmp_path):
    root, source = _source(tmp_path)
    profile = fit_public_equalization_profile(source, ((0, 3, 5),))
    from pllm import load_model

    manifest = load_model(source)
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, public_equalization_digest=profile.digest)
    with pytest.raises(TransformerEngineError, match="profile"):
        asyncio.run(engine.load(manifest))
    assert not engine.models
    altered = PublicEqualizationProfile("0" * 64, profile.calibration_tokens, profile.stage_scales)
    profile_path(root, profile.digest).write_bytes(altered.pack())
    with pytest.raises(TransformerEngineError, match="digest|source"):
        asyncio.run(engine.load(manifest))
    assert not engine.models


@pytest.mark.parametrize("client_owned", [True, False])
def test_calibrated_sdk_and_gateway_follow_experiment_roles(tmp_path, client_owned):
    from fastapi.testclient import TestClient

    from pllm.runtime import build_roles

    root, source = _source(tmp_path)
    profile = fit_public_equalization_profile(source, ((0, 3, 5), (0, 2, 4, 6)))
    profile_path(root, profile.digest).write_bytes(profile.pack())
    experiment = Experiment(
        name="Calibrated tiny client" if client_owned else "Calibrated tiny prepared",
        pipeline=(ClientOnlyCpu if client_owned else MaskedLinearCpu)(
            source, quantization=PublicPerChannelEqualized(profile.digest),
        ),
        deployment=Deployment.local(root=str(tmp_path / "deployment")),
        budget=ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=2),
    )
    with build_roles(experiment) as topology:
        with topology.client() as client:
            result = client.responses.create(
                model="public-calibrated-tiny", input="A", max_output_tokens=1,
            )
            assert result.usage.input_tokens > 0
            assert (client.privacy_audit.inference_stage_calls == 0) == client_owned
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
        with TestClient(topology.gateway_app(local_api_key="equalized-test")) as gateway:
            response = gateway.post(
                "/v1/responses", headers={"Authorization": "Bearer equalized-test"},
                json={"model": "public-calibrated-tiny", "input": "B", "max_output_tokens": 1},
            )
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["input_tokens"] > 0


def test_prepared_numeric_choices_have_matched_online_stage_body_sizes(tmp_path):
    from pllm.runtime import build_roles

    root, source = _source(tmp_path)
    profile = fit_public_equalization_profile(source, ((0, 3, 5), (0, 2, 4, 6)))
    profile_path(root, profile.digest).write_bytes(profile.pack())
    bodies = []
    for index, quantization in enumerate((
        SymmetricPerRow(weight_bits=8, activation_bits=8),
        PublicPerChannelEqualized(profile.digest),
    )):
        experiment = Experiment(
            name=f"Public numeric choice {index}",
            pipeline=MaskedLinearCpu(source, quantization=quantization),
            deployment=Deployment.local(root=str(tmp_path / f"deployment-{index}")),
            budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=1),
        )
        with build_roles(experiment) as topology, topology.client() as client:
            result = client.responses.create(
                model="public-calibrated-tiny", input="A", max_output_tokens=1,
            )
            assert result.usage.input_tokens > 0
            audit = client.privacy_audit
            assert audit.plaintext_prompt_bytes_sent == 0
            bodies.append((
                audit.inference_stage_calls, audit.inference_upload_bytes,
                audit.inference_download_bytes,
            ))
    assert bodies[0] == bodies[1]
    assert bodies[0][0] > 0


@pytest.mark.parametrize("preparation", [False, True])
def test_role_cli_preserves_pinned_hub_profile_and_rejects_changed_source(tmp_path, monkeypatch, preparation):
    from fastapi.testclient import TestClient

    from pllm import load_model
    from pllm._cli.app import build_parser
    from pllm.runtime import cli
    from pllm.runtime.hf_hub import ResolvedModelSource

    commit = "a" * 40
    cache = tmp_path / "hub"
    root = create_tiny_llama_checkpoint(
        cache / "models--public--tiny" / "snapshots" / commit, num_hidden_layers=1,
    )
    calls = []

    def resolve(source, **kwargs):
        calls.append((source, kwargs))
        return ResolvedModelSource(
            path=root, model_id=kwargs["model_id"] or source,
            repo_id=source, revision=kwargs["revision"],
        )

    monkeypatch.setattr("pllm.model_loader.resolve_huggingface_source", resolve)
    source = Model.hf("public/tiny", model_id="hub-profile", revision=commit)
    profile = fit_public_equalization_profile(source, ((0, 3, 5),))
    profile_path(root, profile.digest).write_bytes(profile.pack())
    assert profile.source_lock_digest != load_model(Model.path(str(root))).source_lock_digest
    engines = []

    def engine(**kwargs):
        value = MaskedTransformerEngine(**kwargs)
        engines.append(value)
        return value

    def serve(app, **kwargs):
        with TestClient(app):
            loaded = engines[-1].models["hub-profile"]
            assert loaded.manifest.source_lock_digest == profile.source_lock_digest

    monkeypatch.setattr(cli, "MaskedTransformerEngine", engine)
    monkeypatch.setattr(cli.uvicorn, "run", serve)
    monkeypatch.setattr("pllm.runtime.telemetry.configure_telemetry", lambda *_: None)
    args = build_parser().parse_args([
        "serve", "preparation" if preparation else "inference",
        "--model", source.source, "--model-id", "hub-profile", "--revision", commit,
        "--hf-cache-dir", str(cache), "--public-equalization-digest", profile.digest,
        *(["--inference-url", "http://127.0.0.1:9000", "--push-api-key", "test-push"]
          if preparation else ["--provider-push-api-key", "test-push"]),
    ])
    cli.run_server(args, preparation=preparation)
    assert calls[-1][0] == source.source
    assert calls[-1][1]["revision"] == commit
    assert calls[-1][1]["local_files_only"] is True
    assert calls[-1][1]["cache_dir"] == str(cache)

    # The second resolve must still verify files, not trust the pre-launch lock.
    def mutate_before_start(app, **kwargs):
        path = root / "model.safetensors"
        value = bytearray(path.read_bytes())
        value[-1] ^= 1
        path.write_bytes(value)
        with TestClient(app):
            pytest.fail("changed source reached ready state")

    monkeypatch.setattr(cli.uvicorn, "run", mutate_before_start)
    args.public_equalization_digest = None
    with pytest.raises(ValueError, match="source lock changed"):
        cli.run_server(args, preparation=preparation)
    assert not engines[-1].models

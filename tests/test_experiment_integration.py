from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from pllm import (
    Cpu,
    Deployment,
    ExecutionBudget,
    Experiment,
    ExperimentProfile,
    MaskedLinearCpu,
    Model,
    OpenAI,
    ProprietaryGuarded,
    _native,
)
from pllm.runtime.client import ProtocolError, RuntimeClient


pytestmark = pytest.mark.rust


def baseline_experiment(model: str = "model-a") -> Experiment:
    return Experiment(
        name="baseline",
        pipeline=MaskedLinearCpu(
            Model(model),
            kernels=Cpu(threads=4),
        ),
        deployment=Deployment.local(root=".pllm/baseline"),
        budget=ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=4),
    )


def inert_client(origin: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    return httpx.Client(base_url=origin, transport=httpx.MockTransport(handler))


def test_profile_is_native_resolved_and_immutable():
    experiment = baseline_experiment()
    profile = experiment.resolve()
    native = _native.resolve_experiment(experiment.canonical_bytes())

    assert isinstance(profile, ExperimentProfile)
    assert profile.model == "model-a"
    assert profile.configuration_digest == experiment.configuration_digest()
    assert profile.canonical_composition == (
        b'{"components":{"inference":{"component":"pllm/inference","params":{}},'
        b'"kernels":{"component":"pllm/cpu","params":{"threads":4}},'
        b'"linear":{"component":"pllm/masked-linear","params":{}},'
        b'"preparation":{"component":"pllm/model-aware-corrections","params":{}}},'
        b'"model":{"source":"model-a"}}'
    )
    assert profile.composition_digest == experiment.pipeline.digest()
    with pytest.raises(AttributeError):
        profile.model = "model-b"
    with pytest.raises(AttributeError):
        native.canonical_composition = b"{}"
    with pytest.raises(TypeError):
        _native.ResolvedExperimentComposition()


def test_openai_experiment_locks_default_and_rejects_request_drift():
    http = inert_client("https://inference.example")
    client = OpenAI(
        base_url="https://inference.example",
        api_key="inference-key",
        default_model="model-a",
        http_client=http,
        experiment=baseline_experiment(),
    )
    try:
        assert client._core.default_model == "model-a"
        client._core.default_model = "model-b"
        assert client._core.experiment.model == "model-a"
        with pytest.raises(ValueError, match="request model conflicts"):
            client.responses.create(model="model-b", input="secret")
    finally:
        client.close()
        http.close()

    with pytest.raises(ValueError, match="default_model conflicts"):
        RuntimeClient(
            base_url="https://inference.example",
            api_key="inference-key",
            default_model="model-b",
            http_client=inert_client("https://inference.example"),
            experiment=baseline_experiment(),
        )


def test_one_role_experiment_clears_inherited_preparation_settings(monkeypatch):
    experiment = Experiment(
        name="one-role-client",
        pipeline=ProprietaryGuarded(Model.tiny()),
        deployment=Deployment.local(root=".pllm/one-role"),
        budget=ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=1),
    )
    monkeypatch.setenv("PLLM_PREPARATION_BASE_URL", "https://stale-preparation.example")
    monkeypatch.setenv("PLLM_PREPARATION_API_KEY", "stale-key")
    http = inert_client("https://inference.example")

    client = OpenAI(
        base_url="https://inference.example",
        api_key="secret",
        experiment=experiment,
        http_client=http,
    )
    try:
        assert client._core.preparation_http is None
    finally:
        client.close()
        http.close()


def test_one_role_experiment_rejects_explicit_preparation_settings():
    experiment = Experiment(
        name="one-role-client",
        pipeline=ProprietaryGuarded(Model.tiny()),
        deployment=Deployment.local(root=".pllm/one-role"),
        budget=ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=1),
    )
    http = inert_client("https://inference.example")

    with pytest.raises(ValueError, match="no preparation role"):
        OpenAI(
            base_url="https://inference.example",
            api_key="secret",
            experiment=experiment,
            preparation_base_url="https://preparation.example",
            preparation_api_key="preparation-key",
            http_client=http,
        )
    http.close()


def test_experiment_rejects_nonpublic_transformer_runtime():
    http = inert_client("https://inference.example")
    core = RuntimeClient(
        base_url="https://inference.example",
        api_key="inference-key",
        http_client=http,
        experiment=baseline_experiment(),
    )
    core._client_bundle_descriptor = lambda _model: {
        "sha256": "0" * 64,
        "metadata": {
            "client_runtime": "direct_fhe_transformer_v1",
            "privacy_mode": "proprietary",
        },
    }
    try:
        with pytest.raises(ProtocolError, match="runtime contract"):
            core._transformer_state("model-a")
    finally:
        core.close()
        http.close()


def test_experiment_requires_seeded_inventory_preparation(monkeypatch):
    model = "model-a"
    commitment = {
        "body_fingerprint": "body",
        "stage_commitment": "stages",
        "architecture": "tiny",
        "stage_count": 1,
    }

    def inference_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": model, "runtime": commitment}]})

    def preparation_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": model,
                        "preparation": {
                            **commitment,
                            "protocol": "other/v1",
                            "weight_bits": 4,
                            "activation_bits": 4,
                        },
                    }
                ]
            },
        )

    inference = httpx.Client(
        base_url="https://inference.example", transport=httpx.MockTransport(inference_handler)
    )
    preparation = httpx.Client(
        base_url="https://preparation.example", transport=httpx.MockTransport(preparation_handler)
    )
    core = RuntimeClient(
        base_url="https://inference.example",
        api_key="inference-key",
        preparation_api_key="preparation-key",
        http_client=inference,
        preparation_http_client=preparation,
        experiment=baseline_experiment(),
    )
    bundle = SimpleNamespace(
        privacy={
            "mode": "public",
            "protocol": "masked_w4a4",
            "body_fingerprint": "body",
            "stage_commitment": "stages",
            "weight_bits": 4,
            "activation_bits": 4,
        },
        manifest={"architecture": "tiny", "stages": ["linear"]},
        stages={
            "linear": SimpleNamespace(
                id="linear", client_weight=None, weight_digest="weight", seeded_profile={}
            )
        },
    )
    monkeypatch.setattr(
        core,
        "_client_bundle_descriptor",
        lambda _model: {
            "sha256": "0" * 64,
            "metadata": {"client_runtime": "masked_transformer_v1", "privacy_mode": "public"},
        },
    )
    monkeypatch.setattr(core, "_load_client_bundle_record", lambda *_args: (bundle, "0" * 64))
    try:
        with pytest.raises(ProtocolError, match="commitments do not match"):
            core._transformer_state(model)
    finally:
        core.close()
        inference.close()
        preparation.close()

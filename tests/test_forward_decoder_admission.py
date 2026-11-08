"""Forward intent must fail before decoder resolution or runtime side effects."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import pytest

from pllm._decoder_admission import (
    require_decoder_experiment,
    require_decoder_pipeline,
    require_decoder_profile,
    require_decoder_source_record,
)
from pllm.configuration import (
    ConfigurationError, Deployment, ExecutionBudget, Experiment, Model,
    ForwardBudget,
)
from pllm.numeric import SourceFloat32
from pllm.profiles import ClientOnlyCpu


def forward_experiment():
    return Experiment(
        "forward-admission",
        ClientOnlyCpu(Model("example/model"), numeric=SourceFloat32()),
        Deployment.local(root=".pllm/forward-admission"),
        ForwardBudget(1, 8),
    )


def forbidden(*args, **kwargs):
    raise AssertionError("decoder side effect reached before forward admission")


@pytest.fixture
def decoder_sentinels(monkeypatch):
    from pllm import _native
    from pllm import model_loader
    from pllm.runtime import client, transformer_engine
    from pllm.settings import ClientSettings

    monkeypatch.setattr(_native, "resolve_experiment", forbidden)
    monkeypatch.setattr(ClientSettings, "load", forbidden)
    monkeypatch.setattr(model_loader, "resolve_model", forbidden)
    monkeypatch.setattr(client.httpx, "Client", forbidden)
    monkeypatch.setattr(client.httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(client.ClientBundle, "tokenizer", forbidden)
    monkeypatch.setattr(transformer_engine, "MaskedTransformerEngine", forbidden)


@pytest.mark.parametrize("bypass_budget", [False, True])
@pytest.mark.parametrize("entry", [
    "resolve", "openai", "async-openai", "runtime-client", "roles", "adapter",
])
def test_public_decoder_entry_points_reject_before_side_effects(
    entry, bypass_budget, decoder_sentinels,
):
    from pllm.runtime.client import AsyncOpenAI, OpenAI, RuntimeClient
    from pllm.runtime.servers import build_roles, load_role_adapter

    experiment = forward_experiment()
    before = experiment.canonical_bytes()
    if bypass_budget:
        # Configuration-only corruption: never forge live authentication state.
        experiment = copy.copy(experiment)
        object.__setattr__(experiment, "budget", ExecutionBudget(1, 8, 2))
        before = experiment.canonical_bytes()
    with pytest.raises(ConfigurationError, match="forward-only"):
        if entry == "resolve":
            experiment.resolve()
        elif entry == "openai":
            OpenAI(experiment=experiment, network=object())
        elif entry == "async-openai":
            AsyncOpenAI(experiment=experiment, network=object())
        elif entry == "runtime-client":
            RuntimeClient(experiment=experiment, base_url="invalid", api_key="test")
        elif entry == "roles":
            build_roles(experiment, wan=object())
        else:
            asyncio.run(load_role_adapter(experiment, "inference", "test"))
    assert experiment.canonical_bytes() == before


@pytest.mark.parametrize("entry", ["openai", "async-openai", "runtime-client", "guard"])
def test_profile_checks_canonical_numeric_intent_before_sdk(entry, monkeypatch):
    from pllm.runtime.client import AsyncOpenAI, OpenAI, RuntimeClient
    from pllm.settings import ClientSettings

    decoder = Experiment(
        "decoder", ClientOnlyCpu(Model("example/model")),
        Deployment.local(root=".pllm/decoder"), ExecutionBudget(1, 8, 2),
    )
    profile = copy.copy(decoder.resolve())
    object.__setattr__(profile, "canonical_composition", forward_experiment().pipeline.canonical_bytes())
    # Keep genuine resolved decoder runtime fields; canonical intent wins.
    monkeypatch.setattr(ClientSettings, "load", forbidden)
    with pytest.raises(ConfigurationError, match="forward-only"):
        if entry == "openai":
            OpenAI(experiment=profile)
        elif entry == "async-openai":
            AsyncOpenAI(experiment=profile)
        elif entry == "runtime-client":
            RuntimeClient(experiment=profile, base_url="invalid", api_key="test")
        else:
            require_decoder_profile(profile)


@pytest.mark.parametrize("marker", ["numeric", "other-slot"])
def test_guard_uses_slot_or_component_identity_without_resolution(marker, monkeypatch):
    from pllm import _native
    from pllm.profiles import resolve_decoder_runtime_composition

    pipeline = copy.copy(ClientOnlyCpu(Model("example/model")))
    components = dict(pipeline.components)
    components[marker] = pipeline.components["kernels"] if marker == "numeric" else SourceFloat32()
    object.__setattr__(pipeline, "components", MappingProxyType(components))
    monkeypatch.setattr(_native, "resolve_experiment", forbidden)
    for guard in (require_decoder_pipeline, resolve_decoder_runtime_composition):
        with pytest.raises(ConfigurationError, match="forward-only"):
            guard(pipeline)


def test_build_roles_pipeline_rejects_before_wan_validation(decoder_sentinels):
    from pllm.runtime.servers import build_roles

    with pytest.raises(ConfigurationError, match="forward-only"):
        build_roles(forward_experiment().pipeline, wan=object())


def test_forward_budget_alone_rejects_without_resolving_pipeline():
    experiment = copy.copy(forward_experiment())
    object.__setattr__(experiment, "pipeline", ClientOnlyCpu(Model("example/model")))
    with pytest.raises(ConfigurationError, match="forward-only"):
        require_decoder_experiment(experiment)


def test_settings_derived_experiment_rejects_before_network(tmp_path, monkeypatch):
    from pllm.runtime.client import OpenAI
    from pllm.settings import ClientSettings

    path = tmp_path / "forward.json"
    path.write_text(forward_experiment().to_json())
    settings = ClientSettings(experiment=str(path))
    monkeypatch.setattr(ClientSettings, "load", lambda: settings)
    with pytest.raises(ConfigurationError, match="forward-only"):
        OpenAI(network=object())


def declared_record(kind):
    experiment = forward_experiment()
    if kind == "schema":
        return {"schema": "pllm.experiment.forward.v1", "source": "example/model"}
    if kind == "experiment":
        return {"source": "example/model", "experiment": experiment.to_spec()}
    if kind == "pipeline":
        return {"source": "example/model", "pipeline": experiment.pipeline.to_spec()}
    if kind == "other-slot":
        pipeline = ClientOnlyCpu(Model("example/model")).to_spec()
        pipeline["components"]["other"] = SourceFloat32().to_spec()
        return {"source": "example/model", "pipeline": pipeline}
    return experiment.to_spec()


@pytest.mark.parametrize("kind", ["schema", "experiment", "pipeline", "other-slot", "direct"])
@pytest.mark.parametrize("entry", ["gateway", "preparation", "import"])
def test_serialized_forward_records_prechecked_before_service_setup(kind, entry, monkeypatch):
    from pllm import model_loader
    from pllm.runtime import preparation_server, server
    from pllm.runtime.config import GatewayConfig

    record = declared_record(kind)
    before = copy.deepcopy(record)
    config = GatewayConfig(engine_models=({"source": "ordinary/model"}, record))
    monkeypatch.setattr(server.BackendRegistry, "from_config", forbidden)
    monkeypatch.setattr(server, "BFVCorrelationServer", forbidden)
    monkeypatch.setattr(server, "CorrectionRendezvous", forbidden)
    monkeypatch.setattr(preparation_server, "PreparationSessionRegistry", forbidden)
    monkeypatch.setattr(preparation_server.httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(model_loader.Model, "from_spec", forbidden)
    with pytest.raises(ConfigurationError, match="forward-only"):
        if entry == "gateway":
            server.create_app(config)
        elif entry == "preparation":
            preparation_server.create_preparation_app(config, None)
        else:
            model_loader.model_from_runtime_spec(record)
    assert record == before


@pytest.mark.parametrize("source", ["bert", "roberta", "example/source-float32"])
def test_bare_checkpoint_labels_do_not_declare_numeric_intent(source):
    from pllm.model_loader import model_from_runtime_spec

    record = {"source": source}
    require_decoder_source_record(record)
    assert model_from_runtime_spec(record) == Model(source)
    # Unrelated metadata is not an execution envelope.
    require_decoder_source_record({"source": source, "metadata": declared_record("pipeline")})


def test_nonforward_guard_preserves_composition_and_serialization():
    experiment = Experiment(
        "decoder", ClientOnlyCpu(Model("example/model")),
        Deployment.local(root=".pllm/decoder"), ExecutionBudget(1, 8, 2),
    )
    before = experiment.canonical_bytes()
    require_decoder_experiment(experiment)
    require_decoder_pipeline(experiment.pipeline)
    require_decoder_source_record(experiment.to_spec())
    assert experiment.canonical_bytes() == before


def test_compile_rejects_forward_pipeline_with_genuine_plan_and_bundle(tmp_path, monkeypatch):
    from pllm import _native
    from pllm.runtime.model_binding import compile_runtime_model
    from test_runtime_model_binding import _bundle, _plan

    engine, bundle, config = _bundle(tmp_path)
    plan = _plan(config)
    monkeypatch.setattr(_native, "resolve_experiment", forbidden)
    monkeypatch.setattr(_native, "decoder_coverage", forbidden)
    monkeypatch.setattr(_native, "decoder_runtime_schedule", forbidden)
    monkeypatch.setattr(plan.__class__, "coverage", forbidden)
    with pytest.raises(ConfigurationError, match="forward-only"):
        compile_runtime_model(plan, bundle, composition=forward_experiment().pipeline)


@dataclass(frozen=True)
class _SourceReadProbe:
    """Policy collaborator probe, not a replacement for SourceReadPolicy tests."""

    max_files: int = 8
    max_name_bytes: int = 128
    max_metadata_bytes: int = 4096
    max_file_bytes: int = 4096
    calls: list = field(default_factory=list)

    def read_bytes(self, path, *, limit=None):
        limit = self.max_metadata_bytes if limit is None else limit
        self.calls.append(("bytes", path.name, limit))
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("policy byte limit")
        return raw

    def read_json(self, path, *, limit=None):
        self.calls.append(("json", path.name, limit))
        return json.loads(self.read_bytes(path, limit=limit))

    def hash_file(self, path):
        self.calls.append(("hash", path.name))
        return hashlib.sha256(self.read_bytes(path, limit=self.max_file_bytes)).hexdigest()


def _policy_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    raw = b'{"model_type":"bert"}'
    (root / "config.json").write_bytes(raw)
    metadata = {
        "schema": 1, "repo_id": "example/model", "revision": None, "commit": None,
        "files": [{"path": "config.json", "size": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}],
    }
    (root / ".pllm-model.json").write_text(json.dumps(metadata))
    return root, Model(str(root)), raw


def test_source_lock_policy_preserves_legacy_bytes_and_uses_bounded_reader(tmp_path, monkeypatch):
    from pathlib import Path
    from pllm.model_loader import _sha256_file, _source_lock

    root, model, raw = _policy_source(tmp_path)
    legacy = _source_lock(root, model)
    policy = _SourceReadProbe()
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    assert _source_lock(root, model, read_policy=policy) == legacy
    assert policy.calls.count(("json", ".pllm-model.json", policy.max_metadata_bytes)) == 1
    assert policy.calls.count(("bytes", ".pllm-model.json", policy.max_metadata_bytes)) == 1
    assert ("hash", "config.json") in policy.calls
    assert _sha256_file(root / "config.json", read_policy=policy) == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("limit", ["max_files", "max_name_bytes", "max_metadata_bytes"])
def test_source_inventory_admits_before_retaining_unbounded_names(tmp_path, limit):
    from pllm.model_loader import _source_files
    from pllm.runtime.loaders import ModelLoadError

    root = tmp_path / "source"
    root.mkdir()
    for index in range(3):
        (root / f"file-{index}.json").write_bytes(b"{}")
    policy = replace(_SourceReadProbe(), **{limit: 1})
    with pytest.raises(ModelLoadError, match="inventory exceeds"):
        _source_files(root, read_policy=policy)
    assert not policy.calls


def test_source_hash_policy_rejects_actual_bytes_and_default_is_unchanged(tmp_path):
    from pllm.model_loader import _sha256_file, _source_lock

    root, model, raw = _policy_source(tmp_path)
    policy = replace(_SourceReadProbe(), max_file_bytes=1)
    with pytest.raises(ValueError, match="policy byte limit"):
        _source_lock(root, model, read_policy=policy)
    assert _sha256_file(root / "config.json") == hashlib.sha256(raw).hexdigest()
    assert _source_lock(root, model) == _source_lock(root, model, read_policy=None)


def test_resolver_metadata_delegates_strict_json_rejection_once(tmp_path, monkeypatch):
    from pllm.model_loader import _resolver_metadata
    from pllm.runtime.loaders import ModelLoadError

    root, _, _ = _policy_source(tmp_path)
    calls = []

    def reject_json(self, path, *, limit=None):
        calls.append((path.name, limit))
        raise ValueError("strict duplicate/nonfinite JSON rejection")

    monkeypatch.setattr(_SourceReadProbe, "read_json", reject_json)
    policy = _SourceReadProbe()
    with pytest.raises(ModelLoadError, match="invalid .pllm-model.json"):
        _resolver_metadata(root, read_policy=policy)
    assert calls == [(".pllm-model.json", policy.max_metadata_bytes)]


def test_strict_git_blob_check_keeps_same_sha256_identity(tmp_path):
    from pllm.model_loader import _file_digest

    raw = b"checkpoint bytes"
    path = tmp_path / "weights.safetensors"
    path.write_bytes(raw)
    etag = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
    assert _file_digest(path, expected_sha1=etag, read_policy=_SourceReadProbe()) == (
        _file_digest(path, expected_sha1=etag)
    )

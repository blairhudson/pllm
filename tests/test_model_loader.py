from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import pllm
import pllm.models as public_models
from pllm.model_loader import (
    ModelLoadError,
    coerce_model,
    model_from_runtime_spec,
    resolve_model,
)
from pllm.runtime.hf_hub import ResolvedModelSource
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint


def test_public_loader_locks_checkpoint_contents_without_path_identity(tmp_path: Path) -> None:
    roots = [
        create_tiny_llama_checkpoint(tmp_path / name / "model", seed=31)
        for name in ("first", "second")
    ]
    first = pllm.load_model(pllm.Model.path(str(roots[0])), cache_dir=tmp_path / "cache")
    second = pllm.load_model(pllm.Model.path(str(roots[1])), cache_dir=tmp_path / "other-cache")

    assert isinstance(first, pllm.ModelManifest)
    assert public_models.ModelLoadError is pllm.ModelLoadError
    assert public_models.ModelManifest is pllm.ModelManifest
    assert public_models.load_model is pllm.load_model
    assert len(first.checkpoint_digest or "") == 64
    assert len(first.source_lock_digest or "") == 64
    assert first.checkpoint_digest == second.checkpoint_digest
    assert first.source_lock_digest == second.source_lock_digest
    assert first.metadata["model_spec"] == {
        "source": str(roots[0]),
        "local_files_only": True,
    }
    lock = first.metadata["source_lock"]
    schema = json.loads(
        Path("schemas/model-source-lock.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(lock)
    serialized = json.dumps(lock, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert lock["checkpoint_digest"] == first.checkpoint_digest
    weight = next(row for row in lock["files"] if row["path"] == "model.safetensors")
    assert weight["sha256"] == hashlib.sha256((roots[0] / "model.safetensors").read_bytes()).hexdigest()

    path = roots[1] / "model.safetensors"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    changed = pllm.load_model(pllm.Model.path(str(roots[1])))
    assert changed.checkpoint_digest != first.checkpoint_digest
    assert changed.source_lock_digest != first.source_lock_digest


def test_public_loader_binds_model_id_and_source_format(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)
    model = pllm.Model.path(str(root), format="safetensors", model_id="org/tiny")
    resolved = resolve_model(model)
    assert resolved.model is model
    assert resolved.path == root.resolve()
    assert resolved.manifest.id == "org/tiny"
    assert resolved.manifest.source_format == "safetensors"
    assert resolved.manifest.checkpoint_digest == resolved.checkpoint_digest
    assert resolved.manifest.source_lock_digest == resolved.source_lock_digest
    assert resolved.manifest.metadata["config_only"] is False


def test_remote_huggingface_model_uses_the_canonical_resolver(tmp_path: Path, monkeypatch) -> None:
    root = create_tiny_llama_checkpoint(tmp_path / "resolved", num_hidden_layers=1)
    calls = []

    def fake(source, **kwargs):
        calls.append((source, kwargs))
        return ResolvedModelSource(
            path=root.resolve(),
            model_id=kwargs["model_id"] or source,
            repo_id=source,
            revision=kwargs["revision"],
        )

    monkeypatch.setattr("pllm.model_loader.resolve_huggingface_source", fake)
    model = pllm.Model.hf(
        "org/model",
        model_id="public-id",
        revision="abc123",
        local_files_only=True,
    )
    manifest = pllm.load_model(model, token="secret", cache_dir=tmp_path / "cache")
    assert manifest.id == "public-id"
    assert calls == [
        (
            "org/model",
            {
                "model_id": "public-id",
                "revision": "abc123",
                "token": "secret",
                "cache_dir": tmp_path / "cache",
                "local_files_only": True,
            },
        )
    ]
    serialized = json.dumps(manifest.to_dict())
    assert "secret" not in serialized
    assert str(tmp_path / "cache") not in serialized


def test_runtime_model_request_accepts_canonical_and_legacy_shapes() -> None:
    canonical = model_from_runtime_spec({
        "engine": "masked",
        "kind": "huggingface",
        "source": "org/model",
        "model_id": "public",
        "revision": "abc",
    })
    assert canonical == pllm.Model.hf("org/model", model_id="public", revision="abc")
    assert model_from_runtime_spec({"kind": "huggingface", "repo_id": "org/model"}) == pllm.Model(
        "org/model"
    )
    assert model_from_runtime_spec({"kind": "gguf", "path": "/models/model.gguf"}) == pllm.Model.path(
        "/models/model.gguf", format="gguf"
    )
    assert model_from_runtime_spec({"kind": "ollama", "name": "qwen:latest"}) == pllm.Model.ollama(
        "qwen:latest"
    )
    assert model_from_runtime_spec({"model": {"source": "org/model"}}) == pllm.Model("org/model")
    for invalid in (
        {},
        {"source": "one", "path": "two"},
        {"repo_id": "one", "path": "two"},
        {"model": "org/model"},
        {"model": {"source": "org/model"}, "model_id": "outer"},
        {
            "kind": "ollama",
            "source": "qwen",
            "endpoint": "http://one",
            "base_url": "http://two",
        },
    ):
        with pytest.raises(pllm.ConfigurationError):
            model_from_runtime_spec(invalid)


def test_synchronous_ollama_loader_keeps_credentials_operational(monkeypatch) -> None:
    calls = []

    async def fake(base_url, name, *, api_key=None, client=None):
        calls.append((base_url, name, api_key, client))
        return pllm.ModelManifest(
            id=name,
            architecture="llama",
            source_format="ollama",
            source=f"{base_url}/{name}",
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            context_length=32,
        )

    monkeypatch.setattr("pllm.model_loader.load_ollama_model", fake)
    model = pllm.Model.ollama(
        "qwen:latest",
        endpoint="https://ollama.example",
        model_id="public-model",
    )
    manifest = pllm.load_model(model, api_key="secret")
    assert manifest.id == "public-model"
    assert manifest.metadata["model_spec"] == model.to_spec()
    assert "secret" not in json.dumps(manifest.to_dict())
    assert calls == [("https://ollama.example", "qwen:latest", "secret", None)]

    async def nested():
        with pytest.raises(ModelLoadError, match="running event loop"):
            pllm.load_model(model)

    asyncio.run(nested())


def test_config_only_sources_are_explicitly_internal(tmp_path: Path) -> None:
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = pllm.Model.path(str(tmp_path))
    with pytest.raises(ModelLoadError):
        pllm.load_model(model)
    resolved = resolve_model(model, allow_config_only=True)
    assert resolved.manifest.metadata["config_only"] is True
    assert resolved.checkpoint_digest is not None


def test_huggingface_pull_entry_is_unique() -> None:
    root = Path("python/pllm")
    users = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "resolve_huggingface_source" in path.read_text(encoding="utf-8")
    )
    assert users == ["model_loader.py", "runtime/hf_hub.py"]


def test_recorded_source_digest_mismatch_and_unsupported_kind_fail_closed(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)
    weight = root / "model.safetensors"
    (root / ".pllm-model.json").write_text(
        json.dumps({
            "schema": 1,
            "repo_id": "org/model",
            "revision": "main",
            "commit": "abc",
            "files": [
                {
                    "path": weight.name,
                    "size": weight.stat().st_size,
                    "sha256": "0" * 64,
                }
            ],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError, match="digest mismatch"):
        pllm.load_model(pllm.Model.path(str(root)))
    (root / ".pllm-model.json").write_text(
        json.dumps({
            "schema": 1,
            "repo_id": "org/model",
            "revision": "main",
            "commit": "abc",
            "files": [{"path": "missing.json", "size": 0, "sha256": None}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError, match="missing recorded files"):
        pllm.load_model(pllm.Model.path(str(root)))
    with pytest.raises(ModelLoadError, match="no public loader"):
        pllm.load_model(pllm.Model.tiny())
    with pytest.raises(TypeError):
        coerce_model(object())

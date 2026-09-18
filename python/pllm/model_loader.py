from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pllm.configuration import ConfigurationError, Model
from pllm.runtime.hf_hub import ResolvedModelSource, resolve_huggingface_source
from pllm.runtime.loaders import (
    ModelLoadError,
    load_gguf,
    load_hf_directory,
    load_mlx_directory,
    load_ollama_model,
)
from pllm.runtime.models import ModelManifest

_SOURCE_LOCK_SCHEMA = "pllm.model_source_lock.v1"
_SOURCE_LOCK_DOMAIN = b"pllm.model_source_lock.v1\0"
_CHECKPOINT_DOMAIN = b"pllm.checkpoint.v1\0"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DIRECTORY_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "vocab.*",
)


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    model: Model
    manifest: ModelManifest
    path: Path | None
    checkpoint_digest: str | None
    source_lock_digest: str | None


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + _canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _resolver_metadata(root: Path) -> dict[str, Any]:
    path = root / ".pllm-model.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ModelLoadError("invalid .pllm-model.json") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "repo_id", "revision", "commit", "files"}
        or value.get("schema") != 1
        or not isinstance(value.get("files"), list)
        or any(
            item is not None and not isinstance(item, str)
            for item in (value.get("repo_id"), value.get("revision"), value.get("commit"))
        )
    ):
        raise ModelLoadError("invalid .pllm-model.json")
    paths = []
    for row in value["files"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "size", "sha256"}
            or not isinstance(row.get("path"), str)
            or type(row.get("size")) is not int
            or row["size"] < 0
            or row.get("sha256") is not None
            and (
                not isinstance(row["sha256"], str)
                or _DIGEST.fullmatch(row["sha256"]) is None
            )
        ):
            raise ModelLoadError("invalid .pllm-model.json")
        paths.append(row["path"])
    if len(paths) != len(set(paths)):
        raise ModelLoadError("invalid .pllm-model.json")
    return value


def _source_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    files: set[Path] = set()
    for pattern in _DIRECTORY_PATTERNS:
        files.update(
            candidate
            for candidate in path.glob(pattern)
            if candidate.is_file() and candidate.name != ".pllm-model.json"
        )
    return tuple(sorted(files, key=lambda candidate: candidate.name))


def _source_lock(path: Path, model: Model) -> tuple[str, str, dict[str, Any]]:
    root = path if path.is_dir() else path.parent
    resolver = _resolver_metadata(root)
    recorded = {
        str(row.get("path")): row
        for row in resolver.get("files", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    sources = _source_files(path)
    actual_paths = {
        source.name if path.is_file() else source.relative_to(path).as_posix()
        for source in sources
    }
    missing = set(recorded) - actual_paths
    if missing:
        raise ModelLoadError(f"model source is missing recorded files: {sorted(missing)}")
    rows = []
    for source in sources:
        relative = source.name if path.is_file() else source.relative_to(path).as_posix()
        before = source.stat()
        size = before.st_size
        record = recorded.get(relative, {})
        candidate = record.get("sha256")
        sha256 = _file_digest(source)
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ModelLoadError(f"model source changed while hashing {relative}")
        if candidate is not None and (
            not isinstance(candidate, str)
            or _DIGEST.fullmatch(candidate) is None
            or record.get("size") != size
            or candidate != sha256
        ):
            raise ModelLoadError(f"model source digest mismatch for {relative}")
        rows.append({"path": relative, "size": size, "sha256": sha256})
    if not rows:
        raise ModelLoadError(f"model source has no lockable files: {path}")
    checkpoint = _digest(_CHECKPOINT_DOMAIN, {"files": rows})
    lock = {
        "schema": _SOURCE_LOCK_SCHEMA,
        "kind": model.kind,
        "repo_id": resolver.get("repo_id"),
        "revision": resolver.get("revision") or model.revision,
        "commit": resolver.get("commit"),
        "files": rows,
        "checkpoint_digest": checkpoint,
    }
    return checkpoint, _digest(_SOURCE_LOCK_DOMAIN, lock), lock


def coerce_model(value: Model | str | Mapping[str, Any]) -> Model:
    if isinstance(value, Model):
        return value
    if type(value) is str:
        return Model(value)
    if isinstance(value, Mapping):
        return Model.from_spec(value)
    raise TypeError("model must be a Model, source string, or model specification")


def model_from_runtime_spec(value: Mapping[str, Any]) -> Model:
    if not isinstance(value, Mapping):
        raise ConfigurationError("runtime model request must be a mapping")
    nested = value.get("model")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ConfigurationError("runtime model field must be a mapping")
        duplicates = set(value) & {
            "source",
            "kind",
            "model_id",
            "revision",
            "local_files_only",
            "endpoint",
            "repo_id",
            "path",
            "name",
            "base_url",
        }
        if duplicates:
            raise ConfigurationError(
                f"runtime model request mixes nested and flat fields: {sorted(duplicates)}"
            )
        model = dict(nested)
    else:
        model = {
            key: value[key]
            for key in (
                "source",
                "kind",
                "model_id",
                "revision",
                "local_files_only",
                "endpoint",
            )
            if key in value
        }
    aliases = [
        ("repo_id", value.get("repo_id")),
        ("path", value.get("path")),
        ("name", value.get("name")),
    ]
    supplied = [(name, source) for name, source in aliases if source not in (None, "")]
    if "source" not in model:
        if len(supplied) != 1:
            raise ConfigurationError("runtime model request requires exactly one source")
        alias, source = supplied[0]
        model["source"] = source
        if "kind" not in model and alias == "name":
            model["kind"] = "ollama"
    elif supplied and any(source != model["source"] for _, source in supplied):
        raise ConfigurationError("runtime model request has conflicting sources")
    base_url = value.get("base_url")
    if "endpoint" not in model and base_url is not None:
        model["endpoint"] = base_url
    elif base_url is not None and base_url != model.get("endpoint"):
        raise ConfigurationError("runtime model request has conflicting endpoints")
    return Model.from_spec(model)


def _resolve_model(
    value: Model | str | Mapping[str, Any],
    *,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    api_key: str | None = None,
    allow_config_only: bool = False,
) -> ResolvedModel:
    model = coerce_model(value)
    path: Path | None
    if model.kind in {"huggingface", "safetensors", "vllm"}:
        local = Path(model.source).expanduser()
        if allow_config_only and local.is_dir() and (local / "config.json").is_file():
            path = local.resolve()
            model_id = model.model_id or path.name
            manifest = load_hf_directory(
                path,
                model_id=model_id,
                source_format=model.kind,
            )
        else:
            resolved: ResolvedModelSource = resolve_huggingface_source(
                model.source,
                model_id=model.model_id,
                revision=model.revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=model.local_files_only,
            )
            path = resolved.path
            manifest = load_hf_directory(
                path,
                model_id=resolved.model_id,
                source_format=model.kind,
            )
    elif model.kind in {"mlx", "mlx-lm"}:
        path = Path(model.source).expanduser().resolve()
        manifest = load_mlx_directory(path, model_id=model.model_id)
    elif model.kind in {"gguf", "llama.cpp"}:
        path = Path(model.source).expanduser().resolve()
        manifest = load_gguf(path, model_id=model.model_id)
    elif model.kind == "ollama":
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            manifest = asyncio.run(
                load_ollama_model(
                    model.endpoint or "http://127.0.0.1:11434",
                    model.source,
                    api_key=api_key,
                )
            )
            if model.model_id is not None:
                manifest.id = model.model_id
            manifest.metadata["model_spec"] = model.to_spec()
            return ResolvedModel(model, manifest, None, None, None)
        raise ModelLoadError("load_model cannot query Ollama inside a running event loop")
    else:
        raise ModelLoadError(f"model kind {model.kind!r} has no public loader")
    checkpoint, source_lock_digest, source_lock = _source_lock(path, model)
    manifest.metadata.update({
        "model_spec": model.to_spec(),
        "checkpoint_digest": checkpoint,
        "source_lock_digest": source_lock_digest,
        "source_lock": source_lock,
        "config_only": not any(row["path"].endswith((".safetensors", ".gguf")) for row in source_lock["files"]),
    })
    return ResolvedModel(model, manifest, path, checkpoint, source_lock_digest)


def resolve_model(
    value: Model | str | Mapping[str, Any],
    *,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    api_key: str | None = None,
    allow_config_only: bool = False,
) -> ResolvedModel:
    try:
        return _resolve_model(
            value,
            token=token,
            cache_dir=cache_dir,
            api_key=api_key,
            allow_config_only=allow_config_only,
        )
    except (ConfigurationError, ModelLoadError):
        raise
    except Exception as exc:
        model = coerce_model(value)
        raise ModelLoadError(f"failed to load model {model.source!r}: {exc}") from exc


def load_model(
    value: Model | str | Mapping[str, Any],
    *,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    api_key: str | None = None,
) -> ModelManifest:
    return resolve_model(
        value,
        token=token,
        cache_dir=cache_dir,
        api_key=api_key,
    ).manifest


__all__ = ["ModelLoadError", "ModelManifest", "load_model"]

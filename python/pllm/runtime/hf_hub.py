from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .hf_download import cached_huggingface_model, download_huggingface_model


class HuggingFaceSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResolvedModelSource:
    path: Path
    model_id: str
    repo_id: str | None
    revision: str | None


_HF_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.*",
)

_HF_IGNORE_PATTERNS = (
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.gguf",
)


def _configure_hf_transfer() -> None:
    # HTTP leaves resumable partial files in the Hub cache. Xet can be restored
    # explicitly for installations where its all-or-nothing transfers are useful.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _is_complete_snapshot(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    indexes = tuple(path.glob("*.safetensors.index.json"))
    if not indexes:
        return any(path.glob("*.safetensors"))
    try:
        for index_path in indexes:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            if any(not (path / str(filename)).is_file() for filename in set(weight_map.values())):
                return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def resolve_huggingface_source(
    source: str | os.PathLike[str],
    *,
    model_id: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
) -> ResolvedModelSource:
    """Resolve a local HF directory or download a Hub snapshot.

    The resolver intentionally downloads only Safetensors and tokenizer/config
    assets. Pickle-based ``pytorch_model.bin`` files are rejected by the strict
    compiler.
    """

    raw = str(source)
    path = Path(raw).expanduser()
    repo_id: str | None = None
    if path.exists():
        path = path.resolve()
        resolved_id = model_id or path.name
    else:
        repo_id = raw
        _configure_hf_transfer()
        try:
            cached = cached_huggingface_model(repo_id, revision=revision, cache_dir=cache_dir)
            offline = local_files_only or os.environ.get("HF_HUB_OFFLINE", "").lower() in {
                "1",
                "on",
                "true",
                "yes",
            }
            if cached is not None:
                path = cached
            elif offline:
                from huggingface_hub import snapshot_download

                downloaded = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    token=token,
                    cache_dir=None if cache_dir is None else str(cache_dir),
                    local_files_only=True,
                    allow_patterns=list(_HF_ALLOW_PATTERNS),
                    ignore_patterns=list(_HF_IGNORE_PATTERNS),
                )
                path = Path(downloaded).resolve()
            else:
                path = download_huggingface_model(
                    repo_id,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    allow_patterns=_HF_ALLOW_PATTERNS,
                    ignore_patterns=_HF_IGNORE_PATTERNS,
                )
        except Exception as exc:
            raise HuggingFaceSourceError(f"failed to resolve Hugging Face model {repo_id!r}: {exc}") from exc
        resolved_id = model_id or repo_id

    if not _is_complete_snapshot(path):
        raise HuggingFaceSourceError(f"model source is incomplete or has no Safetensors weights: {path}")
    return ResolvedModelSource(path=path, model_id=str(resolved_id), repo_id=repo_id, revision=revision)

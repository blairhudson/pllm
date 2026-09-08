from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import httpx

from .models import ModelManifest, gemma4_stage_plan, transformer_stage_plan


class ModelLoadError(ValueError):
    pass


def load_hf_directory(path: str | Path, *, model_id: str | None = None, source_format: str = "huggingface") -> ModelManifest:
    root = Path(path)
    config_path = root / "config.json"
    if not config_path.exists():
        raise ModelLoadError(f"missing config.json in {root}")
    raw = json.loads(config_path.read_text())
    cfg = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw
    architecture = _architecture(raw, cfg)
    hidden = _int(cfg, "hidden_size", "d_model", "n_embd")
    intermediate = _int(cfg, "intermediate_size", "ffn_dim", "n_inner", default=hidden * 4)
    layers = _int(cfg, "num_hidden_layers", "n_layer", "num_layers")
    heads = _int(cfg, "num_attention_heads", "n_head", "num_heads")
    kv_heads = _int(cfg, "num_key_value_heads", "n_kv_head", default=heads)
    head_dim = _int(cfg, "head_dim", default=hidden // heads)
    vocab = _int(cfg, "vocab_size")
    context = _int(cfg, "max_position_embeddings", "max_sequence_length", "seq_length", default=2048)
    tied = bool(cfg.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)))
    tokenizer_path = str(root) if any((root / name).exists() for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")) else None
    chat_template = None
    tokenizer_config = root / "tokenizer_config.json"
    if tokenizer_config.exists():
        try:
            chat_template = json.loads(tokenizer_config.read_text()).get("chat_template")
        except Exception:
            pass
    quantization = raw.get("quantization_config") or cfg.get("quantization_config")
    weights = sorted(p.name for p in root.glob("*.safetensors")) + sorted(p.name for p in root.glob("*.bin"))
    model_type = str(cfg.get("model_type", raw.get("model_type", ""))).lower()
    is_gemma4 = "gemma4" in architecture.lower() or model_type in {"gemma4", "gemma4_text"}
    stages = gemma4_stage_plan(cfg, include_lm_head=True) if is_gemma4 else transformer_stage_plan(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
    )
    manifest = ModelManifest(
        id=model_id or str(raw.get("name_or_path") or root.name),
        architecture=architecture,
        source_format=source_format,
        source=str(root.resolve()),
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        context_length=context,
        tokenizer_path=tokenizer_path,
        chat_template=chat_template,
        quantization=quantization,
        tied_embeddings=tied,
        stages=stages,
        metadata={
            "weight_files": weights,
            "model_type": cfg.get("model_type", raw.get("model_type")),
            "architectures": raw.get("architectures", []),
            "text_config": cfg,
            "client_tensor_policy": "embedding_norms_and_ple_lookup",
        },
    )
    return manifest


def load_mlx_directory(path: str | Path, *, model_id: str | None = None) -> ModelManifest:
    manifest = load_hf_directory(path, model_id=model_id, source_format="mlx-lm")
    manifest.metadata["mlx_weight_files"] = sorted(p.name for p in Path(path).glob("*.safetensors"))
    return manifest


class GGUFReader:
    """Minimal GGUF metadata reader for llama.cpp model discovery.

    It parses the official metadata preamble but intentionally doesn't map
    tensors into memory. That makes model inspection safe and cheap even for
    multi-hundred-gigabyte files.
    """

    TYPES = {
        0: ("B", 1),  # uint8
        1: ("b", 1),
        2: ("H", 2),
        3: ("h", 2),
        4: ("I", 4),
        5: ("i", 4),
        6: ("f", 4),
        7: ("?", 1),
        10: ("Q", 8),
        11: ("q", 8),
        12: ("d", 8),
    }

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.version = 0
        self.tensor_count = 0
        self.kv_count = 0
        self.metadata: dict[str, Any] = {}

    def read(self) -> "GGUFReader":
        if self._read_exact(4) != b"GGUF":
            raise ModelLoadError("not a GGUF file")
        self.version = self._u32()
        if self.version not in {2, 3}:
            raise ModelLoadError(f"unsupported GGUF version {self.version}")
        self.tensor_count = self._u64()
        self.kv_count = self._u64()
        for _ in range(self.kv_count):
            key = self._string()
            value_type = self._u32()
            self.metadata[key] = self._value(value_type, keep=key not in {"tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type"})
        return self

    def _value(self, value_type: int, *, keep: bool = True) -> Any:
        if value_type == 8:
            value = self._string()
            return value if keep else None
        if value_type == 9:
            subtype = self._u32()
            count = self._u64()
            # Token arrays can contain hundreds of thousands of strings. Skip
            # their content while retaining a useful count marker.
            if not keep:
                for _ in range(count):
                    self._value(subtype, keep=False)
                return {"count": count, "type": subtype}
            return [self._value(subtype) for _ in range(count)]
        spec = self.TYPES.get(value_type)
        if spec is None:
            raise ModelLoadError(f"unsupported GGUF metadata type {value_type}")
        fmt, size = spec
        value = struct.unpack("<" + fmt, self._read_exact(size))[0]
        return value if keep else None

    def _string(self) -> str:
        length = self._u64()
        return self._read_exact(length).decode("utf-8", errors="replace")

    def _u32(self) -> int:
        return struct.unpack("<I", self._read_exact(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self._read_exact(8))[0]

    def _read_exact(self, length: int) -> bytes:
        value = self.stream.read(length)
        if len(value) != length:
            raise ModelLoadError("truncated GGUF metadata")
        return value


def load_gguf(path: str | Path, *, model_id: str | None = None) -> ModelManifest:
    source = Path(path)
    with source.open("rb") as stream:
        reader = GGUFReader(stream).read()
    meta = reader.metadata
    architecture = str(meta.get("general.architecture", "unknown"))
    prefix = architecture
    hidden = _meta_int(meta, f"{prefix}.embedding_length", "llama.embedding_length")
    intermediate = _meta_int(meta, f"{prefix}.feed_forward_length", "llama.feed_forward_length", default=hidden * 4)
    layers = _meta_int(meta, f"{prefix}.block_count", "llama.block_count")
    heads = _meta_int(meta, f"{prefix}.attention.head_count", "llama.attention.head_count")
    kv_heads = _meta_int(meta, f"{prefix}.attention.head_count_kv", "llama.attention.head_count_kv", default=heads)
    context = _meta_int(meta, f"{prefix}.context_length", "llama.context_length", default=2048)
    head_dim = hidden // heads
    token_marker = meta.get("tokenizer.ggml.tokens")
    vocab = int(token_marker.get("count", 0)) if isinstance(token_marker, dict) else len(token_marker or [])
    if not vocab:
        vocab = _meta_int(meta, f"{prefix}.vocab_size", default=32000)
    quant = source.stem.rsplit(".", 1)[-1] if "." in source.stem else None
    return ModelManifest(
        id=model_id or source.stem,
        architecture=architecture,
        source_format="gguf",
        source=str(source.resolve()),
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        context_length=context,
        quantization={"gguf_file_type": meta.get("general.file_type"), "label": quant},
        tied_embeddings=bool(meta.get("general.tie_word_embeddings", False)),
        stages=transformer_stage_plan(
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=vocab,
        ),
        metadata={
            "gguf_version": reader.version,
            "tensor_count": reader.tensor_count,
            "name": meta.get("general.name"),
        },
    )


async def load_ollama_model(
    base_url: str,
    name: str,
    *,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> ModelManifest:
    owned = client is None
    client = client or httpx.AsyncClient(timeout=30)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = await client.post(f"{base_url.rstrip('/')}/api/show", headers=headers, json={"model": name})
        response.raise_for_status()
        data = response.json()
    finally:
        if owned:
            await client.aclose()
    info = data.get("model_info") or {}
    architecture = str(info.get("general.architecture", data.get("details", {}).get("family", "unknown")))
    prefix = architecture
    hidden = _meta_int(info, f"{prefix}.embedding_length", "llama.embedding_length")
    intermediate = _meta_int(info, f"{prefix}.feed_forward_length", "llama.feed_forward_length", default=hidden * 4)
    layers = _meta_int(info, f"{prefix}.block_count", "llama.block_count")
    heads = _meta_int(info, f"{prefix}.attention.head_count", "llama.attention.head_count")
    kv_heads = _meta_int(info, f"{prefix}.attention.head_count_kv", "llama.attention.head_count_kv", default=heads)
    context = _meta_int(info, f"{prefix}.context_length", "llama.context_length", default=2048)
    vocab = _meta_int(info, f"{prefix}.vocab_size", "tokenizer.ggml.tokens", default=32000)
    if isinstance(info.get("tokenizer.ggml.tokens"), list):
        vocab = len(info["tokenizer.ggml.tokens"])
    return ModelManifest(
        id=name,
        architecture=architecture,
        source_format="ollama",
        source=f"{base_url.rstrip('/')}/{name}",
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=hidden // heads,
        context_length=context,
        quantization={"level": data.get("details", {}).get("quantization_level")},
        tied_embeddings=False,
        stages=transformer_stage_plan(
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=hidden // heads,
            vocab_size=vocab,
        ),
        metadata={"template": data.get("template"), "parameters": data.get("parameters")},
    )


def _architecture(raw: dict[str, Any], cfg: dict[str, Any]) -> str:
    architectures = raw.get("architectures") or cfg.get("architectures") or []
    if architectures:
        return str(architectures[0])
    return str(cfg.get("model_type", raw.get("model_type", "transformer")))


def _int(value: dict[str, Any], *keys: str, default: int | None = None) -> int:
    for key in keys:
        if key in value and value[key] is not None:
            return int(value[key])
    if default is not None:
        return int(default)
    raise ModelLoadError(f"missing required model field: one of {keys}")


def _meta_int(value: dict[str, Any], *keys: str, default: int | None = None) -> int:
    for key in keys:
        item = value.get(key)
        if item is not None and not isinstance(item, (dict, list)):
            return int(item)
    if default is not None:
        return int(default)
    raise ModelLoadError(f"missing required metadata field: one of {keys}")

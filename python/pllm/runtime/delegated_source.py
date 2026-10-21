"""Experiment-bound, bounded float32 source ownership for the GKM runtime.

Native Experiment and complete-budget admission precede hashing, tokenizer
construction and tensor loading. Model probes remain available internally. This
slice accepts local HF/Safetensors directories, complete tokenizer.json vocabularies,
and F32/F16/BF16 tensors converted exactly to little-endian float32.

Locks use the ordinary model-source schema and digest domains. Bounded readers
reject observed filesystem changes; they do not promise an atomic filesystem
snapshot or authenticate a caller-selected local checkpoint against a publisher.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any

import numpy as np

from pllm.configuration import Experiment, Model
from pllm import _native, model_loader
from pllm.modeling import ModelPlan

from .delegated_decoder import _MAX_CONFIG_BYTES, _encode_config, decoder_contract
from .loaders import _architecture, _int
from .models import ModelManifest
from .safetensors_store import SafeTensorStore, TensorLocation
from .semantic_tensors import preflight_semantic_checkpoint, required_checkpoint_tensors
from . import tokenizer_admission
from .transformer_client import TokenizersJSONTokenizer


_MAX_SOURCE_BYTES = 8 << 20
_MAX_METADATA_BYTES = 1 << 20
_MAX_HEADER_BYTES = 64 << 10
_MAX_TOKENIZER_BYTES = 256 << 10
_MAX_FILES = 64
_MAX_EXPERIMENT_BYTES = 32 << 10
_DECODER_SOURCE_BYTES = 4 << 30
_DECODER_METADATA_BYTES = 32 << 20
_DECODER_TOKENIZER_BYTES = 16 << 20
_DECODER_HEADER_BYTES = 4 << 20
_DECODER_SCHEMA = "pllm.gkm.decoder-reference.v1"
_RUNTIME_SCHEMA = "pllm.gkm.sdk-contract.v1"
_DTYPES = {"F32": np.dtype("<f4"), "F16": np.dtype("<f2"),
           "BF16": np.dtype("bfloat16").newbyteorder("<")}
_METADATA_LIMITS = {
    "config.json": _MAX_CONFIG_BYTES,
    "tokenizer.json": _MAX_TOKENIZER_BYTES,
    "tokenizer_config.json": _MAX_CONFIG_BYTES,
    "special_tokens_map.json": _MAX_CONFIG_BYTES,
    "added_tokens.json": _MAX_CONFIG_BYTES,
    "generation_config.json": _MAX_CONFIG_BYTES,
    ".pllm-model.json": _MAX_HEADER_BYTES,
}


class DelegatedSourceError(ValueError):
    """The source is incomplete, changed, or outside the bounded GKM contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON number")

    try:
        value = json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise DelegatedSourceError(f"GKM invalid {label}") from exc
    if type(value) is not dict:
        raise DelegatedSourceError(f"GKM {label} must be a JSON object")
    return value


def _stat_identity(value) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _identity(path: Path) -> tuple[int, ...]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise DelegatedSourceError(f"GKM source requires regular files: {path.name}")
    # Include the directory entry as well as its target (local HF snapshots can
    # use symlinks). ctime catches same-size writes with restored mtime.
    return _stat_identity(path.lstat()) + _stat_identity(info)


def _read_bounded(path: Path, limit: int) -> bytes:
    before = _identity(path)
    if before[9] > limit:
        raise DelegatedSourceError(f"GKM {path.name} exceeds its byte limit")
    return _read_range(path, before, 0, before[9], limit=limit)


def _files(root: Path, source_bytes: int, capacity="capsule") -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Bound directory traversal and every file before any shared hashing/reader."""
    result = []
    total = 0
    expanded = capacity == "decoder"
    metadata_limit = _DECODER_METADATA_BYTES if expanded else _MAX_METADATA_BYTES
    header_limit = _DECODER_HEADER_BYTES if expanded else _MAX_HEADER_BYTES
    limits = dict(_METADATA_LIMITS)
    if expanded:
        limits.update({"tokenizer.json": _DECODER_TOKENIZER_BYTES,
                       "merges.txt": 8 << 20, "vocab.json": 8 << 20})
    with os.scandir(root) as entries:
        for count, entry in enumerate(entries):
            if count >= _MAX_FILES:
                raise DelegatedSourceError("GKM source file cardinality exceeds its limit")
            name = entry.name
            if name in limits:
                limit = limits[name]
            elif name.endswith(".safetensors.index.json"):
                limit = header_limit
            elif name.endswith(".safetensors"):
                limit = source_bytes + header_limit + 8
            elif expanded and not name.endswith((".bin", ".gguf", ".npz", ".json")):
                # Shared source locking ignores documentation and unrelated
                # derived artifacts. Never read them; traversal remains bounded.
                continue
            else:
                raise DelegatedSourceError(f"GKM unsupported source file: {name}")
            identity = _identity(root / name)
            size = identity[9]
            if size > limit:
                raise DelegatedSourceError(f"GKM {name} exceeds its byte limit")
            total += size
            if total > source_bytes + metadata_limit:
                raise DelegatedSourceError("GKM source files exceed their total byte limit")
            result.append((name, identity))
    names = {name for name, _ in result}
    if not {"config.json", "tokenizer.json"} <= names:
        raise DelegatedSourceError("GKM source requires config.json and complete tokenizer.json")
    if not any(name.endswith(".safetensors") for name in names):
        raise DelegatedSourceError("GKM source requires Safetensors weights")
    return tuple(sorted(result))


def _unchanged(root, source_bytes, identities, capacity="capsule"):
    try:
        current = _files(root, source_bytes, capacity)
    except (OSError, DelegatedSourceError) as exc:
        raise DelegatedSourceError("GKM source changed after admission") from exc
    if current != identities:
        raise DelegatedSourceError("GKM source changed after admission")


def _locked(root, model, source_bytes, identities, capacity="capsule"):
    """Run the actual shared lock implementation through a fixed-size file view."""
    _unchanged(root, source_bytes, identities, capacity)
    if not root.is_dir():
        raise DelegatedSourceError("GKM local source changed after admission")
    locked = model_loader._source_lock(_LockPath(root, dict(identities)), model)
    _unchanged(root, source_bytes, identities, capacity)
    return locked


class _FixedReader:
    """Finite stream for the shared EOF-based hasher; never chase an appended tail."""

    def __init__(self, path, identity):
        self.path, self.identity = path, identity
        self.remaining = identity[9]

    def _check(self):
        if (_stat_identity(os.fstat(self.stream.fileno())) != self.identity[6:]
                or _identity(self.path) != self.identity):
            raise DelegatedSourceError("GKM source changed while hashing")

    def __enter__(self):
        self.stream = self.path.open("rb")
        try:
            self._check()
        except BaseException:
            self.stream.close()
            raise
        return self

    def read(self, size=-1):
        self._check()
        if not self.remaining:
            if self.stream.read(1):
                raise DelegatedSourceError("GKM source grew while hashing")
            self._check()
            return b""
        count = min(self.remaining, 64 << 10, self.remaining if size < 0 else size)
        if count == 0:
            return b""
        block = self.stream.read(count)
        if not block or len(block) > count:
            raise DelegatedSourceError("GKM source truncated while hashing")
        self.remaining -= len(block)
        self._check()
        return block

    def __exit__(self, kind, *_exc):
        try:
            if kind is None:
                self._check()
        finally:
            self.stream.close()


@dataclass(frozen=True)
class _LockPath:
    """Restricted Path interface used by model_loader's unchanged lock helpers.

    Glob enumerates only the admitted inventory, read_text uses an exact range,
    and open supplies the finite stream above. No global monkeypatch, copied
    lock schema, downloader, or unchecked filesystem rediscovery is involved.
    """

    root: Path
    identities: Mapping[str, tuple] = field(compare=False, hash=False, repr=False)
    filename: str | None = None

    @property
    def path(self):
        return self.root if self.filename is None else self.root / self.filename

    @property
    def name(self):
        return self.path.name

    @property
    def parent(self):
        return self.path.parent

    def __str__(self):
        return str(self.path)

    def __truediv__(self, name):
        if self.filename is not None or Path(name).name != name:
            raise DelegatedSourceError("GKM source lock requested an unadmitted path")
        return _LockPath(self.root, self.identities, name)

    def is_dir(self):
        return self.filename is None

    def is_file(self):
        return self.filename in self.identities

    def is_symlink(self):
        return self.path.is_symlink()

    def readlink(self):
        return self.path.readlink()

    def resolve(self, *, strict=False):
        return self.path.resolve(strict=strict)

    def glob(self, pattern):
        return (self / name for name in self.identities if fnmatchcase(name, pattern))

    def relative_to(self, other):
        return self.path.relative_to(other.path)

    def stat(self):
        if _identity(self.path) != self.identities[self.filename]:
            raise DelegatedSourceError("GKM source changed during lock inspection")
        return self.path.stat()

    def open(self, mode):
        if mode != "rb":
            raise DelegatedSourceError("GKM source lock is read-only")
        return _FixedReader(self.path, self.identities[self.filename])

    def read_text(self, encoding="utf-8"):
        identity = self.identities[self.filename]
        return _read_range(self.path, identity, 0, identity[9]).decode(encoding)


def _read_range(path, identity, offset, size, *, limit=_MAX_SOURCE_BYTES):
    if (type(offset) is not int or type(size) is not int or offset < 0
            or not 0 <= size <= min(limit, _DECODER_METADATA_BYTES) or offset + size > identity[9]):
        raise DelegatedSourceError("GKM source read range exceeds admitted file bounds")
    with path.open("rb") as stream:
        if _stat_identity(os.fstat(stream.fileno())) != identity[6:]:
            raise DelegatedSourceError("GKM source changed before reading")
        stream.seek(offset)
        raw = stream.read(size)
        if len(raw) != size or _stat_identity(os.fstat(stream.fileno())) != identity[6:]:
            raise DelegatedSourceError("GKM source changed or truncated while reading")
    if _identity(path) != identity:
        raise DelegatedSourceError("GKM source changed after reading")
    return raw


class _AdmittedTensorStore(SafeTensorStore):
    """Shared resolution/conversion with frozen metadata and checked range reads."""

    def __init__(self, root, identities, specs):
        # The ordinary constructor re-opens config/headers. All metadata here
        # already came from bounded reads; inherited metadata access never reopens.
        self.root = root
        self._identities = dict(identities)
        self._specs = specs
        self._locations = {name: TensorLocation(name, root / row[0]) for name, row in specs.items()}
        self._metadata = {name: (row[1], row[2]) for name, row in specs.items()}
        self._cache, self._file_signatures, self.quantization = {}, {}, {}

    def get(self, key_or_suffix, *, dtype=np.float32):
        key = self.resolve(key_or_suffix)
        filename, shape, source_dtype, offset, size = self._specs[key]
        raw = _read_range(self.root / filename, self._identities[filename], offset, size)
        value = np.frombuffer(raw, dtype=_DTYPES[source_dtype]).reshape(shape)
        return self._convert(value, dtype)


def _store(root: Path, identities, contract) -> tuple[SafeTensorStore, tuple]:
    """Bound all headers/ranges before shared shape and float-dtype preflight."""

    capacity = _capacity(contract)
    expanded = capacity == "decoder"
    header_limit = _DECODER_HEADER_BYTES if expanded else _MAX_HEADER_BYTES
    metadata_limit = _DECODER_METADATA_BYTES if expanded else _MAX_METADATA_BYTES
    shapes = {key: tuple(shape) for key, shape in contract["source_shapes"].items()}
    plan = ModelPlan(_canonical(contract["plan"]))  # Metadata only; never a native digest.
    if required_checkpoint_tensors(plan) != shapes:
        raise DelegatedSourceError("GKM native and shared source shapes differ")
    # Native admission owns the source envelope. Derive reader bounds from its
    # checked geometry rather than a second method-specific size table. Shared
    # semantic preflight below still checks every tensor's exact shape and dtype.
    dimension_limit = max(max(shape) for shape in shapes.values())
    element_limit = max(math.prod(shape) for shape in shapes.values())
    _unchanged(root, contract["source_float32_bytes"], identities, capacity)
    locations = {}
    specs = {}
    indexes = []
    header_bytes = 0
    for name, identity in identities:
        if name.endswith(".safetensors.index.json"):
            indexes.append(name)
        if not name.endswith(".safetensors"):
            continue
        path = root / name
        prefix = _read_range(path, identity, 0, 8)
        size = int.from_bytes(prefix, "little")
        header_bytes += size
        if (len(prefix) != 8 or size > header_limit or size + 8 > identity[9]
                or header_bytes > metadata_limit):
            raise DelegatedSourceError("GKM Safetensors header exceeds its byte limit")
        header = _json(_read_range(path, identity, 8, size), "Safetensors header")
        metadata = header.pop("__metadata__", {})
        if type(metadata) is not dict or any(type(value) is not str for value in metadata.values()):
            raise DelegatedSourceError("GKM invalid Safetensors metadata")
        if not header or len(locations) + len(header) > len(shapes):
            raise DelegatedSourceError("GKM source tensor cardinality mismatch")
        spans = []
        for key, item in header.items():
            if key in locations:
                raise DelegatedSourceError("GKM duplicate source tensor")
            if (type(item) is not dict or set(item) != {"dtype", "shape", "data_offsets"}
                    or type(item["dtype"]) is not str or item["dtype"] not in _DTYPES
                    or type(item["shape"]) is not list or not 1 <= len(item["shape"]) <= 2
                    or any(type(d) is not int or not 1 <= d <= dimension_limit
                           for d in item["shape"])
                    or math.prod(item["shape"]) > element_limit):
                raise DelegatedSourceError("GKM source tensor has unsupported shape or dtype")
            shape, dtype = tuple(item["shape"]), item["dtype"]
            offsets = item["data_offsets"]
            if (type(offsets) is not list or len(offsets) != 2
                    or any(type(i) is not int or i < 0 for i in offsets)
                    or offsets[1] - offsets[0] != math.prod(shape) * _DTYPES[dtype].itemsize
                    or offsets[1] > identity[9] - size - 8):
                raise DelegatedSourceError("GKM source tensor byte range mismatch")
            spans.append(tuple(offsets))
            locations[key] = name
            specs[key] = (name, shape, dtype, 8 + size + offsets[0], offsets[1] - offsets[0])
        end = 0
        for begin, stop in sorted(spans):
            if begin != end:
                raise DelegatedSourceError("GKM source has overlapping or gapped tensor ranges")
            end = stop
        if end + size + 8 != identity[9]:
            raise DelegatedSourceError("GKM source payload differs from header byte ranges")
    if len(locations) != len(shapes):
        raise DelegatedSourceError("GKM source tensor cardinality mismatch")
    if len(indexes) > 1:
        raise DelegatedSourceError("GKM multiple Safetensors indexes are unsupported")
    if indexes:
        index = _json(_read_bounded(root / indexes[0], header_limit), "tensor index")
        # Exact equality also rejects external paths, unlisted shards/tensors,
        # hidden extras, duplicate ownership and absent index entries.
        if index.get("weight_map") != locations:
            raise DelegatedSourceError("GKM Safetensors index differs from shard headers")
    store = _AdmittedTensorStore(root, identities, specs)
    preflight_semantic_checkpoint(plan, store)
    if {store.resolve(name) for name in shapes} != set(locations):
        raise DelegatedSourceError("GKM source tensor identities differ")
    _unchanged(root, contract["source_float32_bytes"], identities, capacity)
    return store, tuple(sorted(specs.items()))


def _manifest(root, model, raw, identities):
    """Reuse source dimension parsing without a second, ordinary W8A8 lowering.

    Native admission has already validated this immutable configuration snapshot.
    Calling load_hf_directory here would re-read a mutable path before lowering.
    The GKM contract, rather than ordinary stage specs, owns executable metadata.
    """
    cfg = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw
    hidden = _int(cfg, "hidden_size", "d_model", "n_embd")
    heads = _int(cfg, "num_attention_heads", "n_head", "num_heads")
    return ModelManifest(
        id=model_loader.expected_model_id(model), architecture=_architecture(raw, cfg),
        source_format=model.kind, source=str(root), vocab_size=_int(cfg, "vocab_size"),
        hidden_size=hidden, intermediate_size=_int(cfg, "intermediate_size", "ffn_dim", "n_inner",
                                                  default=hidden * 4),
        num_hidden_layers=_int(cfg, "num_hidden_layers", "n_layer", "num_layers"),
        num_attention_heads=heads, num_key_value_heads=_int(cfg, "num_key_value_heads", "n_kv_head",
                                                          default=heads),
        head_dim=_int(cfg, "head_dim", default=hidden // heads),
        context_length=_int(cfg, "max_position_embeddings", "max_sequence_length", "seq_length",
                            default=2048),
        tokenizer_path=str(root), quantization=raw.get("quantization_config") or cfg.get("quantization_config"),
        tied_embeddings=bool(cfg.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))),
        metadata={"model_type": cfg.get("model_type", raw.get("model_type")), "text_config": cfg,
                  "architectures": raw.get("architectures", []),
                  "weight_files": [name for name, _ in identities if name.endswith(".safetensors")]},
    )


def _tokenizer(root, manifest, config, contract):
    from tokenizers import Tokenizer as HFTokenizer

    expanded = _capacity(contract) == "decoder"
    raw = _read_bounded(root / "tokenizer.json", _DECODER_TOKENIZER_BYTES if expanded else _MAX_TOKENIZER_BYTES)
    document = _json(raw, "tokenizer.json")
    tokenizer_admission.validate_document(
        document, vocabulary=contract["vocabulary"],
        max_entries=contract["vocabulary"] if expanded else 256,
        error=DelegatedSourceError, label="GKM")
    options = {}
    if (root / "tokenizer_config.json").exists():
        options = _json(_read_bounded(root / "tokenizer_config.json", _MAX_CONFIG_BYTES),
                        "tokenizer_config.json")
    context = options.get("model_max_length", manifest.context_length)
    if (type(context) is not int or context < contract["input_rows"] + contract["decode_steps"]
            or manifest.context_length < contract["input_rows"] + contract["decode_steps"]):
        raise DelegatedSourceError("GKM workload exceeds source/tokenizer context")
    if type(options.get("add_bos_token", True)) is not bool:
        raise DelegatedSourceError("GKM add_bos_token must be boolean")
    if config.get("pllm_test_tokenizer") is not None:
        raise DelegatedSourceError("GKM requires genuine tokenizer.json source semantics")
    # Preserve ID zero rather than the ordinary descriptor's historical fallback.
    bos = config.get("bos_token_id", 2)
    eos = config.get("eos_token_id", 1)
    eos_ids = eos if type(eos) is list else [eos]
    if not eos_ids or any(type(i) is not int or not 0 <= i < contract["vocabulary"]
                          for i in [bos, *eos_ids]):
        raise DelegatedSourceError("GKM tokenizer special token IDs are outside the vocabulary")
    try:
        inner = HFTokenizer.from_buffer(raw)
    except Exception as exc:
        raise DelegatedSourceError("GKM invalid tokenizer.json model") from exc
    expected_size = tokenizer_admission.validate_vocabulary(
        inner, vocabulary=contract["vocabulary"], allow_model_padding=expanded,
        error=DelegatedSourceError, label="GKM")
    if any(i >= expected_size for i in [bos, *eos_ids]):
        raise DelegatedSourceError("GKM special tokens require original tokenizer entries")
    if inner.truncation is not None:
        raise DelegatedSourceError("GKM tokenizer must not silently truncate source input")
    # Same descriptor/wrapper contract as the ordinary client, built entirely
    # from bounded snapshots instead of re-reading tokenizer files unboundedly.
    return {
        "type": "tokenizer_json", "model": raw, "vocab_size": contract["vocabulary"],
        "bos_token_id": bos, "eos_token_id": eos_ids[0],
        "chat_template": options.get("chat_template") or manifest.chat_template,
        "bos_token": options.get("bos_token", ""), "eos_token": options.get("eos_token", ""),
        "add_bos_token": options.get("add_bos_token", True),
    }


@dataclass(frozen=True, slots=True)
class AdmittedSource:
    """Metadata-only source owner; all document properties return independent copies."""

    model: Model
    path: Path
    config_bytes: bytes
    config_source_bytes: bytes
    contract_bytes: bytes
    checkpoint_digest: str
    source_lock_digest: str
    tokenizer_bytes: bytes
    tokenizer_digest: str
    _source_lock_bytes: bytes = field(repr=False)
    _manifest_bytes: bytes = field(repr=False)
    _tokenizer_options_bytes: bytes = field(repr=False)
    _identities: tuple = field(repr=False)
    _tensor_specs: tuple = field(repr=False)
    experiment_bytes: bytes | None = field(default=None, repr=False)
    runtime_contract_bytes: bytes | None = field(default=None, repr=False)

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self.config_bytes)

    @property
    def contract(self) -> dict[str, Any]:
        return json.loads(self.contract_bytes)

    def runtime_contract(self, experiment: Experiment, rows: int, outputs: int) -> bytes:
        """Bind an actual request to the exact admitted Experiment; retain native bytes.

        Model-only probes cannot enter SDK transport. The envelope was admitted
        in full before source access, so a smaller request cannot relax its costs.
        """
        encoded = _experiment_bytes(experiment)
        if self.experiment_bytes is None or encoded != self.experiment_bytes:
            raise DelegatedSourceError("GKM runtime requires the exact admitted Experiment")
        if (type(rows) is not int or type(outputs) is not int
                or not 1 <= rows <= experiment.budget.max_input_tokens
                or not 1 <= outputs <= experiment.budget.max_new_tokens):
            raise DelegatedSourceError("GKM actual request exceeds the admitted Experiment budget")
        decoder_bytes = decoder_contract(self.config, input_rows=rows, decode_steps=outputs - 1,
                                         capacity=self.capacity, streamed=True)
        return _bind_runtime(experiment, encoded, self.config_bytes, decoder_bytes,
                             self.source_lock_digest, self.tokenizer_digest)

    @property
    def config_digest(self) -> str:
        return self.contract["plan"]["config_digest"]

    @property
    def contract_digest(self) -> str:
        return hashlib.sha256(_DECODER_SCHEMA.encode() + b"\0" + self.contract_bytes).hexdigest()

    @property
    def capacity(self) -> str:
        return _capacity(self.contract)

    @property
    def runtime_contract_digest(self) -> str:
        if self.runtime_contract_bytes is None:
            raise DelegatedSourceError("GKM Model probe has no SDK runtime contract")
        return hashlib.sha256(_RUNTIME_SCHEMA.encode() + b"\0" + self.runtime_contract_bytes).hexdigest()

    @property
    def configuration_digest(self) -> str:
        if self.runtime_contract_bytes is None:
            raise DelegatedSourceError("GKM Model probe has no SDK runtime contract")
        return json.loads(self.runtime_contract_bytes)["configuration"]

    @property
    def composition_digest(self) -> str:
        if self.runtime_contract_bytes is None:
            raise DelegatedSourceError("GKM Model probe has no SDK runtime contract")
        return json.loads(self.runtime_contract_bytes)["composition"]

    @property
    def source_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(shape) for name, shape in self.contract["source_shapes"].items()}

    @property
    def source_float32_bytes(self) -> int:
        return self.contract["source_float32_bytes"]

    @property
    def execution_payload_estimate_bytes(self) -> int:
        return self.contract["execution_payload_estimate_bytes"]

    @property
    def source_lock(self) -> dict[str, Any]:
        return json.loads(self._source_lock_bytes)

    @property
    def source(self) -> model_loader.ResolvedModel:
        return model_loader.ResolvedModel(
            self.model, ModelManifest.from_dict(json.loads(self._manifest_bytes)), self.path,
            self.checkpoint_digest, self.source_lock_digest,
        )

    @property
    def tokenizer_descriptor(self) -> dict[str, Any]:
        return {**json.loads(self._tokenizer_options_bytes), "model": self.tokenizer_bytes}

    def tokenizer(self) -> TokenizersJSONTokenizer:
        from tokenizers import Tokenizer as HFTokenizer

        descriptor = self.tokenizer_descriptor
        return TokenizersJSONTokenizer(HFTokenizer.from_buffer(self.tokenizer_bytes),
                                       descriptor["bos_token_id"], descriptor["eos_token_id"])

    def load_values(self) -> LoadedSource:
        """Load exact admitted tensors after the caller has reserved its payload."""
        contract = self.contract
        if self.capacity != "capsule":
            raise DelegatedSourceError("GKM decoder capacity requires streaming native source import")
        expected_lock = (self.checkpoint_digest, self.source_lock_digest, self.source_lock)
        if _locked(self.path, self.model, self.source_float32_bytes, self._identities, self.capacity) != expected_lock:
            raise DelegatedSourceError("GKM source lock changed after admission")
        store, specs = _store(self.path, self._identities, contract)
        tensors = {}
        try:
            if specs != self._tensor_specs:
                raise DelegatedSourceError("GKM source tensor metadata changed after admission")
            _unchanged(self.path, self.source_float32_bytes, self._identities)
            for name, shape in self.source_shapes.items():
                value = store.get(name, dtype=np.dtype("<f4"))
                if (value.shape != shape or value.dtype != np.dtype("<f4")
                        or not np.all(np.isfinite(value))):
                    raise DelegatedSourceError(f"GKM source tensor {name!r} has invalid values")
                # F16/BF16 -> F32 conversion is exact; no quantization or arithmetic.
                # F32 bytes, including signed zero, are retained unchanged.
                tensors[name] = value.tobytes(order="C")
                store.drop(name)
            if sum(map(len, tensors.values())) != self.source_float32_bytes:
                raise DelegatedSourceError("GKM source byte accounting differs")
            if _locked(self.path, self.model, self.source_float32_bytes, self._identities, self.capacity) != expected_lock:
                raise DelegatedSourceError("GKM source lock changed while loading values")
            return LoadedSource(self, tensors, {name: tensors[name] for name in contract["local_tensors"]})
        except BaseException:
            tensors.clear()
            raise
        finally:
            store.clear_cache()

    def load_transport(self, experiment: Experiment, rows: int, outputs: int, *,
                       worker=None, cancelled=None):
        """Stream public values into one native SDK owner under caller admission.

        The native importer bounds each original float32 frame to 64 KiB. No
        complete Python float tensor dictionary escapes. Source identities and
        all headers still revalidate before values and again before publication.
        """
        self.runtime_contract(experiment, rows, outputs)
        contract = self.contract
        expected_lock = (self.checkpoint_digest, self.source_lock_digest, self.source_lock)
        if _locked(self.path, self.model, self.source_float32_bytes, self._identities, self.capacity) != expected_lock:
            raise DelegatedSourceError("GKM source lock changed after admission")
        store, specs = _store(self.path, self._identities, contract)
        importer = native = None
        try:
            if specs != self._tensor_specs:
                raise DelegatedSourceError("GKM source tensor metadata changed after admission")
            importer = _native._GkmTransportImport(
                experiment.canonical_bytes(), self.config_bytes, rows, outputs,
                self.source_lock_digest, self.tokenizer_digest,
            )
            identities = dict(self._identities)
            total = 0
            specs_by_name = dict(specs)
            for name in sorted(contract["source_shapes"]):
                filename, shape, dtype, offset, size = specs_by_name[store.resolve(name)]
                itemsize = _DTYPES[dtype].itemsize
                elements = math.prod(shape)
                if size != elements * itemsize:
                    raise DelegatedSourceError("GKM source byte accounting differs")
                for begin in range(0, elements, (64 << 10) // 4):
                    if cancelled is not None and cancelled():
                        raise DelegatedSourceError("GKM source import cancelled")
                    count = min((64 << 10) // 4, elements - begin)
                    # Scope conversions to this frame: no previous tensor/frame
                    # alias survives while reading the next frame.
                    frame = _source_float_frame(self.path / filename, identities[filename],
                                                offset + begin * itemsize, count, dtype)
                    importer.push(name, begin * 4, frame)
                    total += len(frame)
                    del frame
            if total != self.source_float32_bytes:
                raise DelegatedSourceError("GKM source byte accounting differs")
            if _locked(self.path, self.model, self.source_float32_bytes, self._identities, self.capacity) != expected_lock:
                raise DelegatedSourceError("GKM source lock changed while loading values")
            if cancelled is not None and cancelled():
                raise DelegatedSourceError("GKM source import cancelled")
            native = importer.client() if worker is None else importer.worker(*worker)
            return native
        except BaseException:
            if native is not None:
                native.abort()
            raise
        finally:
            if importer is not None:
                importer.abort()
            store.clear_cache()


def _source_float_frame(path, identity, offset, count, dtype):
    raw = _read_range(path, identity, offset, count * _DTYPES[dtype].itemsize)
    value = np.frombuffer(raw, dtype=_DTYPES[dtype])
    # Reuse the source reader's exact F16/BF16 conversion, including signed zero.
    value = SafeTensorStore._convert(value, np.dtype("<f4"))
    if not np.all(np.isfinite(value)):
        raise DelegatedSourceError("GKM source tensor has invalid values")
    return value.tobytes(order="C")


@dataclass(frozen=True, slots=True, init=False)
class LoadedSource:
    """Read-only source bytes; close releases owned references, without secure erasure."""

    admission: AdmittedSource
    _tensors: dict[str, bytes] = field(repr=False)
    _coefficients: dict[str, bytes] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __init__(self, admission: AdmittedSource, tensors: Mapping[str, bytes],
                 coefficients: Mapping[str, bytes]):
        object.__setattr__(self, "admission", admission)
        object.__setattr__(self, "_tensors", dict(tensors))
        object.__setattr__(self, "_coefficients", dict(coefficients))
        object.__setattr__(self, "_closed", False)

    def _check_open(self):
        if self._closed:
            raise DelegatedSourceError("GKM source values are retired")

    @property
    def tensors(self) -> Mapping[str, bytes]:
        self._check_open()
        return MappingProxyType(self._tensors)

    @property
    def coefficients(self) -> Mapping[str, bytes]:
        self._check_open()
        return MappingProxyType(self._coefficients)

    @property
    def tokenizer(self) -> TokenizersJSONTokenizer:
        self._check_open()
        return self.admission.tokenizer()

    def close(self) -> None:
        self._tensors.clear()
        self._coefficients.clear()
        object.__setattr__(self, "_closed", True)

    retire = close

    def __enter__(self) -> LoadedSource:
        self._check_open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _capacity(contract):
    resources = contract.get("decoder_resources")
    if resources is None:
        return "capsule"
    if (contract.get("schema_version") == "pllm.replicated.sdk-contract.v1"
            and contract.get("weights") == "public"
            and resources == {"profile": "pllm.puma.public-source-capacity.v1"}):
        return "decoder"
    if type(resources) is not dict or resources.get("profile") != "pllm.gkm.decoder-capacity.v1":
        raise DelegatedSourceError("GKM invalid native capacity contract")
    return "decoder"


def _validate_contract(contract):
    expanded = _capacity(contract) == "decoder"
    shapes = contract.get("source_shapes")
    if (contract.get("schema_version") != _DECODER_SCHEMA or type(shapes) is not dict
            or not 1 <= len(shapes) <= (2048 if expanded else 256)):
        raise DelegatedSourceError("GKM invalid native source-shape contract")
    for name, shape in shapes.items():
        if (type(name) is not str or not name or type(shape) is not list or not 1 <= len(shape) <= 2
                or any(type(d) is not int or not 1 <= d <= (262144 if expanded else 4096) for d in shape)
                or math.prod(shape) > (1 << 28 if expanded else 262144)):
            raise DelegatedSourceError("GKM native source shape exceeds its limit")
    source_bytes = sum(math.prod(shape) * 4 for shape in shapes.values())
    if (type(contract.get("source_float32_bytes")) is not int
            or source_bytes != contract["source_float32_bytes"]
            or not 0 < source_bytes <= (_DECODER_SOURCE_BYTES if expanded else _MAX_SOURCE_BYTES)
            or type(contract.get("vocabulary")) is not int
            or not 1 <= contract["vocabulary"] <= (262144 if expanded else 256)
            or type(contract.get("execution_payload_estimate_bytes")) is not int
            or not 0 < contract["execution_payload_estimate_bytes"] <= (_DECODER_SOURCE_BYTES if expanded else 64 << 20)
            or (not expanded and contract["execution_payload_estimate_bytes"] < source_bytes)
            or type(contract.get("local_tensors")) is not list
            or any(type(name) is not str or name not in shapes for name in contract["local_tensors"])):
        raise DelegatedSourceError("GKM native source payload exceeds its bounded contract")
    return source_bytes


def _experiment_bytes(experiment: Experiment) -> bytes:
    if not isinstance(experiment, Experiment):
        raise TypeError("GKM runtime requires an Experiment")
    encoded = experiment.canonical_bytes()
    if not 0 < len(encoded) <= _MAX_EXPERIMENT_BYTES:
        raise DelegatedSourceError("GKM Experiment exceeds its 32 KiB limit")
    # Native canonical/composition/deployment/budget validation is mandatory,
    # including when Python's broader runtime resolver would accept a profile.
    _native.resolve_experiment(encoded)
    if not callable(getattr(_native, "gkm_runtime_contract", None)):
        raise DelegatedSourceError("installed native core lacks GKM runtime admission")
    return encoded


def _bind_runtime(experiment, experiment_bytes, config_bytes, decoder_bytes,
                  source_lock_digest, tokenizer_digest) -> bytes:
    decoder = _json(decoder_bytes, "native decoder contract")
    result = _native.gkm_runtime_contract(
        experiment_bytes, config_bytes, decoder["input_rows"], decoder["decode_steps"] + 1,
        source_lock_digest, tokenizer_digest,
    )
    if type(result) is not bytes:
        raise DelegatedSourceError("GKM native runtime contract must be immutable bytes")
    bound = _json(result, "native runtime contract")
    expected = {
        "schema_version": _RUNTIME_SCHEMA, "configuration": experiment.configuration_digest(),
        "composition": experiment.pipeline.digest(), "source_lock": source_lock_digest,
        "tokenizer": tokenizer_digest,
    }
    if experiment.pipeline.components["linear"].params["sharing"] == "seeded":
        expected["sharing_profile"] = "gkm.seeded.aes128.duplicate-le64.two-purpose.window.v1"
    if (set(bound) != {*expected, "decoder"}
            or any(bound[key] != value for key, value in expected.items())
            or bound["decoder"] != decoder):
        raise DelegatedSourceError("GKM native runtime contract differs from its source or Experiment")
    # Native float spelling is part of its canonical digest. Never re-encode
    # either this record or its nested decoder to produce transport/digest bytes.
    return result


def _admit_source(
    model_or_experiment: Model | Experiment | str | Mapping[str, Any], *,
    input_rows: int | None = None, decode_steps: int | None = None,
) -> AdmittedSource:
    """Admit a complete source envelope before tensor values or ordinary lowering.

    Experiment uses its complete budget (max_input_tokens, max_new_tokens - 1),
    with native canonical validation before file access. Model probes require
    both explicit workload sizes and carry no SDK admission. Use the returned
    owner's runtime_contract(experiment, rows, outputs) for actual SDK requests.
    Native source/estimated payload bytes are available before load_values().
    """
    experiment = model_or_experiment if isinstance(model_or_experiment, Experiment) else None
    experiment_bytes = None
    if experiment is not None:
        if input_rows is not None or decode_steps is not None:
            raise DelegatedSourceError("GKM Experiment workload comes from its complete budget")
        experiment_bytes = _experiment_bytes(experiment)
        input_rows, decode_steps = experiment.budget.max_input_tokens, experiment.budget.max_new_tokens - 1
        model_or_experiment = experiment.pipeline.model
    model = model_loader.coerce_model(model_or_experiment)
    if model.kind == "tiny":
        from pllm.sources import _tiny_model_config

        decoder_contract(_tiny_model_config(), input_rows=input_rows, decode_steps=decode_steps,
                         streamed=experiment is not None)
        raise DelegatedSourceError("GKM Model.tiny() full vocabulary is unsupported")
    if model.kind not in {"huggingface", "safetensors"}:
        raise DelegatedSourceError("GKM source requires an HF/Safetensors Model")
    root = Path(model.source).expanduser()
    if model.kind == "huggingface" and not root.exists() and not root.is_absolute():
        # Resolve through the shared Hub cache without constructing an ordinary
        # W8A8 graph. Native geometry admission still precedes tensor decoding.
        from .hf_hub import resolve_huggingface_source

        root = resolve_huggingface_source(
            model.source, model_id=model.model_id, revision=model.revision,
            local_files_only=model.local_files_only,
        ).path
    if not root.is_dir():
        raise DelegatedSourceError("GKM source requires an existing local HF/Safetensors directory")
    root = root.resolve()
    original_identity = _identity(root / "config.json")
    original = _read_bounded(root / "config.json", _MAX_CONFIG_BYTES)
    raw_config = _json(original, "config.json")
    config_bytes = _encode_config(raw_config)
    for cfg in (raw_config, raw_config.get("text_config", {})):
        if isinstance(cfg, dict) and any(cfg.get(key) is not None for key in ("quantization", "quantization_config")):
            raise DelegatedSourceError("GKM requires original unquantized floating-point tensors")
    # Full-envelope geometry precedes hashing, header readers and tokenizers.
    capacity = "capsule" if experiment is None else experiment.pipeline.components["linear"].params.get("capacity", "capsule")
    envelope_bytes = decoder_contract(raw_config, input_rows=input_rows, decode_steps=decode_steps,
                                      capacity=capacity, streamed=experiment is not None)
    envelope = _json(envelope_bytes, "native envelope contract")
    source_bytes = _validate_contract(envelope)
    if _capacity(envelope) != capacity:
        raise DelegatedSourceError("GKM native capacity differs from Experiment")
    identities = _files(root, source_bytes, capacity)
    if dict(identities)["config.json"] != original_identity:
        raise DelegatedSourceError("GKM source configuration changed during admission")
    store, specs = _store(root, identities, envelope)
    store.clear_cache()
    checkpoint, lock_digest, lock = _locked(root, model, source_bytes, identities, capacity)
    hashes = {row["path"]: row["sha256"] for row in lock["files"]}
    if hashes["config.json"] != hashlib.sha256(original).hexdigest():
        raise DelegatedSourceError("GKM source configuration differs from its lock")
    tokenizer_bytes = _read_bounded(root / "tokenizer.json", _DECODER_TOKENIZER_BYTES
                                    if capacity == "decoder" else _MAX_TOKENIZER_BYTES)
    tokenizer_digest = hashlib.sha256(tokenizer_bytes).hexdigest()
    if hashes["tokenizer.json"] != tokenizer_digest:
        raise DelegatedSourceError("GKM tokenizer differs from its source lock")
    runtime_bytes = None if experiment is None else _bind_runtime(
        experiment, experiment_bytes, config_bytes, envelope_bytes, lock_digest, tokenizer_digest,
    )
    manifest = _manifest(root, model, raw_config, identities)
    if (manifest.vocab_size != envelope["vocabulary"] or manifest.hidden_size != envelope["width"]
            or manifest.context_length < input_rows + decode_steps):
        raise DelegatedSourceError("GKM source table/head geometry or context differs")
    descriptor = _tokenizer(root, manifest, manifest.metadata["text_config"], envelope)
    if descriptor.pop("model") != tokenizer_bytes:
        raise DelegatedSourceError("GKM tokenizer changed during construction")
    if _locked(root, model, source_bytes, identities, capacity) != (checkpoint, lock_digest, lock):
        raise DelegatedSourceError("GKM source lock changed during admission")
    manifest.metadata.update(model_spec=model.to_spec(), checkpoint_digest=checkpoint,
                             source_lock_digest=lock_digest, source_lock=lock, config_only=False)
    manifest.chat_template = descriptor["chat_template"]
    return AdmittedSource(
        model, root, config_bytes, original, envelope_bytes, checkpoint, lock_digest,
        tokenizer_bytes, tokenizer_digest, _canonical(lock),
        _canonical(manifest.to_dict(include_fingerprint=False)), _canonical(descriptor), identities, specs,
        experiment_bytes, runtime_bytes,
    )


def admit_source(model_or_experiment, *, input_rows=None, decode_steps=None) -> AdmittedSource:
    """Admit source metadata and the complete native envelope before tensor reads."""
    if (isinstance(model_or_experiment, Experiment)
            and model_or_experiment.pipeline.components.get("linear") is not None
            and model_or_experiment.pipeline.components["linear"].params.get("capacity") == "decoder"):
        _experiment_bytes(model_or_experiment)
        # Full-vocabulary JSON parsing, reverse indexes, tokenizer construction,
        # source graph copies and retained raw bytes are real metadata owners.
        with tokenizer_admission.reserve_decoder_metadata():
            return _admit_source(model_or_experiment, input_rows=input_rows, decode_steps=decode_steps)
    return _admit_source(model_or_experiment, input_rows=input_rows, decode_steps=decode_steps)


__all__ = ["AdmittedSource", "DelegatedSourceError", "LoadedSource", "admit_source"]

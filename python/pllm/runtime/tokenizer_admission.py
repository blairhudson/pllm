"""Shared bounded tokenizer admission, independent of model execution semantics.

Model vocabulary can include rows without tokenizer entries. Admission never
adds entries, reindexes IDs, or removes model rows. Callers retain the original
tokenizer bytes and use the ordinary TokenizersJSONTokenizer for text decoding.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import stat


DECODER_METADATA_MEMORY_BYTES = 384 << 20


@contextmanager
def reserve_decoder_metadata():
    """Shared source-graph, full-vocabulary JSON and reverse-index workspace.

    Charge before source inspection/parsing. Retained source bytes have their
    ordinary caller ownership; this allowance covers transient metadata work.
    """
    from pllm.deployment.resources import reserve_incremental_memory
    from .delegated_protocol import clear_failure_frames

    with reserve_incremental_memory(DECODER_METADATA_MEMORY_BYTES):
        try:
            yield
        except BaseException as error:
            clear_failure_frames(error)
            raise


def validate_document(document, *, vocabulary, max_entries, error, label):
    """Bound IDs and shape before the tokenizer allocates its reverse indexes."""
    if document.get("padding") is not None:
        raise error(f"{label} tokenizer padding is not admitted")
    if document.get("truncation") is not None:
        raise error(f"{label} tokenizer must not silently truncate source input")
    model = document.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    added = document.get("added_tokens", [])
    if (not isinstance(vocab, (dict, list)) or not 1 <= len(vocab) <= max_entries
            or not isinstance(added, list) or len(added) > max_entries):
        raise error(f"{label} tokenizer vocabulary exceeds its complete-table limit")
    ids = vocab.values() if isinstance(vocab, dict) else range(len(vocab))
    if any(type(i) is not int or not 0 <= i < vocabulary for i in ids):
        raise error(f"{label} tokenizer IDs exceed the complete vocabulary")
    if any(type(row) is not dict or type(row.get("id")) is not int
           or not 0 <= row["id"] < vocabulary for row in added):
        raise error(f"{label} added tokenizer IDs exceed the complete vocabulary")


def validate_vocabulary(inner, *, vocabulary, allow_model_padding, error, label):
    """Require a contiguous, reversible original ID space within the model head."""
    complete = inner.get_vocab(with_added_tokens=True)
    expected_size = int(inner.get_vocab_size()) if allow_model_padding else vocabulary
    if (not 1 <= expected_size <= vocabulary or len(complete) != expected_size
            or set(complete.values()) != set(range(expected_size))
            or any(inner.id_to_token(i) != text for text, i in complete.items())):
        raise error(f"{label} tokenizer must cover the complete table/head vocabulary")
    return expected_size


# Private factory policies, not request options or source-authentication tokens.


class _DescriptorPolicy(Enum):
    DELEGATED = (2, True)
    FSS = (0, False)


def _json_metadata(raw, *, error, label):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise error(f"{label} duplicate metadata key")
            result[key] = value
        return result

    def invalid(_):
        raise error(f"{label} nonfinite metadata")

    try:
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise error(f"{label} invalid metadata JSON") from exc
    if type(result) is not dict:
        raise error(f"{label} metadata must be an object")
    return result


def _source_descriptor(raw, config, options, *, policy, vocabulary, max_entries,
                       allow_model_padding, manifest_template, error, label,
                       document=None):
    """Original tokenizer metadata contract shared by real factories and preview.

    No native geometry/contract is created. Factory callers keep their original
    source/context admission separately. This function never renders/encodes.
    """
    from tokenizers import Tokenizer

    if type(policy) is not _DescriptorPolicy:
        raise error(f"{label} unknown tokenizer metadata policy")
    if type(options.get("add_bos_token", True)) is not bool:
        raise error(f"{label} add_bos_token must be boolean")
    if policy is _DescriptorPolicy.DELEGATED and config.get("pllm_test_tokenizer") is not None:
        raise error(f"{label} requires genuine tokenizer.json source semantics")
    doc = _json_metadata(raw, error=error, label=label) if document is None else document
    validate_document(doc, vocabulary=vocabulary, max_entries=max_entries, error=error, label=label)
    del doc, document  # Retire parsed vocabulary before reverse-index overlap.
    bos = config.get("bos_token_id", policy.value[0])
    eos = config.get("eos_token_id", 1)
    eos_ids = eos if type(eos) is list else [eos]
    if not eos_ids or any(type(i) is not int or not 0 <= i < vocabulary for i in [bos, *eos_ids]):
        raise error(f"{label} special token IDs exceed complete vocabulary")
    try:
        inner = Tokenizer.from_buffer(raw)
    except Exception as exc:
        raise error(f"{label} invalid tokenizer.json model") from exc
    size = validate_vocabulary(inner, vocabulary=vocabulary,
        allow_model_padding=allow_model_padding, error=error, label=label)
    if any(i >= size for i in [bos, *eos_ids]):
        raise error(f"{label} special tokens require original tokenizer entries")
    if inner.truncation is not None:
        raise error(f"{label} tokenizer must not silently truncate source input")
    template = options.get("chat_template")
    if policy.value[1]:
        template = template or manifest_template
    return {"type": "tokenizer_json", "model": raw, "vocab_size": vocabulary,
            "bos_token_id": bos, "eos_token_id": eos_ids[0], "chat_template": template,
            "bos_token": options.get("bos_token", ""), "eos_token": options.get("eos_token", ""),
            "add_bos_token": options.get("add_bos_token", True)}


class _PromptMetadataError(ValueError):
    pass


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _named_identity(path, *, optional):
    try:
        link = path.lstat()
    except FileNotFoundError:
        if optional:
            return None
        raise _PromptMetadataError(f"missing prompt metadata: {path.name}") from None
    try:
        target = path.stat()
    except OSError as exc:
        raise _PromptMetadataError(f"invalid prompt metadata: {path.name}") from exc
    if not stat.S_ISREG(target.st_mode):
        raise _PromptMetadataError(f"prompt metadata is not a regular file: {path.name}")
    return (_identity(link), _identity(target))


def _read_named(root, name, limit, *, optional=False):
    # These names are closed at _capture_prompt_metadata, never supplied by a request.
    path = root / name
    before = _named_identity(path, optional=optional)
    if before is None:
        return None, None
    if not 0 < before[1][2] <= limit:
        raise _PromptMetadataError(f"oversize/empty prompt metadata: {name}")
    import os
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nonblock is None:
        raise _PromptMetadataError("nonblocking metadata open is unavailable")
    descriptor = None
    try:
        # Follow legitimate HF-cache FILE symlinks, but never block opening a
        # raced FIFO. Only an opened regular file with the checked target
        # identity may reach read(). O_NOFOLLOW would reject supported caches.
        descriptor = os.open(path, os.O_RDONLY | nonblock | getattr(os, "O_CLOEXEC", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != before[1]:
            raise _PromptMetadataError(f"changed/nonregular prompt metadata: {name}")
        stream = os.fdopen(descriptor, "rb")
        descriptor = None  # stream now owns it, including exceptional exits.
        with stream:
            raw = stream.read(limit + 1)
            if (len(raw) != before[1][2] or len(raw) > limit
                    or _identity(os.fstat(stream.fileno())) != before[1]):
                raise _PromptMetadataError(f"changed prompt metadata: {name}")
    except OSError as exc:
        raise _PromptMetadataError(f"cannot open/read prompt metadata: {name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if _named_identity(path, optional=optional) != before:
        raise _PromptMetadataError(f"changed prompt metadata: {name}")
    return raw, before


@dataclass(frozen=True, slots=True)
class _PromptMetadataSnapshot:
    root: Path
    root_identity: tuple
    model_spec: tuple
    config_bytes: bytes
    tokenizer_bytes: bytes
    options_bytes: bytes | None
    descriptor_bytes: bytes
    identities: tuple
    expected_source_lock: str | None

    @property
    def tokenizer_descriptor(self):
        return {**json.loads(self.descriptor_bytes), "model": self.tokenizer_bytes}

    @property
    def config(self):
        return _json_metadata(self.config_bytes, error=_PromptMetadataError, label="prompt config")

    def association(self):
        from .responses import _prompt_snapshot
        return _prompt_snapshot(("provisional-local-metadata", self.model_spec,
            self.config_bytes, self.tokenizer_descriptor, self.identities))

    def check_unchanged(self):
        info = self.root.stat()
        if (info.st_dev, info.st_ino) != self.root_identity:
            raise _PromptMetadataError("prompt metadata directory changed")
        for name, identity in self.identities:
            if _named_identity(self.root / name, optional=name == "tokenizer_config.json") != identity:
                raise _PromptMetadataError(f"prompt metadata changed before binding: {name}")

    def check_source(self, source):
        """One-way comparison AFTER original real source admission, not its replacement."""
        from .responses import _prompt_snapshot
        self.check_unchanged()
        if (_prompt_snapshot(source.model.to_spec()) != self.model_spec
                or source.config_source_bytes != self.config_bytes
                or _prompt_snapshot(source.tokenizer_descriptor)
                != _prompt_snapshot(self.tokenizer_descriptor)
                or (self.expected_source_lock is not None
                    and source.source_lock_digest != self.expected_source_lock)):
            raise _PromptMetadataError("prepared metadata/source association changed")

    def tokenizer(self):
        from tokenizers import Tokenizer
        from .transformer_client import TokenizersJSONTokenizer
        descriptor = self.tokenizer_descriptor
        return _PromptTokenizer(TokenizersJSONTokenizer(Tokenizer.from_buffer(self.tokenizer_bytes),
            descriptor["bos_token_id"], descriptor["eos_token_id"]))


class _PromptTokenizer:
    """Retire parsed tokenizer BEFORE the request's metadata allowance."""
    def __init__(self, inner):
        self._inner = inner

    def encode(self, text, *, add_bos):
        if self._inner is None:
            raise _PromptMetadataError("prompt tokenizer retired")
        return self._inner.encode(text, add_bos=add_bos)

    def decode(self, ids):
        if self._inner is None:
            raise _PromptMetadataError("prompt tokenizer retired")
        return self._inner.decode(ids)

    def close(self):
        self._inner = None


def _capture_prompt_metadata(model, *, policy, capacity, input_rows, outputs,
                             manifest_template=None, expected_source_lock=None):
    """Provisional named metadata only. Caller holds existing metadata reservation.

    Does not resolve models/HF refs, enumerate files, hash checkpoints, inspect
    headers/indexes/stores or invoke native contracts. No cache download fallback.
    """
    from .responses import _prompt_snapshot

    if type(policy) is not _DescriptorPolicy or capacity not in {"capsule", "decoder"}:
        raise _PromptMetadataError("unknown prompt metadata contract")
    path = Path(model.source).expanduser()
    if model.kind not in {"huggingface", "safetensors"} or not path.is_dir():
        raise _PromptMetadataError("prompt preview needs an explicit existing local/cached metadata directory")
    root = path.resolve(strict=True)
    info = root.stat()
    root_identity = (info.st_dev, info.st_ino)
    raw_config, cfg_id = _read_named(root, "config.json", 16 << 10)
    raw_tokenizer, tok_id = _read_named(root, "tokenizer.json",
        16 << 20 if capacity == "decoder" else 256 << 10)
    raw_options, opt_id = _read_named(root, "tokenizer_config.json", 16 << 10, optional=True)
    config = _json_metadata(raw_config, error=_PromptMetadataError, label="prompt config")
    cfg = config.get("text_config") if type(config.get("text_config")) is dict else config
    options = {} if raw_options is None else _json_metadata(raw_options,
        error=_PromptMetadataError, label="tokenizer options")
    vocabulary = cfg.get("vocab_size")
    maximum = 262144 if capacity == "decoder" else 256
    if type(vocabulary) is not int or not 1 <= vocabulary <= maximum:
        raise _PromptMetadataError("prompt vocabulary exceeds source metadata bounds")
    required = input_rows + outputs - 1
    if policy is _DescriptorPolicy.FSS:
        context = required  # Same absent-options default as the original FSS loader.
    else:
        context = next((cfg[k] for k in ("max_position_embeddings", "max_sequence_length", "seq_length")
                        if cfg.get(k) is not None), 2048)
    available = options.get("model_max_length", context)
    if (type(context) is not int or type(available) is not int
            or context < required or available < required):
        raise _PromptMetadataError("prompt exceeds source/tokenizer metadata context")
    descriptor = _source_descriptor(raw_tokenizer, cfg, options, policy=policy,
        vocabulary=vocabulary, max_entries=vocabulary if capacity == "decoder" else 256,
        allow_model_padding=capacity == "decoder", manifest_template=manifest_template,
        error=_PromptMetadataError, label="prompt")
    del descriptor["model"]
    result = _PromptMetadataSnapshot(root, root_identity, _prompt_snapshot(model.to_spec()),
        raw_config, raw_tokenizer, raw_options,
        json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
        (("config.json", cfg_id), ("tokenizer.json", tok_id), ("tokenizer_config.json", opt_id)),
        expected_source_lock)
    result.check_unchanged()
    return result

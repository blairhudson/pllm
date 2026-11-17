"""Dormant local toy attempt: SDK lock -> pinned files -> exact native append.

Trusted, already resolved temporary ancestors only. No HF authority, production
issuer, worker activation, resource payer, RSS bound or trained-model approval.
The finite envelopes below count logical objects/work, not allocator overhead.
No owner, fd, selectable reader pair or append callback escapes this attempt.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import stat
import struct
import tempfile
import threading

from pllm.configuration import Experiment, _bounded_canonical_preflight
from pllm.deployment.resources import _CleanupSlot
from . import delegated_source as source_io
from ._authenticated_source_file import (
    _AuthenticatedSourceFile, _FileBounds, _ReadAllowance, _SelectedFile,
    _requirements,
)

_FILES = 8
_FILE_BYTES = 256 << 10
_TOTAL_BYTES = 512 << 10
_RETURN_BYTES = 64 << 10
_CALLS = 4096
_LEAF_WORK = 64 << 20
_TENSORS = 256
_FRAME_ELEMENTS = 4096


def _refuse(message):
    raise source_io.DelegatedSourceError(f"local toy generation: {message}")


def _local_inventory(root):
    """No ignored entries, aliases, traversal or payload reads in toy preflight."""
    rows = []
    total = indexes = 0
    with os.scandir(root) as entries:
        for entry in entries:
            if len(rows) == _FILES:
                _refuse("file count exceeds envelope")
            name = entry.name
            index = name.endswith(".safetensors.index.json")
            if (not name or name in {".", ".."} or any(c in name for c in "/\\\0")
                    or not (name in {"config.json", "tokenizer.json", "tokenizer_config.json"}
                            or name.endswith(".safetensors") or index)):
                _refuse("unsupported inventory")
            info = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                _refuse("inventory requires regular entries, not aliases")
            if info.st_nlink != 1:
                _refuse("inventory requires singly linked files")
            if not 0 <= info.st_size <= _FILE_BYTES:
                _refuse("file bytes exceed envelope")
            # Retain stricter ordinary limits for known metadata kinds.
            if info.st_size > source_io._METADATA_LIMITS.get(name, _FILE_BYTES):
                _refuse("metadata bytes exceed ordinary limit")
            total += info.st_size
            indexes += index
            if total > _TOTAL_BYTES or indexes > 1:
                _refuse("inventory exceeds envelope")
            identity = source_io._stat_identity(info)
            rows.append((name, identity + identity))
    names = {name for name, _ in rows}
    if not {"config.json", "tokenizer.json"} <= names or not any(
            name.endswith(".safetensors") for name in names):
        _refuse("incomplete inventory")
    return tuple(sorted(rows))


def _preflight(experiment, ids, outputs):
    """Bound the local domain BEFORE ordinary SDK admission can resolve a path."""
    from .replicated_protocol import storage_policy

    if type(experiment) is not Experiment:
        _refuse("requires exact Experiment")
    census = _bounded_canonical_preflight(experiment, max_bytes=32 << 10)
    encoded = experiment.canonical_bytes()
    if not 0 < len(encoded) == census.encoded_bytes <= 32 << 10:
        _refuse("Experiment exceeds envelope")
    if (experiment.resolve().client_runtime != "compiled_replicated_v1"
            or storage_policy(experiment) != "expanded"):
        _refuse("requires expanded replicated workload")
    if (type(ids) is not tuple or not 1 <= len(ids) <= 8
            or any(type(token) is not int or not 0 <= token < (1 << 32) for token in ids)
            or type(outputs) is not int or not 1 <= outputs <= 2
            or not len(ids) <= experiment.budget.max_input_tokens <= 8
            or not outputs <= experiment.budget.max_new_tokens <= 2):
        _refuse("workload exceeds envelope")
    model = experiment.pipeline.model
    if model.kind not in {"safetensors", "huggingface"}:
        _refuse("requires local Safetensors directory")
    raw = model.source
    if (type(raw) is not str or not raw.startswith("/") or len(raw) > 4096
            or "\0" in raw or any(part in {"", ".", ".."} for part in raw[1:].split("/"))):
        _refuse("requires resolved absolute local path")
    # This is intentionally NOT a hostile-ancestor walk. The caller must own
    # these private temporary ancestors. Stable aliases are still forbidden.
    if os.path.realpath(raw, strict=True) != raw:
        _refuse("source path is aliased")
    root = Path(raw)
    temporary = Path(tempfile.gettempdir()).resolve(strict=True)
    if root == temporary or not root.is_relative_to(temporary) or not root.is_dir():
        _refuse("source is outside trusted temporary toy domain")
    info = root.stat()
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        _refuse("temporary source directory is not privately controlled")
    inventory = _local_inventory(root)
    identity = dict(inventory)["config.json"]
    original = source_io._read_range(root / "config.json", identity, 0, identity[9],
                                     limit=source_io._MAX_CONFIG_BYTES)
    raw_config = source_io._json(original, "config.json")
    # Shared alias-aware dimension parsing, not a paper-specific tensor/name
    # table. Bound native plan construction as well as the subsequent readers.
    manifest = source_io._manifest(root, model, raw_config, inventory)
    for value, maximum in ((manifest.vocab_size, 256), (manifest.hidden_size, 16),
            (manifest.intermediate_size, 64), (manifest.num_hidden_layers, 2),
            (manifest.num_attention_heads, 8), (manifest.num_key_value_heads, 8),
            (manifest.head_dim, 16), (manifest.context_length, 256)):
        if type(value) is not int or not 1 <= value <= maximum:
            _refuse("configuration geometry exceeds toy envelope")
    return encoded, root, inventory


@dataclass(frozen=True, slots=True)
class _Envelope:
    file_slots: int
    table_bytes: int
    active_payload_bytes: int
    seal_requested_bytes: int
    read_calls: int
    leaf_work_bytes: int
    json_assembly_bytes: int
    tensor_slots: int
    frame_bytes: int
    comparison_scratch_bytes: int


def _envelope(rows):
    needs = tuple(_requirements(identity[9], _RETURN_BYTES) for _, identity in rows)
    tables = sum(item.retained_bytes for item in needs)
    peak = max((tables - item.retained_bytes + max(
        item.seal_payload_bytes, item.read_payload_bytes) for item in needs), default=0)
    # JSON decode objects are additionally bounded by <=512 KiB source, <=256
    # tensor records and existing JSON/header/tokenizer policies. These logical
    # counts are not a measured allocation budget, nor payer/funding credit.
    return _Envelope(len(rows), tables, peak, sum(item.seal_requested_bytes for item in needs),
        len(rows) * _CALLS, len(rows) * _LEAF_WORK, 4 * _TOTAL_BYTES + _RETURN_BYTES,
        _TENSORS, 4 * _FRAME_ELEMENTS * 4, 4096)


def _selected_rows(source, inventory):
    """Bind every ordered real SourceLock row to exactly one local identity."""
    if source._identities != inventory:
        _refuse("inventory changed across SDK admission")
    lock = source.source_lock
    files = lock.get("files")
    if type(files) is not list or len(files) != len(inventory):
        _refuse("lock inventory differs")
    selected = []
    for item, (name, identity) in zip(files, inventory, strict=True):
        if (type(item) is not dict or set(item) != {"path", "size", "sha256"}
                or type(item["path"]) is not str or item["path"] != name
                or type(item["size"]) is not int or item["size"] != identity[9]
                or type(item["sha256"]) is not str or len(item["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in item["sha256"])
                or type(identity) is not tuple or len(identity) != 12
                or identity[:6] != identity[6:] or not stat.S_ISREG(identity[2])):
            _refuse("invalid or unlocked local inventory row")
        selected.append(_SelectedFile(str(source.path / name), name, item["size"],
                                      identity[:6], bytes.fromhex(item["sha256"])))
    return tuple(selected)


def _widen_reference(raw, dtype):
    """Independent explicit-LE IEEE widening. No NumPy or SDK converter."""
    width = 4 if dtype == "F32" else 2
    if dtype not in {"F32", "F16", "BF16"} or len(raw) % width:
        _refuse("invalid reference words")
    result = bytearray()
    try:
        for offset in range(0, len(raw), width):
            word = int.from_bytes(raw[offset:offset + width], "little")
            if dtype == "F32":
                if word & 0x7F800000 == 0x7F800000:
                    _refuse("nonfinite reference word")
                result.extend(word.to_bytes(4, "little"))
            elif dtype == "BF16":
                if word & 0x7F80 == 0x7F80:
                    _refuse("nonfinite reference word")
                result.extend((word << 16).to_bytes(4, "little"))
            else:
                if word & 0x7C00 == 0x7C00:
                    _refuse("nonfinite reference word")
                result.extend(struct.pack("<f", struct.unpack_from("<e", raw, offset)[0]))
        return bytes(result)
    finally:
        result.clear()


class _Generation:
    """One serialized attempt, rooted before acquisition, all-slot retirement.

    Internal only: the entrypoint never returns this object. A busy close stops
    the attempt and defers retirement. Ambiguous closes are never retried, and
    other slots are still visited. No GC cleanup or implicit refund.
    """
    def __init__(self, cleanup_scope, cancelled):
        if type(cleanup_scope) is not _CleanupSlot or cleanup_scope.closed:
            _refuse("requires admitted cleanup custody")
        if not callable(cancelled):
            _refuse("requires cancellation hook")
        self.__scope, self.__cancelled = cleanup_scope, cancelled
        self.__gate = threading.Lock()
        self.__busy, self.__stopped = True, False
        self.__io_active = self.__ready = False
        self.__slots = []  # [owner, terminal ambiguity]; staged before any seal
        self.__publisher = None
        self.__publisher_unknown = False
        self.__by_name = {}
        self.inventory = ()
        self.envelope = None
        self.mapping = ()
        self.__cursor = (0, 0)
        cleanup_scope.retain(self, self._retire)

    def _checkpoint(self):
        if self.__stopped or self.__cancelled() is not False or self.__stopped:
            _refuse("cancelled or retired")

    def _stage(self, source, inventory):
        if (not 1 <= len(source.contract["source_shapes"]) <= _TENSORS
                or source.source_float32_bytes > 2 * _TOTAL_BYTES):
            _refuse("native geometry exceeds toy envelope")
        self.envelope = _envelope(inventory)
        rows = _selected_rows(source, inventory)
        # Every inert slot is retained before the first descriptor acquisition.
        for row in rows:
            needs = _requirements(row.size, _RETURN_BYTES)
            owner = _AuthenticatedSourceFile(row,
                _FileBounds(_FILE_BYTES, needs.retained_bytes, _RETURN_BYTES,
                            max(needs.seal_payload_bytes, needs.read_payload_bytes)),
                _ReadAllowance(_CALLS, _LEAF_WORK))
            self.__slots.append([owner, False])
            self.__by_name[row.basename] = owner
        self.inventory = tuple((row.basename, row.size) for row in rows)
        for owner, _ in self.__slots:
            with self._file_operation():
                self._checkpoint()
                owner.seal(cancelled=self._cancel_check)

    @contextmanager
    def _file_operation(self):
        with self.__gate:
            if self.__io_active or self.__stopped:
                self.__stopped = True
                _refuse("generation I/O busy or stopped")
            self.__io_active = True
        try:
            yield
        finally:
            with self.__gate:
                self.__io_active = False

    def _cancel_check(self):
        self._checkpoint()
        return False

    def _read_at(self, name, offset, count):
        with self._file_operation():
            self._checkpoint()
            return self.__by_name[name].read_verified(offset, count, cancelled=self._cancel_check)

    def _document(self, name, limit):
        size = dict(self.inventory)[name]
        if size > min(limit, _FILE_BYTES):
            _refuse("document exceeds assembly limit")
        result = bytearray()
        block = None
        try:
            for offset in range(0, max(1, size), _RETURN_BYTES):
                block = self._read_at(name, offset, min(_RETURN_BYTES, size - offset))
                result.extend(block)
            return bytes(result)
        finally:
            result.clear()
            del block

    def _join(self, source, experiment, encoded, ids, outputs):
        if source.experiment_bytes != encoded or experiment.canonical_bytes() != encoded:
            _refuse("Experiment changed")
        contract = source.contract
        if (not 1 <= len(contract["source_shapes"]) <= _TENSORS
                or source.source_float32_bytes > 2 * _TOTAL_BYTES
                or any(token >= contract["vocabulary"] for token in ids)):
            _refuse("native geometry exceeds toy envelope")
        store, specs = source_io._parse_tensor_inventory(
            self.inventory, self._read_at, source_io._tensor_geometry(contract))
        try:
            if specs != source._tensor_specs:
                _refuse("fresh tensor metadata differs from admission")
            parsed = dict(specs)
            mapping = []
            for key, shape in sorted(contract["source_shapes"].items()):
                physical = store.resolve(key)
                filename, actual, dtype, offset, size = parsed[physical]
                if actual != tuple(shape) or size != math.prod(shape) * source_io._DTYPES[dtype].itemsize:
                    _refuse("resolved tensor geometry differs")
                mapping.append((key, physical, filename, actual, dtype, offset, size))
            if {row[1] for row in mapping} != set(parsed) or len(mapping) != len(parsed):
                _refuse("logical mapping is not bijective")
            self.mapping = tuple(mapping)
        finally:
            store.clear_cache()
        original = self._document("config.json", source_io._MAX_CONFIG_BYTES)
        raw = source_io._json(original, "config.json")
        if original != source.config_source_bytes or source_io._encode_config(raw) != source.config_bytes:
            _refuse("configuration differs from admission")
        tokenizer = self._document("tokenizer.json", source_io._DECODER_TOKENIZER_BYTES)
        if (tokenizer != source.tokenizer_bytes
                or hashlib.sha256(tokenizer).hexdigest() != source.tokenizer_digest):
            _refuse("tokenizer differs from admission")
        options = (source_io._json(self._document("tokenizer_config.json", source_io._MAX_CONFIG_BYTES),
                                  "tokenizer_config.json")
                   if "tokenizer_config.json" in dict(self.inventory) else {})
        # expected_model_id() probes paths for ordinary Model.path callers. Bind
        # its already-admitted value instead; no sealed-generation path lookup.
        admitted_manifest = source_io._json(source._manifest_bytes, "admitted manifest")
        model_id = admitted_manifest["id"]
        manifest = source_io._manifest(source.path, source.model, raw, source._identities,
                                        _model_id=model_id)
        # Creation time is admission transport metadata, not derived from the
        # locked config. A new ModelManifest gets a fresh clock sample, but the
        # exact serialized admission snapshot below includes created_at. Retain
        # that admitted value; independently rebuild every source-derived field.
        manifest.created_at = admitted_manifest["created_at"]
        descriptor = source_io._tokenizer_from_bytes(tokenizer, options, manifest,
                                                     manifest.metadata["text_config"], contract)
        if descriptor.pop("model") != tokenizer:
            _refuse("descriptor model differs")
        manifest.metadata.update(model_spec=source.model.to_spec(), checkpoint_digest=source.checkpoint_digest,
            source_lock_digest=source.source_lock_digest, source_lock=source.source_lock, config_only=False)
        manifest.chat_template = descriptor["chat_template"]
        if source_io._canonical(descriptor) != source._tokenizer_options_bytes:
            _refuse("tokenizer options differ from admission")
        if source_io._canonical(manifest.to_dict(include_fingerprint=False)) != source._manifest_bytes:
            _refuse("manifest differs from admission")
        if (manifest.vocab_size != contract["vocabulary"] or manifest.hidden_size != contract["width"]
                or manifest.context_length < len(ids) + outputs - 1):
            _refuse("manifest geometry differs")
        if source.runtime_contract(experiment, experiment.budget.max_input_tokens,
                                   experiment.budget.max_new_tokens) != source.runtime_contract_bytes:
            _refuse("native admission binding differs")
        actual = source_io._json(source.runtime_contract(experiment, len(ids), outputs), "native runtime contract")
        source_io._tensor_geometry(actual)
        if actual["source_shapes"] != contract["source_shapes"]:
            _refuse("actual request source geometry differs")
        self._checkpoint()
        self.__ready = True

    def _reference_frame(self, row, begin, count):
        # Separate cursor and separately positioned small-block reads of the
        # SAME closed-over generation/map. No selectable second source owner.
        index, expected = self.__cursor
        if (type(begin) is not int or index >= len(self.mapping)
                or self.mapping[index] != row or begin != expected):
            _refuse("reference frame order or offset differs")
        _, _, filename, shape, dtype, offset, span = row
        elements = math.prod(shape)
        if type(count) is not int or count != min(_FRAME_ELEMENTS, elements - begin) or count <= 0:
            _refuse("reference frame count differs")
        # Reference byte stride is format-defined, not the SDK dtype table.
        width = {"F32": 4, "F16": 2, "BF16": 2}.get(dtype)
        if width is None or span != elements * width:
            _refuse("reference tensor span differs")
        result = bytearray()
        raw = widened = None
        try:
            for lane in range(0, count, 256):
                length = min(256, count - lane)
                raw = self._read_at(filename, offset + (begin + lane) * width, length * width)
                widened = _widen_reference(raw, dtype)
                result.extend(widened)
            return bytes(result)
        finally:
            result.clear()
            del raw, widened

    def _publish(self, source, experiment, ids, outputs):
        from .replicated_source import _expanded_publisher
        from .replicated_protocol import validate_execution_binding

        if not self.__ready:
            _refuse("metadata join is not ready")
        self.__ready = False  # one publication attempt, never replay
        self._checkpoint()
        self.__publisher = _expanded_publisher(source, experiment, len(ids), outputs)
        owner = self.__publisher
        append = owner.append
        expected_shapes = tuple((row[0], row[3]) for row in self.mapping)
        if tuple((key, tuple(shape)) for key, shape in owner.shapes) != expected_shapes:
            _refuse("publisher order/shape/coverage differs")
        for index, row in enumerate(self.mapping):
            key, _, filename, shape, dtype, offset, _ = row
            width = source_io._DTYPES[dtype].itemsize
            elements = math.prod(shape)
            for begin in range(0, elements, _FRAME_ELEMENTS):
                self._checkpoint()
                count = min(_FRAME_ELEMENTS, elements - begin)
                raw = self._read_at(filename, offset + begin * width, count * width)
                frame = source_io._source_float_frame_from_bytes(raw, dtype)
                del raw
                reference = self._reference_frame(row, begin, count)
                if type(frame) is not bytes or len(frame) != count * 4 or frame != reference:
                    _refuse("pre-append F32 word comparison failed")
                # Identical immutable object, exactly once. No observer, callback
                # or converter between comparison and this bound real append.
                append(key, begin, frame)
                self.__cursor = ((index + 1, 0) if begin + count == elements
                                 else (index, begin + count))
                del frame, reference
        if self.__cursor != (len(self.mapping), 0):
            _refuse("incomplete append coverage")
        self._checkpoint()  # includes cancellation after the last append
        manifest = owner.seal(b"".join(token.to_bytes(4, "little") for token in ids))
        validate_execution_binding(experiment, source_io._json(manifest, "publisher manifest"))
        return manifest

    def _finish_operation(self):
        with self.__gate:
            self.__busy = False

    def _retire(self):
        with self.__gate:
            self.__stopped = True
            if self.__busy:
                _refuse("operation in flight; retirement deferred")
            self.__busy = True
        failed = False
        try:
            # Attempt every staged slot even when an earlier close fails. Never
            # retain exception tracebacks as the durable ownership root.
            for slot in self.__slots:
                owner, unknown = slot
                if owner is None:
                    continue
                if unknown:
                    failed = True
                    continue
                try:
                    if owner.close() is True:
                        slot[0] = None
                    else:
                        failed = True
                except BaseException:
                    slot[1] = True
                    failed = True
            if self.__publisher is not None:
                if self.__publisher_unknown:
                    failed = True
                else:
                    try:
                        if self.__publisher.close() is False:
                            failed = True
                        else:
                            self.__publisher = None
                    except BaseException:
                        self.__publisher_unknown = True
                        failed = True
            if failed:
                _refuse("retirement pending or ambiguous; custody retained")
            self.__by_name.clear()
            self.mapping = self.inventory = ()
        finally:
            self._finish_operation()


def _run_local_toy(experiment, ids, outputs, cancelled, *, cleanup_scope):
    from .replicated_source import admit_source

    attempt = _Generation(cleanup_scope, cancelled)
    try:
        attempt._checkpoint()
        # Run the caller hook BEFORE the final local checks, never between those
        # checks and ordinary admission. The private tree and resolved ancestors
        # must remain trusted and stable throughout ordinary admission; this is
        # not a general TOCTOU proof or a hostile-filesystem admission route.
        encoded, root, inventory = _preflight(experiment, ids, outputs)
        source = admit_source(experiment)  # actual SDK admission, never caller rows
        if source.path != root or experiment.canonical_bytes() != encoded:
            _refuse("SDK source changed")
        attempt._stage(source, inventory)
        attempt._join(source, experiment, encoded, ids, outputs)
        manifest = attempt._publish(source, experiment, ids, outputs)
    except BaseException as error:
        attempt._finish_operation()
        try:
            cleanup_scope.drain()
        except BaseException:
            # Optional enrichment cannot replace the original operation error.
            try:
                error.add_note("local toy generation cleanup unconfirmed; custody retained")
            except BaseException:
                pass
        raise
    attempt._finish_operation()
    cleanup_scope.drain()  # no success result unless all cleanup is confirmed
    return manifest

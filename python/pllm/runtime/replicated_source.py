"""Bounded source ownership shared with the existing Safetensors admission path."""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path

from pllm import _native, model_loader
from pllm.configuration import Experiment
from pllm.deployment.resources import _CleanupSlot, reserve_incremental_disk, reserve_incremental_memory
from . import delegated_source as source_io
from .replicated_protocol import execution_binding, storage_policy, validate_execution_binding, STREAM_MEMORY, REQUEST_MEMORY


class _CompactPublisherOwner:
    """Retain caller/disk charges until native compact payload retirement.

    Construct BEFORE the native factory. Adopt its original result, then admit
    its aggregate disk quote BEFORE appending source bytes. The directory must
    be the actual native snapshot filesystem, not the checkpoint filesystem.
    No backing credit is claimed. The existing coarse REQUEST_MEMORY allowance
    is charged independently, including for direct publisher callers; it is not
    free credit against the transport's separate reservation.

    Snapshot directory/quantity come from the actual live native owner.
    This owner itself grants no native storage/source/execution authority.
    """
    __slots__ = ("_owner", "_leases", "_disk_admitted", "_closed", "_cleanup_scope")

    def __init__(self, cleanup_scope):
        if not isinstance(cleanup_scope, _CleanupSlot) or cleanup_scope.closed:
            raise ValueError("compact publisher requires an admitted cleanup scope")
        self._owner = None
        self._leases = ExitStack()
        self._disk_admitted = False
        self._closed = False
        self._cleanup_scope = cleanup_scope
        cleanup_scope.retain(self, self._retire)
        try:
            self._leases.enter_context(reserve_incremental_memory(REQUEST_MEMORY))
        except BaseException as error:
            self._close_preserving(error)
            raise

    def _retain(self, owner):
        if self._owner is not None or self._closed:
            raise ValueError("compact publisher already retained")
        self._owner = owner

    def _admit_disk(self, amount, *, directory):
        if self._owner is None or self._closed or self._disk_admitted or type(amount) is not int or amount <= 0:
            raise ValueError("invalid compact publisher disk admission")
        self._leases.enter_context(reserve_incremental_disk(amount, directory=directory))
        self._disk_admitted = True

    def __getattr__(self, name):
        if self._owner is None:
            raise ValueError("compact publisher retired")
        if not self._disk_admitted:
            raise ValueError("compact publisher disk not admitted")
        return getattr(self._owner, name)

    def close(self):
        self._cleanup_scope.drain()

    def _retire(self):
        # If native close fails, retain both owner and leases; never pretend the
        # disk/caller backing retired. Caller preserves original error identity.
        if self._owner is not None:
            self._owner.close()
            self._owner = None
        self._leases.close()
        self._closed = True

    def _close_preserving(self, error):
        try:
            self.close()
        except BaseException as cleanup:
            error.add_note(f"compact publisher cleanup failed: {type(cleanup).__name__}")


def _compact_snapshot_directory(owner):
    # I frozen source03 readonly property, not a method or ambient tempdir.
    # Native owns this location for publisher AND importer snapshots. It is
    # location metadata, never permission/backing credit/source authority.
    directory = owner.collection_disk_directory
    if type(directory) is not str or not directory:
        raise ValueError("compact publisher snapshot directory must be a native string")
    return directory


def _independently_rooted_compact_publisher(owner):
    # Actual owner+live admission-token membership, NEVER requested storage or
    # a duck-typed/config echo flag. Observer proxies conservatively return false.
    return (isinstance(owner, _CompactPublisherOwner)
            and type(owner._cleanup_scope) is _CleanupSlot
            and _CleanupSlot.owns(owner._cleanup_scope, owner))


def validate_execution(experiment, contract):
    from pllm.profiles import resolve_runtime_composition

    expected = execution_binding(experiment)
    workspace = STREAM_MEMORY if expected and expected["peer_transport"] == "streams" else 0
    actual_workspace = contract.get("peer_stream_workspace_bytes_per_worker", 0)
    validate_execution_binding(experiment, contract)
    if (contract.get("numeric_profile") != resolve_runtime_composition(experiment.pipeline).numeric_profile
            or type(actual_workspace) is not int or actual_workspace != workspace):
        raise ValueError("replicated source execution contract differs from Experiment")


class ReplicatedSource(source_io.AdmittedSource):
    __slots__ = ()

    def runtime_contract(self, experiment, rows, outputs):
        if experiment.canonical_bytes() != self.experiment_bytes:
            raise ValueError("replicated source requires its exact Experiment")
        contract = _native.replicated_source_contract(
            self.experiment_bytes, self.config_bytes, rows, outputs,
            self.source_lock_digest, self.tokenizer_digest,
        )
        validate_execution(experiment, json.loads(contract))
        return contract

    @property
    def runtime_contract_digest(self):
        return hashlib.sha256(b"pllm.replicated.sdk-contract.v1\0" + self.runtime_contract_bytes).hexdigest()

    def _append_source(self, shapes, append, cancelled):
        expected = (self.checkpoint_digest, self.source_lock_digest, self.source_lock)
        if source_io._locked(self.path, self.model, self.source_float32_bytes, self._identities, self.capacity) != expected:
            raise ValueError("replicated source changed after admission")
        store, specs = source_io._store(self.path, self._identities, self.contract)
        try:
            if specs != self._tensor_specs:
                raise ValueError("replicated source metadata changed")
            identities, specs = dict(self._identities), dict(specs)
            for key, shape in shapes:
                filename, actual_shape, dtype, offset, size = specs[store.resolve(key)]
                if tuple(shape) != actual_shape:
                    raise ValueError("replicated source shape changed")
                itemsize = source_io._DTYPES[dtype].itemsize
                for begin in range(0, math.prod(shape), 4096):
                    if cancelled():
                        raise ValueError("replicated source import cancelled")
                    count = min(4096, math.prod(shape) - begin)
                    frame = source_io._source_float_frame(self.path / filename, identities[filename],
                                                         offset + begin * itemsize, count, dtype)
                    append(key, begin, frame)
                    del frame
            if source_io._locked(self.path, self.model, self.source_float32_bytes, self._identities, self.capacity) != expected:
                raise ValueError("replicated source changed during import")
        finally:
            store.clear_cache()

    def populate_public_worker(self, owner, cancelled):
        try:
            if self.contract.get("weights") != "public":
                raise ValueError("public worker requires public weights")
            self._append_source(sorted(self.contract["source_shapes"].items()), owner.append_public_source, cancelled)
            if cancelled():
                raise ValueError("replicated source import cancelled")
            owner.seal_public_source()
        except BaseException as error:
            try:
                owner.close()
            except BaseException as cleanup:
                error.add_note(f"replicated public import cleanup failed: {type(cleanup).__name__}")
            raise

    def publisher(self, experiment, ids, outputs, cancelled, *, cleanup_scope=None):
        if storage_policy(experiment) == "compact":
            if outputs != 2 or type(outputs) is not int:
                raise ValueError("compact PUMA publisher requires exactly two outputs")
            if experiment.canonical_bytes() != self.experiment_bytes:
                raise ValueError("replicated source requires its exact Experiment")
            try:
                factory = _native.ReplicatedPublisher._from_compact_source_request
            except AttributeError as error:
                raise RuntimeError("compact PUMA native publisher capability unavailable") from error
            if cleanup_scope is None:
                client = getattr(cancelled, "owner", None)
                register = getattr(client, "_new_compact_cleanup_scope", None)
                if register is None:
                    raise ValueError("direct compact publisher requires an explicit cleanup scope")
                cleanup_scope = register(cancelled)
            if not isinstance(cleanup_scope, _CleanupSlot) or cleanup_scope.closed:
                raise ValueError("compact publisher requires an admitted cleanup scope")
            owner = None
            try:
                owner = _CompactPublisherOwner(cleanup_scope)  # registered BEFORE factory
                owner._retain(factory(self.experiment_bytes, self.config_bytes, len(ids), outputs,
                    bytes.fromhex(self.source_lock_digest), bytes.fromhex(experiment.pipeline.digest())))
                native = owner._owner
                owner._admit_disk(native.collection_disk_bytes,
                    directory=_compact_snapshot_directory(native))
                self.runtime_contract(experiment, len(ids), outputs)
                self._append_source(owner.shapes, owner.append, cancelled)
                if cancelled():
                    raise ValueError("replicated source import cancelled")
                manifest = owner.seal(b"".join(token.to_bytes(4, "little") for token in ids))
                validate_execution_binding(experiment, json.loads(manifest))
                return owner, manifest
            except BaseException as error:
                if owner is not None:
                    owner._close_preserving(error)
                raise
        self.runtime_contract(experiment, len(ids), outputs)
        owner = _native.ReplicatedPublisher(
            self.config_bytes, len(ids), outputs, bytes.fromhex(self.source_lock_digest),
            bytes.fromhex(experiment.pipeline.digest()), self.contract.get("weights", "private"),
            **(execution_binding(experiment) or {}),
        )
        try:
            self._append_source(owner.shapes, owner.append, cancelled)
            manifest = owner.seal(b"".join(token.to_bytes(4, "little") for token in ids))
            value = json.loads(manifest)
            validate_execution_binding(experiment, value)
            return owner, manifest
        except BaseException:
            if owner is not None:
                owner.close()
            raise


def admit_source(experiment: Experiment) -> ReplicatedSource:
    if not isinstance(experiment, Experiment) or experiment.resolve().client_runtime != "compiled_replicated_v1":
        raise ValueError("replicated source requires its bounded Experiment")
    encoded = experiment.canonical_bytes()
    model = experiment.pipeline.model
    root = Path(model.source).expanduser()
    if model.kind == "huggingface" and not root.exists() and not root.is_absolute():
        from .hf_hub import resolve_huggingface_source
        root = resolve_huggingface_source(model.source, model_id=model.model_id, revision=model.revision,
                                         local_files_only=model.local_files_only).path
    if not root.is_dir():
        raise ValueError("replicated source requires an admitted HF/Safetensors directory")
    root = root.resolve()
    original_identity = source_io._identity(root / "config.json")
    original = source_io._read_bounded(root / "config.json", 16384)
    raw = source_io._json(original, "config.json")
    config = source_io._encode_config(raw)
    for value in (raw, raw.get("text_config", {})):
        if isinstance(value, dict) and any(value.get(key) is not None for key in ("quantization", "quantization_config")):
            raise ValueError("replicated source requires original floating-point tensors")
    rows, outputs = experiment.budget.max_input_tokens, experiment.budget.max_new_tokens
    contract = _native.replicated_source_contract(encoded, config, rows, outputs, "0" * 64, "0" * 64)
    metadata = json.loads(contract)
    validate_execution(experiment, metadata)
    capacity = source_io._capacity(metadata)
    size = metadata["source_float32_bytes"]
    identities = source_io._files(root, size, capacity)
    if dict(identities)["config.json"] != original_identity:
        raise ValueError("replicated configuration changed during admission")
    store, specs = source_io._store(root, identities, metadata)
    store.clear_cache()
    checkpoint, lock_digest, lock = source_io._locked(root, model, size, identities, capacity)
    hashes = {row["path"]: row["sha256"] for row in lock["files"]}
    tokenizer = source_io._read_bounded(root / "tokenizer.json", (16 << 20) if capacity == "decoder" else (256 << 10))
    tokenizer_digest = hashlib.sha256(tokenizer).hexdigest()
    if hashes["config.json"] != hashlib.sha256(original).hexdigest() or hashes["tokenizer.json"] != tokenizer_digest:
        raise ValueError("replicated source bytes differ from lock")
    contract = _native.replicated_source_contract(encoded, config, rows, outputs, lock_digest, tokenizer_digest)
    validate_execution(experiment, json.loads(contract))
    manifest = source_io._manifest(root, model, raw, identities)
    descriptor = source_io._tokenizer(root, manifest, manifest.metadata["text_config"], metadata)
    if descriptor.pop("model") != tokenizer or source_io._locked(root, model, size, identities, capacity) != (checkpoint, lock_digest, lock):
        raise ValueError("replicated source changed during admission")
    manifest.metadata.update(model_spec=model.to_spec(), checkpoint_digest=checkpoint,
                             source_lock_digest=lock_digest, source_lock=lock, config_only=False)
    manifest.chat_template = descriptor["chat_template"]
    return ReplicatedSource(model, root, config, original, contract, checkpoint, lock_digest, tokenizer,
        tokenizer_digest, source_io._canonical(lock), source_io._canonical(manifest.to_dict(include_fingerprint=False)),
        source_io._canonical(descriptor), identities, specs, encoded, contract)

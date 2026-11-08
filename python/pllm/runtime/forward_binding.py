"""Authenticated immutable coefficients for native-owned public forward plans.

No decoder manifest, tokenizer, quantization, or protected-source admission lives
on this path. Hashes authenticate bytes, never cached file timestamps.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import tempfile
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from filelock import FileLock

from .bundle_storage import CHUNK_BYTES, FilePayload, f32_chunks, immutable_f32_array
from .source_io import SourceReadPolicy, _JSONCensus, _json_frames_supported


class ForwardBindingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Fixed augmented output contract. Independent immutable bytes survive close."""

    _storage: object
    identities: object

    @property
    def outputs(self):
        return MappingProxyType({name: immutable_f32_array(raw, shape)
                                 for name, (raw, shape) in self._storage.items()})

    @property
    def last_hidden_state(self):
        raw, shape = self._storage["last_hidden_state"]
        return immutable_f32_array(raw, shape)

    @property
    def pooler_output(self):
        raw, shape = self._storage["pooler_output"]
        return immutable_f32_array(raw, shape)

    @property
    def logits(self):
        raw, shape = self._storage["logits"]
        return immutable_f32_array(raw, shape)


# Quote schema has <=7 fields, <=20 decimal digits/value and short ASCII keys.
# 2048 bytes/32 lexical nodes/depth 2 cover it without a plan-sized allowance.
# Use the SAME component census as production parsing, never a second estimator.
_QUOTE_POLICY = SourceReadPolicy(metadata_bytes=2048, inventory_nodes=32, bounded_json=True)
_QUOTE_CENSUS = _JSONCensus(2048, 32, 32, 32, 32, 2, 1024)
_QUOTE_PARSE_BYTES = _QUOTE_CENSUS.parse_peak_bytes
# Include retained transfer alongside parse scratch, without relying on release
# timing. Any future production-census growth automatically tightens bootstrap.
_QUOTE_BOOTSTRAP_BYTES = _QUOTE_PARSE_BYTES + _QUOTE_CENSUS.retained_bytes
_METADATA_PHASE_BYTES = 8 * CHUNK_BYTES
_NATIVE_NUMERICAL_FLOOR_BYTES = 8 * CHUNK_BYTES


def _native_bootstrap_budget(memory_bytes, held_bytes):
    """Pure checked partition, not an authority or an observed memory peak.

    B = caller cap, U = floor(4B/5), H = live Python holdings, A = phase
    allowance. Native receives U-H-A, not B. Its universal numerical floor W
    absorbs first-quote parse/retained envelope Q before native M is readable:
    M + W <= native budget and Q <= W imply H + A + M + Q <= U.
    """
    if (type(memory_bytes) is not int or not 0 < memory_bytes <= sys.maxsize
            or type(held_bytes) is not int or not 0 <= held_bytes <= sys.maxsize
            or not 0 < _QUOTE_BOOTSTRAP_BYTES <= _NATIVE_NUMERICAL_FLOOR_BYTES):
        raise ForwardBindingError("invalid forward native bootstrap byte envelope")
    budget = (4 * memory_bytes) // 5 - held_bytes - _METADATA_PHASE_BYTES
    if not 0 < budget <= sys.maxsize:
        raise ForwardBindingError("forward native bootstrap has no addressable budget")
    return budget


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def _admit(required, memory_bytes, *, retained=0):
    """Ordinary process/host headroom, separate from native logical quote."""
    from .benchmark_memory import host_memory, process_memory
    if (type(required) is not int or not 0 <= required <= sys.maxsize
            or type(memory_bytes) is not int or not 0 < memory_bytes <= sys.maxsize
            or type(retained) is not int or not 0 <= retained <= sys.maxsize):
        raise ForwardBindingError("forward storage exceeds addressable byte envelope")
    host, process = host_memory(), process_memory()
    rss = process["rss_bytes"]
    if type(rss) is not int or rss < 0:
        raise ForwardBindingError("process memory telemetry unavailable")
    peak = (required * 125 + 99) // 100
    incremental = max(0, peak - retained)
    if (peak > memory_bytes or incremental > max(0, host.available - host.reserve)
            or host.pressure in {"warning", "critical"}):
        raise ForwardBindingError("forward working set exceeds process/host memory admission")
    # A finite OS address-space ceiling remains binding even with spare RAM.
    import resource
    ceiling, _ = resource.getrlimit(resource.RLIMIT_AS)
    if ceiling != resource.RLIM_INFINITY and rss + incremental > ceiling:
        raise ForwardBindingError("forward working set exceeds process address-space ceiling")


def _source_path(model, policy):
    from .hf_hub import resolve_huggingface_source
    if model.kind not in {"huggingface", "safetensors"}:
        raise ForwardBindingError("forward requires a Hugging Face/local Safetensors source")
    return resolve_huggingface_source(
        model.source, model_id=model.model_id, revision=model.revision,
        local_files_only=model.local_files_only,
        read_policy=policy,
    ).path


def _source_lock_storage(lock):
    """Known SourceLock backing/records only; no generic object hook traversal."""
    fields = {"schema", "kind", "repo_id", "revision", "commit", "files", "checkpoint_digest"}
    if type(lock) is not dict or set(lock) != fields or type(lock["files"]) is not list:
        raise ForwardBindingError("invalid retained SourceLock structure")
    def scalar(value):
        if type(value) is str:
            # Unicode backing needs at most four bytes/code point. Do not first
            # allocate an encoded copy of potentially large resolver identities.
            return 4 * len(value) + 256
        if value is None or type(value) is int:
            return 256
        raise ForwardBindingError("invalid retained SourceLock scalar")
    size = 256 * (2 + len(fields) + len(lock["files"]))
    for name in ("schema", "kind", "repo_id", "revision", "commit", "checkpoint_digest"):
        size += scalar(lock[name])
    for row in lock["files"]:
        if type(row) is not dict or set(row) != {"path", "size", "sha256"}:
            raise ForwardBindingError("invalid retained SourceLock file record")
        size += 4 * 256 + scalar(row["path"]) + scalar(row["size"]) + scalar(row["sha256"])
    if size > sys.maxsize:
        raise ForwardBindingError("SourceLock storage exceeds addressable byte envelope")
    return size


@dataclass(slots=True)
class _MetadataAdmission:
    """Observed logical phase storage; no nominal cap-sized extrapolation."""
    memory_bytes: int
    held_bytes: int = 0
    peak_bytes: int = 0
    lock_scratch: int | None = None

    def admit(self, scratch):
        if type(scratch) is not int or not 0 <= scratch <= sys.maxsize:
            raise ForwardBindingError("metadata scratch exceeds addressable byte envelope")
        if self.lock_scratch is not None:
            # The resolver DOM remains live inside _source_lock while later
            # descriptor hashes/serialization run. Retain its actually observed
            # read/parse allowance until that entire source-lock phase ends.
            self.lock_scratch = max(self.lock_scratch, scratch)
            scratch = self.lock_scratch
        required = self.held_bytes + scratch + _METADATA_PHASE_BYTES
        if required > sys.maxsize:
            raise ForwardBindingError("metadata phase exceeds addressable byte envelope")
        self.peak_bytes = max(self.peak_bytes, required)
        _admit(required, self.memory_bytes)

    def hold(self, size):
        if type(size) is not int or not 0 <= size <= sys.maxsize - self.held_bytes:
            raise ForwardBindingError("retained metadata exceeds addressable byte envelope")
        self.held_bytes += size
        self.admit(0)

    @contextmanager
    def source_lock(self, root, model, policy):
        from pllm.model_loader import _source_lock
        previous = self.lock_scratch
        self.lock_scratch = 0
        try:
            yield _source_lock(root, model, read_policy=policy)
        finally:
            self.lock_scratch = previous


def _disk_admit(directory, expected_identity, required):
    sample = directory.stat()
    if (directory.resolve() != directory or (sample.st_dev, sample.st_ino) != expected_identity):
        raise ForwardBindingError("snapshot destination changed")
    if required + 64 * CHUNK_BYTES > shutil.disk_usage(directory).free:
        raise ForwardBindingError("insufficient disk for authenticated checkpoint snapshots")


def _headers(root, files, stack, policy, admission):
    """Duplicate-safe headers from already hash-locked open descriptors."""
    source_bytes = sum(row["size"] for name, row in files.items() if name.endswith(".safetensors"))
    directory = Path(tempfile.gettempdir()).resolve(strict=True)
    sample = directory.stat()
    directory_identity = (sample.st_dev, sample.st_ino)
    _disk_admit(directory, directory_identity, source_bytes)
    policy._admit(3 * CHUNK_BYTES)
    handles, entries, header_bytes = {}, {}, 0
    remaining = source_bytes
    for name, row in files.items():
        if not name.endswith(".safetensors"):
            continue
        # Hash the exact bytes copied to a private snapshot, not the pathname
        # before/after an unrelated tensor read. Even transient source rewrites
        # cannot substitute unauthenticated tensor contents between those reads.
        _disk_admit(directory, directory_identity, remaining)
        snapshot = FilePayload(policy.chunks(root / name), size=row["size"],
                               digest=row["sha256"], temp_directory=directory)
        stack.callback(snapshot.close)
        if Path(snapshot.path).parent.resolve() != directory:
            raise ForwardBindingError("snapshot escaped admitted destination")
        _disk_admit(directory, directory_identity, 0)
        remaining -= row["size"]
        prefix = snapshot.read_at(0, min(8, row["size"]))
        size = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 2 <= size <= 16 * CHUNK_BYTES or size + 8 > row["size"]:
            raise ForwardBindingError("invalid or oversized Safetensors header")
        policy._admit(16 * size + 3 * CHUNK_BYTES)
        raw = snapshot.read_at(8, size)
        if len(raw) != size:
            raise ForwardBindingError("truncated Safetensors header")
        header_bytes += size
        header, _ = policy._parse_json(raw, retain=admission.hold)
        if type(header) is not dict:
            raise ForwardBindingError("invalid Safetensors header object")
        spans = []
        for key, metadata in header.items():
            if key == "__metadata__":
                if type(metadata) is not dict or any(type(value) is not str for value in metadata.values()):
                    raise ForwardBindingError("invalid Safetensors string metadata")
                continue
            if (type(key) is not str or key in entries or type(metadata) is not dict
                    or set(metadata) != {"dtype", "shape", "data_offsets"}):
                raise ForwardBindingError("duplicate or invalid checkpoint tensor metadata")
            shape, offsets = metadata["shape"], metadata["data_offsets"]
            if (type(metadata["dtype"]) is not str
                    or type(shape) is not list or any(type(n) is not int or n < 0 for n in shape)
                    or type(offsets) is not list or len(offsets) != 2
                    or any(type(n) is not int or n < 0 for n in offsets)
                    or not offsets[0] <= offsets[1] <= row["size"] - 8 - size):
                raise ForwardBindingError("invalid checkpoint tensor shape/offset")
            spans.append(tuple(offsets))
            entries[key] = (name, metadata, size + 8)
        # No aliased, overlapping, or unindexed payload bytes in a shard.
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ForwardBindingError("ambiguous Safetensors payload spans")
            cursor = end
        if cursor != row["size"] - size - 8:
            raise ForwardBindingError("Safetensors payload has trailing bytes")
        handles[name] = snapshot
    if not handles:
        raise ForwardBindingError("source has no Safetensors checkpoint")
    indexes = [name for name in files if name.endswith(".safetensors.index.json")]
    if len(indexes) > 1:
        raise ForwardBindingError("ambiguous Safetensors indexes")
    if indexes:
        name = indexes[0]
        raw = policy.read_bytes(root / name)
        if hashlib.sha256(raw).hexdigest() != files[name]["sha256"]:
            raise ForwardBindingError("checkpoint index changed")
        header_bytes += len(raw)
        index, _ = policy._parse_json(raw, retain=admission.hold)
        weight_map = index.get("weight_map") if type(index) is dict else None
        if (type(weight_map) is not dict or set(weight_map) != set(entries)
                or any(type(name) is not str or Path(name).name != name
                       or entries[key][0] != name for key, name in weight_map.items())):
            raise ForwardBindingError("checkpoint index and actual shard keys differ")
    return handles, entries, header_bytes, source_bytes, directory


def _quote(contract, sequence, *, admit, retain=None):
    if not _FORWARD_JSON_FRAMES_SUPPORTED:
        raise ForwardBindingError("forward JSON frame layout unsupported")
    # Admission precedes even the getter. Native owns the PyBytes transfer;
    # frontend reserves its small bounded parse envelope alongside live owners.
    admit(_QUOTE_BOOTSTRAP_BYTES)
    policy = SourceReadPolicy(metadata_bytes=_QUOTE_POLICY.metadata_bytes,
                              inventory_nodes=_QUOTE_POLICY.inventory_nodes,
                              bounded_json=True, metadata_admit=admit)
    policy._require_bounded_backend()
    raw = contract.quote(sequence)
    census = policy._json_census(raw)
    if census.max_depth > 2:
        raise ForwardBindingError("native forward quote exceeds depth envelope")
    quote, _ = policy._parse_json(raw, retain=retain)
    _validate_native_quote(quote, sequence, contract.memory_bytes())
    return quote


def _validate_native_quote(quote, sequence, native_budget):
    """Validate inspection data only; never constructs or adopts an owner."""
    fields = {"sequence_length", "input_bytes", "retained_tensor_bytes", "output_bytes",
              "workspace_bytes", "required_bytes", "metadata_bytes"}
    if (type(native_budget) is not int or not 0 < native_budget <= sys.maxsize
            or type(quote) is not dict or set(quote) != fields
            or any(type(value) is not int or not 0 <= value <= sys.maxsize for value in quote.values())
            or quote["sequence_length"] != sequence
            or not 0 < quote["metadata_bytes"] < quote["required_bytes"] <= native_budget
            or quote["required_bytes"] - quote["metadata_bytes"] < _NATIVE_NUMERICAL_FLOOR_BYTES):
        raise ForwardBindingError("native forward quote is invalid")


def _reservation(plan, quote, *, requests, metadata_bytes, metadata_phase_peak_bytes,
                 native_compile_prerequisite_bytes, weight_bytes, memory_bytes):
    from .forward_execution import _python_workspace, _shape
    sequence = quote["sequence_length"]
    shapes = [_shape(spec["tensor"], sequence) for spec in plan["outputs"].values()]
    output_bytes = sum(4 * math.prod(shape) for shape in shapes)
    if output_bytes != quote["output_bytes"]:
        raise ForwardBindingError("native quote and fixed published output census differ")
    # The existing 256-byte container/scalar allowance prices the result record,
    # mapping/proxy, each bytes header/entry, and all shape metadata. Identities
    # are shared immutable objects already priced in metadata_bytes. Accessor
    # clones/views and caller traceback retention are not runtime result owners.
    record_bytes = 256 * (4 + sum(4 + len(shape) for shape in shapes))
    supplement = _python_workspace(plan, sequence, metadata_bytes)
    max_call = quote["required_bytes"] + supplement
    import_peak = weight_bytes + metadata_bytes + 8 * CHUNK_BYTES
    per_result = output_bytes + record_bytes
    ceiling = min(sys.maxsize, memory_bytes)
    if (max_call > ceiling or import_peak > ceiling or per_result > ceiling
            or requests > ceiling // per_result):
        raise ForwardBindingError("forward lifetime result reservation exceeds byte envelope")
    # Deliberately double-count the current result's payload: native/Python call
    # storage already includes publication, plus R complete persistent results.
    required = max(metadata_phase_peak_bytes, native_compile_prerequisite_bytes,
                   import_peak, max_call + requests * per_result)
    if required > ceiling:
        raise ForwardBindingError("forward lifetime result reservation exceeds byte envelope")
    result = {"requests": requests, "max_sequence": sequence,
              "output_bytes_per_result": output_bytes,
              "result_metadata_bytes_per_result": record_bytes,
              "max_call_bytes": max_call, "import_peak_bytes": import_peak,
               "metadata_phase_peak_bytes": metadata_phase_peak_bytes,
               "native_compile_prerequisite_bytes": native_compile_prerequisite_bytes,
              "required_bytes": required, "admission_bytes": (required * 125 + 99) // 100}
    _admit(required, memory_bytes)
    return MappingProxyType(result), supplement


@dataclass(frozen=True, slots=True)
class _PreloadedForward:
    root: Path
    contract: object
    plan: dict
    expected: dict
    handles: dict
    entries: dict
    policy: SourceReadPolicy
    checkpoint: str
    source_lock: str
    metadata_bytes: int
    maximum: int
    requests: int
    execution_metadata: object
    identities: object
    reservation: object
    quote: dict
    header_bytes: int
    snapshot_bytes: int
    snapshot_directory: Path
    python_workspace_bytes: int


def _preload(experiment, *, memory_bytes, stack):
    """Single production source/quote path. Never constructs a numerical array."""
    from pllm import _native
    from pllm.configuration import Experiment, ForwardBudget
    from pllm.profiles import resolve_forward_runtime_composition

    if (type(experiment) is not Experiment or type(experiment.budget) is not ForwardBudget
            or experiment.deployment.kind != "local"
            or type(memory_bytes) is not int or not 0 < memory_bytes <= sys.maxsize):
        raise ForwardBindingError("forward requires local Experiment, ForwardBudget, memory_bytes")
    runtime = resolve_forward_runtime_composition(experiment.pipeline)
    if not _FORWARD_JSON_FRAMES_SUPPORTED:
        raise ForwardBindingError("forward JSON frame layout unsupported")
    native_type = getattr(_native, "ForwardContract", None)
    if native_type is None:
        raise ForwardBindingError("native ForwardContract is unavailable; rebuild extension")
    # This precedes the normal resolver, including its first index read.
    metadata = _MetadataAdmission(memory_bytes)
    metadata.admit(0)
    policy = SourceReadPolicy(bounded_json=True, metadata_admit=metadata.admit)
    policy._require_bounded_backend()
    root = _source_path(experiment.pipeline.model, policy)
    stack.enter_context(FileLock(str(root / ".pllm-forward.lock")))
    # Pre-admit names/rows before the shared source lock accumulates its lists.
    inventory = policy.directory_files(root, patterns=("*",))
    metadata.hold(16 * sum(4 * len(path.name) + 256 for path in inventory))
    del inventory
    with metadata.source_lock(root, experiment.pipeline.model, policy) as locked:
        checkpoint, source_lock, lock = locked
    metadata.hold(_source_lock_storage(lock))
    files = {row["path"]: row for row in lock["files"]}
    raw_config = policy.read_bytes(root / "config.json", limit=CHUNK_BYTES)
    if ("config.json" not in files or hashlib.sha256(raw_config).hexdigest()
            != files["config.json"]["sha256"]):
        raise ForwardBindingError("source config changed after source locking")
    config, _ = policy._parse_json(raw_config, limit=CHUNK_BYTES, retain=metadata.hold)
    if type(config) is not dict:
        raise ForwardBindingError("source config must be an object")
    for section in (config, config.get("text_config")):
        if type(section) is dict:
            for name in ("quantization", "quantization_config"):
                if name in section and section[name] is not None and section[name] != {}:
                    raise ForwardBindingError("forward does not admit quantized checkpoints")
    handles, entries, header_bytes, snapshot_bytes, directory = _headers(
        root, files, stack, policy, metadata)
    composition = experiment.pipeline.canonical_bytes()
    metadata.hold(16 * len(composition))
    maximum = experiment.budget.max_input_tokens
    held_at_compile = metadata.held_bytes
    native_budget = _native_bootstrap_budget(memory_bytes, held_at_compile)
    # Ordinary admission covers construction/first quote alongside Python H+A.
    # Do NOT put this nominal available budget into observed metadata.peak_bytes.
    _admit(held_at_compile + _METADATA_PHASE_BYTES + native_budget, memory_bytes)
    # Native reconstructs source semantics. No Python model/geometry fallback.
    try:
        contract = native_type.compile(raw_config, composition, max_sequence=maximum,
                                       memory_bytes=native_budget)
    except ValueError as exc:
        # Partitioning can now reject inside native admission before Python's
        # final reservation check; preserve the runtime's ordinary error type.
        raise ForwardBindingError(f"native forward compilation or memory admission failed: {exc}") from exc
    if type(contract) is not native_type:
        raise ForwardBindingError("forward requires exact native contract ownership")
    quote = _quote(contract, maximum, admit=metadata.admit, retain=metadata.hold)
    _validate_native_quote(quote, maximum, native_budget)
    # Native-owned retained graph/requirements/source metadata is mandatory in
    # the native quote and stays charged through all subsequent Python inspection
    # and final verification. Python trees below have separate real backings.
    metadata.hold(quote["metadata_bytes"])
    native_compile_prerequisite = held_at_compile + _METADATA_PHASE_BYTES + quote["required_bytes"]
    if native_compile_prerequisite > sys.maxsize:
        raise ForwardBindingError("native compile prerequisite exceeds addressable byte envelope")
    # Actual native requirement makes reported-cap recompilation pass the same
    # partition. Unlike native_budget, this prerequisite is cap-independent.
    _admit(native_compile_prerequisite, memory_bytes)
    metadata.peak_bytes = max(metadata.peak_bytes, native_compile_prerequisite)
    # Native retention is learned only from its authoritative bounded quote.
    # Retained quote is now held, so add only parse scratch here: retained+parse
    # is bounded by Q, already absorbed by native's numerical floor before decode.
    metadata.admit(_QUOTE_PARSE_BYTES)
    if (contract.source_config_bytes() != raw_config or contract.composition_bytes() != composition
            or contract.max_sequence() != maximum or contract.memory_bytes() != native_budget):
        raise ForwardBindingError("native forward source/composition identity differs")
    documents = []
    for inspect in (contract.plan_bytes, contract.schedule_bytes, contract.requirements_bytes):
        raw = inspect()
        value, census = policy._parse_json(raw, retain=metadata.hold)
        # Original DOM remains live in loaded/documents while frozen plan and
        # derived requirements are built. Prepay destinations, not postcharge.
        metadata.hold(census.freeze_bytes)
        documents.append(value)
    del raw
    plan, schedule, requirement_document = documents
    if (plan["schema_version"] != "pllm.forward_plan.v1"
            or plan["numeric"] != "pllm.numeric.source_float32.v1"
            or plan["output_contract"] != "pllm.forward.encoder_outputs.v1"
            or plan["max_sequence"] != maximum):
        raise ForwardBindingError("unsupported native forward contract")
    if (schedule["schema_version"] != "pllm.forward_runtime_schedule.v1"
            or schedule["plan_digest"] != contract.plan_digest()
            or schedule["composition_digest"] != contract.composition_digest()
            or schedule["numeric_digest"] != contract.numeric_digest()
            or schedule["max_sequence"] != maximum or schedule["complete"] is not True
            or schedule["protected_execution"] is not False
            or schedule["steps"] != [{"operation_id": op["id"],
                                      "executor": "client_local_source_f32_v1"}
                                     for op in plan["operations"]]):
        raise ForwardBindingError("native forward schedule differs from public local execution")
    if (type(requirement_document) is not dict
            or set(requirement_document) != {"schema_version", "tensors", "weight_bytes"}
            or requirement_document["schema_version"] != "pllm.forward_requirements.v1"
            or type(requirement_document["weight_bytes"]) is not int):
        raise ForwardBindingError("invalid native forward requirements document")
    requirements = requirement_document["tensors"]
    if type(requirements) is not list or not requirements:
        raise ForwardBindingError("native forward requirements must be nonempty records")
    expected = {}
    for row in requirements:
        if (type(row) is not dict or set(row) != {"name", "dtype", "shape"}
                or type(row["name"]) is not str or row["name"] in expected
                or row["dtype"] != "float32" or type(row["shape"]) is not list
                or not row["shape"] or any(type(n) is not int or n <= 0 for n in row["shape"])):
            raise ForwardBindingError("invalid native forward tensor requirement")
        expected[row["name"]] = tuple(row["shape"])
    weight_bytes = sum(4 * math.prod(shape) for shape in expected.values())
    if requirement_document["weight_bytes"] != weight_bytes:
        raise ForwardBindingError("native forward weight byte census differs")
    for name, shape in expected.items():
        if name not in entries:
            raise ForwardBindingError(f"required exact checkpoint key missing: {name}")
        _, tensor_metadata, _ = entries[name]
        if (tensor_metadata["dtype"] != "F32" or tuple(tensor_metadata["shape"]) != shape
                or tensor_metadata["data_offsets"][1] - tensor_metadata["data_offsets"][0] != 4 * math.prod(shape)):
            raise ForwardBindingError(f"required exact F32 tensor differs: {name}")
    # Fixed description/identity/reservation records plus <=3 published outputs:
    # <128 fixed headers/edges at 256 bytes each, doubled for original/frozen
    # records and identity-serialization scratch. Paths get separate 8x backing
    # (Unicode plus join scratch, including separators); scalars share on freeze.
    # Shape destinations already covered by plan census freeze allowance. Charge
    # both original and frozen records before building either; compile/describe
    # share this phase peak, independent of requested memory cap.
    # Metadata-only description has at most three recursive dict/list levels
    # (output shapes are already tuples). Its <=7 freeze frames fit the prepaid
    # native plan's F (plan outputs/operations require >=3 levels). Each preload
    # freezes either this description OR the plan, never both concurrently.
    description_bytes = (65536 + 8 * sum(len(part) + 1 for path in (root, directory)
                                        for part in path.parts))
    metadata.hold(description_bytes)
    execution = MappingProxyType({"backend": "numpy_einsum_unoptimized_f32_v1",
        "requested_compute_threads": runtime.threads, "effective_compute_threads": 1,
        "thread_policy": "serial_compute_within_requested_budget"})
    identities = {"source_digest": checkpoint, "checkpoint_digest": checkpoint,
        "source_lock_digest": source_lock, "plan_digest": contract.plan_digest(),
        "schedule_digest": contract.schedule_digest(), "composition_digest": contract.composition_digest(),
        "numeric_digest": contract.numeric_digest(), "configuration_digest": experiment.configuration_digest()}
    binding = {"identities": identities, "execution_metadata": dict(execution)}
    identities["binding_digest"] = hashlib.sha256(b"pllm.forward_binding.v1\0" +
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata.hold(256 * (len(expected) * 4 + len(identities) * 4 + len(execution) * 4 + 16))
    # Final original-file verification is deliberately BEFORE numerical RAM.
    # After this boundary, coefficients are read solely from authenticated owned
    # snapshot descriptors; later original-path mutation cannot change binding.
    _verify_source(root, experiment.pipeline.model, checkpoint=checkpoint,
                   source_lock=source_lock, policy=policy, metadata=metadata)
    metadata_bytes = metadata.held_bytes
    reservation, supplement = _reservation(plan, quote, requests=experiment.budget.requests,
        metadata_bytes=metadata_bytes, metadata_phase_peak_bytes=metadata.peak_bytes,
        native_compile_prerequisite_bytes=native_compile_prerequisite,
        weight_bytes=weight_bytes, memory_bytes=memory_bytes)
    return _PreloadedForward(root, contract, plan, expected, handles, entries, policy,
        checkpoint, source_lock, metadata_bytes, maximum, experiment.budget.requests,
        execution, MappingProxyType(identities), reservation, quote, header_bytes,
        snapshot_bytes, directory, supplement)


def _verify_source(root, model, *, checkpoint, source_lock, policy, metadata):
    with metadata.source_lock(root, model, policy) as (final_checkpoint, final_lock, final_record):
        # Keep the observed final resolver-DOM scratch allowance through this
        # comparison; it covers new temporary SourceLock strings/serialization.
        if final_checkpoint != checkpoint or final_lock != source_lock:
            raise ForwardBindingError("source changed during forward binding")


def describe_forward_source(experiment, *, memory_bytes: int) -> Mapping[str, object]:
    """Metadata-only production preflight; no coefficient reads or finite claim.

    Streams/hash-verifies private disk snapshots and closes them before return.
    The immutable scalar tree is descriptive, never an executable binding or an
    authority accepted by a factory. Compile repeats this authenticated preload.
    """
    from .forward_execution import _shape
    with ExitStack() as stack:
        loaded = _preload(experiment, memory_bytes=memory_bytes, stack=stack)
        return _freeze({"schema_version": "pllm.forward_source_description.v1",
            "execution_admitted": False, "numerical_finite_validated": False,
            "source_path": str(loaded.root), "identities": dict(loaded.identities),
            "max_sequence": loaded.maximum, "requests": loaded.requests, "memory_bytes": memory_bytes,
            "source_tensor_bytes": sum(4 * math.prod(shape) for shape in loaded.expected.values()),
            "tensor_count": len(loaded.expected), "metadata_bytes": loaded.metadata_bytes,
            "header_bytes": loaded.header_bytes, "snapshot_bytes": loaded.snapshot_bytes,
            "snapshot_directory": str(loaded.snapshot_directory), "quote": loaded.quote,
            "python_workspace_bytes": loaded.python_workspace_bytes,
            "metadata_phase_peak_bytes": loaded.reservation["metadata_phase_peak_bytes"],
            "lifetime_reservation": dict(loaded.reservation),
            "execution_metadata": dict(loaded.execution_metadata),
            "outputs": {name: {"dtype": "float32", "shape": _shape(spec["tensor"], loaded.maximum),
                               "nbytes": 4 * math.prod(_shape(spec["tensor"], loaded.maximum))}
                        for name, spec in loaded.plan["outputs"].items()}})


class CompiledForwardModel:
    """One native contract, authenticated private weights, one in-flight lease."""

    __slots__ = ("_contract", "_plan", "_weights", "_identities", "_domains",
                 "_memory_bytes", "_metadata_bytes", "_maximum", "_requests",
                 "_condition", "_active", "_closed", "_execution_metadata", "_lifetime_reservation")

    def __init__(self):
        raise ForwardBindingError("forward models require Experiment.compile_forward()")

    @classmethod
    def _from_experiment(cls, experiment, *, memory_bytes):
        if cls is not CompiledForwardModel:
            raise ForwardBindingError("forward binding cannot be subclassed")
        with ExitStack() as stack:
            loaded = _preload(experiment, memory_bytes=memory_bytes, stack=stack)
            weights = {}
            for name, shape in loaded.expected.items():
                shard, metadata, base = loaded.entries[name]
                start, end = metadata["data_offsets"]
                raw = loaded.handles[shard].read_at(base + start, end - start)
                array = immutable_f32_array(raw, shape)
                if not all(np.all(np.isfinite(chunk)) for chunk in f32_chunks(array)):
                    raise ForwardBindingError(f"checkpoint tensor is nonfinite: {name}")
                weights[name] = (raw, shape)
        self = object.__new__(cls)
        self._contract, self._plan = loaded.contract, _freeze(loaded.plan)
        self._weights = MappingProxyType(weights)
        self._domains = {}
        for op in loaded.plan["operations"]:
            if op["operator"] == "token_lookup":
                name = op["inputs"][0].removeprefix("input.")
                self._domains[name] = loaded.expected[op["attributes"]["weight"]][0]
        if set(self._domains) != {"tokens", "positions", "segments"}:
            raise ForwardBindingError("native forward plan lacks exact input lookup domains")
        self._domains = MappingProxyType(self._domains)
        self._execution_metadata, self._identities = loaded.execution_metadata, loaded.identities
        self._lifetime_reservation = loaded.reservation
        self._memory_bytes, self._metadata_bytes = memory_bytes, loaded.metadata_bytes
        self._maximum, self._requests = loaded.maximum, loaded.requests
        self._condition = threading.Condition()
        self._active = self._closed = False
        return self

    @property
    def identities(self):
        return self._identities

    @property
    def lifetime_reservation(self):
        """Immutable accounted lifetime bytes, retained even after failures/close.

        Includes the conservative 25% admission margin separately. Not a memory
        allocation, RSS measurement, hard allocator cap, or disk reservation.
        """
        return self._lifetime_reservation

    @property
    def execution_metadata(self):
        """Serial numerical kernels stay within the requested maximum budget.

        Not a utilization, total-process-thread, multicore, or third-party NumPy
        patch guarantee. Logical storage admission is not an allocator/RSS cap.
        """
        return self._execution_metadata

    @property
    def digest(self):
        return self._identities["binding_digest"]

    @property
    def contract(self):
        return self._contract

    def local_tensor(self, name):
        """Fresh array metadata; no mutable backing or retained ndarray escapes."""
        with self._condition:
            if self._closed:
                raise ForwardBindingError("forward model is closed")
            raw, shape = self._weights[name]
            return immutable_f32_array(raw, shape)

    def forward(self, inputs):
        from .forward_execution import execute_forward
        return execute_forward(self, inputs)

    def close(self):
        with self._condition:
            self._closed = True
            while self._active:
                self._condition.wait()
            self._weights = MappingProxyType({})

    def __enter__(self):
        with self._condition:
            if self._closed:
                raise ForwardBindingError("forward model is closed")
        return self

    def __exit__(self, *_):
        self.close()


# Parser callback frames belong to ordinary metadata admission (A); recursive
# freeze frames belong to F. Check actual code, including generator bodies,
# rather than assuming a source-independent frame size. Upstream caller frames
# are existing caller state, not implicitly covered by the parser's 2D+12 bound.
_FORWARD_JSON_FRAME_FUNCTIONS = (
    _freeze, _quote, _validate_native_quote, _native_bootstrap_budget,
    _MetadataAdmission.admit, _MetadataAdmission.hold, _admit,
    _MetadataAdmission.source_lock, _MetadataAdmission.source_lock.__wrapped__,
    _source_path, _source_lock_storage, _verify_source,
    _preload, _headers, _reservation, describe_forward_source,
    CompiledForwardModel._from_experiment.__func__,
)
_FORWARD_JSON_FRAMES_SUPPORTED = _json_frames_supported(_FORWARD_JSON_FRAME_FUNCTIONS)


__all__ = ["CompiledForwardModel", "ForwardBindingError", "ForwardResult", "describe_forward_source"]

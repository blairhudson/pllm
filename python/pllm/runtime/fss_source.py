"""Bounded public FuseFSS source import, not an SDK serving qualification.

Native geometry precedes hashing and tensor reads. Checkpoint IEEE bytes retain
their original dtype and orientation; Python never quantizes or owns a complete
tensor copy. Source locks detect observed mutations, not atomic filesystem
snapshots or publisher authenticity. Closing releases references; no Python
secure-erasure claim is made.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import struct

from pllm import _native
from pllm.configuration import Experiment
from pllm.protocols import FuseFssSourceDecoder

from . import delegated_source as io
from . import tokenizer_admission

_PROFILES = frozenset(FuseFssSourceDecoder.NUMERIC_PROFILES)
_FRAME = 64 << 10
_SOURCE = 8 << 20
_METADATA = 1 << 20
_TABLE_PROCEDURE = "pllm.fss.host-libm-f32-split-half.v1"


class FssSourceError(ValueError):
    """Source or native admission is unavailable, changed or unsupported."""


def _document(payload, label, limit=_METADATA):
    if type(payload) is not bytes or not 0 < len(payload) <= limit:
        raise FssSourceError(f"FuseFSS invalid bounded {label} bytes")
    return io._json(payload, label)


def _pin(value):
    return (type(value) is str and len(value) == 64 and value != "0" * 64
            and all(c in "0123456789abcdef" for c in value))


def _cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise FssSourceError("FuseFSS source import cancelled")


def _capacity(record):
    capacity = record.get("capacity", "capsule")
    if type(capacity) is not str or capacity not in {"capsule", "decoder"}:
        raise FssSourceError("FuseFSS capacity must be capsule or decoder")
    return capacity


def _material_delivery(record):
    from .fss_protocol import material_delivery
    try:
        return material_delivery(record.get("material_delivery", "single"))
    except ValueError as error:
        raise FssSourceError("FuseFSS material delivery must be single or batch-v1") from error


def _online_schedule(record):
    schedule = record.get("online_schedule", "serial")
    if type(schedule) is not str or schedule not in {"serial", "wavefront-v1", "tensor-v1"}:
        raise FssSourceError("FuseFSS online schedule must be serial, wavefront-v1 or tensor-v1")
    return schedule


def _experiment(experiment):
    if not isinstance(experiment, Experiment):
        raise TypeError("FuseFSS source requires an Experiment")
    if not callable(getattr(_native, "fss_source_contract", None)) or not callable(
            getattr(_native, "FssSourceBuilder", None)):
        raise FssSourceError("installed native core lacks FuseFSS source admission; rebuild required")
    encoded = experiment.canonical_bytes()
    if not 0 < len(encoded) <= 32 << 10:
        raise FssSourceError("FuseFSS Experiment exceeds its byte limit")
    _native.resolve_experiment(encoded)
    method = experiment.pipeline.components.get("linear")
    profile = method.params.get("numeric_profile") if method else None
    if (method is None or method.component != "pllm/fusefss-source-decoder/v1"
            or profile not in _PROFILES):
        raise FssSourceError("FuseFSS source requires its explicit numeric profile")
    _material_delivery(method.params)
    _online_schedule(method.params)
    model = experiment.pipeline.model
    if model.kind not in {"huggingface", "safetensors"}:
        raise FssSourceError("FuseFSS requires public HF/Safetensors source")
    return encoded, model, profile, _capacity(method.params)


def _geometry(config, rows, outputs, profile, capacity="capsule"):
    # SourceBuilder::new lowers/bounds geometry before reserving metadata vectors;
    # it allocates no tensor/rotary payload until append. No fabricated source pin.
    kwargs = {"capacity": capacity} if capacity == "decoder" else {}
    owner = _native.FssSourceBuilder(config, rows, outputs, profile, **kwargs)
    owner.close()


def _files(root, size, capacity="capsule"):
    # The explicit decoder uses the shared expanded metadata envelope. Capsule
    # retains its original additional limits and documentation-ignore behavior.
    identities = io._files(root, size, "decoder")
    if capacity == "decoder":
        return identities
    limits = {**io._METADATA_LIMITS, "vocab.json": 256 << 10, "merges.txt": 256 << 10}
    total = 0
    for name, identity in identities:
        limit = (size + (64 << 10) + 8 if name.endswith(".safetensors") else
                 (64 << 10) if name.endswith(".safetensors.index.json") else limits[name])
        if identity[9] > limit:
            raise FssSourceError("FuseFSS source file exceeds bounded geometry/metadata")
        total += identity[9]
    if total > size + _METADATA:
        raise FssSourceError("FuseFSS source inventory exceeds total bounds")
    return identities


def _unchanged(root, size, identities, capacity="capsule"):
    if _files(root, size, capacity) != identities:
        raise FssSourceError("FuseFSS source changed after admission")


def _locked(root, model, size, identities, capacity="capsule"):
    _unchanged(root, size, identities, capacity)
    result = io._locked(root, model, size, identities, "decoder")
    _unchanged(root, size, identities, capacity)
    return result


def _contract(experiment, encoded, config, rows, outputs, sourcepin, tokenizerpin):
    payload = _native.fss_source_contract(encoded, config, rows, outputs, sourcepin, tokenizerpin)
    record = _document(payload, "native source contract")
    expected = {"schema_version": "pllm.fss.sdk-contract.v1", "weights": "public",
                "configuration": experiment.configuration_digest(),
                "composition": experiment.pipeline.digest(), "source_lock": sourcepin,
                "tokenizer": tokenizerpin, "input_rows": rows, "output_tokens": outputs,
                "numeric_profile": experiment.pipeline.components["linear"].params["numeric_profile"]}
    if any(record.get(k) != v for k, v in expected.items()):
        raise FssSourceError("FuseFSS native contract differs from source or Experiment")
    capacity = _capacity(record)
    if capacity != _capacity(experiment.pipeline.components["linear"].params):
        raise FssSourceError("FuseFSS native capacity differs from Experiment")
    if _material_delivery(record) != _material_delivery(experiment.pipeline.components["linear"].params):
        raise FssSourceError("FuseFSS native material delivery differs from Experiment")
    if _online_schedule(record) != _online_schedule(experiment.pipeline.components["linear"].params):
        raise FssSourceError("FuseFSS native online schedule differs from Experiment")
    expanded = capacity == "decoder"
    shapes, layouts = record.get("source_shapes"), record.get("source_layouts")
    if (type(shapes) is not dict or not 1 <= len(shapes) <= 1024
            or type(layouts) is not dict or set(layouts) != set(shapes)):
        raise FssSourceError("FuseFSS invalid source inventory")
    for name, shape in shapes.items():
        if (type(name) is not str or not name or len(name.encode()) > 1024
                or type(shape) is not list or not 1 <= len(shape) <= 2
                or any(type(d) is not int or d < 1 for d in shape)
                or (not expanded and math.prod(shape) > 32768)
                or layouts[name] not in {"features", "rows_features", "input_output", "output_input"}
                or (len(shape) == 1) != (layouts[name] == "features")):
            raise FssSourceError("FuseFSS invalid source shape/layout")
    total = sum(math.prod(shape) * 4 for shape in shapes.values())
    source_limit = io._DECODER_SOURCE_BYTES if expanded else _SOURCE
    if type(record.get("source_float32_bytes")) is not int or not 0 < total <= source_limit or total != record["source_float32_bytes"]:
        raise FssSourceError("FuseFSS source byte accounting differs")
    if expanded:
        # Native storage admission owns these exact bounds. They must be present;
        # source geometry never grants an expanded activation/material allowance.
        matrix_bytes = sum(math.prod(s) * 8 for s in shapes.values() if len(s) == 2)
        if (type(record.get("source_q12_value_bytes")) is not int
                or record["source_q12_value_bytes"] != 2 * total
                or type(record.get("source_disk_bytes")) is not int
                or not matrix_bytes <= record["source_disk_bytes"] <= 2 * source_limit
                or type(record.get("source_resident_bound_bytes")) is not int
                or not 0 < record["source_resident_bound_bytes"] <= _SOURCE):
            raise FssSourceError("FuseFSS missing or invalid native source storage admission")
    vocabulary_limit = max(max(s) for s in shapes.values()) if expanded else 32768
    if (type(rows) is not int or type(outputs) is not int or not 1 <= outputs <= 4
            or not 1 <= rows <= rows + outputs - 1 <= 16
            or record.get("maximum_sequence") != rows + outputs - 1
            or type(record.get("vocabulary")) is not int or not 1 <= record["vocabulary"] <= vocabulary_limit):
        raise FssSourceError("FuseFSS invalid request geometry")
    _rotary_specs(record)  # Validate the semantic label and explicit D13 override.
    return payload, record


def _store(root, identities, contract):
    """Shared frozen store/geometry access, with native authoritative inventory.

    The older shared semantic inventory cannot describe positional lookups,
    LayerNorm bias or input_output Conv1D. Never reinterpret these as dense OI.
    """
    shapes = contract["source_shapes"]
    expanded = _capacity(contract) == "decoder"
    metadata_limit = io._DECODER_METADATA_BYTES if expanded else _METADATA
    header_limit = min(io._DECODER_HEADER_BYTES if expanded else 64 << 10, max(1024, 1024 + sum(
        len(name.encode()) + 192 + 24 * len(shape) for name, shape in shapes.items())))
    specs, locations, indexes = {}, {}, []
    header_bytes = 0
    for filename, identity in identities:
        if filename.endswith(".safetensors.index.json"):
            indexes.append(filename)
        if not filename.endswith(".safetensors"):
            continue
        path = root / filename
        size = int.from_bytes(io._read_range(path, identity, 0, 8), "little")
        header_bytes += size
        if not 0 < size <= header_limit or 8 + size > identity[9] or header_bytes > metadata_limit:
            raise FssSourceError("FuseFSS Safetensors header exceeds geometry-derived bounds")
        header = _document(io._read_range(path, identity, 8, size), "Safetensors header", header_limit)
        metadata = header.pop("__metadata__", {})
        if type(metadata) is not dict or any(type(v) is not str for v in metadata.values()):
            raise FssSourceError("FuseFSS malformed Safetensors metadata")
        if not header or len(specs) + len(header) > len(shapes):
            raise FssSourceError("FuseFSS source tensor cardinality differs")
        spans = []
        for name, item in header.items():
            if (name not in shapes or name in specs or type(item) is not dict
                    or set(item) != {"shape", "dtype", "data_offsets"}
                    or item["shape"] != shapes[name]
                    or type(item["shape"]) is not list
                    or any(type(d) is not int for d in item["shape"])
                    or type(item["dtype"]) is not str or item["dtype"] not in io._DTYPES):
                raise FssSourceError("FuseFSS source tensor shape/dtype/identity differs")
            offsets, dtype = item["data_offsets"], item["dtype"]
            if (type(offsets) is not list or len(offsets) != 2
                    or any(type(i) is not int or i < 0 for i in offsets)
                    or offsets[1] - offsets[0] != math.prod(shapes[name]) * io._DTYPES[dtype].itemsize
                    or offsets[1] > identity[9] - 8 - size):
                raise FssSourceError("FuseFSS source byte range differs")
            specs[name] = (filename, tuple(shapes[name]), dtype, 8 + size + offsets[0], offsets[1] - offsets[0])
            locations[name] = filename
            spans.append(offsets)
        end = 0
        for begin, stop in sorted(spans):
            if begin != end:
                raise FssSourceError("FuseFSS source has gapped/overlapping ranges")
            end = stop
        if end + 8 + size != identity[9]:
            raise FssSourceError("FuseFSS source payload differs from header")
    if set(specs) != set(shapes) or len(indexes) > 1:
        raise FssSourceError("FuseFSS source tensor inventory differs")
    if indexes:
        index = _document(io._read_bounded(root / indexes[0], header_limit), "tensor index", header_limit)
        if index.get("weight_map") != locations:
            raise FssSourceError("FuseFSS shard index differs from headers")
    store = io._AdmittedTensorStore(root, identities, specs)
    for name, shape in shapes.items():
        if store.tensor_shape(name) != tuple(shape) or store.tensor_dtype(name) not in io._DTYPES:
            raise FssSourceError("FuseFSS shared reader geometry differs")
    return store, tuple(sorted(specs.items()))


def _rotary_specs(contract):
    shapes = contract.get("rotary_shapes")
    plan = contract.get("plan")
    if type(shapes) is not dict or type(plan) is not dict:
        raise FssSourceError("FuseFSS missing semantic rotary inventory")
    operations = plan.get("prefill", {}).get("operations")
    if type(operations) is not list or len(operations) > 1024:
        raise FssSourceError("FuseFSS invalid semantic view")
    rotary = [op for op in operations if op.get("operator") == "rotary_embedding"]
    if len(rotary) != len(shapes) or {op["id"] for op in rotary} != set(shapes):
        raise FssSourceError("FuseFSS rotary inventory differs")
    result = []
    for op in sorted(rotary, key=lambda op: op["id"]):
        attrs, shape = op["attributes"], shapes[op["id"]]
        width = op["output_shape"][-1]
        if (type(shape) is not list or len(shape) != 2 or any(type(d) is not int for d in shape)
                or shape != [contract["maximum_sequence"], width] or math.prod(shape) > 32768
                or type(width) is not int or not 2 <= width <= 4096 or width % 2
                or type(attrs.get("theta")) is not int or not 1 <= attrs["theta"] <= 1000000
                or attrs.get("rotary_dimensions") != width or attrs.get("pairing") != "split_half"
                or attrs.get("position_policy") != "sequential_absolute"
                or attrs.get("coefficient_profile") != "pllm.numeric.rope.q30.libm.v1"
                or attrs.get("tail_policy") != "unchanged" or attrs.get("frequency_scaling") is not None):
            raise FssSourceError("FuseFSS requires approved unscaled full split-half rotary semantics")
        result.append((op["id"], shape, attrs["theta"]))
    return result


def _rotary_frames(shape, theta):
    # Binary64 host pow/multiply/libm, then exactly one binary32 packing. This is
    # the frozen example's procedure, not a portable exact-real libm specification.
    # Fresh coefficient hashes authenticate this host's actual F32 bytes.
    rows, width = shape
    half = width // 2
    for begin in range(0, rows * width, _FRAME // 8):
        end = min(rows * width, begin + _FRAME // 8)
        cosine, sine = bytearray(), bytearray()
        for index in range(begin, end):
            position, feature = divmod(index, width)
            angle = position * (theta ** (-2.0 * (feature % half) / width))
            cosine.extend(struct.pack("<f", math.cos(angle)))
            sine.extend(struct.pack("<f", math.sin(angle)))
        yield begin, bytes(cosine), bytes(sine)


def _tokenizer(root, config, contract):
    from tokenizers import Tokenizer

    expanded = _capacity(contract) == "decoder"
    tokenizer_limit = io._DECODER_TOKENIZER_BYTES if expanded else io._MAX_TOKENIZER_BYTES
    raw = io._read_bounded(root / "tokenizer.json", tokenizer_limit)
    doc = _document(raw, "tokenizer.json", tokenizer_limit)
    maximum = contract["vocabulary"]
    tokenizer_admission.validate_document(
        doc, vocabulary=maximum, max_entries=maximum, error=FssSourceError, label="FuseFSS")
    inner = Tokenizer.from_buffer(raw)
    actual_size = tokenizer_admission.validate_vocabulary(
        inner, vocabulary=maximum, allow_model_padding=expanded,
        error=FssSourceError, label="FuseFSS")
    options = {}
    if (root / "tokenizer_config.json").exists():
        options = _document(io._read_bounded(root / "tokenizer_config.json", 16384), "tokenizer options", 16384)
    if type(options.get("add_bos_token", True)) is not bool:
        raise FssSourceError("FuseFSS add_bos_token must be boolean")
    context = options.get("model_max_length", contract["maximum_sequence"])
    if type(context) is not int or context < contract["maximum_sequence"]:
        raise FssSourceError("FuseFSS tokenizer context is too small")
    bos, eos = config.get("bos_token_id", 0), config.get("eos_token_id", 1)
    eos = eos if type(eos) is list else [eos]
    if not eos or any(type(i) is not int or not 0 <= i < maximum for i in [bos, *eos]):
        raise FssSourceError("FuseFSS special token IDs exceed complete vocabulary")
    if expanded and any(i >= actual_size for i in [bos, *eos]):
        raise FssSourceError("FuseFSS special tokens require original tokenizer entries")
    # Descriptor retains the complete model vocabulary. Unmapped model IDs remain
    # valid final IDs; ordinary TokenizersJSONTokenizer delegates text rendering
    # to tokenizers.decode (which omits unmapped IDs), without rewriting feedback.
    descriptor = {"type": "tokenizer_json", "vocab_size": maximum, "bos_token_id": bos,
                  "eos_token_id": eos[0], "chat_template": options.get("chat_template"),
                  "bos_token": options.get("bos_token", ""), "eos_token": options.get("eos_token", ""),
                  "add_bos_token": options.get("add_bos_token", True)}
    return raw, io._canonical(descriptor)


@dataclass(frozen=True, slots=True)
class FrozenFssSource:
    """Metadata/tokenizer only; no client source tensor payload owner."""

    model: object
    path: Path
    experiment_bytes: bytes
    config_bytes: bytes
    config_source_bytes: bytes
    contract_bytes: bytes
    checkpoint_digest: str
    source_lock_digest: str
    tokenizer_digest: str
    tokenizer_bytes: bytes = field(repr=False)
    _lock: bytes = field(repr=False)
    _tokenizer_options: bytes = field(repr=False)
    _identities: tuple = field(repr=False)
    _specs: tuple = field(repr=False)

    @property
    def contract(self):
        return _document(self.contract_bytes, "source contract")

    @property
    def capacity(self):
        return _capacity(self.contract)

    @property
    def material_delivery(self):
        return _material_delivery(self.contract)

    @property
    def snapshot_pins(self):
        return {"checkpoint": self.checkpoint_digest, "source_lock": self.source_lock_digest,
                "tokenizer": self.tokenizer_digest,
                "config": hashlib.sha256(self.config_source_bytes).hexdigest()}

    @property
    def tokenizer_descriptor(self):
        return {**_document(self._tokenizer_options, "tokenizer options"), "model": self.tokenizer_bytes}

    def tokenizer(self):
        from tokenizers import Tokenizer
        from .transformer_client import TokenizersJSONTokenizer

        descriptor = self.tokenizer_descriptor
        return TokenizersJSONTokenizer(Tokenizer.from_buffer(self.tokenizer_bytes),
                                       descriptor["bos_token_id"], descriptor["eos_token_id"])

    def runtime_contract(self, experiment, rows, outputs):
        encoded, _, _, _ = _experiment(experiment)
        if encoded != self.experiment_bytes:
            raise FssSourceError("FuseFSS requires exact admitted Experiment")
        if (type(rows) is not int or type(outputs) is not int
                or not 1 <= rows <= experiment.budget.max_input_tokens
                or not 1 <= outputs <= experiment.budget.max_new_tokens):
            raise FssSourceError("FuseFSS request exceeds saved envelope")
        if io._canonical(_document(self.config_source_bytes, "original config", 16384)) != self.config_bytes:
            raise FssSourceError("FuseFSS canonical config differs from original snapshot")
        _unchanged(self.path, self.contract["source_float32_bytes"], self._identities, self.capacity)
        return _contract(experiment, encoded, self.config_bytes, rows, outputs,
                         self.source_lock_digest, self.tokenizer_digest)[0]


def _check_pins(source, expected):
    if not isinstance(expected, Mapping) or dict(expected) != source.snapshot_pins:
        raise FssSourceError("FuseFSS worker original snapshot pins differ")


def _resolve_client_source(experiment, *, expected_pins=None):
    """Resolve original source pins and tokenizer without decoding any weight values."""
    encoded, model, profile, capacity = _experiment(experiment)
    root = Path(model.source).expanduser()
    if model.kind == "huggingface" and not root.exists() and not root.is_absolute():
        from .hf_hub import resolve_huggingface_source
        root = resolve_huggingface_source(model.source, model_id=model.model_id,
                                         revision=model.revision, local_files_only=model.local_files_only).path
    if not root.is_dir():
        raise FssSourceError("FuseFSS requires an existing HF/Safetensors directory")
    root = root.resolve()
    identity = io._identity(root / "config.json")
    original = io._read_bounded(root / "config.json", 16384)
    raw = _document(original, "config.json", 16384)
    for cfg in (raw, raw.get("text_config", {})):
        if isinstance(cfg, dict) and any(cfg.get(key) is not None for key in ("quantization", "quantization_config")):
            raise FssSourceError("FuseFSS requires original unquantized floating-point source")
    config = io._canonical(raw)
    rows, outputs = experiment.budget.max_input_tokens, experiment.budget.max_new_tokens
    _geometry(config, rows, outputs, profile, capacity)
    source_limit = io._DECODER_SOURCE_BYTES if capacity == "decoder" else _SOURCE
    tokenizer_limit = io._DECODER_TOKENIZER_BYTES if capacity == "decoder" else io._MAX_TOKENIZER_BYTES
    identities = _files(root, source_limit, capacity)
    if dict(identities)["config.json"] != identity:
        raise FssSourceError("FuseFSS config changed during admission")
    checkpoint, lock_digest, lock = _locked(root, model, source_limit, identities, capacity)
    hashes = {row["path"]: row["sha256"] for row in lock["files"]}
    tokenizer = io._read_bounded(root / "tokenizer.json", tokenizer_limit)
    tokenizer_digest = hashlib.sha256(tokenizer).hexdigest()
    if hashes.get("config.json") != hashlib.sha256(original).hexdigest() or hashes.get("tokenizer.json") != tokenizer_digest:
        raise FssSourceError("FuseFSS config/tokenizer differ from original source lock")
    contract, record = _contract(experiment, encoded, config, rows, outputs, lock_digest, tokenizer_digest)
    size = record["source_float32_bytes"]
    _unchanged(root, size, identities, capacity)
    store, specs = _store(root, identities, record)
    store.clear_cache()
    tokenizer_again, descriptor = _tokenizer(root, raw.get("text_config", raw), record)
    if tokenizer_again != tokenizer or _locked(root, model, size, identities, capacity) != (checkpoint, lock_digest, lock):
        raise FssSourceError("FuseFSS source changed during admission")
    result = FrozenFssSource(model, root, encoded, config, original, contract, checkpoint,
                             lock_digest, tokenizer_digest, tokenizer, io._canonical(lock), descriptor,
                             identities, specs)
    if expected_pins is not None:
        _check_pins(result, expected_pins)
    return result


def resolve_client_source(experiment, *, expected_pins=None):
    """Reserve expanded source metadata before hashing, headers or tokenizers."""
    if (isinstance(experiment, Experiment)
            and experiment.pipeline.components.get("linear") is not None
            and experiment.pipeline.components["linear"].params.get("capacity") == "decoder"):
        _experiment(experiment)  # Validate the explicit native policy before reservation.
        with tokenizer_admission.reserve_decoder_metadata():
            return _resolve_client_source(experiment, expected_pins=expected_pins)
    return _resolve_client_source(experiment, expected_pins=expected_pins)


@dataclass(slots=True)
class LoadedFssSource:
    """One native program owner; close is idempotent and releases native ownership."""

    source: FrozenFssSource
    contract_bytes: bytes
    native_program: object
    manifest_bytes: bytes
    _rotary_source_bytes: bytes = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def manifest(self):
        return _document(self.manifest_bytes, "program manifest", 1024)

    @property
    def rotary_source(self):
        return _document(self._rotary_source_bytes, "F32 rotary source descriptor")

    def close(self):
        if not self._closed:
            self._closed = True
            self.native_program.close()

    def __enter__(self):
        if self._closed:
            raise FssSourceError("FuseFSS program is closed")
        return self

    def __exit__(self, *_exc):
        self.close()


def load_role_source(experiment, *, expected_pins, input_rows=None, outputs=None, cancelled=None, source=None):
    """Import a role's own public checkpoint, pinned to coordinator's original snapshot.

    This does not send weights from client or manufacture serving admission.
    Native seal owns Q12 conversion, numeric/range rejection and program creation.
    """
    _cancelled(cancelled)
    source = source if source is not None else resolve_client_source(experiment, expected_pins=expected_pins)
    if not isinstance(source, FrozenFssSource):
        raise TypeError("FuseFSS source requires FrozenFssSource metadata")
    _check_pins(source, expected_pins)
    rows = experiment.budget.max_input_tokens if input_rows is None else input_rows
    outputs = experiment.budget.max_new_tokens if outputs is None else outputs
    contract_bytes = source.runtime_contract(experiment, rows, outputs)
    record = _document(contract_bytes, "source contract")
    capacity = _capacity(record)
    size = record["source_float32_bytes"]
    expected_lock = (source.checkpoint_digest, source.source_lock_digest, _document(source._lock, "source lock"))
    if _locked(source.path, source.model, size, source._identities, capacity) != expected_lock:
        raise FssSourceError("FuseFSS source changed after admission")
    store, specs = _store(source.path, source._identities, record)
    builder = program = None
    try:
        if specs != source._specs:
            raise FssSourceError("FuseFSS tensor headers changed after admission")
        kwargs = {"capacity": capacity} if capacity == "decoder" else {}
        builder = _native.FssSourceBuilder(source.config_bytes, rows, outputs, record["numeric_profile"], **kwargs)
        original_hash = hashlib.sha256(b"pllm.fss.original-ieee-stream.v1\0")
        identities, specs = dict(source._identities), dict(specs)
        for name in sorted(record["source_shapes"]):
            filename, shape, dtype, offset, size_bytes = specs[name]
            itemsize = io._DTYPES[dtype].itemsize
            original_hash.update(struct.pack("<Q", len(name.encode())) + name.encode())
            original_hash.update(bytes([{"F32": 0, "F16": 1, "BF16": 2}[dtype]]))
            original_hash.update(struct.pack("<Q", len(shape)))
            for dimension in shape:
                original_hash.update(struct.pack("<Q", dimension))
            for begin in range(0, size_bytes // itemsize, _FRAME // itemsize):
                _cancelled(cancelled)
                count = min(_FRAME // itemsize, size_bytes // itemsize - begin)
                frame = io._read_range(source.path / filename, identities[filename],
                                       offset + begin * itemsize, count * itemsize, limit=_FRAME)
                builder.append(name, dtype, begin, frame)
                original_hash.update(frame)
                del frame
        tables = []
        for operation, shape, theta in _rotary_specs(record):
            cosine_hash, sine_hash = hashlib.sha256(), hashlib.sha256()
            original_hash.update(struct.pack("<Q", len(operation.encode())) + operation.encode())
            original_hash.update(struct.pack("<QQQ", theta, *shape))
            for begin, cosine, sine in _rotary_frames(shape, theta):
                _cancelled(cancelled)
                builder.append_rotary(operation, begin, cosine, sine)
                cosine_hash.update(cosine)
                sine_hash.update(sine)
                for i in range(0, len(cosine), 4):
                    original_hash.update(cosine[i:i+4])
                    original_hash.update(sine[i:i+4])
                del cosine, sine
            tables.append({"operation": operation, "shape": shape, "theta": theta,
                           "cosine_sha256": cosine_hash.hexdigest(), "sine_sha256": sine_hash.hexdigest()})
        if _locked(source.path, source.model, size, source._identities, capacity) != expected_lock:
            raise FssSourceError("FuseFSS source changed during import")
        _cancelled(cancelled)
        seal = {"serial": "seal", "wavefront-v1": "seal_wavefront", "tensor-v1": "seal_tensor"}[_online_schedule(record)]
        program = getattr(builder, seal)()
        manifest_bytes = program.manifest(contract_bytes)
        manifest = _document(manifest_bytes, "program manifest", 1024)
        expected = {"schema": "pllm.fss.channel.program.v1", "contract": hashlib.sha256(contract_bytes).hexdigest(),
                    "input_digest": original_hash.hexdigest(), "profile": record["numeric_profile"],
                    "initial": rows, "selections": outputs, "vocabulary": record["vocabulary"]}
        if (set(manifest) != {*expected, "source", "program"}
                or any(manifest.get(k) != v for k, v in expected.items())
                or not _pin(manifest["source"]) or not _pin(manifest["program"])):
            raise FssSourceError("FuseFSS native program manifest differs from original source/contract")
        _cancelled(cancelled)
        descriptor = {"procedure": _TABLE_PROCEDURE, "tables": tables,
                      "semantic_coefficient_profile": "pllm.numeric.rope.q30.libm.v1" if tables else None,
                      "numeric_override": "D13:F32-to-Q12;split-half;per-product-floor" if tables else None}
        descriptor["sha256"] = hashlib.sha256(b"pllm.fss.public-f32-rotary-source.v1\0" + io._canonical(descriptor)).hexdigest()
        return LoadedFssSource(source, contract_bytes, program, manifest_bytes, io._canonical(descriptor))
    except BaseException:
        if program is not None:
            program.close()
        raise
    finally:
        if builder is not None:
            builder.close()
        store.clear_cache()


admit_source = resolve_client_source

__all__ = ["FrozenFssSource", "FssSourceError", "LoadedFssSource", "admit_source",
           "load_role_source", "resolve_client_source"]

"""Exact public prepared-bundle artifacts and a bounded immutable object cache.

Only schema-2 public compiled bundles qualify. MessagePack skeletons retain map
order and numeric representations; the original raw commitment remains the
authority. Objects contain public tensors, never KV or preprocessing material.
Cache ownership covers serialized objects only, not NumPy/native snapshots or
source checkpoint storage. Cache eviction is safe for readers holding copies.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import stat
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import msgpack

ENCODING = "pllm-public-artifacts/1"
MAX_RAW_BYTES = 4 << 30
MAX_MANIFEST_BYTES = 16 << 20
MAX_OBJECTS = 16384
DEFAULT_CACHE_BYTES = 2 << 30
_REF = 42
_DOMAIN = b"pllm-public-object/1\0"
_ITEMSIZE = {"i1": 1, "u1": 1, "<f4": 4}
_MAX_CACHE_OBJECTS = MAX_OBJECTS


class ArtifactError(ValueError):
    """Malformed, private, unsupported, or uncommitted artifact data."""


def _integer(value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ArtifactError("artifact integer exceeds its bound")
    return value


def _digest(value: Any) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ArtifactError("invalid artifact digest")
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactError("invalid artifact metadata") from exc


def _unpack(payload: bytes | bytearray, *, maximum: int) -> Any:
    if type(payload) not in (bytes, bytearray) or not 0 < len(payload) <= maximum:
        raise ArtifactError("artifact document exceeds its byte bound")

    def pairs(rows):
        result = {}
        for key, value in rows:
            if type(key) is not str or key in result:
                raise ArtifactError("artifact maps require unique string keys")
            result[key] = value
        return result

    try:
        return msgpack.unpackb(payload, raw=False, object_pairs_hook=pairs,
                               max_bin_len=maximum, max_str_len=MAX_MANIFEST_BYTES,
                               max_array_len=65536, max_map_len=65536, max_ext_len=32)
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise ArtifactError("invalid artifact MessagePack") from exc


def _tree(value: Any, depth: int = 0, budget: list[int] | None = None) -> None:
    budget = [250000] if budget is None else budget
    budget[0] -= 1
    if depth > 64 or budget[0] < 0:
        raise ArtifactError("artifact graph exceeds its node/depth bound")
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ArtifactError("artifact graph key must be a string")
            _tree(item, depth + 1, budget)
    elif type(value) in (list, tuple):
        for item in value:
            _tree(item, depth + 1, budget)
    elif type(value) not in (str, bytes, int, float, bool, type(None), msgpack.ExtType):
        raise ArtifactError("unsupported artifact graph value")


def _shape(value: Any, *, rank: int | None = None) -> list[int]:
    if type(value) is not list or not 1 <= len(value) <= 8 or rank and len(value) != rank:
        raise ArtifactError("invalid artifact tensor shape")
    size = 1
    for dim in value:
        size *= _integer(dim, 1, MAX_RAW_BYTES)
        if size > MAX_RAW_BYTES:
            raise ArtifactError("artifact tensor shape exceeds byte bound")
    return value


def _public(value: Any) -> None:
    if (type(value) is not dict or type(value.get("v")) is not int or value["v"] != 2
            or value.get("runtime") != "masked_transformer"
            or set(value) != {"v", "runtime", "model", "manifest", "config", "tokenizer",
                             "stages", "local_tensors", "client_weights", "privacy"}):
        raise ArtifactError("artifacts require a compiled schema-2 prepared bundle")
    privacy = value["privacy"]
    if (type(privacy) is not dict or privacy.get("mode") != "public"
            or privacy.get("preprocessed") is not True
            or privacy.get("model_privacy_threat_model") != "public_weights"):
        raise ArtifactError("artifacts require public weights; private bundles cannot be exported")
    if (type(value["manifest"]) is not dict or type(value["manifest"].get("metadata")) is not dict
            or value["manifest"]["metadata"].get("privacy_mode") != "public"):
        raise ArtifactError("artifact manifest is not public")


def _binary_metadata(value: dict[str, Any], object_rows: list[dict[str, Any]] | None = None) -> dict[tuple[str, ...], dict[str, Any]]:
    """Validate schema binary fields; infer physical metadata without family/path rules."""
    _public(value)
    source = value["manifest"].get("metadata", {}).get("source_lock_digest")
    _digest(source)
    result = {}
    numeric_context = hashlib.sha256(_canonical({
        "config": value["config"],
        "weight_bits": value["privacy"].get("weight_bits"),
        "activation_bits": value["privacy"].get("activation_bits"),
    })).hexdigest()

    def add(path, shape, dtype, orientation, source_keys, numeric):
        count = 1
        for dim in _shape(shape):
            count *= dim
        result[path] = {"shape": shape, "dtype": dtype, "orientation": orientation,
                        "source": {"lock": source, "keys": source_keys},
                        "numeric": {**numeric, "runtime_context": numeric_context}}
        return count * _ITEMSIZE[dtype]

    weights = value["client_weights"]
    tensors = value["local_tensors"]
    stages = value["stages"]
    if any(type(v) is not dict for v in (weights, tensors, stages)):
        raise ArtifactError("invalid artifact matrix schema")
    if len(weights) + len(tensors) + len(stages) > MAX_OBJECTS:
        raise ArtifactError("too many artifact records")
    contexts = {}
    for sid, stage in stages.items():
        if type(stage) is not dict:
            raise ArtifactError("invalid artifact stage")
        out_dim = _integer(stage.get("out_features"), 1, MAX_RAW_BYTES)
        in_dim = _integer(stage.get("in_features"), 1, MAX_RAW_BYTES)
        keys = stage.get("source_keys")
        if type(keys) is not list or not keys or any(type(k) is not str for k in keys):
            raise ArtifactError("artifact stage lacks source identity")
        numeric = {k: stage.get(k) for k in ("weight_bits", "activation_bits", "ring", "modulus",
                                             "wire_bits", "equalization_profile_digest")}
        for k in ("weight_bits", "activation_bits", "wire_bits", "modulus"):
            _integer(numeric[k], 1, 1 << 64)
        if stage.get("weight_scales") is None:
            raise ArtifactError("artifact stage lacks public weight scales")
        for field, shape, orientation in (("weight_scales", [out_dim], "row-scale"),
                                           ("bias", [out_dim], "output-vector"),
                                           ("input_equalization", [in_dim], "input-vector")):
            if stage.get(field) is not None:
                add(("stages", sid, field), shape, "<f4", orientation, keys, numeric)
        for field in ("client_weight", "client_aux_weight"):
            ref = stage.get(field)
            if ref is not None:
                if type(ref) is not dict or type(ref.get("ref")) is not str:
                    raise ArtifactError("artifact matrix requires schema-2 reference")
                wid = ref["ref"]
                # Tied weights have two graph orientations, one physical storage.
                context = (keys, numeric)
                # Use the physical linear storage producer for tied matrices;
                # embedding descriptors may have different arithmetic domains
                # and additional fused lookup source keys (e.g. auxiliary rows).
                contexts.setdefault(wid, []).append((ref.get("layout") == "linear", context))
    if set(contexts) != set(weights):
        raise ArtifactError("unreferenced or missing artifact matrix")
    for wid, row in weights.items():
        if type(row) is not dict or set(row) != {"dtype", "shape", "data", "scales"} or row["dtype"] != "i1":
            raise ArtifactError("artifact matrix dtype/schema mismatch")
        shape = _shape(row["shape"], rank=2)
        producers = contexts[wid]
        linear = [context for is_linear, context in producers if is_linear]
        candidates = linear or [context for _, context in producers]
        keys, numeric = candidates[0]
        if any(context != candidates[0] for context in candidates):
            raise ArtifactError("ambiguous artifact physical matrix domain")
        add(("client_weights", wid, "data"), shape, "i1", "row-major", keys, numeric)
        add(("client_weights", wid, "scales"), [shape[0]], "<f4", "row-scale", keys, numeric)
    for key, row in tensors.items():
        if type(row) is not dict or set(row) != {"shape", "dtype", "data"} or row["dtype"] != "f4":
            raise ArtifactError("artifact local tensor dtype/schema mismatch")
        add(("local_tensors", key, "data"), _shape(row["shape"]), "<f4", "row-major", [key],
            {"storage": "float32-le"})
    tokenizer = value["tokenizer"]
    if type(tokenizer) is not dict:
        raise ArtifactError("invalid artifact tokenizer descriptor")
    if "model" in tokenizer:
        kind = tokenizer.get("kind", tokenizer.get("type"))
        if kind not in {"tokenizer_json", "sentencepiece"}:
            raise ArtifactError("artifact tokenizer binary lacks a public schema")
        model = tokenizer["model"]
        if type(model) is bytes:
            length = len(model)
        elif type(model) is msgpack.ExtType and model.code == _REF and len(model.data) == 4 and object_rows is not None:
            index = int.from_bytes(model.data, "big")
            if index >= len(object_rows):
                raise ArtifactError("invalid artifact tokenizer reference")
            length = object_rows[index]["size"]
        else:
            raise ArtifactError("artifact tokenizer binary/reference is invalid")
        add(("tokenizer", "model"), [length], "u1", "serialized-tokenizer", ["tokenizer", kind],
            {"codec": kind})
    return result


def _key(content_digest: str, domain_hash: str) -> str:
    return hashlib.sha256(_DOMAIN + bytes.fromhex(domain_hash) + bytes.fromhex(content_digest)).hexdigest()


def verify_object(row: dict[str, Any], payload: bytes) -> None:
    key, content, domain = (_digest(row[k]) for k in ("sha256", "content_sha256", "domain_hash"))
    size = _integer(row["size"], 1, MAX_RAW_BYTES)
    if domain != hashlib.sha256(_canonical(row["metadata"])).hexdigest() or key != _key(content, domain):
        raise ArtifactError("artifact object domain mismatch")
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != content:
        raise ArtifactError("artifact object content digest mismatch")


@dataclass(frozen=True)
class BundleArtifacts:
    manifest: bytes
    objects: dict[str, bytes]


def export_bundle(payload: bytes) -> BundleArtifacts:
    value = _unpack(payload, maximum=MAX_RAW_BYTES)
    _tree(value)
    metadata = _binary_metadata(value)
    # Reject noncanonical binary representations rather than silently changing
    # map order, integer widths, or float widths and breaking the raw commitment.
    if msgpack.packb(value, use_bin_type=True) != payload:
        raise ArtifactError("bundle cannot be reconstructed byte-exactly")
    objects, rows, indices = {}, [], {}

    def visit(node, path=()):
        if type(node) is dict:
            return {k: visit(v, (*path, k)) for k, v in node.items()}
        if type(node) is list:
            return [visit(v, (*path, str(i))) for i, v in enumerate(node)]
        if type(node) is msgpack.ExtType:
            raise ArtifactError("bundle contains reserved artifact extension")
        if path in metadata:
            if type(node) is not bytes:
                raise ArtifactError("artifact binary field is not bytes")
            meta = metadata[path]
            expected = _ITEMSIZE[meta["dtype"]]
            for dim in meta["shape"]:
                expected *= dim
            if len(node) != expected:
                raise ArtifactError("artifact tensor byte length mismatch")
            content = hashlib.sha256(node).hexdigest()
            domain = hashlib.sha256(_canonical(meta)).hexdigest()
            key = _key(content, domain)
            if key not in indices:
                indices[key] = len(rows)
                rows.append({"sha256": key, "content_sha256": content, "domain_hash": domain,
                             "size": len(node), "metadata": meta})
                objects[key] = node
            return msgpack.ExtType(_REF, indices[key].to_bytes(4, "big"))
        if type(node) is bytes:
            raise ArtifactError("binary field outside public artifact schema")
        return node

    skeleton = msgpack.packb(visit(value), use_bin_type=True)
    manifest = msgpack.packb({"schema": 1, "encoding": ENCODING,
                             "raw": {"schema": 2, "size": len(payload),
                                     "sha256": hashlib.sha256(payload).hexdigest()},
                             "skeleton": skeleton, "objects": rows}, use_bin_type=True)
    if len(manifest) > MAX_MANIFEST_BYTES or len(rows) > MAX_OBJECTS:
        raise ArtifactError("artifact manifest exceeds bounds")
    return BundleArtifacts(manifest, objects)


def parse_manifest(payload: bytes, *, fingerprint: str, size: int) -> dict[str, Any]:
    manifest = _unpack(payload, maximum=MAX_MANIFEST_BYTES)
    _tree(manifest)
    if (type(manifest) is not dict or set(manifest) != {"schema", "encoding", "raw", "skeleton", "objects"}
            or type(manifest["schema"]) is not int or manifest["schema"] != 1
            or manifest["encoding"] != ENCODING):
        raise ArtifactError("unsupported artifact manifest")
    raw = manifest["raw"]
    if (type(raw) is not dict or set(raw) != {"schema", "size", "sha256"}
            or type(raw["schema"]) is not int or raw["schema"] != 2
            or _integer(raw["size"], 1, MAX_RAW_BYTES) != size
            or _digest(raw["sha256"]) != fingerprint):
        raise ArtifactError("artifact original raw descriptor mismatch")
    rows = manifest["objects"]
    if type(rows) is not list or not 1 <= len(rows) <= MAX_OBJECTS:
        raise ArtifactError("invalid artifact object count")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"sha256", "content_sha256", "domain_hash", "size", "metadata"}:
            raise ArtifactError("invalid artifact object descriptor")
        key, content, domain = (_digest(row[k]) for k in ("sha256", "content_sha256", "domain_hash"))
        _integer(row["size"], 1, min(size, MAX_RAW_BYTES))
        if domain != hashlib.sha256(_canonical(row["metadata"])).hexdigest() or key != _key(content, domain) or key in seen:
            raise ArtifactError("artifact object domain mismatch")
        seen.add(key)
    skeleton = _unpack(manifest["skeleton"], maximum=MAX_MANIFEST_BYTES)
    _tree(skeleton)
    metadata = _binary_metadata(skeleton, rows)
    used = set()

    def visit(node, path=()):
        if type(node) is dict:
            for k, v in node.items():
                visit(v, (*path, k))
        elif type(node) is list:
            for i, v in enumerate(node):
                visit(v, (*path, str(i)))
        elif path in metadata:
            if type(node) is not msgpack.ExtType or node.code != _REF or len(node.data) != 4:
                raise ArtifactError("artifact binary reference required")
            index = int.from_bytes(node.data, "big")
            if index >= len(rows) or rows[index]["metadata"] != metadata[path]:
                raise ArtifactError("artifact tensor domain/shape/dtype mismatch")
            expected = _ITEMSIZE[metadata[path]["dtype"]]
            for dim in metadata[path]["shape"]:
                expected *= dim
            if rows[index]["size"] != expected:
                raise ArtifactError("artifact tensor byte length mismatch")
            used.add(index)
        elif type(node) in (msgpack.ExtType, bytes):
            raise ArtifactError("artifact reference outside public schema")

    visit(skeleton)
    if used != set(range(len(rows))) or sum(row["size"] for row in rows) > size:
        raise ArtifactError("artifact unreferenced objects or oversized graph")
    manifest["graph"] = skeleton
    return manifest


def reconstruct_bundle(manifest: dict[str, Any], read: Callable[[dict[str, Any]], bytes]) -> bytearray:
    """Bounded output, one object reader at a time; hash before ClientBundle import."""
    packer = msgpack.Packer(use_bin_type=True)
    output, digest = bytearray(), hashlib.sha256()
    bound = manifest["raw"]["size"]

    def chunks(node) -> Iterator[bytes | memoryview]:
        if type(node) is dict:
            yield packer.pack_map_header(len(node))
            for key, value in node.items():
                yield packer.pack(key)
                yield from chunks(value)
        elif type(node) is list:
            yield packer.pack_array_header(len(node))
            for value in node:
                yield from chunks(value)
        elif type(node) is msgpack.ExtType:
            row = manifest["objects"][int.from_bytes(node.data, "big")]
            payload = read(row)
            verify_object(row, payload)
            # Standard bin header, followed by bounded memoryview slices. Avoid
            # Packer duplicating a whole multi-GB matrix into a second buffer.
            n = len(payload)
            yield (b"\xc4" + n.to_bytes(1, "big") if n <= 255 else
                   b"\xc5" + n.to_bytes(2, "big") if n <= 65535 else
                   b"\xc6" + n.to_bytes(4, "big"))
            view = memoryview(payload)
            for offset in range(0, n, 65536):
                yield view[offset:offset + 65536]
        else:
            yield packer.pack(node)

    for chunk in chunks(manifest["graph"]):
        if len(output) + len(chunk) > bound:
            raise ArtifactError("artifact reconstruction exceeds raw admission cap")
        output.extend(chunk)
        digest.update(chunk)
    if len(output) != bound or digest.hexdigest() != manifest["raw"]["sha256"]:
        raise ArtifactError("artifact reconstructed original raw digest mismatch")
    return output


@dataclass
class ArtifactCacheStats:
    hits: int = 0
    misses: int = 0
    corruptions: int = 0
    evictions: int = 0
    object_requests: int = 0
    manifest_requests: int = 0
    object_download_bytes: int = 0
    manifest_download_bytes: int = 0
    retained_payload_bytes: int = 0
    hash_io_seconds: float = 0.0
    hash_io_cpu_seconds: float = 0.0
    manifest_parse_seconds: float = 0.0
    manifest_parse_cpu_seconds: float = 0.0
    reconstruction_seconds: float = 0.0
    reconstruction_cpu_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@contextlib.contextmanager
def _directory(path: Path, *, create: bool):
    """Resolve each directory relative to a no-follow fd, including ancestors."""
    absolute = path.absolute()
    # Darwin's system /var and /tmp aliases are lexical aliases, not cache
    # symlinks. Address their known canonical locations directly; never follow
    # user-controlled symlinks in any cache component.
    if sys.platform == "darwin" and len(absolute.parts) > 1 and absolute.parts[1] in {"var", "tmp"}:
        absolute = Path("/private").joinpath(*absolute.parts[1:])
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            if part in {".", ".."}:
                raise OSError("artifact cache path must be normalized")
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _open_lock(directory: int, name: str, *, create: bool) -> int:
    # Separate exclusive creation from opening existing locks. Concurrent
    # openat(O_CREAT | O_NOFOLLOW) can spuriously return ENOENT on Darwin.
    try:
        descriptor = os.open(name, (os.O_RDWR if create else os.O_RDONLY) | os.O_NOFOLLOW,
                             dir_fd=directory)
    except FileNotFoundError:
        if not create:
            raise
        try:
            descriptor = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
        except FileExistsError:
            descriptor = os.open(name, os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise OSError("invalid artifact cache lock")
    return descriptor


class ArtifactObjectCache:
    """Byte-capped public objects only; no manifests, KV, masks, or material IDs.

    Reads return verified owned bytes, so eviction cannot invalidate an active
    reader. Objects larger than cap are used transiently and never admitted.
    A single directory flock serializes admission and LRU across SDK processes.
    """

    def __init__(self, root: str | Path, *, max_bytes: int = DEFAULT_CACHE_BYTES,
                 mode: str = "read-write", stats: ArtifactCacheStats | None = None):
        self.root = Path(root).expanduser()
        self.max_bytes = _integer(max_bytes, 1, MAX_RAW_BYTES)
        if mode not in {"read-write", "read-only", "refresh", "off"}:
            raise ArtifactError("invalid artifact cache mode")
        self.mode, self.stats = mode, stats or ArtifactCacheStats()

    @contextlib.contextmanager
    def _locked(self, *, create=True):
        with _directory(self.root, create=create) as directory:
            lock = _open_lock(directory, ".lock", create=create)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
                yield directory
            finally:
                os.close(lock)

    def _inventory(self, directory, *, clean_pending=False):
        rows = []
        with os.scandir(directory) as entries:
            for count, entry in enumerate(entries):
                if count > 2 * _MAX_CACHE_OBJECTS + 4:
                    raise ArtifactError("artifact cache directory exceeds its entry bound")
                name = entry.name
                is_object = len(name) == 69 and name.endswith(".blob")
                is_pending = len(name) == 41 and name.startswith(".pending-") and all(
                    c in "0123456789abcdef" for c in name[9:])
                if is_object or is_pending:
                    item = entry.stat(follow_symlinks=False)
                    if not stat.S_ISREG(item.st_mode):
                        raise OSError("artifact cache object is not a regular file")
                    if is_object:
                        _digest(name[:-5])
                        rows.append((item.st_mtime_ns, name, item.st_size))
                    elif clean_pending:
                        # Admission lock excludes active writers; these are our
                        # own abandoned atomic-write records after process death.
                        os.unlink(name, dir_fd=directory)
        return sorted(rows)

    def get(self, row: dict[str, Any]) -> bytes | None:
        _digest(row["sha256"])
        if self.mode in {"off", "refresh"}:
            self.stats.misses += 1
            return None
        started, cpu_started = time.perf_counter(), time.process_time()
        try:
            with self._locked(create=self.mode == "read-write") as directory:
                inventory = self._inventory(directory, clean_pending=self.mode == "read-write")
                total = sum(r[2] for r in inventory)
                if self.mode == "read-write":
                    for _, victim, size in inventory:
                        if total <= self.max_bytes:
                            break
                        os.unlink(victim, dir_fd=directory)
                        total -= size
                        self.stats.evictions += 1
                self.stats.retained_payload_bytes = total
                name = row["sha256"] + ".blob"
                try:
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                except FileNotFoundError:
                    self.stats.misses += 1
                    return None
                try:
                    item = os.fstat(fd)
                    if not stat.S_ISREG(item.st_mode):
                        raise OSError("artifact cache object is not a regular file")
                    if item.st_size != row["size"]:
                        raise ArtifactError("artifact cache object size mismatch")
                    with os.fdopen(fd, "rb", closefd=False) as stream:
                        payload = stream.read(row["size"] + 1)
                    verify_object(row, payload)
                except ArtifactError:
                    self.stats.corruptions += 1
                    if self.mode == "read-write":
                        os.unlink(name, dir_fd=directory)
                    return None
                finally:
                    os.close(fd)
                if self.mode == "read-write":
                    os.utime(name, dir_fd=directory, follow_symlinks=False)
                self.stats.hits += 1
                self.stats.retained_payload_bytes = sum(r[2] for r in self._inventory(directory))
                return payload
        except FileNotFoundError:
            self.stats.misses += 1
            return None
        finally:
            self.stats.hash_io_seconds += time.perf_counter() - started
            self.stats.hash_io_cpu_seconds += time.process_time() - cpu_started

    def put(self, row: dict[str, Any], payload: bytes) -> None:
        _digest(row["sha256"])
        started, cpu_started = time.perf_counter(), time.process_time()
        verify_object(row, payload)
        if self.mode in {"off", "read-only"} or len(payload) > self.max_bytes:
            self.stats.hash_io_seconds += time.perf_counter() - started
            self.stats.hash_io_cpu_seconds += time.process_time() - cpu_started
            return
        with self._locked() as directory:
            name = row["sha256"] + ".blob"
            inventory = self._inventory(directory, clean_pending=True)
            total = sum(r[2] for r in inventory if r[1] != name)
            count = sum(r[1] != name for r in inventory)
            for _, victim, size in inventory:
                if total + len(payload) <= self.max_bytes and count < _MAX_CACHE_OBJECTS:
                    break
                if victim != name:
                    os.unlink(victim, dir_fd=directory)
                    total -= size
                    count -= 1
                    self.stats.evictions += 1
            temporary = ".pending-" + secrets.token_hex(16)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
            try:
                with os.fdopen(fd, "wb", closefd=False) as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(fd)
                os.rename(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
                self.stats.retained_payload_bytes = total + len(payload)
            finally:
                os.close(fd)
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=directory)
        self.stats.hash_io_seconds += time.perf_counter() - started
        self.stats.hash_io_cpu_seconds += time.process_time() - cpu_started

    def fetch(self, row: dict[str, Any], download: Callable[[], bytes]) -> bytes:
        """Cross-process single-flight admission; no duplicate cold object fetches."""
        if self.mode != "read-write":
            payload = self.get(row)
            if payload is None:
                payload = download()
                self.put(row, payload)
            return payload
        with _directory(self.root, create=True) as directory:
            lock = _open_lock(directory, ".download-lock", create=True)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
                payload = self.get(row)
                if payload is None:
                    payload = download()
                    self.put(row, payload)
                return payload
            finally:
                os.close(lock)


def configure_artifact_cache(client: Any, *, max_bytes: int = DEFAULT_CACHE_BYTES) -> ArtifactObjectCache:
    """Set public-object cache cap using the existing SDK ClientBundleCachePath.

    Call before first request. Selection still belongs to ClientBundleTransport;
    this facade changes ownership/cap only, never inference or private caches.
    """
    core = getattr(client, "_core", client)
    cache = ArtifactObjectCache(core.bundle_cache_dir / "public-artifacts", max_bytes=max_bytes,
                                mode=core.bundle_cache_mode)
    core.bundle_artifact_cache = cache
    core.artifact_cache_stats = cache.stats
    return cache

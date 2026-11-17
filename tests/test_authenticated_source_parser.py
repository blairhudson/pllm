"""Literal parser oracles; no source generation owner or native execution needed."""

import json
import os
from pathlib import Path
import struct

import pytest

from pllm.runtime import delegated_source as source
from pllm.runtime.safetensors_store import SafeTensorStore, TensorStoreError
from pllm.runtime.semantic_tensors import SemanticTensorError


_A = "a.safetensors"
_B = "b.safetensors"
_INDEX = "model.safetensors.index.json"
_NORM = "wrapper.norm.weight"
_TABLE = "wrapper.table.weight"
_HEAD = "wrapper.head.weight"
_HEADER = 512
# Absolute positions and payload words are independent of production specs.
_EXPECTED = tuple(sorted({
    _NORM: (_A, (2,), "F16", 520, 4),
    _TABLE: (_A, (3, 2), "BF16", 524, 12),
    _HEAD: (_B, (3, 2), "F32", 520, 24),
}.items()))
_A_PAYLOAD = struct.pack("<8H", 0x0001, 0x8000, 0x0000, 0x8000,
                         0x3F80, 0xBF00, 0x3E80, 0x4000)
_B_PAYLOAD = struct.pack("<6I", 0, 0x80000000, 0x3F800000,
                         0xBF800000, 0x3E800000, 0x40000000)


def _contract(*, expanded=False):
    operations = [
        {"id": "lookup", "operator": "token_lookup", "inputs": ["input.tokens"],
         "output_shape": [1, 2], "attributes": {
             "weight": "table.weight", "index_domain": "tokens", "weight_shape": [3, 2]}},
        {"id": "norm", "operator": "rms_norm", "inputs": ["lookup"],
         "output_shape": [1, 2], "attributes": {"weight": "norm.weight"}},
        {"id": "head", "operator": "output_head", "inputs": ["norm"],
         "output_shape": [1, 3], "attributes": {"weight": "head.weight"}},
    ]
    result = {
        "source_shapes": {"norm.weight": [2], "table.weight": [3, 2], "head.weight": [3, 2]},
        "source_float32_bytes": 56,
        "plan": {"schema_version": "pllm.decoder_plan.v1",
                 "prefill": {"operations": operations}, "decode": {"operations": operations}},
    }
    if expanded:
        result["decoder_resources"] = {"profile": "pllm.gkm.decoder-capacity.v1"}
    return result


def _headers():
    return {
        _A: {_NORM: {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
             _TABLE: {"dtype": "BF16", "shape": [3, 2], "data_offsets": [4, 16]}},
        _B: {_HEAD: {"dtype": "F32", "shape": [3, 2], "data_offsets": [0, 24]}},
    }


def _literal_files(headers=None, *, header_size=_HEADER, index=True):
    headers = _headers() if headers is None else headers
    files = {"config.json": b"{}", "tokenizer.json": b"{}"}
    for name, payload in ((_A, _A_PAYLOAD), (_B, _B_PAYLOAD)):
        encoded = json.dumps(headers[name], separators=(",", ":")).encode()
        assert len(encoded) <= header_size
        files[name] = struct.pack("<Q", header_size) + encoded.ljust(header_size, b" ") + payload
    if index:
        files[_INDEX] = json.dumps({"weight_map": {
            _NORM: _A, _TABLE: _A, _HEAD: _B,
        }}, separators=(",", ":")).encode()
    return files


class _LiteralReader:
    """Independent positioned byte slices, never production range/parser helpers."""

    def __init__(self, files):
        self.files = dict(files)
        self.inventory = tuple(sorted((name, len(raw)) for name, raw in files.items()))
        self.calls = []

    def __call__(self, name, offset, count):
        assert type(offset) is type(count) is int
        assert 0 <= count <= 65536
        assert 0 <= offset <= offset + count <= len(self.files[name])
        self.calls.append((name, offset, count))
        return self.files[name][offset:offset + count]


def _parse(reader, contract=None):
    geometry = source._tensor_geometry(_contract() if contract is None else contract)
    return source._parse_tensor_inventory(reader.inventory, reader, geometry)


def _forbidden(*args, **kwargs):
    raise AssertionError("metadata parser attempted path discovery/read")


@pytest.mark.parametrize("index", [False, True])
def test_literal_parser_resolves_exact_spans_without_filesystem(monkeypatch, index):
    reader = _LiteralReader(_literal_files(index=index))
    monkeypatch.setattr(Path, "open", _forbidden)
    monkeypatch.setattr(Path, "exists", _forbidden)
    monkeypatch.setattr(Path, "glob", _forbidden)
    monkeypatch.setattr("safetensors.safe_open", _forbidden)
    store, specs = _parse(reader)
    assert isinstance(store, SafeTensorStore)
    assert specs == _EXPECTED
    for logical, physical in (("norm.weight", _NORM), ("table.weight", _TABLE),
                              ("head.weight", _HEAD)):
        assert store.resolve(logical) == physical
        assert store.resolve(physical) == physical
        assert store.tensor_shape(logical) == dict(_EXPECTED)[physical][1]
        assert store.tensor_dtype(logical) == dict(_EXPECTED)[physical][2]
    assert reader.calls[:4] == [(_A, 0, 8), (_A, 8, 512), (_B, 0, 8), (_B, 8, 512)]
    assert all(name not in {"config.json", "tokenizer.json"} for name, _, _ in reader.calls)


def test_metadata_only_store_refuses_every_inherited_path_escape(monkeypatch):
    store, _ = _parse(_LiteralReader(_literal_files()))
    monkeypatch.setattr(Path, "open", _forbidden)
    monkeypatch.setattr(Path, "stat", _forbidden)
    monkeypatch.setattr("safetensors.safe_open", _forbidden)
    for name, args in (
        ("_discover", ()), ("_load_quantization_config", ()),
        ("get", ("norm.weight",)), ("get_slice", ("norm.weight", slice(None))),
        ("iter_slices", ("norm.weight", [slice(None)])),
        ("source_signature", (["norm.weight"],)),
        ("get_optional", (["norm.weight"],)), ("get_linear", (["norm.weight"],)),
    ):
        with pytest.raises(source.DelegatedSourceError, match="metadata-only"):
            getattr(store, name)(*args)
    with pytest.raises(TensorStoreError, match="not found"):
        store.tensor_shape("missing.weight")
    del store._metadata[_NORM]
    with pytest.raises(source.DelegatedSourceError, match="metadata is unavailable"):
        store.tensor_shape("norm.weight")


@pytest.mark.parametrize("change,match", [
    ("overlap", "overlapping or gapped"), ("gap", "overlapping or gapped"),
    ("trailing", "payload differs"), ("past_eof", "byte range mismatch"),
    ("short_span", "byte range mismatch"), ("bool_offset", "byte range mismatch"),
    ("dtype", "unsupported shape or dtype"), ("shape", "invalid shape or dtype"),
    ("duplicate", "duplicate source tensor"), ("ambiguous", "missing 'norm.weight'"),
])
def test_literal_parser_rejects_malformed_spans_and_suffixes(change, match):
    headers = _headers()
    item = headers[_A][_TABLE]
    if change == "overlap":
        item["data_offsets"] = [2, 14]
    elif change == "gap":
        item["data_offsets"] = [6, 18]
    elif change == "past_eof":
        item["data_offsets"] = [8, 20]
    elif change == "short_span":
        item["data_offsets"] = [4, 15]
    elif change == "bool_offset":
        headers[_A][_NORM]["data_offsets"] = [False, 4]
    elif change == "dtype":
        item["dtype"] = "I16"
    elif change == "shape":
        item["shape"] = [2, 3]  # Same bytes, wrong semantic orientation.
    elif change == "duplicate":
        headers[_B][_NORM] = headers[_B].pop(_HEAD)
    elif change == "ambiguous":
        headers[_A]["other.norm.weight"] = headers[_A].pop(_TABLE)
    files = _literal_files(headers, index=False)
    if change == "gap":
        files[_A] += b"\0\0"
    elif change == "trailing":
        files[_A] += b"\0"
    with pytest.raises((source.DelegatedSourceError, SemanticTensorError), match=match):
        _parse(_LiteralReader(files))


@pytest.mark.parametrize("change", ["missing", "extra", "ownership", "external", "duplicate_json"])
def test_literal_parser_requires_exact_index_ownership(change):
    files = _literal_files()
    index = json.loads(files[_INDEX])
    if change == "missing":
        del index["weight_map"][_NORM]
    elif change == "extra":
        index["weight_map"]["extra.weight"] = _A
    elif change == "ownership":
        index["weight_map"][_TABLE] = _B
    elif change == "external":
        index["weight_map"][_TABLE] = "../a.safetensors"
    files[_INDEX] = (b'{"weight_map":{},"weight_map":{}}' if change == "duplicate_json"
                     else json.dumps(index).encode())
    with pytest.raises(source.DelegatedSourceError, match=(
            "invalid tensor index" if change == "duplicate_json" else "index differs")):
        _parse(_LiteralReader(files))


def test_literal_parser_rejects_duplicate_header_json_and_multiple_indexes():
    files = _literal_files()
    header = b'{"x":{},"x":{}}'.ljust(512, b" ")
    files[_A] = struct.pack("<Q", 512) + header + _A_PAYLOAD
    with pytest.raises(source.DelegatedSourceError, match="invalid Safetensors header"):
        _parse(_LiteralReader(files))
    files = _literal_files()
    files["second.safetensors.index.json"] = files[_INDEX]
    with pytest.raises(source.DelegatedSourceError, match="multiple Safetensors indexes"):
        _parse(_LiteralReader(files))


@pytest.mark.parametrize("kind", ["short", "long", "mutable"])
def test_positioned_reader_must_return_exact_immutable_bytes(kind):
    reader = _LiteralReader(_literal_files())

    def broken(name, offset, count):
        raw = reader(name, offset, count)
        return raw[:-1] if kind == "short" else raw + b"x" if kind == "long" else bytearray(raw)

    with pytest.raises(source.DelegatedSourceError, match="inexact range"):
        source._parse_tensor_inventory(reader.inventory, broken, source._tensor_geometry(_contract()))
    assert reader.calls == [(_A, 0, 8)]


@pytest.mark.parametrize("inventory", [
    (("../a.safetensors", 8),), (("a.safetensors", True),),
    (("a.safetensors", -1),), (("a.safetensors", 8), ("a.safetensors", 8)),
    (("literal\\name.safetensors", 8),),
    [("a.safetensors", 8)], tuple((f"{i}.safetensors", 8) for i in range(65)),
])
def test_parser_inventory_is_frozen_flat_unique_and_bounded(inventory):
    with pytest.raises(source.DelegatedSourceError, match="invalid parser inventory"):
        source._parse_tensor_inventory(inventory, _forbidden, source._tensor_geometry(_contract()))


def test_large_metadata_uses_bounded_pieces_and_caps_assembly_first():
    files = _literal_files(header_size=65544)
    files[_INDEX] = files[_INDEX].ljust(65544, b" ")
    reader = _LiteralReader(files)
    _, specs = _parse(reader, _contract(expanded=True))
    assert dict(specs)[_NORM][3] == 65552
    assert (_A, 8, 65536) in reader.calls and (_A, 65544, 8) in reader.calls
    assert (_INDEX, 0, 65536) in reader.calls and (_INDEX, 65536, 8) in reader.calls
    reader = _LiteralReader(files)
    with pytest.raises(source.DelegatedSourceError, match="header exceeds"):
        _parse(reader)
    assert reader.calls == [(_A, 0, 8)]
    files = _literal_files()
    files[_INDEX] = b" " * 65537
    reader = _LiteralReader(files)
    with pytest.raises(source.DelegatedSourceError, match="exceeds its byte limit"):
        _parse(reader)
    assert not any(name == _INDEX for name, _, _ in reader.calls)


def test_aggregate_header_limit_refuses_before_next_header_read(monkeypatch):
    reader = _LiteralReader(_literal_files())
    monkeypatch.setattr(source, "_MAX_METADATA_BYTES", 1023)
    with pytest.raises(source.DelegatedSourceError, match="header exceeds"):
        _parse(reader)
    assert reader.calls == [(_A, 0, 8), (_A, 8, 512), (_B, 0, 8)]


def test_index_refusal_precedes_semantic_shape_preflight():
    headers = _headers()
    headers[_A][_TABLE]["shape"] = [2, 3]
    files = _literal_files(headers)
    files[_INDEX] = b'{"weight_map":{}}'
    with pytest.raises(source.DelegatedSourceError, match="index differs"):
        _parse(_LiteralReader(files))


def test_failure_in_later_metadata_piece_discards_parse_without_fallback(monkeypatch):
    reader = _LiteralReader(_literal_files(header_size=65544))
    monkeypatch.setattr(Path, "open", _forbidden)
    monkeypatch.setattr("safetensors.safe_open", _forbidden)

    def fail_tail(name, offset, count):
        raw = reader(name, offset, count)
        if name == _A and offset == 65544:
            raise source.DelegatedSourceError("sealed reader refused tail")
        return raw

    with pytest.raises(source.DelegatedSourceError, match="sealed reader refused tail"):
        source._parse_tensor_inventory(
            reader.inventory, fail_tail, source._tensor_geometry(_contract(expanded=True)))
    assert reader.calls == [(_A, 0, 8), (_A, 8, 65536), (_A, 65544, 8)]


@pytest.mark.parametrize("failure", ["raised", "short"])
def test_failed_assembly_clears_its_own_traceback_locals_preserving_error_and_custody(failure):
    reader = _LiteralReader(_literal_files(header_size=65544))
    custody = object()
    reader.custody = custody
    original_error = source.DelegatedSourceError("positioned read failed")

    def fail_tail(name, offset, count):
        if name == _A and offset == 65544:
            if failure == "raised":
                raise original_error
            return b""
        return reader(name, offset, count)

    with pytest.raises(source.DelegatedSourceError) as caught:
        source._parse_tensor_inventory(
            reader.inventory, fail_tail, source._tensor_geometry(_contract(expanded=True)))
    if failure == "raised":
        assert caught.value is original_error
    else:
        assert "inexact range" in str(caught.value)
    traceback = caught.value.__traceback__
    assembly_frames = []
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_filename == source.__file__ and frame.f_code.co_name == "read":
            assembly_frames.append(frame)
        traceback = traceback.tb_next
    assert len(assembly_frames) == 1
    # Only this reader's assembly frame is covered, not arbitrary callback locals.
    assert assembly_frames[0].f_locals["raw"] == bytearray()
    assert "block" not in assembly_frames[0].f_locals
    assert reader.custody is custody


def _write_files(root, files):
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    return source._files(root, 56)


def test_ordinary_adapter_matches_literal_metadata_and_checked_value_reads(tmp_path):
    files = _literal_files()
    identities = _write_files(tmp_path, files)
    store, specs = source._store(tmp_path, identities, _contract())
    _, parsed = _parse(_LiteralReader(files))
    assert specs == parsed == _EXPECTED
    assert store.keys == (_NORM, _TABLE, _HEAD)
    # Signed zero and half subnormal are exact F32 words, not numerical equality.
    assert store.get("norm.weight").tobytes() == struct.pack("<2I", 0x33800000, 0x80000000)
    assert store.get("table.weight").tobytes() == struct.pack(
        "<6I", 0, 0x80000000, 0x3F800000, 0xBF000000, 0x3E800000, 0x40000000)
    assert store.get("head.weight").tobytes() == _B_PAYLOAD
    old = tmp_path / _A
    old.rename(tmp_path / "old")
    old.write_bytes(files[_A])
    with pytest.raises(source.DelegatedSourceError, match="source changed"):
        store.get("norm.weight")


@pytest.mark.skipif(os.name != "posix", reason="literal backslash is a POSIX basename character")
def test_ordinary_admitted_backslash_basename_does_not_relax_sealed_inventory(tmp_path):
    files = _literal_files()
    basename = "literal\\a.safetensors"
    files[basename] = files.pop(_A)
    index = json.loads(files[_INDEX])
    index["weight_map"][_NORM] = basename
    index["weight_map"][_TABLE] = basename
    files[_INDEX] = json.dumps(index).encode()
    identities = _write_files(tmp_path, files)
    assert basename in dict(identities)
    store, specs = source._store(tmp_path, identities, _contract())
    expected = tuple((key, (basename if row[0] == _A else row[0], *row[1:]))
                     for key, row in _EXPECTED)
    assert specs == expected
    assert store.get("norm.weight").tobytes() == struct.pack("<2I", 0x33800000, 0x80000000)
    with pytest.raises(source.DelegatedSourceError, match="invalid parser inventory"):
        _parse(_LiteralReader(files))


@pytest.mark.parametrize("header_order", [(_NORM, _TABLE), (_TABLE, _NORM)])
def test_ordinary_keys_keep_shard_and_header_insertion_order(tmp_path, header_order):
    headers = _headers()
    headers[_A] = {key: headers[_A][key] for key in header_order}
    identities = _write_files(tmp_path, _literal_files(headers))
    store, specs = source._store(tmp_path, identities, _contract())
    assert store.keys == (*header_order, _HEAD)
    assert specs == _EXPECTED


class _DirectorySwap:
    """Temporarily change all leaf identities without changing original leaves."""

    def __init__(self, root, alternate, parked):
        self.root, self.alternate, self.parked = root, alternate, parked
        self.active = False

    def swap(self):
        assert not self.active
        self.root.rename(self.parked)
        self.alternate.rename(self.root)
        self.active = True

    def restore(self):
        if self.active:
            self.root.rename(self.alternate)
            self.parked.rename(self.root)
            self.active = False


def _swap_fixture(tmp_path):
    root, alternate = tmp_path / "source", tmp_path / "alternate"
    root.mkdir()
    alternate.mkdir()
    files = _literal_files()
    identities = _write_files(root, files)
    _write_files(alternate, files)
    return root, identities, _DirectorySwap(root, alternate, tmp_path / "parked")


def test_initial_identity_check_cannot_be_replaced_by_reads_or_final_check(tmp_path, monkeypatch):
    root, identities, swap = _swap_fixture(tmp_path)
    original = source._read_range
    reads = []

    def restore_before_read(*args, **kwargs):
        reads.append(True)
        swap.restore()
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "_read_range", restore_before_read)
    swap.swap()
    try:
        with pytest.raises(source.DelegatedSourceError, match="source changed after admission"):
            source._store(root, identities, _contract())
        assert reads == []
    finally:
        swap.restore()
    assert source._files(root, 56) == identities
    # Without the initial check, first read restores identities and every range
    # check plus final inventory check would succeed: this test must then fail.


@pytest.mark.parametrize("boundary", [(_A, 0), (_A, 8), (_INDEX, 0)])
def test_per_read_identity_check_cannot_be_replaced_by_initial_or_final_check(
        tmp_path, monkeypatch, boundary):
    root, identities, swap = _swap_fixture(tmp_path)
    original = source._read_range
    touched = []

    def swap_only_during_read(path, identity, offset, size, **kwargs):
        if (path.name, offset) != boundary:
            return original(path, identity, offset, size, **kwargs)
        touched.append(boundary)
        swap.swap()
        try:
            return original(path, identity, offset, size, **kwargs)
        finally:
            swap.restore()

    monkeypatch.setattr(source, "_read_range", swap_only_during_read)
    with pytest.raises(source.DelegatedSourceError, match="source changed before reading"):
        source._store(root, identities, _contract())
    assert touched == [boundary]
    assert source._files(root, 56) == identities
    # Both boundary inventory checks see the original files. The alternate
    # contains identical parser bytes, so only range identity checks distinguish.


def _change_inside_stream_read(monkeypatch, root, boundary, change, restore=None):
    """Keep real open/seek/read/stat calls; change only after the fd precheck."""
    original_open = Path.open
    original_range = source._read_range
    reads = []

    class ChangingStream:
        def __init__(self, stream, path):
            self.stream, self.path = stream, path

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size):
            position = self.stream.tell()
            if reads or (self.path.name, position) != boundary:
                return self.stream.read(size)
            # Reaching here proves the real admitted-fd precheck succeeded.
            change(self.stream)
            raw = self.stream.read(size)
            reads.append((self.path.name, position, size, raw))
            return raw

    def open_stream(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == root / boundary[0] and args == ("rb",) and not kwargs:
            return ChangingStream(stream, path)
        return stream

    def range_then_restore(*args, **kwargs):
        try:
            return original_range(*args, **kwargs)
        finally:
            # Restore only after the range function returns or raises, never
            # before either post-read check. Outer inventory sees original files.
            if restore is not None:
                restore()

    monkeypatch.setattr(Path, "open", open_stream)
    monkeypatch.setattr(source, "_read_range", range_then_restore)
    return reads


@pytest.mark.parametrize("boundary", [(_A, 0), (_A, 8), (_INDEX, 0)])
def test_post_read_fd_check_is_independent_of_path_and_inventory_checks(
        tmp_path, monkeypatch, boundary):
    root, identities, swap = _swap_fixture(tmp_path)
    files = _literal_files()

    def replace_open_descriptor(stream):
        # Change only the live descriptor, not the admitted pathname or inode.
        # A real same-byte alternate gives an exact read, not a truncation oracle.
        before = os.fstat(stream.fileno())
        alternate = os.open(swap.alternate / boundary[0], os.O_RDONLY)
        try:
            os.lseek(alternate, stream.tell(), os.SEEK_SET)
            os.dup2(alternate, stream.fileno())
        finally:
            os.close(alternate)
        assert os.fstat(stream.fileno()).st_ino != before.st_ino
        assert source._identity(root / boundary[0]) == dict(identities)[boundary[0]]
        # The ordinary stream context closes the substituted descriptor on exit;
        # no filesystem identity needs restoration and no descriptor escapes.

    reads = _change_inside_stream_read(
        monkeypatch, root, boundary, replace_open_descriptor)
    with pytest.raises(source.DelegatedSourceError,
                       match="^GKM source changed or truncated while reading$"):
        source._store(root, identities, _contract())
    name, offset = boundary
    size = 8 if offset == 0 and name == _A else 512 if name == _A else len(files[name])
    assert reads == [(name, offset, size, files[name][offset:offset + size])]
    assert source._files(root, 56) == identities
    store, specs = source._store(root, identities, _contract())
    assert specs == _EXPECTED
    assert store.get("norm.weight").tobytes() == struct.pack("<2I", 0x33800000, 0x80000000)


@pytest.mark.parametrize("boundary", [(_A, 0), (_A, 8), (_INDEX, 0)])
def test_post_read_path_check_is_independent_of_fd_and_inventory_checks(
        tmp_path, monkeypatch, boundary):
    root, identities, swap = _swap_fixture(tmp_path)
    files = _literal_files()

    def replace_path_only(stream):
        before = os.fstat(stream.fileno())
        swap.swap()
        # Moving parent directories leaves the original open leaf unchanged.
        assert os.fstat(stream.fileno()) == before
        assert source._identity(root / boundary[0]) != dict(identities)[boundary[0]]

    reads = _change_inside_stream_read(
        monkeypatch, root, boundary, replace_path_only, swap.restore)
    try:
        with pytest.raises(source.DelegatedSourceError,
                           match="^GKM source changed after reading$"):
            source._store(root, identities, _contract())
    finally:
        swap.restore()
    name, offset = boundary
    size = 8 if offset == 0 and name == _A else 512 if name == _A else len(files[name])
    assert reads == [(name, offset, size, files[name][offset:offset + size])]
    assert source._files(root, 56) == identities
    store, specs = source._store(root, identities, _contract())
    assert specs == _EXPECTED
    assert store.get("norm.weight").tobytes() == struct.pack("<2I", 0x33800000, 0x80000000)


def test_ordinary_adapter_keeps_geometry_before_inventory_refusal(tmp_path):
    identities = _write_files(tmp_path, _literal_files())
    (tmp_path / "unexpected.bin").write_bytes(b"x")
    contract = _contract()
    contract["source_shapes"]["norm.weight"] = [1]
    with pytest.raises(source.DelegatedSourceError, match="native and shared source shapes differ"):
        source._store(tmp_path, identities, contract)
    with pytest.raises(source.DelegatedSourceError, match="source changed after admission"):
        source._store(tmp_path, identities, _contract())


def test_ordinary_adapter_retains_post_parse_inventory_check(tmp_path, monkeypatch):
    identities = _write_files(tmp_path, _literal_files())
    original = source._parse_tensor_inventory

    def parse_then_change(*args, **kwargs):
        result = original(*args, **kwargs)
        (tmp_path / "tokenizer.json").write_bytes(b'{"changed":true}')
        return result

    monkeypatch.setattr(source, "_parse_tensor_inventory", parse_then_change)
    with pytest.raises(source.DelegatedSourceError, match="source changed after admission"):
        source._store(tmp_path, identities, _contract())


def test_ordinary_adapter_detects_shard_change_at_actual_range_boundary(tmp_path, monkeypatch):
    files = _literal_files()
    identities = _write_files(tmp_path, files)
    original = source._read_range

    def change_before_range(path, identity, offset, size, **kwargs):
        if path.name == _A and offset == 8:
            path.write_bytes(files[_A][:-1] + b"x")
        return original(path, identity, offset, size, **kwargs)

    monkeypatch.setattr(source, "_read_range", change_before_range)
    with pytest.raises(source.DelegatedSourceError, match="source changed"):
        source._store(tmp_path, identities, _contract())


@pytest.mark.parametrize("change", ["overlap", "ownership", "shape"])
def test_ordinary_and_positioned_adapters_share_malformed_contract(tmp_path, change):
    headers = _headers()
    if change == "overlap":
        headers[_A][_TABLE]["data_offsets"] = [2, 14]
    elif change == "shape":
        headers[_A][_TABLE]["shape"] = [2, 3]
    files = _literal_files(headers)
    if change == "ownership":
        index = json.loads(files[_INDEX])
        index["weight_map"][_TABLE] = _B
        files[_INDEX] = json.dumps(index).encode()
    identities = _write_files(tmp_path, files)
    with pytest.raises((source.DelegatedSourceError, SemanticTensorError)) as positioned:
        _parse(_LiteralReader(files))
    with pytest.raises(type(positioned.value), match=str(positioned.value)):
        source._store(tmp_path, identities, _contract())

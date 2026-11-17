"""Finite, non-trained SDK-relative lineage and aggregate custody definitions.

Real generated OS files and real lock/admission/append on the happy path. Literal
word/span oracles do not use parsed specs, NumPy conversion or lock output.
Seam stress tests are explicitly not native append coverage.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import psutil
import pytest

from pllm.deployment import resources
from pllm.deployment.resources import reserve_cleanup_slot
from pllm.runtime import benchmark_memory
from pllm.runtime import delegated_source as source_io
from pllm.runtime import replicated_source as replicated
from pllm.runtime import _authenticated_source_generation as generation
from pllm.runtime._authenticated_source_file import _AuthenticatedSourceFile, _SourceFileError
from test_replicated_sdk import experiment_fixture
from test_replicated_literal_worker_lineage import _RECORDS, _NORM, _GATE, _UP
from test_authenticated_source_parser import _contract, _literal_files, _EXPECTED


_A, _B = "a.safetensors", "b.safetensors"
_INDEX = "model.safetensors.index.json"
_PREFIX = "wrapper."
_RECORD_ORACLE = tuple(replace(record, words=(0x0001, 0x8000),
    widened=(0x33800000, 0x80000000)) if record.name == _NORM else record
    for record in _RECORDS)
# Fixed physical split at byte 108, independent of production metadata.
_SPANS = {r.name: (_A if r.position < 108 else _B,
    2056 + r.position - (0 if r.position < 108 else 108), r.width * len(r.words))
    for r in _RECORD_ORACLE}
_WORDS = {r.name: struct.pack(f"<{len(r.words)}I", *(r.widened or r.words))
          for r in _RECORD_ORACLE}


@pytest.fixture(autouse=True)
def _toy_memory_telemetry(monkeypatch):
    """Deterministic samples for this module only, not real host fit evidence."""
    # Keep observe(), byte pricing, accounting and actual cleanup-slot lifecycle
    # real. Replace only machine-dependent samples, never disk policy or limits.
    memory = benchmark_memory.HostMemory(8 << 30, 6 << 30, 0, 0)
    with monkeypatch.context() as scoped:
        scoped.setattr(benchmark_memory, "host_memory", lambda: memory)
        scoped.setattr(psutil.Process, "memory_info", lambda self: SimpleNamespace(rss=64 << 20))
        scoped.setattr(resources, "_linux_memory_headroom", lambda: 6 << 30)
        yield


def _fixture(tmp_path, *, options=True, weights="private"):
    experiment = experiment_fixture(tmp_path.resolve(), weights=weights)
    root = Path(experiment.pipeline.model.source)
    (root / "model.safetensors").unlink()
    files = {}
    for filename in (_A, _B):
        header, payload = {}, bytearray()
        for r in _RECORD_ORACLE:
            shard, absolute, size = _SPANS[r.name]
            if shard != filename:
                continue
            assert len(payload) + 2056 == absolute
            header[_PREFIX + r.name] = {"dtype": r.dtype, "shape": r.shape,
                "data_offsets": [len(payload), len(payload) + size]}
            payload.extend(struct.pack(f"<{len(r.words)}{r.fmt}", *r.words))
        # Reverse JSON key order, keeping physical positions unchanged.
        raw = json.dumps(dict(reversed(tuple(header.items()))), separators=(",", ":")).encode()
        assert len(raw) <= 2048
        files[filename] = struct.pack("<Q", 2048) + raw.ljust(2048, b" ") + payload
    files[_INDEX] = json.dumps({"weight_map": {
        _PREFIX + r.name: _SPANS[r.name][0] for r in _RECORD_ORACLE
    }}).encode()
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    if not options:
        (root / "tokenizer_config.json").unlink()
    return experiment, root


def _forbidden(*args, **kwargs):
    raise AssertionError("sealed generation attempted ordinary path metadata/range I/O")


def _run(experiment, cancelled=lambda: False):
    scope = reserve_cleanup_slot()
    try:
        result = replicated._local_toy_generation(experiment, (3, 4), 2, cancelled,
                                                   cleanup_scope=scope)
        assert scope.closed
        assert type(result) is bytes
        return result
    finally:
        if not scope.closed:
            scope.drain()


@pytest.mark.parametrize("options", [False, True])
@pytest.mark.parametrize("weights", ["private", "public"])
def test_real_local_generation_exact_words_map_lock_and_native_seal(tmp_path, monkeypatch, options, weights):
    experiment, root = _fixture(tmp_path, options=options, weights=weights)
    literal_digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    stage, reference = generation._Generation._stage, generation._Generation._reference_frame
    observed, locks = {}, []
    guarded_root = Path(os.path.abspath(root))

    def source_path_guard(original):
        def guarded(path, *args, **kwargs):
            # Lexical normalization only: deciding whether to forbid an access
            # must not itself stat/resolve/reopen the sealed source. Include the
            # directory, descendants and relative spellings of those paths.
            candidate = Path(os.path.abspath(path))
            if candidate.is_relative_to(guarded_root):
                raise AssertionError(
                    f"sealed generation attempted Path.{original.__name__} on source: {path}")
            # Native destination admission and unrelated pytest/runtime paths
            # retain their real behavior; they are not source authorities.
            return original(path, *args, **kwargs)
        return guarded

    def seal_then_forbid(self, source, inventory):
        locks.extend(source.source_lock["files"])
        assert {r["path"]: r["sha256"] for r in locks} == literal_digests
        stage(self, source, inventory)
        assert self.envelope.file_slots == len(literal_digests)
        assert self.envelope.seal_requested_bytes == sum(p.stat().st_size + 1 for p in root.iterdir())
        for name in ("_files", "_locked", "_read_range", "_read_bounded", "_identity"):
            scoped.setattr(source_io, name, _forbidden)
        for name in ("open", "exists", "glob", "stat", "lstat", "resolve"):
            scoped.setattr(Path, name, source_path_guard(getattr(Path, name)))

    def literal_reference(self, row, begin, count):
        key, physical, filename, shape, dtype, offset, size = row
        assert begin == 0 and physical == _PREFIX + key
        assert (filename, offset, size) == _SPANS[key]
        golden = next(r for r in _RECORD_ORACLE if r.name == key)
        assert shape == golden.shape and dtype == golden.dtype
        result = reference(self, row, begin, count)
        assert result == _WORDS[key]
        observed[key] = result
        return result

    # Destination stays the unwrapped real native owner, including inline and
    # out-of-line native source shapes. This stops at seal, not inference/Q18.
    # Undo even on failure before pytest constructs a traceback/report.
    with monkeypatch.context() as scoped:
        scoped.setattr(generation._Generation, "_stage", seal_then_forbid)
        scoped.setattr(generation._Generation, "_reference_frame", literal_reference)
        manifest = _run(experiment)
    assert json.loads(manifest)
    assert observed == _WORDS


def test_sealed_manifest_retains_admission_timestamp_not_reconstruction_clock(tmp_path, monkeypatch):
    experiment, _ = _fixture(tmp_path)
    admit, build_manifest = replicated.admit_source, source_io._manifest
    admitted, rebuilt = [], []

    def admit_then_distinguish_clock(value):
        source = admit(value)  # real admission, immutable expected snapshot
        expected = json.loads(source._manifest_bytes)
        admitted.append(source._manifest_bytes)

        def different_clock(*args, **kwargs):
            manifest = build_manifest(*args, **kwargs)
            manifest.created_at = expected["created_at"] + 1000.0
            assert manifest.created_at != expected["created_at"]
            rebuilt.append(manifest)
            return manifest

        scoped.setattr(source_io, "_manifest", different_clock)
        return source

    with monkeypatch.context() as scoped:
        scoped.setattr(replicated, "admit_source", admit_then_distinguish_clock)
        manifest = _run(experiment)
    assert json.loads(manifest)
    assert len(admitted) == len(rebuilt) == 1
    assert source_io._canonical(rebuilt[0].to_dict(include_fingerprint=False)) == admitted[0]


@pytest.mark.parametrize("mismatch", ["manifest", "tokenizer_options"])
def test_sealed_reconstruction_rejects_independent_semantic_mismatch_before_destination(
        tmp_path, monkeypatch, mismatch):
    experiment, _ = _fixture(tmp_path)
    stage = generation._Generation._stage
    describe = source_io._tokenizer_from_bytes
    reached = []

    def alter_reconstructed_semantics(*args, **kwargs):
        descriptor = describe(*args, **kwargs)
        manifest = args[2]
        if mismatch == "manifest":
            manifest.metadata["text_config"]["sealed_only_extra"] = True
        else:
            descriptor["add_bos_token"] = not descriptor["add_bos_token"]
        reached.append(True)
        return descriptor

    def stage_then_inject(self, source, inventory):
        stage(self, source, inventory)
        # Inject only after real admission and sealing. Expected immutable
        # manifest/options and every locked source byte remain untouched.
        scoped.setattr(source_io, "_tokenizer_from_bytes", alter_reconstructed_semantics)

    message = ("manifest differs from admission" if mismatch == "manifest"
               else "tokenizer options differ from admission")
    with monkeypatch.context() as scoped:
        scoped.setattr(generation._Generation, "_stage", stage_then_inject)
        scoped.setattr(replicated, "_expanded_publisher", _forbidden)
        with pytest.raises(source_io.DelegatedSourceError, match=message):
            _run(experiment)
    assert reached == [True]


@pytest.mark.parametrize("weights", ["private", "public"])
def test_forwarding_real_native_append_receives_literal_words_and_exact_converted_object(
        tmp_path, monkeypatch, weights):
    experiment, _ = _fixture(tmp_path, weights=weights)
    factory = replicated._expanded_publisher
    converter = source_io._source_float_frame_from_bytes
    converted, appended, sealed, closed = [], [], [], []

    def observe_conversion(raw, dtype):
        frame = converter(raw, dtype)
        assert type(frame) is bytes
        converted.append(frame)
        return frame  # preserve the original immutable converted object

    class ForwardingDestination:
        def __init__(self, real):
            self.real, self.shapes = real, real.shapes
            self.native_append = real.append

        def append(self, key, begin, frame):
            assert begin == 0 and len(converted) == len(appended) + 1
            assert frame is converted[-1]
            assert frame == _WORDS[key]  # independent literal pre-Q18 words
            self.native_append(key, begin, frame)  # identical object, real native append
            appended.append((key, begin, frame))

        def seal(self, ids):
            assert len(appended) == len(_WORDS)
            manifest = self.real.seal(ids)
            sealed.append(manifest)
            return manifest

        def close(self):
            result = self.real.close()
            closed.append(True)
            return result

    monkeypatch.setattr(source_io, "_source_float_frame_from_bytes", observe_conversion)
    monkeypatch.setattr(replicated, "_expanded_publisher",
                        lambda *args: ForwardingDestination(factory(*args)))
    manifest = _run(experiment)
    assert sealed == [manifest] and closed == [True]
    assert tuple(key for key, _, _ in appended) == tuple(sorted(_WORDS))
    assert all(frame is converted[index] for index, (_, _, frame) in enumerate(appended))


@pytest.mark.parametrize("dtype,raw,expected", [
    ("F16", struct.pack("<5H", 0, 0x8000, 1, 0x3C00, 0xBC00),
     struct.pack("<5I", 0, 0x80000000, 0x33800000, 0x3F800000, 0xBF800000)),
    ("BF16", struct.pack("<5H", 0, 0x8000, 1, 0x3F80, 0xBF80),
     struct.pack("<5I", 0, 0x80000000, 0x10000, 0x3F800000, 0xBF800000)),
    ("F32", struct.pack("<5I", 0, 0x80000000, 1, 0x3F800000, 0xBF800000),
     struct.pack("<5I", 0, 0x80000000, 1, 0x3F800000, 0xBF800000)),
])
def test_independent_ieee_widening_literal_words(dtype, raw, expected):
    assert generation._widen_reference(raw, dtype) == expected
    assert source_io._source_float_frame_from_bytes(raw, dtype) == expected


@pytest.mark.parametrize("dtype,fmt,word", [
    ("F16", "H", 0x7C00), ("F16", "H", 0xFE01),
    ("BF16", "H", 0x7F80), ("BF16", "H", 0xFFC1),
    ("F32", "I", 0x7F800000), ("F32", "I", 0xFFC00001),
])
def test_both_converters_refuse_nonfinite(dtype, fmt, word):
    raw = struct.pack("<" + fmt, word)
    with pytest.raises(source_io.DelegatedSourceError, match="nonfinite"):
        generation._widen_reference(raw, dtype)
    with pytest.raises(source_io.DelegatedSourceError, match="invalid values"):
        source_io._source_float_frame_from_bytes(raw, dtype)


@pytest.mark.parametrize("entry", ["README.md", ".pllm-model.json", "extra.bin", "extra.json"])
def test_unsupported_inventory_refuses_before_real_admission(tmp_path, monkeypatch, entry):
    experiment, root = _fixture(tmp_path)
    (root / entry).write_bytes(b"ignored by some ordinary inventories")
    monkeypatch.setattr(replicated, "admit_source", _forbidden)
    with pytest.raises(source_io.DelegatedSourceError, match="unsupported inventory"):
        _run(experiment)


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "tokenizer_config.json", _A, _B, _INDEX])
def test_externally_hard_linked_valid_entry_refuses_before_normal_reads_and_admission(
        tmp_path, monkeypatch, name):
    experiment, root = _fixture(tmp_path)
    original = (root / name).read_bytes()
    external = tmp_path / "external-valid-entry"
    os.link(root / name, external)
    assert external.read_bytes() == original
    assert (root / name).stat().st_nlink == 2
    assert external.stat().st_ino == (root / name).stat().st_ino
    # This is an otherwise valid, allowed basename with unchanged valid bytes,
    # not an extra inventory name or a symlink that another guard would reject.
    monkeypatch.setattr(source_io, "_read_range", _forbidden)
    monkeypatch.setattr(source_io, "_read_bounded", _forbidden)
    monkeypatch.setattr(replicated, "admit_source", _forbidden)
    with pytest.raises(source_io.DelegatedSourceError, match="singly linked"):
        _run(experiment)


@pytest.mark.parametrize("change", ["root_alias", "unsupported_entry", "external_link", "geometry"])
def test_caller_callback_changes_are_checked_before_ordinary_admission(tmp_path, monkeypatch, change):
    experiment, root = _fixture(tmp_path)
    config = json.loads((root / "config.json").read_bytes())
    calls = []

    def change_tree():
        calls.append(True)
        assert len(calls) == 1
        if change == "root_alias":
            parked = tmp_path / "parked-source"
            root.rename(parked)
            root.symlink_to(parked, target_is_directory=True)
        elif change == "unsupported_entry":
            (root / "extra.bin").write_bytes(b"callback-added entry")
        elif change == "external_link":
            os.link(root / _A, tmp_path / "callback-external-link")
        else:
            config["hidden_size"] = 32  # outside the unchanged toy ceiling
            (root / "config.json").write_text(json.dumps(config))
        return False

    monkeypatch.setattr(replicated, "admit_source", _forbidden)
    if change != "geometry":
        monkeypatch.setattr(source_io, "_read_range", _forbidden)
        monkeypatch.setattr(source_io, "_read_bounded", _forbidden)
    match = {"root_alias": "aliased", "unsupported_entry": "unsupported inventory",
             "external_link": "singly linked", "geometry": "geometry exceeds"}[change]
    with pytest.raises(source_io.DelegatedSourceError, match=match):
        _run(experiment, change_tree)
    assert calls == [True]


def test_final_preflight_to_real_admission_has_no_caller_callback(tmp_path, monkeypatch):
    experiment, _ = _fixture(tmp_path)
    preflight, admit = generation._preflight, replicated.admit_source
    events = []

    def cancelled():
        assert not events or events[-1] != "checked"
        events.append("cancel")
        return False

    def checked(*args):
        result = preflight(*args)
        assert events == ["cancel"]
        events.append("checked")
        return result

    def admitted(value):
        assert events == ["cancel", "checked"]
        events.append("admit")
        return admit(value)  # actual SDK admission, not substituted rows/bytes

    monkeypatch.setattr(generation, "_preflight", checked)
    monkeypatch.setattr(replicated, "admit_source", admitted)
    assert json.loads(_run(experiment, cancelled))
    assert events[:3] == ["cancel", "checked", "admit"]


@pytest.mark.parametrize("change", ["alias", "large_file", "count", "total", "second_index"])
def test_finite_inventory_limits_precede_real_admission(tmp_path, monkeypatch, change):
    experiment, root = _fixture(tmp_path)
    if change == "alias":
        (root / "alias.safetensors").symlink_to(root / _A)
    elif change == "large_file":
        with (root / _A).open("ab") as stream:
            stream.truncate((256 << 10) + 1)
    elif change == "count":
        for i in range(9):
            (root / f"{i}.safetensors").write_bytes(b"")
    elif change == "total":
        for name in (_A, _B):
            with (root / name).open("ab") as stream:
                stream.truncate(256 << 10)
    else:
        (root / "other.safetensors.index.json").write_bytes(b"{}")
    monkeypatch.setattr(replicated, "admit_source", _forbidden)
    with pytest.raises(source_io.DelegatedSourceError, match="alias|envelope"):
        _run(experiment)


@pytest.mark.parametrize("source", ["org/not-local", "/definitely/not/a/local/toy"])
def test_remote_and_outside_paths_never_reach_admission(tmp_path, monkeypatch, source):
    experiment, _ = _fixture(tmp_path)
    experiment = experiment.with_params(pipeline__model__source=source)
    monkeypatch.setattr(replicated, "admit_source", _forbidden)
    with pytest.raises((source_io.DelegatedSourceError, FileNotFoundError)):
        _run(experiment)


def test_compact_selection_is_not_connected(tmp_path, monkeypatch):
    experiment, _ = _fixture(tmp_path, weights="public")
    experiment = experiment.with_params(pipeline__components__linear__storage="compact",
        pipeline__components__linear__range_policy="checked")
    monkeypatch.setattr(replicated, "admit_source", _forbidden)
    with pytest.raises(source_io.DelegatedSourceError, match="expanded replicated"):
        _run(experiment)


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "tokenizer_config.json", _A, _INDEX])
def test_equal_length_mutation_after_admission_before_seal_refuses(tmp_path, monkeypatch, name):
    experiment, root = _fixture(tmp_path)
    admit = replicated.admit_source

    def admit_then_mutate(value):
        source = admit(value)
        raw = (root / name).read_bytes()
        # Equal length, even a syntactically valid whitespace replacement cannot
        # become the admitted A generation after its identity/hash changed.
        (root / name).write_bytes(raw[:-1] + (b" " if raw[-1:] != b" " else b"\n"))
        return source

    monkeypatch.setattr(replicated, "admit_source", admit_then_mutate)
    monkeypatch.setattr(replicated, "_expanded_publisher", _forbidden)
    with pytest.raises((_SourceFileError, source_io.DelegatedSourceError)):
        _run(experiment)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unlocked", "size_bool", "digest"])
def test_lock_inventory_must_match_one_to_one(tmp_path, kind):
    experiment, root = _fixture(tmp_path)
    source = replicated.admit_source(experiment)
    lock = source.source_lock
    rows = lock["files"]
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = dict(rows[0])
    elif kind == "unlocked":
        rows[-1]["path"] = "unlocked.safetensors"
    elif kind == "size_bool":
        rows[-1]["size"] = True
    else:
        rows[-1]["sha256"] = "g" * 64
    # Negative-only forged metadata; never used to supply happy-path bytes.
    forged = replace(source, _source_lock_bytes=source_io._canonical(lock))
    with pytest.raises(source_io.DelegatedSourceError, match="inventory|row"):
        generation._selected_rows(forged, generation._local_inventory(root))


def _staged_parser_fixture(tmp_path, *, files=None, contract=None):
    """Seam-only fixture: real shared lock and owners, no native publication."""
    from pllm.configuration import Model

    root = tmp_path.resolve()
    selected_contract = _contract() if contract is None else contract
    source_bytes = selected_contract["source_float32_bytes"]
    for name, raw in (_literal_files() if files is None else files).items():
        (root / name).write_bytes(raw)
    identities = source_io._files(root, source_bytes)
    _, _, lock = source_io._locked(root, Model.path(str(root)), source_bytes, identities)
    # _stage consumes only these genuine already-admitted lock/identity fields;
    # the public dormant entry cannot be constructed from this helper record.
    class Source:
        path = root
        _identities = identities
        source_lock = lock
        contract = selected_contract
        source_float32_bytes = source_bytes
    scope = reserve_cleanup_slot()
    attempt = generation._Generation(scope, lambda: False)
    attempt._stage(Source(), identities)
    return root, attempt, scope


def test_sealed_parser_literal_map_reverse_endpoint_reads_and_path_replacement(tmp_path, monkeypatch):
    root, attempt, scope = _staged_parser_fixture(tmp_path)
    try:
        store, specs = source_io._parse_tensor_inventory(attempt.inventory, attempt._read_at,
                                                        source_io._tensor_geometry(_contract()))
        assert specs == _EXPECTED
        for key, (name, shape, dtype, offset, size) in reversed(_EXPECTED):
            original = (root / name).read_bytes()
            width = 4 if dtype == "F32" else 2
            for position in (offset + size - width, offset):
                assert attempt._read_at(name, position, width) == original[position:position + width]
        old = root / "a.safetensors"
        old.rename(root / "parked")
        old.write_bytes(b"B" * 536)
        monkeypatch.setattr(source_io, "_read_range", _forbidden)
        assert attempt._read_at("a.safetensors", 520, 4) == struct.pack("<2H", 1, 0x8000)
        store.clear_cache()
    finally:
        attempt._finish_operation()
        scope.drain()


def test_in_place_change_outside_requested_slice_same_leaf_refuses(tmp_path):
    root, attempt, scope = _staged_parser_fixture(tmp_path)
    try:
        with (root / "a.safetensors").open("r+b") as stream:
            stream.seek(535)
            stream.write(b"x")
        with pytest.raises(_SourceFileError, match="SHA-256"):
            attempt._read_at("a.safetensors", 520, 2)
    finally:
        attempt._finish_operation()
        scope.drain()


@pytest.mark.parametrize("fault", ["sign", "append", "seal", "cancel_after_last"])
def test_preappend_failure_never_returns_manifest_and_preserves_append_count(tmp_path, monkeypatch, fault):
    experiment, _ = _fixture(tmp_path)
    factory = replicated._expanded_publisher
    converter = source_io._source_float_frame_from_bytes
    calls, seals, closes = [], [], []
    error = RuntimeError("injected native failure")

    class NegativeDestination:
        def __init__(self, real):
            self.real, self.shapes = real, real.shapes

        def append(self, key, begin, frame):
            if fault == "append" and len(calls) == 2:
                raise error
            self.real.append(key, begin, frame)
            calls.append((key, begin, frame))

        def seal(self, ids):
            seals.append(True)
            if fault == "seal":
                raise error
            return self.real.seal(ids)

        def close(self):
            closes.append(True)
            return self.real.close()

    def wrong_sign(raw, dtype):
        frame = converter(raw, dtype)
        return (0x80000000).to_bytes(4, "little") + frame[4:]

    if fault == "sign":
        # First canonical tensor is lm_head: its first word is +0.5. Force a
        # mismatch; signed-zero-specific equality is tested separately below.
        monkeypatch.setattr(source_io, "_source_float_frame_from_bytes", wrong_sign)
    monkeypatch.setattr(replicated, "_expanded_publisher",
        lambda *args: NegativeDestination(factory(*args)))
    cancelled = lambda: fault == "cancel_after_last" and len(calls) == len(_RECORD_ORACLE)
    with pytest.raises((source_io.DelegatedSourceError, RuntimeError)) as caught:
        _run(experiment, cancelled)
    if fault in {"append", "seal"}:
        assert caught.value is error
    expected = 0 if fault == "sign" else 2 if fault == "append" else len(_RECORD_ORACLE)
    assert len(calls) == expected
    assert len(seals) == (1 if fault == "seal" else 0)
    assert closes == [True]


def test_wrong_zero_sign_fails_before_its_append_despite_equal_q18(tmp_path, monkeypatch):
    experiment, _ = _fixture(tmp_path)
    original = source_io._source_float_frame_from_bytes
    references = []
    reference = generation._Generation._reference_frame

    def signed_zero_fault(raw, dtype):
        frame = original(raw, dtype)
        if dtype == "BF16":
            assert frame[:8] == struct.pack("<2I", 0, 0x80000000)
            return struct.pack("<2I", 0, 0) + frame[8:]
        return frame

    def observe(self, *args):
        result = reference(self, *args)
        references.append(args[0][0])
        return result

    monkeypatch.setattr(source_io, "_source_float_frame_from_bytes", signed_zero_fault)
    monkeypatch.setattr(generation._Generation, "_reference_frame", observe)
    with pytest.raises(source_io.DelegatedSourceError, match="F32 word comparison"):
        _run(experiment)
    assert references[-1] == "model.embed_tokens.weight"


@pytest.mark.parametrize("fault", ["pending", "ambiguous", "partial_seal"])
def test_aggregate_cleanup_visits_all_staged_slots_and_retains_uncertain_ones(tmp_path, monkeypatch, fault):
    experiment, root = _fixture(tmp_path)
    source = replicated.admit_source(experiment)
    scope = reserve_cleanup_slot()
    attempt = generation._Generation(scope, lambda: False)
    close, seal = _AuthenticatedSourceFile.close, _AuthenticatedSourceFile.seal
    visited, staged = [], []
    primary = RuntimeError("partial seal primary")

    def fail_seal(self, **kwargs):
        staged.append(self)
        if len(staged) == 2:
            raise primary
        return seal(self, **kwargs)

    def fail_one_close(self):
        visited.append(self)
        if self is staged[0]:
            if fault == "pending":
                return False
            if fault == "ambiguous":
                # Negative-only fake ambiguity closes the real descriptor first
                # so this regression definition itself does not leak an OS fd.
                close(self)
                raise OSError("ambiguous retirement")
        return close(self)

    if fault == "partial_seal":
        monkeypatch.setattr(_AuthenticatedSourceFile, "seal", fail_seal)
    try:
        if fault == "partial_seal":
            with pytest.raises(RuntimeError) as caught:
                attempt._stage(source, generation._local_inventory(root))
            assert caught.value is primary
        else:
            attempt._stage(source, generation._local_inventory(root))
            # Identify one staged owner without exporting any fd/table.
            staged.append(next(iter(attempt._Generation__by_name.values())))
        monkeypatch.setattr(_AuthenticatedSourceFile, "close", fail_one_close)
        attempt._finish_operation()
        if fault == "partial_seal":
            scope.drain()
            assert scope.closed
        else:
            with pytest.raises(source_io.DelegatedSourceError, match="custody retained"):
                scope.drain()
            assert not scope.closed and scope._payload is attempt
        assert len(visited) == len(source._identities)
        if fault == "pending":
            monkeypatch.setattr(_AuthenticatedSourceFile, "close", close)
            scope.drain()
            assert scope.closed
        elif fault == "ambiguous":
            before = tuple(visited)
            with pytest.raises(source_io.DelegatedSourceError, match="custody retained"):
                scope.drain()
            assert tuple(visited) == before  # terminal ambiguity: no retry
            # Test-only disposal of an already OS-retired fake-ambiguous slot.
            attempt._Generation__slots[0][0] = None
            scope.drain()
    finally:
        monkeypatch.setattr(_AuthenticatedSourceFile, "close", close)
        attempt._finish_operation()
        if not scope.closed and fault != "ambiguous":
            scope.drain()


def test_reentrant_close_stops_attempt_without_abandoning_slots(tmp_path):
    experiment, _ = _fixture(tmp_path)
    scope = reserve_cleanup_slot()
    armed = False
    calls = 0

    def cancel():
        nonlocal calls, armed
        calls += 1
        if calls == 8:
            armed = True
            with pytest.raises(source_io.DelegatedSourceError, match="retirement deferred"):
                scope.drain()
        return False

    with pytest.raises(source_io.DelegatedSourceError, match="cancelled or retired"):
        replicated._local_toy_generation(experiment, (3, 4), 2, cancel, cleanup_scope=scope)
    assert armed and scope.closed


def test_work_exhaustion_refuses_without_native_destination(tmp_path, monkeypatch):
    experiment, _ = _fixture(tmp_path)
    monkeypatch.setattr(generation, "_CALLS", 0)
    monkeypatch.setattr(replicated, "_expanded_publisher", _forbidden)
    with pytest.raises(_SourceFileError, match="allowance exhausted"):
        _run(experiment)


def test_self_consistent_equal_shape_swap_needs_independent_literal_map(tmp_path):
    experiment, root = _fixture(tmp_path)
    name = _SPANS[_GATE][0]
    raw = (root / name).read_bytes()
    header = json.loads(raw[8:2056])
    a, b = header[_PREFIX + _GATE], header[_PREFIX + _UP]
    a["data_offsets"], b["data_offsets"] = b["data_offsets"], a["data_offsets"]
    encoded = json.dumps(header, separators=(",", ":")).encode().ljust(2048, b" ")
    (root / name).write_bytes(raw[:8] + encoded + raw[2056:])
    admitted = replicated.admit_source(experiment)  # authentic, valid semantic shapes
    specs = dict(admitted._tensor_specs)
    assert specs[_PREFIX + _GATE][3] == _SPANS[_UP][1]
    assert specs[_PREFIX + _GATE][3] != _SPANS[_GATE][1]


@pytest.mark.parametrize("change", ["dtype", "gap", "trailing", "ownership", "shape"])
def test_freshly_locked_malformed_source_refuses_before_destination(tmp_path, monkeypatch, change):
    experiment, root = _fixture(tmp_path)
    raw = (root / _A).read_bytes()
    header = json.loads(raw[8:2056])
    item = header[_PREFIX + _GATE]
    if change == "dtype":
        item["dtype"] = "I32"
    elif change == "gap":
        item["data_offsets"] = [29, 45]
    elif change == "shape":
        header[_PREFIX + "model.embed_tokens.weight"]["shape"] = [2, 5]
    elif change == "ownership":
        index = json.loads((root / _INDEX).read_bytes())
        index["weight_map"][_PREFIX + _GATE] = _B
        (root / _INDEX).write_text(json.dumps(index))
    header_bytes = json.dumps(header, separators=(",", ":")).encode().ljust(2048, b" ")
    (root / _A).write_bytes(raw[:8] + header_bytes + raw[2056:] + (b"x" if change == "trailing" else b""))
    identities = source_io._files(root, 4096)
    source_io._locked(root, experiment.pipeline.model, 4096, identities)  # malformed but authentic
    monkeypatch.setattr(replicated, "_expanded_publisher", _forbidden)
    with pytest.raises(ValueError):
        _run(experiment)


def test_seam_only_long_rows_split_frames_and_cross_leaf_with_independent_words(tmp_path):
    # This geometry deliberately exceeds native toy width. It proves the shared
    # parser/frame seam, not native append coverage or native geometry admission.
    width = 5000
    shapes = {"norm.weight": [width], "table.weight": [1, width], "head.weight": [1, width]}
    operations = [
        {"id": "lookup", "operator": "token_lookup", "inputs": ["input.tokens"],
         "output_shape": [1, width], "attributes": {"weight": "table.weight",
          "index_domain": "tokens", "weight_shape": [1, width]}},
        {"id": "norm", "operator": "rms_norm", "inputs": ["lookup"],
         "output_shape": [1, width], "attributes": {"weight": "norm.weight"}},
        {"id": "head", "operator": "output_head", "inputs": ["norm"],
         "output_shape": [1, 1], "attributes": {"weight": "head.weight"}},
    ]
    contract = {"source_shapes": shapes, "source_float32_bytes": 60000,
        "plan": {"schema_version": "pllm.decoder_plan.v1",
                 "prefill": {"operations": operations}, "decode": {"operations": operations}}}
    # Norm, table, head are physical order; canonical order starts with head.
    original = (0, 0x8000, 0x3F80, 0xBF00, 1)
    widened = (0, 0x80000000, 0x3F800000, 0xBF000000, 0x10000)
    payload = struct.pack("<5000H", *(original * 1000))
    golden = struct.pack("<5000I", *(widened * 1000))
    header = {key: {"dtype": "BF16", "shape": shape,
        "data_offsets": [index * 10000, (index + 1) * 10000]}
        for index, (key, shape) in enumerate(shapes.items())}
    encoded = json.dumps(header).encode().ljust(65520, b" ")
    files = {"config.json": b"{}", "tokenizer.json": b"{}",
             _A: struct.pack("<Q", 65520) + encoded + payload * 3}
    root, attempt, scope = _staged_parser_fixture(tmp_path, files=files, contract=contract)
    try:
        store, specs = source_io._parse_tensor_inventory(attempt.inventory, attempt._read_at,
                                                        source_io._tensor_geometry(contract))
        literal_spans = {"norm.weight": 65528, "table.weight": 75528, "head.weight": 85528}
        assert {key: spec[3] for key, spec in specs} == literal_spans
        mapping = tuple((key, key, _A, tuple(shapes[key]), "BF16", literal_spans[key], 10000)
                        for key in sorted(shapes))
        attempt.mapping = mapping  # seam-only metadata installation, no publisher
        seen = []
        for index, row in enumerate(mapping):
            for begin, count in ((0, 4096), (4096, 904)):
                raw = attempt._read_at(_A, row[5] + begin * 2, count * 2)
                frame = source_io._source_float_frame_from_bytes(raw, "BF16")
                assert frame == attempt._reference_frame(row, begin, count) == golden[begin * 4:(begin + count) * 4]
                seen.append((row[0], begin, count))
                attempt._Generation__cursor = ((index + 1, 0) if begin + count == width
                                               else (index, begin + count))
        assert len(seen) == 6 and attempt._Generation__cursor == (3, 0)
        # Literal norm first frame [65528,73720) spans both 64-KiB leaves.
        assert 65528 // 65536 != (65528 + 8192 - 1) // 65536
        store.clear_cache()
    finally:
        attempt._finish_operation()
        scope.drain()


@pytest.mark.parametrize("fault", ["reorder", "repeat", "omit", "wrong_count"])
def test_reference_cursor_refuses_reordered_repeated_omitted_frames(tmp_path, fault):
    root, attempt, scope = _staged_parser_fixture(tmp_path)
    row = ("norm.weight", "wrapper.norm.weight", "a.safetensors", (2,), "F16", 520, 4)
    other = ("table.weight", "wrapper.table.weight", "a.safetensors", (3, 2), "BF16", 524, 12)
    attempt.mapping = (row, other)
    try:
        if fault == "repeat":
            assert attempt._reference_frame(row, 0, 2) == struct.pack("<2I", 0x33800000, 0x80000000)
            attempt._Generation__cursor = (1, 0)  # previous append succeeded
        selected = other if fault == "reorder" else row
        begin = 1 if fault == "omit" else 0
        count = 1 if fault == "wrong_count" else 2
        with pytest.raises(source_io.DelegatedSourceError, match="order|offset|count"):
            attempt._reference_frame(selected, begin, count)
    finally:
        attempt._finish_operation()
        scope.drain()


@pytest.mark.parametrize("result", [None, 0, "", True])
def test_toy_cancellation_requires_exact_false_before_admission(tmp_path, monkeypatch, result):
    experiment, _ = _fixture(tmp_path)
    monkeypatch.setattr(replicated, "admit_source", _forbidden)
    with pytest.raises(source_io.DelegatedSourceError, match="cancelled"):
        _run(experiment, lambda: result)


@pytest.mark.parametrize("result", [None, 0, "", False, True])
def test_ordinary_expanded_final_guard_keeps_legacy_truthiness_and_selection(tmp_path, monkeypatch, result):
    from types import SimpleNamespace

    experiment, _ = _fixture(tmp_path)
    calls = []
    class Destination:
        shapes = ()

        def append(self, *args):
            _forbidden()

        def seal(self, ids):
            calls.append("seal")
            return b"{}"

        def close(self):
            calls.append("close")

    destination = Destination()
    source = SimpleNamespace(runtime_contract=lambda *args: calls.append("contract"),
        _append_source=lambda *args: calls.append("append_source"))
    monkeypatch.setattr(replicated, "_expanded_publisher", lambda *args: destination)
    if result:
        with pytest.raises(ValueError, match="cancelled"):
            replicated.ReplicatedSource.publisher(source, experiment, (3, 4), 2, lambda: result)
        assert calls == ["contract", "append_source", "close"]
    else:
        owner, manifest = replicated.ReplicatedSource.publisher(source, experiment, (3, 4), 2, lambda: result)
        assert owner is destination and manifest == b"{}"
        assert calls == ["contract", "append_source", "seal"]


def test_append_primary_survives_ambiguous_file_cleanup_and_other_slots_retire(tmp_path, monkeypatch):
    experiment, root = _fixture(tmp_path)
    scope = reserve_cleanup_slot()
    factory, close = replicated._expanded_publisher, _AuthenticatedSourceFile.close
    primary = RuntimeError("original append error")
    closed, appended = [], []

    class FailingDestination:
        def __init__(self, real):
            self.real, self.shapes = real, real.shapes

        def append(self, *args):
            if len(appended) == 2:
                raise primary
            self.real.append(*args)
            appended.append(True)

        def seal(self, *args):
            _forbidden()

        def close(self):
            return self.real.close()

    def ambiguous_first(self):
        closed.append(self)
        result = close(self)
        if len(closed) == 1:
            raise OSError("test-only ambiguity after actual OS retirement")
        return result

    monkeypatch.setattr(replicated, "_expanded_publisher", lambda *args: FailingDestination(factory(*args)))
    monkeypatch.setattr(_AuthenticatedSourceFile, "close", ambiguous_first)
    with pytest.raises(RuntimeError) as caught:
        replicated._local_toy_generation(experiment, (3, 4), 2, lambda: False, cleanup_scope=scope)
    assert caught.value is primary and appended == [True, True]
    assert len(closed) == len(tuple(root.iterdir()))
    assert not scope.closed and "custody retained" in primary.__notes__[-1]
    attempt = scope._payload
    with pytest.raises(source_io.DelegatedSourceError, match="custody retained"):
        scope.drain()
    assert len(closed) == len(tuple(root.iterdir()))  # no ambiguous retry
    # Dispose only the fake ambiguity whose real descriptor was already closed.
    attempt._Generation__slots[0][0] = None
    scope.drain()
    assert scope.closed


def test_generation_serializes_reentrant_cross_file_io_before_any_descriptor_read(tmp_path, monkeypatch):
    experiment, _ = _fixture(tmp_path)
    scope = reserve_cleanup_slot()
    pread, started = os.pread, []
    reentered = False

    def observing_pread(*args):
        started.append(True)
        return pread(*args)  # negative-path observation, never substitute bytes

    def cancelled():
        nonlocal reentered
        attempt = scope._payload
        if attempt.inventory and not reentered:
            reentered = True
            with pytest.raises(source_io.DelegatedSourceError, match="I/O busy"):
                attempt._read_at(_B, 0, 8)
        return False

    monkeypatch.setattr(os, "pread", observing_pread)
    with pytest.raises(source_io.DelegatedSourceError, match="cancelled or retired"):
        replicated._local_toy_generation(experiment, (3, 4), 2, cancelled, cleanup_scope=scope)
    assert reentered and started == [] and scope.closed

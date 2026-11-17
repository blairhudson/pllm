"""Independent arithmetic and full-prefix controls; no wheel/checkpoint needed."""

import importlib.util
import gc
from copy import deepcopy
from itertools import repeat
from pathlib import Path
import struct
import weakref
import io
import json
import sys
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import replicated_decoder_oracle as oracle
import replicated_oracle_source as source
from dataclasses import replace

from replicated_decoder_oracle import (
    CHECKED_RANGE_LIMIT, GAP, Q, CheckedRangeSchedule, OracleError, PublicMatrix,
    RangeCheck, Tape, _assert_fixture_gqa_coverage, adapt_tapes, array, compare_trace, linear,
    generated_native_layout, mul, reciprocal, reconstruct_test_shares, ring, rmsnorm, rotary_half,
    rsqrt, run_decoder, silu, softmax, source_f32, stable_argmax,
    StreamingTape, StreamingTraceComparator, _ROOT_ALIASES, _LAYER_ALIASES,
    _native_checkpoint_name, _native_checkpoint_value, _check_native_final,
    NativeMaskReplay, native_mask_tapes, GeneratedNativeStream, _StreamProvider,
    _PIPE_HEADER, _pipe_receive, Result,
)


def _pipe_frame(kind, seq, index=0, count=0, payload=b""):
    return _PIPE_HEADER.pack(b"POT1", 1, kind, 0, seq, index, count, len(payload)) + payload


def _lexical_meta():
    """Small schema-valid public metadata, not native execution evidence."""
    config = {"model_type": "qwen3", "hidden_size": 2,
              "intermediate_size": 2, "num_hidden_layers": 1}
    meta = {"config": config, "config_bytes": list(json.dumps(config).encode()),
            "policy": {"range_policy": "static"}, "provenance": "generated",
            "recipe": {}, "program": {}, "sources": {"w": {"shape": [1], "digest": [0]*32}},
            "boundaries": [{"selection": 0, "id": "output_head", "shape": [1, 1],
                            "batch_dimension": True, "oracle_name": None, "coordinates": [[0, 0]]}],
            "spans": [{"start": 0, "end": 1}], "guards": [],
            "demands": [{"kind": "T", "name": "s0/head", "shape": [2], "span": 0,
                         "shift": 18, "end": 1},
                        {"kind": "C", "name": "s0/logits", "shape": [1], "end": 1, "aliases": [0]}],
            "frontiers": [[1, 1, 1], [1, 1, 0]], "terminal": None,
            "capture_peak": 1, "pending_peak": 1, "tape_words": [2], "source_words": 1,
            "checkpoint_words": 1, "total_tape": 2, "invocations": 1, "logical_len": 1,
            "outputs": 2, "input_rows": 2, "fixture_mode": "ordinary", "expected_accepted": True}
    init = (0, 1, 1, 2, 1, 1, 1, 2, 2, 2)
    return meta, init


def _lexical_wire(payload, init=None):
    values = (len(payload), 1, 1, 1, 1, 1, 1, 1, 2, 2) if init is None else (len(payload), *init[1:])
    wire = _pipe_frame(1, 0, count=10, payload=struct.pack("<10Q", *values))
    for offset in range(0, len(payload), 4096):
        chunk = payload[offset:offset+4096]
        wire += _pipe_frame(2, 0, index=offset, count=len(chunk), payload=chunk)
    return wire


@pytest.mark.parametrize("field,cap", [("byte_limit", 16_777_216), ("node_limit", 65_536),
                                      ("depth_limit", 17), ("token_limit", 1024),
                                      ("requested_copy_bytes", 4096)])
@pytest.mark.parametrize("bad", [True, 0, -1, 1.0, float("inf"), float("nan"), "1"])
def test_json_lexical_invalid_policy_precedes_first_init_read(monkeypatch, field, cap, bad):
    policy = oracle.JSONLexicalPolicy()
    object.__setattr__(policy, field, bad)  # Entry must revalidate even exact type.
    touched = []
    monkeypatch.setattr(oracle, "_pipe_receive", lambda *a, **k: touched.append("read"))
    monkeypatch.setattr(json, "loads", lambda *a: touched.append("decode"))
    monkeypatch.setattr(oracle, "_json_census_reference", lambda: touched.append("load"))
    with pytest.raises(OracleError, match="^invalid JSON lexical policy$"):
        oracle._native_stream_main(64 << 20, "none", json_policy=policy)
    with pytest.raises(OracleError, match="^invalid JSON lexical policy$"):
        oracle._stream_public_schema({}, (), json_policy=policy)
    assert touched == []


@pytest.mark.parametrize("field,cap", [("byte_limit", 16_777_216), ("node_limit", 65_536),
                                      ("depth_limit", 17), ("token_limit", 1024),
                                      ("requested_copy_bytes", 4096)])
def test_json_lexical_policy_cannot_raise_verified_caps(field, cap):
    with pytest.raises(OracleError, match="^invalid JSON lexical policy$"):
        oracle.JSONLexicalPolicy(**{field: cap+1})


@pytest.mark.parametrize("policy", [object(), SimpleNamespace(byte_limit=1), False])
def test_json_lexical_policy_rejects_arbitrary_attribute_owners(monkeypatch, policy):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid policy reached read/load/decode")
    monkeypatch.setattr(oracle, "_pipe_receive", forbidden)
    monkeypatch.setattr(oracle, "_json_census_reference", forbidden)
    with pytest.raises(OracleError, match="invalid JSON lexical policy"):
        oracle._native_stream_main(64 << 20, "none", json_policy=policy)


def test_json_lexical_policy_immutable_and_hook_callable():
    from dataclasses import FrozenInstanceError
    policy = oracle.JSONLexicalPolicy()
    with pytest.raises(FrozenInstanceError):
        policy.node_limit = 1
    with pytest.raises(OracleError, match="invalid JSON lexical policy"):
        oracle.JSONLexicalPolicy(copy_admit=1)


def test_json_lexical_policy_uninitialized_exact_instance_refused_before_read(monkeypatch):
    policy = object.__new__(oracle.JSONLexicalPolicy)
    def forbidden(*args, **kwargs):
        pytest.fail("uninitialized policy reached INIT")
    monkeypatch.setattr(oracle, "_pipe_receive", forbidden)
    with pytest.raises(OracleError, match="^invalid JSON lexical policy$"):
        oracle._native_stream_main(64 << 20, "none", json_policy=policy)


def test_json_lexical_direct_loader_cached_without_package_import(monkeypatch):
    import builtins
    before = set(sys.modules)
    importer = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "pllm" or name.startswith("pllm."):
            pytest.fail("oracle census loaded production package")
        return importer(name, *args, **kwargs)
    monkeypatch.setattr(oracle, "_JSON_CENSUS_REFERENCE", None)
    monkeypatch.setattr(builtins, "__import__", guarded)
    reference = oracle._json_census_reference()
    assert oracle._json_census_reference() is reference
    assert reference.json_census(b"[]") == (2, 1, 0, 0, 1, 1, 0)
    assert not any(n == "pllm" or n.startswith("pllm.") for n in set(sys.modules)-before)


@pytest.mark.parametrize("short", [True, False])
def test_json_lexical_zero_list_refuses_before_outer_loads(monkeypatch, short):
    # Lower-limit analogue of valid POT1 M=16,000,001 / 8m-zero counterexample.
    q = 8
    payload = b"[" + b",".join([b"0"]*q) + b"]"
    reader = io.BytesIO(_lexical_wire(payload))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=reader))
    decoded = []
    class AtDecode(Exception):
        pass
    def loads(raw):
        decoded.append(type(raw))
        raise AtDecode
    monkeypatch.setattr(json, "loads", loads)
    policy = oracle.JSONLexicalPolicy(node_limit=q+int(not short))
    with pytest.raises(OracleError if short else AtDecode):
        oracle._native_stream_main(64 << 20, "none", json_policy=policy)
    assert decoded == ([] if short else [bytearray])
    assert reader.tell() == len(reader.getvalue())


@pytest.mark.parametrize("short", [True, False])
def test_json_lexical_multichunk_scan_release_no_outer_copy(monkeypatch, short):
    payload = b'["' + b"x"*5000 + b'"]'
    reader = io.BytesIO(_lexical_wire(payload))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=reader))
    events, views, owned = [], [], []
    scan = oracle._lexical_json_census
    load = oracle._json_census_reference
    def startup():
        assert reader.tell() == 0
        events.append("startup")
        return load()
    def census(raw, policy, reference):
        assert type(raw) is memoryview and raw.readonly and type(raw.obj) is bytearray
        views.append(raw)
        owned.append(raw.obj)
        events.append("census")
        return scan(raw, policy, reference)
    def forbidden_copy(*args):
        pytest.fail("outer immutable copy forbidden")
    class AtDecode(Exception):
        pass
    def loads(raw):
        assert raw is owned[0] and raw == payload
        with pytest.raises(ValueError):
            views[0].obj  # Released before decoder on success.
        events.append("loads")
        raise AtDecode
    monkeypatch.setattr(oracle, "_json_census_reference", startup)
    monkeypatch.setattr(oracle, "_lexical_json_census", census)
    monkeypatch.setattr(oracle, "bytes", forbidden_copy, raising=False)
    monkeypatch.setattr(json, "loads", loads)
    policy = oracle.JSONLexicalPolicy(byte_limit=len(payload)-int(short))
    with pytest.raises(OracleError if short else AtDecode):
        oracle._native_stream_main(64 << 20, "none", json_policy=policy)
    with pytest.raises(ValueError):
        views[0].obj  # Also released on lexical failure.
    assert events == ["startup", "census"] + ([] if short else ["loads"])


@pytest.mark.parametrize("field", ["byte_limit", "requested_copy_bytes"])
def test_json_lexical_config_copy_budget_refuses_before_conversion(monkeypatch, field):
    meta, init = _lexical_meta()
    calls = []
    def forbidden(*args):
        calls.append(True)
        pytest.fail("denied config copy reached conversion/decode")
    monkeypatch.setattr(oracle, "bytes", forbidden, raising=False)
    monkeypatch.setattr(json, "loads", forbidden)
    policy = oracle.JSONLexicalPolicy(**{field: len(meta["config_bytes"])-1})
    with pytest.raises(OracleError, match="^JSON requested immutable-copy envelope refused$"):
        oracle._stream_public_schema(meta, init, json_policy=policy)
    assert calls == []


@pytest.mark.parametrize("raw", [None, b"{}", [], [True], [256], [-1], [1.0]])
def test_json_lexical_config_original_shortcircuit_precedes_copy_hook(monkeypatch, raw):
    meta, init = _lexical_meta()
    meta["config_bytes"] = raw
    def forbidden(*args):
        pytest.fail("invalid config representation reached admission/copy/decode")
    monkeypatch.setattr(oracle, "bytes", forbidden, raising=False)
    monkeypatch.setattr(json, "loads", forbidden)
    with pytest.raises(OracleError, match="^actual generated config bytes required$"):
        oracle._stream_public_schema(meta, init, json_policy=oracle.JSONLexicalPolicy(copy_admit=forbidden))


@pytest.mark.parametrize("mutation", ["provenance", "unexpected_key"])
def test_json_lexical_config_copy_hook_revalidates_root_before_copy_and_decode(monkeypatch, mutation):
    meta, init = _lexical_meta()
    events = []

    def admit(n, held):
        assert held is meta and n == len(meta["config_bytes"])
        events.append("admit")
        if mutation == "provenance":
            held["provenance"] = "not-generated"
        else:
            held["unexpected"] = 1

    def forbidden(*args):
        pytest.fail("mutated root reached config copy/second decode")

    monkeypatch.setattr(oracle, "bytes", forbidden, raising=False)
    monkeypatch.setattr(json, "loads", forbidden)
    with pytest.raises(OracleError, match="^generated public metadata schema$"):
        oracle._stream_public_schema(meta, init, json_policy=oracle.JSONLexicalPolicy(copy_admit=admit))
    assert events == ["admit"]


@pytest.mark.parametrize("mutation", ["grow", "shrink", "bool", "range", "replace", "policy", "deny"])
def test_json_lexical_config_copy_hook_observes_held_graph_revalidates(monkeypatch, mutation):
    meta, init = _lexical_meta()
    events = []
    def admit(n, held):
        assert held is meta and n == len(meta["config_bytes"])
        assert held["sources"] and held["config"]
        events.append("admit")
        if mutation == "grow":
            held["config_bytes"].append(32)
        elif mutation == "shrink":
            held["config_bytes"].pop()
        elif mutation == "bool":
            held["config_bytes"][0] = True
        elif mutation == "range":
            held["config_bytes"][0] = 256
        elif mutation == "replace":
            held["config_bytes"] = list(held["config_bytes"])
        elif mutation == "policy":
            object.__setattr__(policy, "byte_limit", float("inf"))
        else:
            raise RuntimeError("private callback detail")
    policy = oracle.JSONLexicalPolicy(copy_admit=admit)
    def forbidden(*args):
        pytest.fail("mutated config reached copy/decode")
    monkeypatch.setattr(oracle, "bytes", forbidden, raising=False)
    monkeypatch.setattr(json, "loads", forbidden)
    with pytest.raises(OracleError) as caught:
        oracle._stream_public_schema(meta, init, json_policy=policy)
    assert str(caught.value) in ("invalid JSON lexical policy", "JSON requested immutable-copy envelope refused")
    assert events == ["admit"]


@pytest.mark.parametrize("short", [True, False])
def test_json_lexical_nested_config_census_precedes_second_decoder(monkeypatch, short):
    meta, init = _lexical_meta()
    meta["config_bytes"] = list(b"[0,0,0]")
    meta["config"] = [0, 0, 0]
    calls = []
    loads = json.loads
    def decode(raw):
        calls.append(raw)
        return loads(raw)
    monkeypatch.setattr(json, "loads", decode)
    policy = oracle.JSONLexicalPolicy(node_limit=4-int(short))
    with pytest.raises(OracleError, match="lexical envelope" if short else "generated stream geometry"):
        oracle._stream_public_schema(meta, init, json_policy=policy)
    assert calls == ([] if short else [b"[0,0,0]"])


@pytest.mark.parametrize("policy_enabled", [False, True])
def test_json_lexical_success_keeps_schema_and_exact_copy_order(monkeypatch, policy_enabled):
    meta, init = _lexical_meta()
    expected = oracle._stream_public_schema(deepcopy(meta), init)
    events = []
    scan = oracle._lexical_json_census
    loads = json.loads
    def census(raw, policy, reference):
        assert type(raw) is bytes
        events.append("config-census")
        return scan(raw, policy, reference)
    def convert(raw):
        assert raw is meta["config_bytes"]
        events.append("copy")
        return bytes(raw)
    def decode(raw):
        events.append("config-loads")
        return loads(raw)
    def admit(n, held):
        assert held is meta and n == len(meta["config_bytes"])
        events.append("copy-admit")
    policy = oracle.JSONLexicalPolicy(requested_copy_bytes=len(meta["config_bytes"]), copy_admit=admit)
    monkeypatch.setattr(oracle, "_lexical_json_census", census)
    monkeypatch.setattr(oracle, "bytes", convert, raising=False)
    monkeypatch.setattr(json, "loads", decode)
    assert oracle._stream_public_schema(meta, init, json_policy=policy if policy_enabled else None) == expected
    assert events == (["copy-admit", "copy", "config-census", "config-loads"] if policy_enabled
                      else ["copy", "config-loads"])


@pytest.mark.parametrize("raw", [b"[1,]", b"\xff"])
def test_json_lexical_leaves_decoder_exceptions_unchanged(raw):
    meta, init = _lexical_meta()
    meta["config_bytes"] = list(raw)
    # Invalid UTF-8 must be inside quotes to pass ASCII lexical policy.
    if raw == b"\xff":
        meta["config_bytes"] = list(b'"\xff"')
    with pytest.raises(json.JSONDecodeError if raw != b"\xff" else UnicodeDecodeError):
        oracle._stream_public_schema(meta, init, json_policy=oracle.JSONLexicalPolicy())


def test_json_lexical_config_mismatch_message_unchanged():
    meta, init = _lexical_meta()
    meta["config_bytes"] = list(b"{}")
    with pytest.raises(OracleError, match="^actual generated config bytes required$"):
        oracle._stream_public_schema(meta, init, json_policy=oracle.JSONLexicalPolicy())


def test_json_lexical_full_outer_config_schema_order_before_source(monkeypatch):
    meta, init = _lexical_meta()
    meta["recipe"] = {"padding": "x"*5000}  # Real multi-chunk POT1 assembly.
    payload = json.dumps(meta).encode()
    reader = io.BytesIO(_lexical_wire(payload, init))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=reader))
    events, outer = [], []
    scan, loads = oracle._lexical_json_census, json.loads
    schema, receive = oracle._stream_public_schema, oracle._pipe_receive
    def census(raw, policy, reference):
        events.append("outer-census" if type(raw) is memoryview else "config-census")
        return scan(raw, policy, reference)
    def decode(raw):
        if type(raw) is bytearray:
            events.append("outer-loads")
            result = loads(raw)
            outer.append(result)
            return result
        assert type(raw) is bytes
        events.append("config-loads")
        return loads(raw)
    def copy(raw):
        assert type(raw) is list and raw is outer[0]["config_bytes"]
        events.append("copy")
        return bytes(raw)
    def admit(n, held):
        assert held is outer[0] and n == len(held["config_bytes"])
        events.append("copy-admit")
    def validate(meta, init, *, json_policy=None):
        result = schema(meta, init, json_policy=json_policy)
        events.append("rest-of-schema")
        return result
    class AtSource(Exception):
        pass
    def recv(reader, sequence, expected, **kwargs):
        if expected[0] == 3:
            raise AtSource
        return receive(reader, sequence, expected, **kwargs)
    monkeypatch.setattr(oracle, "_lexical_json_census", census)
    monkeypatch.setattr(json, "loads", decode)
    monkeypatch.setattr(oracle, "bytes", copy, raising=False)
    monkeypatch.setattr(oracle, "_stream_public_schema", validate)
    monkeypatch.setattr(oracle, "_pipe_receive", recv)
    with pytest.raises(AtSource):
        oracle._native_stream_main(64 << 20, "none", json_policy=oracle.JSONLexicalPolicy(copy_admit=admit))
    assert events == ["outer-census", "outer-loads", "copy-admit", "copy",
                      "config-census", "config-loads", "rest-of-schema"]


@pytest.mark.parametrize("seam", ["outer", "config"])
def test_json_lexical_refusal_never_reaches_sources_replay_or_success(monkeypatch, seam):
    meta, init = _lexical_meta()
    if seam == "config":
        meta["config_bytes"] = list(b"0"*1025)
    payload = json.dumps(meta).encode()
    reader = io.BytesIO(_lexical_wire(payload, init))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=reader))
    receive = oracle._pipe_receive
    def recv(reader, sequence, expected, **kwargs):
        assert expected[0] in (1, 2), "preparse refusal reached SOURCE"
        return receive(reader, sequence, expected, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("preparse refusal reached replay/client/success")
    monkeypatch.setattr(oracle, "_pipe_receive", recv)
    monkeypatch.setattr(oracle, "NativeMaskReplay", forbidden)
    monkeypatch.setattr(oracle, "GeneratedNativeStream", forbidden)
    policy = oracle.JSONLexicalPolicy(node_limit=1) if seam == "outer" else oracle.JSONLexicalPolicy()
    with pytest.raises(OracleError, match="^JSON lexical envelope refused$"):
        oracle._native_stream_main(64 << 20, "none", json_policy=policy)


def _raw_read_spies(monkeypatch, wire):
    allocations, reads = [], []

    def allocate(count):
        allocations.append(count)
        return bytearray(count)

    class Reader(io.BytesIO):
        def read(self, count=-1):
            reads.append(count)
            return super().read(count)

    monkeypatch.setattr(oracle, "bytearray", allocate, raising=False)
    return Reader(wire), allocations, reads


@pytest.mark.parametrize("short", [False, True])
def test_pipe_raw_exact_threshold_precedes_allocator_and_read(monkeypatch, short):
    reader, allocations, reads = _raw_read_spies(monkeypatch, b"payload")
    if short:
        with pytest.raises(OracleError, match="raw requested-payload envelope"):
            oracle._pipe_exact(reader, 7, raw_limit=20)
        assert allocations == reads == []
    else:
        assert oracle._pipe_exact(reader, 7, raw_limit=21) == b"payload"
        assert allocations == reads == [7]


@pytest.mark.parametrize("limit", [True, False, -1, 0.0, float("inf"), "3"])
def test_pipe_raw_invalid_limits_precede_allocator_and_read(monkeypatch, limit):
    reader, allocations, reads = _raw_read_spies(monkeypatch, b"")
    with pytest.raises(OracleError, match="raw requested-payload limit"):
        oracle._pipe_exact(reader, 0, raw_limit=limit)
    with pytest.raises(OracleError, match="raw requested-payload limit"):
        _pipe_receive(reader, 1, (12, 0, 0, 0), raw_limit=limit)
    with pytest.raises(OracleError, match="raw requested-payload limit"):
        GeneratedNativeStream(reader, io.BytesIO(), {}, None, raw_limit=limit)
    assert allocations == reads == []


def test_pipe_raw_zero_legacy_and_existing_count_cap(monkeypatch):
    reader, allocations, reads = _raw_read_spies(monkeypatch, b"x")
    assert oracle._pipe_exact(reader, 0, raw_limit=0) == b""
    assert allocations == [0] and reads == []
    assert oracle._pipe_exact(reader, 1) == b"x"  # Optional None preserves callers.
    assert oracle._pipe_exact(io.BytesIO(b"y"), 1, raw_limit=1 << 200) == b"y"
    before = (list(allocations), list(reads))
    with pytest.raises(OracleError, match="allocation bound"):
        oracle._pipe_exact(reader, (64 << 20)+1, raw_limit=1 << 60)
    assert (allocations, reads) == before


@pytest.mark.parametrize("size", [0, 80, 92])
@pytest.mark.parametrize("short", [False, True])
def test_pipe_raw_frame_header_and_body_thresholds(monkeypatch, size, short):
    limit = max(72, 3*size)-int(short)
    reader, allocations, reads = _raw_read_spies(monkeypatch, _pipe_frame(8, 1, count=size, payload=b"x"*size))
    if short:
        with pytest.raises(OracleError, match="raw requested-payload envelope"):
            _pipe_receive(reader, 1, (8, 0, size, size), raw_limit=limit)
        assert allocations == reads == ([] if size == 0 else [24])
    else:
        assert _pipe_receive(reader, 1, (8, 0, size, size), raw_limit=limit) == b"x"*size
        assert allocations == [24, size]
        assert reads == ([24, size] if size else [24])


def test_pipe_raw_wrong_geometry_never_allocates_body(monkeypatch):
    reader, allocations, reads = _raw_read_spies(monkeypatch, _pipe_frame(8, 1, count=11, payload=b"x"*88))
    with pytest.raises(OracleError, match="geometry"):
        _pipe_receive(reader, 1, (8, 0, 10, 80), raw_limit=264)
    assert allocations == reads == [24]


@pytest.mark.parametrize("limit", [None, 72])
def test_pipe_raw_failure_forwarding_uses_header_dominated_limit(monkeypatch, limit):
    reader, allocations, reads = _raw_read_spies(monkeypatch, _pipe_frame(13, 1, count=1, payload=struct.pack("<H", 1)))
    exact, calls = oracle._pipe_exact, []

    def observe(reader, count, *, raw_limit=None):
        calls.append((count, raw_limit))
        return exact(reader, count, raw_limit=raw_limit)

    monkeypatch.setattr(oracle, "_pipe_exact", observe)
    with pytest.raises(OracleError, match="native pipe failure"):
        _pipe_receive(reader, 1, (12, 0, 0, 0), raw_limit=limit)
    assert calls == [(24, limit), (2, limit)]
    assert allocations == reads == [24, 2]  # No invented six-byte frame threshold.


@pytest.mark.parametrize("short", [False, True])
def test_pipe_raw_event_forwarding_and_next_response(monkeypatch, short):
    wire = _pipe_frame(6, 1, count=1, payload=b"e"*92) + _pipe_frame(12, 1)
    reader, allocations, reads = _raw_read_spies(monkeypatch, wire)
    events = []

    def event(index, payload):
        events.append((index, len(payload)))  # Does NOT retain raw callback argument.

    if short:
        with pytest.raises(OracleError, match="raw requested-payload envelope"):
            _pipe_receive(reader, 1, (12, 0, 0, 0), event=event, spans=[{}], raw_limit=275)
        assert allocations == reads == [24] and events == []
    else:
        assert _pipe_receive(reader, 1, (12, 0, 0, 0), event=event, spans=[{}], raw_limit=276) == b""
        assert events == [(0, 92)]
        assert allocations == [24, 92, 24, 0] and reads == [24, 92, 24]


def test_pipe_raw_fragmented_reads_and_empty_fragment(monkeypatch):
    requests = []
    fragments = iter([b"a", b"bc", b"d", b"efg"])

    class Reader:
        def read(self, count):
            requests.append(count)
            return next(fragments)

    assert oracle._pipe_exact(Reader(), 7, raw_limit=21) == b"abcdefg"
    assert requests == [7, 6, 4, 3]
    reader, allocations, reads = _raw_read_spies(monkeypatch, b"")
    with pytest.raises(OracleError, match="truncated"):
        oracle._pipe_exact(reader, 7, raw_limit=21)
    assert allocations == reads == [7]


def test_pipe_raw_oversized_return_rejected_before_assignment(monkeypatch):
    destinations, requests, assignments = [], [], []

    class Destination(bytearray):
        def __setitem__(self, key, value):
            assignments.append(key)
            raise AssertionError("oversized reader reached slice assignment")

    def allocate(count):
        result = Destination(count)
        destinations.append(result)
        return result

    class Reader:
        def read(self, count):
            requests.append(count)
            return b"x"*(count+1)  # External excessive allocation already exists.

    monkeypatch.setattr(oracle, "bytearray", allocate, raising=False)
    with pytest.raises(OracleError, match="oversized pipe reader"):
        oracle._pipe_exact(Reader(), 7, raw_limit=21)
    assert requests == [7] and assignments == []
    assert len(destinations) == 1 and len(destinations[0]) == 7


@pytest.mark.parametrize("mbytes,threshold", [(1, 73), (24, 96), (4096, 16384), (4097, 16385)])
def test_metadata_raw_pure_threshold_before_any_allocation(monkeypatch, mbytes, threshold):
    reader, allocations, reads = _raw_read_spies(monkeypatch, b"")
    assert oracle._metadata_raw_remaining(mbytes, threshold) == threshold-mbytes
    with pytest.raises(OracleError, match="metadata raw requested-payload envelope"):
        oracle._metadata_raw_remaining(mbytes, threshold-1)
    assert allocations == reads == [] and reader.tell() == 0


@pytest.mark.parametrize("mbytes", [4096, 4097])
@pytest.mark.parametrize("short", [False, True])
def test_native_metadata_raw_admission_and_direct_chunk_assembly(monkeypatch, mbytes, short):
    # Stop at JSON entry; no schema, sources, AES or decoder execution.
    payload = b"x"*mbytes
    init = struct.pack("<10Q", mbytes, 1, 1, 1, 1, 1, 1, 1, 2, 2)
    wire = _pipe_frame(1, 0, count=10, payload=init)
    for offset in range(0, mbytes, 4096):
        part = payload[offset:offset+4096]
        wire += _pipe_frame(2, 0, index=offset, count=len(part), payload=part)
    reader, allocations, reads = _raw_read_spies(monkeypatch, wire)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=reader))
    parsed, receives = [], []
    receive = oracle._pipe_receive

    def observe(reader, sequence, expected, **kwargs):
        receives.append((expected, kwargs.get("raw_limit")))
        return receive(reader, sequence, expected, **kwargs)

    monkeypatch.setattr(oracle, "_pipe_receive", observe)

    class AtJSON(Exception):
        pass

    def decode(metadata):
        parsed.append(bytes(metadata))
        raise AtJSON

    monkeypatch.setattr(json, "loads", decode)
    limit = mbytes+3*4096-int(short)
    if short:
        with pytest.raises(OracleError, match="metadata raw requested-payload envelope"):
            oracle._native_stream_main(limit, "none")
        assert allocations == reads == [24, 80] and parsed == []
        assert receives == [((1, 0, 10, 80), limit)]
    else:
        with pytest.raises(AtJSON):
            oracle._native_stream_main(limit, "none")
        assert parsed == [payload]
        sizes = [24, 4096] + ([24, 1] if mbytes == 4097 else [])
        assert allocations == [24, 80, mbytes] + sizes
        assert reads == [24, 80] + sizes
        assert receives == [((1, 0, 10, 80), limit)] + [
            ((2, offset, min(4096, mbytes-offset), min(4096, mbytes-offset)), limit-mbytes)
            for offset in range(0, mbytes, 4096)]


@pytest.mark.parametrize("short", [False, True])
def test_native_init_raw_threshold_is_240_before_metadata(monkeypatch, short):
    init = struct.pack("<10Q", 1, 1, 1, 1, 1, 1, 1, 1, 2, 2)
    reader, allocations, reads = _raw_read_spies(monkeypatch, _pipe_frame(1, 0, count=10, payload=init))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=reader))
    admitted = []
    admission = oracle._metadata_raw_remaining

    class AtMetadata(Exception):
        pass

    def stop(mbytes, limit):
        admission(mbytes, limit)
        admitted.append((mbytes, limit))
        raise AtMetadata

    monkeypatch.setattr(oracle, "_metadata_raw_remaining", stop)
    if short:
        with pytest.raises(OracleError, match="raw requested-payload envelope"):
            oracle._native_stream_main(239, "none")
        assert allocations == reads == [24] and admitted == []
    else:
        with pytest.raises(AtMetadata):
            oracle._native_stream_main(240, "none")
        assert allocations == reads == [24, 80] and admitted == [(1, 240)]


def test_native_main_all_direct_receives_and_client_forward_raw_limit(monkeypatch):
    """Routing control with schema/replay/client stubs, NOT native qualification."""
    source = struct.pack("<q", 7)
    meta = {"sources": {"w": {"shape": [1], "digest": list(hashlib.sha256(source).digest())}},
            "config": {"vocab_size": 3}}
    encoded = json.dumps(meta).encode()
    mbytes = len(encoded)
    limit = max(240, mbytes+max(72, 3*min(mbytes, 4096)))
    init = struct.pack("<10Q", mbytes, 1, 1, 1, 1, 1, 1, 1, 2, 2)
    payloads = {(1, 0, 10, 80): init, (2, 0, mbytes, mbytes): encoded,
                (3, 0, 1, 16): struct.pack("<Q", 0)+source,
                (4, 0, 2, 16): struct.pack("<2Q", 0, 1), (5, 0, 4, 64): bytes(64)}
    receives, clients = [], []

    def receive(reader, sequence, expected, *, raw_limit=None):
        assert sequence == 0
        receives.append((expected, raw_limit))
        return payloads[expected]

    class AtClient(Exception):
        pass

    def client(reader, writer, metadata, replay, *, mutation="none", raw_limit=None):
        clients.append((raw_limit, mutation))
        raise AtClient

    monkeypatch.setattr(oracle, "_pipe_receive", receive)
    monkeypatch.setattr(oracle, "_stream_public_schema", lambda meta, init: ((), (), None))
    monkeypatch.setattr(oracle, "NativeMaskReplay", lambda *args, **kwargs: None)
    monkeypatch.setattr(oracle, "GeneratedNativeStream", client)
    with pytest.raises(AtClient):
        oracle._native_stream_main(limit, "none")
    assert receives == [(expected, limit-mbytes if expected[0] == 2 else limit)
                        for expected in payloads]
    assert clients == [(limit, "none")]


@pytest.mark.parametrize("kind,response,count,length", [
    (32, 7, 0, 0), (33, 8, 5, 40), (34, 9, 6, 48), (34, 9, 9, 72),
    (35, 10, 0, 0), (36, 11, 0, 0), (37, 12, 0, 0)])
@pytest.mark.parametrize("short", [False, True])
def test_native_client_raw_policy_reaches_checkpoint_final_eof_success(monkeypatch, kind, response, count, length, short):
    reader, allocations, reads = _raw_read_spies(monkeypatch, _pipe_frame(response, 1, count=count, payload=b"x"*length))
    output = io.BytesIO()
    meta = {"spans": [], "demands": []}
    client = GeneratedNativeStream(reader, output, meta, None, raw_limit=max(72, 3*length)-int(short))
    if short:
        with pytest.raises(OracleError, match="raw requested-payload envelope"):
            client.request(kind, 0, response, count, length)
        assert allocations == reads == ([] if length == 0 else [24])
        assert client.sequence == 1
    else:
        assert client.request(kind, 0, response, count, length) == b"x"*length
        assert allocations == [24, length] and client.sequence == 2
    assert _PIPE_HEADER.unpack(output.getvalue())[2] == kind


@pytest.mark.parametrize("short", [False, True])
def test_native_client_raw_policy_reaches_event_callback(monkeypatch, short):
    wire = _pipe_frame(6, 1, count=1, payload=b"e"*92) + _pipe_frame(7, 1)
    reader, allocations, reads = _raw_read_spies(monkeypatch, wire)
    client = GeneratedNativeStream(reader, io.BytesIO(), {"spans": [{}], "demands": []}, None,
                                   raw_limit=276-int(short))
    events = []
    client._event = lambda index, payload: events.append((index, len(payload)))
    if short:
        with pytest.raises(OracleError, match="raw requested-payload envelope"):
            client.request(32, 0, 7)
        assert events == [] and allocations == reads == [24]
    else:
        assert client.request(32, 0, 7) == b""
        assert events == [(0, 92)] and allocations == [24, 92, 24, 0]


@pytest.mark.parametrize("change", ["oversized", "wrong_sequence", "reserved", "wrong_count", "unknown"])
def test_generated_pipe_header_rejects_before_payload_allocation(change):
    header = list(_PIPE_HEADER.unpack(_pipe_frame(8, 1, count=1, payload=b"\0"*8)[:24]))
    if change == "oversized":
        header[-1] = (1 << 32)-1
    elif change == "wrong_sequence":
        header[4] = 2
    elif change == "reserved":
        header[3] = 1
    elif change == "wrong_count":
        header[-2] = 2
    else:
        header[2] = 99

    class HeaderOnly(io.BytesIO):
        def read(self, count=-1):
            assert self.tell() == 0 and count == 24, "payload read before header admission"
            return super().read(count)

    with pytest.raises(OracleError):
        _pipe_receive(HeaderOnly(_PIPE_HEADER.pack(*header)), 1, (8, 0, 1, 8))


@pytest.mark.parametrize("tail", [0, 1, 7])
def test_generated_pipe_truncated_payload_is_terminal(tail):
    wire = _pipe_frame(8, 1, count=1, payload=b"\0"*8)
    with pytest.raises(OracleError, match="truncated"):
        _pipe_receive(io.BytesIO(wire[:24+tail]), 1, (8, 0, 1, 8))


def _synthetic_replay_group(replay, binding, lanes, public, pair):
    """Synthetic pure transport control, NOT genuine native execution evidence."""
    pivot = replay.reference.source_fill(replay.seed.to_bytes(16, "little"), public-1, 1)[0] % 3
    a, b, h = pivot, (pivot+1)%3, (pivot+2)%3
    return [dict(binding=binding, role=role, to=to, phase=phase, lanes=lanes,
                 public_end=public, pair_end=pair, node=public, loop_offset=0,
                 tensor_step=None, invocation=None) for role, to, phase in
            ((a, b, 10), (b, a, 10), (h, a, 11), (h, b, 11))]


def test_incremental_native_aes_odd_lanes_matches_complete_replay():
    keys = [bytes([i])*16 for i in (1, 2, 3)]
    replay = NativeMaskReplay(keys, 19, max_invocations=2)
    groups = [_synthetic_replay_group(replay, "a", 3, 1, 4),
              _synthetic_replay_group(replay, "b", 2, 2, 6)]
    observed = {}
    for group in groups:
        for event in group:
            result = replay.ingest(event, 18)
            if result is not None:
                observed[event["binding"]] = result
    replay.finish()
    assert not replay.pending
    assert observed == native_mask_tapes([e for g in groups for e in g], keys, 19, {"a": 18, "b": 18})


@pytest.mark.parametrize("second_end", [6, 8, 12])
def test_native_pair_reservation_overlap_adjacent_and_gap(second_end):
    replay = NativeMaskReplay([bytes([i])*16 for i in (1, 2, 3)], 19, max_invocations=2)
    for event in _synthetic_replay_group(replay, "a", 3, 1, 4):
        replay.ingest(event, 18)
    second = _synthetic_replay_group(replay, "b", 3, 2, second_end)
    if second_end == 6:
        # Increasing endpoints, but [2,6) overlaps burned mask/output [0,4).
        with pytest.raises(OracleError, match="overlapping native PRSS pair reservation"):
            replay.ingest(second[0], 18)
        assert not replay.pending and len(replay.seen_bindings) == 1
    else:
        for event in second:  # Includes both same-binding helper routes.
            replay.ingest(event, 18)
        replay.finish()
        assert len(replay.seen_bindings) == 2


def _direct_test_aes(key, block):
    """Test-local FIPS-197 cipher; no replay/fixture/source_fill dependency.

    Literal standard S-box, byte key schedule, separate SubBytes/ShiftRows and
    explicit MixColumns equations. Anchored by published AES-128 vectors below.
    Not production crypto or a second native PRSS implementation.
    """
    box = bytes.fromhex(
        "637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0"
        "b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275"
        "09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf"
        "d0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2"
        "cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb"
        "e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08"
        "ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e"
        "e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16")
    def twice(n):
        return ((n << 1) ^ (0x11b if n & 128 else 0)) & 255
    expanded, rcon = bytearray(key), 1
    while len(expanded) < 176:
        word = list(expanded[-4:])
        if len(expanded) % 16 == 0:
            word = [box[n] for n in word[1:]+word[:1]]
            word[0] ^= rcon
            rcon = twice(rcon)
        for n in word:
            expanded.append(expanded[-16] ^ n)
    state = [a ^ b for a, b in zip(block, expanded[:16])]
    for round_number in range(1, 11):
        state = [box[n] for n in state]
        state = [state[4*((column+row) % 4)+row] for column in range(4) for row in range(4)]
        if round_number != 10:
            for at in range(0, 16, 4):
                a, b, c, d = state[at:at+4]
                state[at:at+4] = [twice(a)^twice(b)^b^c^d, a^twice(b)^twice(c)^c^d,
                                  a^b^twice(c)^twice(d)^d, twice(a)^a^b^c^twice(d)]
        state = [n ^ expanded[16*round_number+i] for i, n in enumerate(state)]
    return bytes(state)


def _direct_test_fill(key, cursor, lanes):
    initial = int.from_bytes(cursor.to_bytes(16, "little"), "big")
    result = []
    for within_fill in range((lanes+1)//2):
        cipher = _direct_test_aes(key, (initial+within_fill).to_bytes(16, "big"))
        result.extend(struct.unpack("<QQ", cipher))
    return result[:lanes]


def test_direct_cipher_expected_masks_has_published_anchors():
    assert _direct_test_aes(bytes(range(16)), bytes.fromhex("00112233445566778899aabbccddeeff")) == bytes.fromhex(
        "69c4e0d86a7b0430d8cdb78070b4c55a")
    assert _direct_test_aes(bytes(16), bytes(16)) == bytes.fromhex("66e94bd4ef8a2c3b884cfa59ca342b2e")
    assert _direct_test_aes(bytes(16), bytes(15)+b"\1") == bytes.fromhex("58e2fccefa7e3061367f1d57a4e7455a")


def _direct_test_routes(seed, binding, lanes, public_end, pair_end):
    a = _direct_test_fill(seed.to_bytes(16, "little"), public_end-1, 1)[0] % 3
    b, h = (a+1)%3, (a+2)%3
    events = [dict(binding=binding, role=role, to=to, phase=phase, lanes=lanes,
                   public_end=public_end, pair_end=pair_end, node=public_end, loop_offset=0,
                   tensor_step=None, invocation=None)
              for role, to, phase in ((a, b, 10), (b, a, 10), (h, a, 11), (h, b, 11))]
    return events, a, h


@pytest.mark.parametrize("mutation", ["none", "counter", "odd_half"])
def test_incremental_masks_independent_cipher_edges_odd_half_and_fill_boundary(monkeypatch, mutation):
    # Distinct keys; deliberately force a first-lane edge-sum carry regardless of
    # public pivot. Selection and ALL expected words use only direct cipher above.
    keys = []
    for i in range(256):
        key = bytes([i])*16
        if _direct_test_fill(key, 0, 1)[0] >= 1 << 63:
            keys.append(key)
            if len(keys) == 3:
                break
    assert len(set(keys)) == 3
    replay = NativeMaskReplay(keys, 19, max_invocations=2)
    first, a, h = _direct_test_routes(19, "first", 3, 1, 4)
    left, right = _direct_test_fill(keys[a], 0, 4), _direct_test_fill(keys[h], 0, 4)
    assert left[0]+right[0] >= 1 << 64
    expected_first = [(x+y) % (1 << 64) for x, y in zip(left[:3], right[:3])]
    for event in first:
        result = replay.ingest(event, 18)
    assert result["masks"] == expected_first

    second, a, h = _direct_test_routes(19, "second", 3, 2, 8)
    expected = [(x+y) % (1 << 64) for x, y in zip(
        _direct_test_fill(keys[a], 4, 3), _direct_test_fill(keys[h], 4, 3))]
    # Within-fill block increment is BE, next call resets from LE cursor. Lost
    # fourth mask half and the TWO output blocks cannot feed the next mask fill.
    same_fill_second = (int.from_bytes((4).to_bytes(16, "little"), "big")+1).to_bytes(16, "big")
    assert same_fill_second != (5).to_bytes(16, "little")
    assert _direct_test_aes(keys[a], same_fill_second) != _direct_test_aes(keys[a], (5).to_bytes(16, "little"))
    discarded = (_direct_test_fill(keys[a], 0, 4)[3]+_direct_test_fill(keys[h], 0, 4)[3]) % (1 << 64)
    assert expected[0] != discarded
    if mutation == "counter":
        second, _, _ = _direct_test_routes(19, "second", 3, 2, 10)  # Legal gap, wrong captured counter.
    elif mutation == "odd_half":
        original = replay.reference.source_fill
        def reuse_discarded_half(key, cursor, lanes):
            if cursor == 4 and lanes == 3:
                return tuple(original(key, 0, 4)[3:])+tuple(original(key, 4, 2))
            return original(key, cursor, lanes)
        monkeypatch.setattr(replay.reference, "source_fill", reuse_discarded_half)
    for event in second:
        result = replay.ingest(event, 18)
    replay.finish()
    if mutation == "none":
        assert result["masks"] == expected
    else:
        assert result["masks"] != expected  # Independent expected tape rejects mutation.


@pytest.mark.parametrize("cursor", [255, 256])
def test_independent_native_masks_across_counter_byte_boundary(cursor):
    keys, seed = [bytes(range(i, i+16)) for i in (0, 16, 32)], 19
    events, a, h = _direct_test_routes(seed, "boundary", 3, 1, cursor+4)
    # LE call reset crosses byte boundary; BE increment stays within that fill.
    encoded = cursor.to_bytes(16, "little")
    expected = []
    for offset in (0, 1):
        block = (int.from_bytes(encoded, "big")+offset).to_bytes(16, "big")
        x, y = (_direct_test_aes(keys[edge], block) for edge in (a, h))
        expected.extend((left+right) % (1 << 64) for left, right in
                        zip(struct.unpack("<QQ", x), struct.unpack("<QQ", y)))
    replay = NativeMaskReplay(keys, seed, max_invocations=1)
    for event in events:
        observed = replay.ingest(event, 18)
    replay.finish()
    assert observed["masks"] == expected[:3]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "cursor_reuse", "reverse", "owner"])
def test_incremental_native_replay_keeps_cross_group_checks(mutation):
    replay = NativeMaskReplay([bytes([i])*16 for i in (1, 2, 3)], 19, max_invocations=2)
    first = _synthetic_replay_group(replay, "a", 3, 1, 4)
    second = _synthetic_replay_group(replay, "b", 2, 2, 6)
    events = first+second
    if mutation == "missing":
        events.pop()
    elif mutation == "duplicate":
        events.insert(1, deepcopy(first[0]))
    elif mutation == "cursor_reuse":
        events = first+_synthetic_replay_group(replay, "b", 3, 1, 4)
    elif mutation == "reverse":
        events = second+first
    else:
        events[1] = dict(events[1], node=99)
    with pytest.raises(OracleError):
        for event in events:
            replay.ingest(event, 18)
        replay.finish()


def test_generated_provider_releases_operator_arrays_at_last_consumer():
    meta = {"spans": [{}], "demands": [
        {"kind": "T", "name": "a", "span": 0, "end": 1, "shape": [3], "shift": 18},
        {"kind": "T", "name": "b", "span": 0, "end": 1, "shape": [2], "shift": 18}]}
    client = GeneratedNativeStream(io.BytesIO(), io.BytesIO(), meta, None)
    first, last = array([1, 2, 3]), array([4, 5])
    refs = [weakref.ref(first), weakref.ref(last)]
    client.mapped[0] = {"a": {"shift": 18, "residues": first}, "b": {"shift": 18, "residues": last}}
    client.pending_words, client.span_words, client.frontier = 5, [5], 1
    del first, last
    provider = _StreamProvider(client, True)
    name, batch = next(provider)
    assert name == "a" and client.pending_words == 2 and refs[0]() is not None
    del batch
    gc.collect()
    assert refs[0]() is None and refs[1]() is not None
    name, batch = next(provider)
    assert name == "b" and not client.mapped and not client.pending_words and client.span_words == [0]
    del batch
    gc.collect()
    assert refs[1]() is None  # provider/client still alive; no yielded-array history


@pytest.mark.parametrize("limit", [2, 3])
def test_physical_ingestion_exact_and_one_short_pending_bound(limit):
    replay = NativeMaskReplay([bytes([i])*16 for i in (1, 2, 3)], 19, max_invocations=1)
    events = _synthetic_replay_group(replay, "00"*32, 3, 1, 4)
    meta = {"spans": [{"start": 0, "end": 10}], "demands": [
        {"kind": "T", "span": 0}], "pending_peak": limit, "tape_words": [3], "invocations": 1}
    client = GeneratedNativeStream(io.BytesIO(), io.BytesIO(), meta, replay)

    def ingest():
        for e in events:
            payload = (bytes.fromhex(e["binding"]) + struct.pack("<BBBBQQQQQQII",
                e["role"], e["to"], e["phase"], 0, e["node"], 0, (1 << 64)-1, (1 << 64)-1,
                e["public_end"], e["pair_end"], e["lanes"], 0))
            client._event(0, payload)

    if limit == 2:
        with pytest.raises(OracleError, match="bound"):
            ingest()
        assert not replay.pending and not client.pending and not client.pending_words
    else:
        ingest()
        client.census()
        assert client.pending_words == client.peak_words == 3 and not replay.pending


def _synthetic_terminal_client(acceptance=1, validity=1, *, missing=False, mutation="none"):
    """Pure state-machine fixture. Genuine mutation controls live in Rust selector."""
    meta = {"spans": [], "demands": [{"kind": "C", "name": "acceptance.valid", "shape": [],
            "aliases": [0], "end": 1}], "terminal": 0,
            "boundaries": [{"selection": 0, "id": "acceptance.valid", "shape": [],
                            "oracle_name": "acceptance.valid", "batch_dimension": False}],
            "policy": {"range_policy": "checked"}}
    final = struct.pack("<9Q", 1, 2, validity, 0, 0, 0, 0, 0, 0)
    incoming = _pipe_frame(9, 1, count=9, payload=final)
    if not missing:
        incoming += (_pipe_frame(8, 2, count=1, payload=struct.pack("<q", acceptance))
                     + _pipe_frame(11, 3) + _pipe_frame(12, 4))
    output = io.BytesIO()
    client = GeneratedNativeStream(io.BytesIO(incoming), output, meta, None, mutation=mutation)
    client.tape_eof = True
    comparator = StreamingTraceComparator(_StreamProvider(client, False),
        declaration=(("acceptance.valid", ((0, "acceptance.valid", (), False, "acceptance.valid"),)),),
        source_provenance=(("kind", "synthetic-state-control"),), checked_ranges=CheckedRangeSchedule(),
        outputs=2, max_records=1, max_total_values=1)
    return client, comparator, output


def test_checked_empty_pipe_final_precedes_retained_acceptance_and_success():
    client, comparator, output = _synthetic_terminal_client()
    components = client.final()
    assert client.state == "components" and client.at == 0
    comparator.finish(Result([1, 2], {}, True), final_components=components)
    assert client.checkpoint_eof and client.at == 1
    client.success()
    commands = [_PIPE_HEADER.unpack_from(output.getvalue(), at)[2] for at in range(0, len(output.getvalue()), 24)]
    assert commands == [34, 33, 36, 37] and client.state == "closed"


@pytest.mark.parametrize("mutation", ["acceptance", "validity", "missing", "pipe_after_final"])
def test_checked_empty_pipe_mutations_never_send_success(mutation):
    client, comparator, output = _synthetic_terminal_client(
        acceptance=0 if mutation == "acceptance" else 1,
        validity=2 if mutation == "validity" else 1,
        missing=mutation == "missing", mutation="pipe_after_final" if mutation == "pipe_after_final" else "none")
    with pytest.raises(OracleError):
        components = client.final()
        comparator.finish(Result([1, 2], {}, True), final_components=components)
    commands = [_PIPE_HEADER.unpack_from(output.getvalue(), at)[2] for at in range(0, len(output.getvalue()), 24)]
    assert 37 not in commands


def test_generated_final_refuses_other_missing_alias_without_sending_command():
    client, _, output = _synthetic_terminal_client()
    client.meta["terminal"] = None
    with pytest.raises(OracleError, match="nonterminal"):
        client.final()
    assert not output.getvalue()


class FixtureTape(Tape):
    """Test-data authoring ONLY. Replay always uses strict Tape, never this class.

    Named deterministic nonzero low residues provide fully explicit captured
    batches after authoring, without coupling to production PRSS or arithmetic.
    """

    def __init__(self, mode="mixed"):
        super().__init__({})
        self.mode = mode

    def truncate(self, name, raw, shift=18):
        raw = np.asarray(raw, dtype=object)
        if name in self.batches:
            raise OracleError("duplicate fixture operation")
        unit = 1 << shift
        start = sum((i+1)*ord(c) for i, c in enumerate(name))
        residues = [0 if self.mode == "zero" else unit-1 if self.mode == "ceil"
                    else (start + i*104729) % unit for i in range(raw.size)]
        self.batches[name] = {"shift": shift, "residues": np.array(residues, dtype=object).reshape(raw.shape)}
        return super().truncate(name, raw, shift)


def offline_module(relative):
    path = Path(__file__).resolve().parents[1] / "paper/reproduction-contracts" / relative
    spec = importlib.util.spec_from_file_location("independent_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("raw", [-GAP, -Q-1, -Q, -1, 0, 1, Q-1, Q, GAP-1])
@pytest.mark.parametrize("residue", [0, 1, Q//2, Q-1])
def test_signed_rounding_matches_independent_source_high_bit_formula(raw, residue):
    shared = offline_module("shared_profile_oracle.py")
    tape = Tape({"op": {"shift": 18, "residues": residue}})
    assert tape.truncate("op", raw).item() == shared.puma_trunc(raw, 64, 18, residue + (12345 << 18))
    tape.finish()


def test_rounding_exact_distribution_and_negative_floor():
    # Exhaust all low residues for a smaller shift, including negative values.
    for value in (-33, -16, -1, 0, 7, 15, 31):
        results = []
        for residue in range(16):
            results.append(Tape({"x": {"shift": 4, "residues": residue}}).truncate("x", value, 4).item())
        floor, rem = divmod(value, 16)
        assert results.count(floor+1) == rem
        assert set(results) <= {floor, floor+1}


def test_truncation_rejects_wrap_missing_excess_reuse_shape_and_noninteger():
    for raw in (GAP, -GAP-1, 1 << 64):
        with pytest.raises(OracleError, match="quarter-gap"):
            Tape({"x": {"shift": 18, "residues": 0}}).truncate("x", raw)
    with pytest.raises(OracleError, match="missing"):
        Tape({}).truncate("x", 0)
    tape = Tape({"x": {"shift": 18, "residues": 0}})
    with pytest.raises(OracleError, match="excess"):
        tape.finish()
    tape.truncate("x", 0)
    with pytest.raises(OracleError, match="repeated"):
        tape.truncate("x", 0)
    for residues in ([0], -1, Q):
        with pytest.raises(OracleError, match="shape/residue"):
            Tape({"x": {"shift": 18, "residues": residues}}).truncate("x", 0)
    with pytest.raises(OracleError, match="integer"):
        Tape({"x": {"shift": 18, "residues": 0}}).truncate("x", 0.5)


def test_explicit_physical_adapter_reorders_tiles_and_rejects_uncounted_draws():
    physical = {"tile.a": {"shift": 18, "masks": [2**64-1, 2**40+7]},
                "tile.b": {"shift": 18, "masks": [9]}}
    layout = {"dot": {"shift": 18, "shape": [3], "lanes": [["tile.b", 0], ["tile.a", 1], ["tile.a", 0]]}}
    tape = adapt_tapes(physical, layout)
    assert tape.batches["dot"]["residues"].tolist() == [9, 7, Q-1]
    tape.truncate("dot", array([0, 0, 0]))
    tape.finish()
    with pytest.raises(OracleError, match="excess"):
        adapt_tapes(physical, {})
    repeated = {"dot": {"shift": 18, "shape": [3], "lanes": [["tile.a", 0]]*3}}
    with pytest.raises(OracleError, match="reused"):
        adapt_tapes(physical, repeated)
    mismatch = {"dot": {"shift": 5, "shape": [3], "lanes": layout["dot"]["lanes"]}}
    with pytest.raises(OracleError, match="shift mismatch"):
        adapt_tapes(physical, mismatch)


def test_complete_dot_truncates_once_not_per_element_or_tile():
    # Individually both terms round down to zero; full contraction rounds to one.
    tape = Tape({"dot": {"shift": 18, "residues": [[0]]}})
    assert linear([[1, 1]], [[Q-1, Q-1]], tape, "dot").tolist() == [[1]]
    tape.finish()
    # A long contraction exercises exact object accumulation beyond native tiles.
    n = 2051
    tape = Tape({"dot": {"shift": 18, "residues": [[Q-1]]}})
    assert linear([[1]*n], [[Q-1]*n], tape, "dot").item() == n
    # Cancelling unwrapped dot is legal; a wrapped *final* dot is not.
    tape = Tape({"dot": {"shift": 18, "residues": [0]}})
    assert linear([1 << 32, 1 << 32], [[1 << 32, -(1 << 32)]], tape, "dot").item() == 0
    with pytest.raises(OracleError, match="quarter-gap"):
        linear([1 << 32], [[1 << 32]], FixtureTape(), "dot")


def test_source_conversion_binary32_toward_zero_and_range():
    assert source_f32(struct.pack("<4f", 0.1, -0.1, 1/Q, -1/Q)).tolist() == [26214, -26214, 1, -1]
    assert source_f32(struct.pack("<f", 1 << 22)).item() == 1 << 40
    for value in (float("nan"), float("inf"), 1e38, (1 << 22)+1):
        with pytest.raises(OracleError, match="source Q18 range"):
            source_f32(struct.pack("<f", value))
    with pytest.raises(OracleError):
        source_f32(b"123")


@pytest.mark.parametrize("value", [1, Q//4, Q//2, Q, 2*Q, 4*Q, (1 << 34)-1])
def test_hal_reciprocal_and_rsqrt_match_preexisting_independent_fixture(value):
    shared = offline_module("shared_profile_oracle.py")
    t = FixtureTape("zero")
    assert rsqrt(value, t, "rs").item() == shared.puma_rsqrt(value, repeat(0))
    assert reciprocal(value, t, "rec").item() == shared.puma_reciprocal(value, repeat(0))
    assert len([name for name in t.batches if name.startswith("rs/")]) == 6
    assert len([name for name in t.batches if name.startswith("rec/")]) == 6


def test_mixed_tapes_match_independent_scalar_hal_not_only_zero_rounding():
    shared = offline_module("shared_profile_oracle.py")
    for fn, reference in ((rsqrt, shared.puma_rsqrt), (reciprocal, shared.puma_reciprocal)):
        for value in (37, Q-1, Q+1, 9*Q+7):
            t = FixtureTape()
            actual = fn(value, t, "op").item()
            residues = [int(batch["residues"]) for batch in t.batches.values()]
            assert actual == reference(value, iter(residues))


@pytest.mark.parametrize("partitioned", [False, True])
def test_rmsnorm_epsilon_one_separate_rounding_and_public_partitions(partitioned):
    shared = offline_module("shared_profile_oracle.py")
    row, gamma = [Q//4, -Q//2, Q//8], [Q, Q//2, 2*Q]
    t = FixtureTape("zero")
    actual = rmsnorm(row, gamma, t, "norm", partitioned)
    assert actual.tolist() == shared.puma_rmsnorm(row, gamma, repeat(0))
    replay = Tape(t.batches)
    assert rmsnorm(row, gamma, replay, "norm", not partitioned).tolist() == actual.tolist()
    replay.finish()
    zero = rmsnorm([0, 0], [Q, Q], FixtureTape(), "zero", partitioned)
    assert zero.tolist() == [0, 0]


def test_rotary_separate_rounding_not_accumulated_product():
    # Each half sums two separately rounded signed products, not a joint dot.
    t = FixtureTape("zero")
    actual = rotary_half([1, 1], [Q//2], [Q//2], t, "rope")
    assert actual.tolist() == [-1, 0]  # combined-dot wrong result would [0,1]
    assert len(t.batches) == 2


def test_silu_guard_cutoffs_identity_zero_and_per_coefficient_truncation():
    xs = [-1000000*Q, -8*Q-1, -8*Q, -4*Q-1, -4*Q, 0, 4*Q, 4*Q+1, 1000000*Q]
    t = FixtureTape("zero")
    out = silu(xs, t, "act").tolist()
    assert out[0:2] == [0, 0]
    assert out[5] == 2229  # pinned approximation, NOT exact analytic SiLU(0)
    assert out[-2:] == xs[-2:]
    assert len(t.batches) == 9  # 3 powers, 2 low terms, 4 high terms
    # Independently expand floor equations at a nonintegral coefficient point.
    x = -4*Q-1
    x2 = x*x//Q
    assert out[3] == -80413 + (-21489*x//Q) + (-1453*x2//Q)
    replay = Tape(t.batches)
    assert silu(xs, replay, "act").tolist() == out
    replay.finish()


def test_softmax_strict_minus14_cutoff_guard_and_all_draws():
    t = FixtureTape("ceil")
    values = softmax([0, -14*Q, -14*Q+1, -1000000*Q], t, "sm")
    assert values[0] > 0
    assert values[1] == values[3] == 0
    assert values[2] == 1  # ceil tape distinguishes strict cutoff by one Q18 unit
    assert t.batches["sm/reduce"]["shift"] == 5
    assert len(t.batches) == 13
    assert all(b["residues"].size == (1 if "/reciprocal/" in name else 4)
               for name, b in t.batches.items())
    replay = Tape(t.batches)
    assert softmax([0, -14*Q, -14*Q+1, -1000000*Q], replay, "sm").tolist() == values.tolist()
    replay.finish()


def test_streamed_public_weight_blocks_keep_one_complete_contraction_round():
    reads = []
    shape = (10000, 257)  # source exceeds per-array oracle bound

    def reader(start, stop):
        reads.append((start, stop))
        return np.array([[Q-1]*257 for _ in range(stop-start)], dtype=object)

    weight = PublicMatrix(shape, reader)
    t = FixtureTape("zero")
    out = linear([[1]*257], weight, t, "stream")
    assert out.shape == (1, 10000)
    assert set(out.flat) == {256}
    assert len(t.batches) == 1
    assert len(reads) > 1 and all((b-a)*shape[1] <= 65536 for a, b in reads)
    assert reads[0][0] == 0 and reads[-1][1] == shape[0]
    with pytest.raises(OracleError, match="reader shape"):
        PublicMatrix((2, 3), lambda a, b: [[0]]).rows(0, 1)
    with pytest.raises(OracleError, match="bounded public"):
        weight.rows(0, 129)


def generated_fixture(family, layers=2, *, wide=False, head_dim=18):
    """Nonconstant source-derived-compatible geometry, no production imports."""
    hidden, mid, heads, kv, dim = (72, 131, 4, 2, 18) if wide else (4, 6, 2, 1, 2)
    if wide:
        dim, hidden = head_dim, 4*head_dim
    cfg = dict(model_type=family, hidden_size=hidden, intermediate_size=mid,
               num_hidden_layers=layers, num_attention_heads=heads,
               num_key_value_heads=kv, head_dim=dim, vocab_size=5,
               attention_bias=family == "qwen2", hidden_act="silu",
               tie_word_embeddings=False)
    weights = {}

    def put(name, shape, salt=0):
        count = int(np.prod(shape))
        weights[name] = np.array([((i*7+salt)%13-6)*(Q//(1024 if wide else 128)) for i in range(count)], dtype=object).reshape(shape)

    weights["model.embed_tokens.weight"] = array([
        [Q//2, Q//4, -Q//8, Q//16], [-Q//3, Q//5, Q//7, -Q//9],
        [Q//6, -Q//4, Q//3, Q//8], [Q//8, Q//9, -Q//7, Q//6],
        [-Q//5, -Q//6, Q//7, Q//8]])
    if wide:
        weights["model.embed_tokens.weight"] = array([
            [(row*11+col*3)%17*(Q//128)-Q//16 for col in range(hidden)] for row in range(5)])
        weights["model.embed_tokens.weight"][0, 0] = Q//4
        weights["model.embed_tokens.weight"][1, 0] = -Q//4
    weights["model.norm.weight"] = array([Q]*hidden)
    put("lm_head.weight", (5, hidden), 4)
    for layer in range(layers):
        src = f"model.layers.{layer}."
        weights[src + "input_layernorm.weight"] = array([Q]*hidden)
        weights[src + "post_attention_layernorm.weight"] = array([Q]*hidden)
        for kind, count in (("q", heads), ("k", kv), ("v", kv)):
            put(src + f"self_attn.{kind}_proj.weight", (count*dim, hidden), ord(kind)+layer)
            if family == "qwen2":
                put(src + f"self_attn.{kind}_proj.bias", (count*dim,), ord(kind))
        put(src + "self_attn.o_proj.weight", (hidden, heads*dim), layer)
        for kind in ("gate", "up"):
            put(src + f"mlp.{kind}_proj.weight", (mid, hidden), len(kind)+layer)
        put(src + "mlp.down_proj.weight", (hidden, mid), layer+2)
        if family == "qwen3":
            weights[src + "self_attn.q_norm.weight"] = array([Q]*dim)
            weights[src + "self_attn.k_norm.weight"] = array([Q]*dim)
    # Frozen public rows for width=2 (frequency=1); no host libm at replay.
    rotary = {0: ([Q], [0]), 1: ([141637], [220586]),
              2: ([-109090], [238366]), 3: ([-259520], [36994])}
    if wide:
        rotary = {position: (cos*(dim//2), sin*(dim//2)) for position, (cos, sin) in rotary.items()}
    return cfg, weights, rotary


def _synthetic_final(result, checked):
    """Hand-authored additive components, NOT native final-response evidence."""
    words = list(result.tokens) if result.accepted else [0, 0]
    if checked:
        words.append(int(result.accepted))
    return [[int(word) % (1 << 64) for word in words], [0]*len(words), [0]*len(words)]


def _stream_fixture(family="qwen3", profile="ordinary", *, wide=False, late=False):
    """Small synthetic comparator controls; no source/native capture claim.

    Freeze declaration from pristine generated geometry BEFORE any mutations.
    All payload materialization lives in this test fixture, never comparator.
    """
    cfg, weights, rotary = generated_fixture(family, wide=wide)
    placements = guard_placements(cfg, 2)
    if late:
        placements["s1/final_norm/input"] = RangeCheck(0, 0)
    schedule = (None if profile == "ordinary" else CheckedRangeSchedule() if profile == "empty"
                else CheckedRangeSchedule.from_mapping(placements))
    authored = FixtureTape()
    expected = run_decoder(cfg, weights, [0, 1], 2, authored, rotary, checked_ranges=schedule)
    trace = dict(expected.trace)
    if profile == "empty":
        # Expected-side conjunction; generated actual record below is independent
        # synthetic evidence that must still be consumed, never inserted by sink.
        trace["acceptance.valid"] = array(1)
    aliases = {}
    for selection in range(2):
        for native_id, semantic in _ROOT_ALIASES.items():
            aliases.setdefault(f"s{selection}/{semantic}", []).append((selection, native_id))
        for layer in range(cfg["num_hidden_layers"]):
            for native_id, semantic in _LAYER_ALIASES.items():
                aliases.setdefault(f"s{selection}/layer.{layer}/{semantic}", []).append(
                    (selection, f"layer.{layer}.{native_id}"))
    declaration, records = [], []
    for name, value in trace.items():
        # Public synthetic producer declares global acceptance at selection ZERO;
        # do not infer a convention from accepted bits or last selection's IDs.
        specs = []
        identities = aliases.get(name, [(0 if name == "acceptance.valid" else int(name[1]),
                                         "diagnostic:" + name)])
        for selection, native_id in identities:
            diagnostic = name if name not in aliases else None
            batch = len(records) % 2 == 0
            shape = ((1,) if batch else ()) + value.shape
            if native_id == "token_feedback":
                batch, shape = True, (1, 1)
            spec = (selection, native_id, shape, batch, diagnostic)
            specs.append(spec)
            record = dict(selection=selection, id=native_id, shape=list(shape), batch_dimension=batch)
            if diagnostic is not None:
                record["oracle_name"] = diagnostic
            words = [int(word) for word in value.flat]
            if len(records) % 3 == 0:
                record["components"] = [[word % (1 << 64) for word in words], [0]*len(words), [0]*len(words)]
            else:
                record["values"] = words
            records.append(record)
        declaration.append((name, tuple(specs)))
    provenance = (("kind", "generated"), ("revision", "synthetic-fixture-v1"), ("family", family),
                  ("source_identities", tuple((name, value.shape, "synthetic-q18-integers")
                                              for name, value in weights.items())),
                  ("tied_head", False), ("config", tuple(cfg.items())),
                  ("numeric", "synthetic-integer-stochastic-tape"),
                  ("rotary", tuple((position, (tuple(cos), tuple(sin)))
                                   for position, (cos, sin) in rotary.items())),
                  ("schedule", tuple((name, (rule.lower, rule.upper, rule.fallback))
                                     for name, rule in schedule.entries) if schedule else None),
                  ("counts", (2, 2, len(records))))
    return dict(cfg=cfg, weights=weights, rotary=rotary, authored=authored, expected=expected,
                schedule=schedule, declaration=tuple(declaration), records=records,
                provenance=provenance, final=_synthetic_final(expected, schedule is not None))


def _comparator(fixture, *, records=None, **changes):
    declaration = fixture["declaration"]
    options = dict(declaration=declaration, source_provenance=fixture["provenance"],
                   checked_ranges=fixture["schedule"], outputs=2,
                   max_records=sum(len(specs) for _, specs in declaration),
                   max_total_values=sum(int(np.prod(spec[2])) for _, specs in declaration for spec in specs))
    options.update(changes)
    return StreamingTraceComparator(iter(fixture["records"] if records is None else records), **options)


def _stream_replay(fixture, comparator, *, batches=None):
    tape = StreamingTape(iter(fixture["authored"].batches.items() if batches is None else batches))
    result = run_decoder(fixture["cfg"], fixture["weights"], [0, 1], 2, tape,
                         fixture["rotary"], checked_ranges=fixture["schedule"], trace_sink=comparator)
    return result, tape


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("profile", ["ordinary", "empty", "checked"])
def test_streaming_comparator_generated_parity(family, profile):
    fixture = _stream_fixture(family, profile)
    comparator = _comparator(fixture)
    assert comparator.declaration is fixture["declaration"]
    assert comparator.source_provenance is fixture["provenance"]
    result, tape = _stream_replay(fixture, comparator)
    assert result.trace == {} and result.tokens == fixture["expected"].tokens
    assert result.accepted == fixture["expected"].accepted
    assert comparator.finish(result, final_components=fixture["final"]) is None
    assert comparator.consumed_records == len(fixture["records"])
    assert comparator.consumed_values == comparator.declared_values
    assert comparator.peak_record_values <= comparator.peak_record_words <= 4*comparator.peak_record_values
    assert tape.batches == {} and tape.used == set(fixture["authored"].batches)
    assert comparator._checkpoints is None
    assert not any(isinstance(value, np.ndarray) for value in vars(comparator).values())
    with pytest.raises(OracleError, match="terminal"):
        comparator("s0/input.tokens", array([0, 1]))
    with pytest.raises(OracleError, match="terminal"):
        comparator.finish(result, final_components=fixture["final"])


@pytest.mark.parametrize("fail_callback", [False, True], ids=["finish", "callback-failure"])
def test_streaming_comparator_releases_callback_arrays_while_live(monkeypatch, fail_callback):
    import replicated_decoder_oracle as oracle

    fixture = _stream_fixture()
    actual_refs, expected_refs = [], []
    original = oracle._native_checkpoint_value

    def observe_value(checkpoint):
        actual = original(checkpoint)
        # Observe actual helper output AND ndarray backing views, without keeping
        # any payload alive in the observer. Fixture records contain flat lists,
        # not these newly reconstructed arrays or their NumPy backing storage.
        backing = actual
        while isinstance(backing, np.ndarray):
            actual_refs.append(weakref.ref(backing))
            backing = backing.base
        return actual

    monkeypatch.setattr(oracle, "_native_checkpoint_value", observe_value)
    comparator = _comparator(fixture)

    def send(name, *, corrupt=False):
        # Fresh callback-owned allocation: fixture.trace and the provider never
        # own it. Retaining fixture arrays would not establish a comparator leak.
        expected = fixture["expected"].trace[name].copy()
        expected_refs.append(weakref.ref(expected))
        if corrupt:
            expected.flat[0] += 1
            with pytest.raises(OracleError, match="trace mismatch"):
                comparator(name, expected)
        else:
            comparator(name, expected)
        # Return no payload or exception/traceback. Check only AFTER this caller
        # frame unwinds, including pytest's temporary ExceptionInfo on failure.

    def assert_released():
        gc.collect()
        assert actual_refs and expected_refs
        assert all(reference() is None for reference in actual_refs)
        assert all(reference() is None for reference in expected_refs)

    for name, _ in fixture["declaration"]:
        corrupt = fail_callback and name == "s0/logits"
        send(name, corrupt=corrupt)
        assert_released()
        if corrupt:
            assert comparator._state == "failed" and comparator._checkpoints is None
            break
        # Even with a live provider and comparator, every completed callback's
        # reconstructed and expected arrays must already be gone.
        assert comparator._state == "active" and comparator._checkpoints is not None
    if fail_callback:
        with pytest.raises(OracleError, match="terminal"):
            comparator.finish(fixture["expected"], final_components=fixture["final"])
        with pytest.raises(OracleError, match="terminal"):
            comparator("s0/input.tokens", array([0, 1]))
    else:
        comparator.finish(fixture["expected"], final_components=fixture["final"])
        assert comparator._state == "finished" and comparator._checkpoints is None
    assert_released()


@pytest.mark.parametrize("error_type", [RuntimeError, BaseException], ids=["runtime-error", "base-exception"])
@pytest.mark.parametrize("phase", ["callback", "finish-exhaustion"])
def test_streaming_comparator_iterator_exception_is_terminal(error_type, phase):
    fixture = _stream_fixture()
    failure = error_type("synthetic checkpoint provider failure")
    # Fail inside a multi-alias callback AFTER its first alias was compared.
    before_selected = 0
    for name, specs in fixture["declaration"]:
        if name == "s0/selected":
            assert len(specs) == 2
            break
        before_selected += len(specs)
    fail_after = before_selected+1 if phase == "callback" else len(fixture["records"])

    class RaisingProvider:
        def __init__(self):
            self.provider = iter(fixture["records"])
            self.yielded = self.calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.yielded == fail_after:
                raise failure
            checkpoint = next(self.provider)
            self.yielded += 1
            return checkpoint

    provider = RaisingProvider()
    # Public schema errors still fail BEFORE private iteration. This provider
    # must not get a chance to mask malformed public metadata with its exception.
    for invalid in (dict(max_records=True), dict(declaration=[])):
        with pytest.raises(OracleError):
            _comparator(fixture, records=provider, **invalid)
        assert provider.calls == provider.yielded == 0
    comparator = _comparator(fixture, records=provider)
    with pytest.raises(error_type) as caught:
        result, _ = _stream_replay(fixture, comparator)
        assert phase == "finish-exhaustion"
        assert comparator._state == "active"
        comparator.finish(result, final_components=fixture["final"])
    assert caught.value is failure  # Original exception, not wrapping/replacement.
    assert comparator._checkpoints is None and comparator._state == "failed"
    assert comparator.consumed_records == provider.yielded == fail_after
    assert provider.calls == fail_after+1
    calls = provider.calls
    with pytest.raises(OracleError, match="terminal"):
        comparator("s0/input.tokens", array([0, 1]))
    with pytest.raises(OracleError, match="terminal"):
        comparator.finish(fixture["expected"], final_components=fixture["final"])
    assert provider.calls == calls  # Terminal calls cannot resume/drain provider.


def test_streaming_comparator_guard_tiles_and_feedback_scalar_shapes():
    fixture = _stream_fixture(profile="checked", wide=True)
    assert fixture["expected"].trace[
        "s0/layer.0/input_norm/input/checked_range/all_valid"].shape == (2,)
    small = _stream_fixture(profile="checked")
    assert small["expected"].trace[
        "s0/layer.0/input_norm/input/checked_range/all_valid"].shape == ()
    for current in (fixture, small):
        feedback = next(record for record in current["records"] if record["id"] == "token_feedback")
        assert feedback["shape"] == [1, 1] and _native_checkpoint_value(feedback).shape == ()
        comparator = _comparator(current)
        result, _ = _stream_replay(current, comparator)
        comparator.finish(result, final_components=current["final"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "omit_alias", "swap"])
def test_streaming_comparator_missing_duplicate_boundary_and_order(mutation):
    fixture = _stream_fixture()
    records = deepcopy(fixture["records"])
    index = next(i for i, record in enumerate(records) if record["id"] == "layer.1.q_linear")
    if mutation == "missing":
        records.pop(index)
    elif mutation == "duplicate":
        records.insert(index, deepcopy(records[index]))
    elif mutation == "omit_alias":
        records.pop(next(i for i, record in enumerate(records) if record["id"] == "layer.0.key_view"))
    else:
        records[index], records[index+1] = records[index+1], records[index]
    comparator = _comparator(fixture, records=records)
    with pytest.raises(OracleError, match="identity/order"):
        _stream_replay(fixture, comparator)
    assert comparator._checkpoints is None
    with pytest.raises(OracleError, match="terminal"):
        comparator.finish(fixture["expected"], final_components=fixture["final"])


@pytest.mark.parametrize("word", [2, 4])
def test_streaming_comparator_logit_mutation(word):
    fixture = _stream_fixture()
    records = deepcopy(fixture["records"])
    record = next(record for record in records if record["id"] == "output_head")
    values = [int(value) for value in _native_checkpoint_value(record).flat]
    values[word] += 1
    record.pop("components", None)
    record["values"] = values
    with pytest.raises(OracleError, match="trace mismatch: s0/logits"):
        _stream_replay(fixture, _comparator(fixture, records=records))


@pytest.mark.parametrize("word", [0, -1])
def test_streaming_comparator_feedback_id_mutation(word):
    fixture = _stream_fixture()
    records = deepcopy(fixture["records"])
    record = next(record for record in records if record["id"] == "feedback.history")
    values = [int(value) for value in _native_checkpoint_value(record).flat]
    values[word] = (values[word]+1) % 5
    record.pop("components", None)
    record["values"] = values
    with pytest.raises(OracleError, match="trace mismatch: s0/feedback"):
        _stream_replay(fixture, _comparator(fixture, records=records))


@pytest.mark.parametrize("profile", ["empty", "checked"])
@pytest.mark.parametrize("mutation", ["checkpoint_bit", "final_bit", "missing_checkpoint", "missing_final"])
def test_streaming_comparator_final_validity(profile, mutation):
    fixture = _stream_fixture(profile=profile)
    records, final = deepcopy(fixture["records"]), deepcopy(fixture["final"])
    if mutation == "checkpoint_bit":
        records[-1].pop("components", None)
        records[-1]["values"] = [0]
    elif mutation == "final_bit":
        final[0][-1] = 0
    elif mutation == "missing_checkpoint":
        records.pop()
    else:
        final = None
    comparator = _comparator(fixture, records=records)
    with pytest.raises(OracleError):
        result, _ = _stream_replay(fixture, comparator)
        comparator.finish(result, final_components=final)
    with pytest.raises(OracleError, match="terminal"):
        comparator.finish(fixture["expected"], final_components=fixture["final"])


@pytest.mark.parametrize("mutation", ["extra", "missing_terminal", "early_finish"])
def test_streaming_comparator_extra_event_and_finality(mutation):
    fixture = _stream_fixture()
    records = deepcopy(fixture["records"])
    if mutation == "extra":
        records.append(deepcopy(records[-1]))
    elif mutation == "missing_terminal":
        records.pop()
    comparator = _comparator(fixture, records=records)
    with pytest.raises(OracleError, match="excess|missing|unfinished"):
        result = fixture["expected"]
        if mutation != "early_finish":
            result, _ = _stream_replay(fixture, comparator)
        comparator.finish(result, final_components=fixture["final"])
    with pytest.raises(OracleError, match="terminal"):
        comparator("s0/input.tokens", array([0, 1]))


@pytest.mark.parametrize("mutation", ["missing", "extra", "repeat", "swap", "shift", "shape", "range", "dtype"])
def test_streaming_comparator_tape_exhaustion(mutation):
    fixture = _stream_fixture()
    batches = deepcopy(list(fixture["authored"].batches.items()))
    if mutation == "missing":
        batches.pop()
    elif mutation == "extra":
        batches.append(("extra", deepcopy(batches[-1][1])))
    elif mutation == "repeat":
        batches.insert(1, deepcopy(batches[0]))
    elif mutation == "swap":
        batches[0], batches[1] = batches[1], batches[0]
    elif mutation == "shift":
        batches[0][1]["shift"] += 1
    elif mutation == "shape":
        batches[0][1]["residues"] = array([0])
    elif mutation == "range":
        batches[0][1]["residues"].flat[0] = Q
    else:
        batches[0][1]["residues"].flat[0] = 0.5
    comparator = _comparator(fixture)
    with pytest.raises(OracleError):
        _stream_replay(fixture, comparator, batches=batches)
    # Tape failure may precede a callback, or extra tape may fail after the last
    # callback. No successful decoder return exists, so do NOT call sink.finish.


def test_streaming_comparator_late_rejection():
    fixture = _stream_fixture(profile="checked", late=True)
    assert not fixture["expected"].accepted and fixture["expected"].tokens == []
    assert "s1/logits" in fixture["expected"].trace and "s1/feedback" in fixture["expected"].trace
    assert fixture["expected"].trace["acceptance.valid"].item() == 0
    for corrupt in (False, True):
        comparator = _comparator(fixture)
        result, tape = _stream_replay(fixture, comparator)
        assert tape.used == set(fixture["authored"].batches)
        final = deepcopy(fixture["final"])
        assert final[0] == [0, 0, 0]
        if corrupt:
            final[0][0] = 1
            with pytest.raises(OracleError, match="masking"):
                comparator.finish(result, final_components=final)
        else:
            comparator.finish(result, final_components=final)


def test_streaming_comparator_bounds_and_order(monkeypatch):
    import replicated_decoder_oracle as oracle

    fixture = _stream_fixture()
    comparator = _comparator(fixture)
    monkeypatch.setattr(oracle, "MAX_TRACE_VALUES", 1)
    with pytest.raises(OracleError, match="bounded trace"):
        run_decoder(fixture["cfg"], fixture["weights"], [0, 1], 2,
                    Tape(fixture["authored"].batches), fixture["rotary"])
    result, _ = _stream_replay(fixture, comparator)
    comparator.finish(result, final_components=fixture["final"])
    assert comparator.consumed_values > oracle.MAX_TRACE_VALUES
    for change in (dict(max_records=comparator.declared_records-1),
                   dict(max_total_values=comparator.declared_values-1)):
        with pytest.raises(OracleError, match="budget"):
            _comparator(fixture, **change)
    # Per-array admission stays distinct from cumulative stream admission.
    declaration = (("s0/logits", ((0, "output_head", (oracle.MAX_VALUES+1,), False, None),)),)
    with pytest.raises(OracleError, match="bounded shape"):
        _comparator(fixture, declaration=declaration, max_total_values=10**9)
    wrong_order = _comparator(fixture)
    with pytest.raises(OracleError, match="order"):
        wrong_order("s0/embedding", fixture["expected"].trace["s0/embedding"])
    # Physical row-first ordering is not a semantic stream. No reorder queue.
    records = deepcopy(fixture["records"])
    q = next(i for i, record in enumerate(records) if record["id"] == "layer.0.q_linear")
    k = next(i for i, record in enumerate(records) if record["id"] == "layer.0.k_linear")
    records.insert(q+1, records.pop(k))
    with pytest.raises(OracleError, match="identity/order"):
        _stream_replay(fixture, _comparator(fixture, records=records))


@pytest.mark.parametrize("change", [
    dict(max_records=True), dict(max_total_values=False), dict(outputs=True),
    dict(declaration=[]), dict(source_provenance=[]),
    dict(source_provenance=(("kind", {"mutable": 1}),)),
    dict(source_provenance=(("kind", lambda: None),)),
    dict(source_provenance=(("kind", "generated"), ("kind", "duplicate"))),
])
def test_streaming_comparator_immutable_and_budget_schema(change):
    fixture = _stream_fixture()
    with pytest.raises(OracleError):
        _comparator(fixture, **change)


@pytest.mark.parametrize("mutation", ["selection_bool", "selection_oob", "layer", "shape_bool", "shape_zero",
                                      "batch", "dtype", "flat", "schema", "diagnostic", "both"])
def test_streaming_comparator_record_schema(mutation):
    fixture = _stream_fixture()
    records = deepcopy(fixture["records"])
    record = records[0]
    if mutation == "selection_bool":
        record["selection"] = False
    elif mutation == "selection_oob":
        record["selection"] = 2
    elif mutation == "layer":
        record["id"] = "layer.-1.q_linear"
    elif mutation == "shape_bool":
        record["shape"][0] = True
    elif mutation == "shape_zero":
        record["shape"][0] = 0
    elif mutation == "batch":
        record["batch_dimension"] = 1
    elif mutation == "dtype":
        record.pop("components", None)
        record["values"] = [False, 1]
    elif mutation == "flat":
        record.pop("components", None)
        record["values"] = [[0, 1]]
    elif mutation == "schema":
        record["unexpected"] = 1
    elif mutation == "diagnostic":
        record["oracle_name"] = "s0/invented/input"
    else:
        record["values"] = [0, 1]
    with pytest.raises(OracleError):
        _stream_replay(fixture, _comparator(fixture, records=records))


def test_streaming_comparator_declaration_negatives_precede_iteration():
    fixture = _stream_fixture(profile="checked")

    def forbidden_iteration():
        raise AssertionError("invalid public metadata consumed private stream")
        yield

    name, specs = fixture["declaration"][0]
    mutations = [((name, specs), (name, specs)),
                 ((name, (specs[0], specs[0])),),
                 ((name, ((False, *specs[0][1:]),)),),
                 ((name, ((2, *specs[0][1:]),)),),
                 ((name, ((0, "unsupported", (2,), False, None),)),),
                 ((name, ((0, "input.tokens", [2], False, None),)),),
                 (("s0/invented/input", ((0, "guard", (2,), False, "s0/invented/input"),)),)]
    for declaration in mutations:
        with pytest.raises(OracleError):
            _comparator(fixture, records=forbidden_iteration(), declaration=declaration)


def test_native_final_exact_shape_and_result_geometry():
    fixture = _stream_fixture(profile="checked")
    result = deepcopy(fixture["expected"])
    with pytest.raises(OracleError, match="flat|masking"):
        _check_native_final(result, 2, True, [[[word for word in part]] for part in fixture["final"]])
    result.tokens.pop()
    with pytest.raises(OracleError, match="geometry"):
        _check_native_final(result, 2, True, fixture["final"])
    assert _native_checkpoint_name(0, "token_feedback") == "s0/selected"


def test_streaming_comparator_global_acceptance_uses_public_producer_convention():
    fixture = _stream_fixture(profile="empty")
    name, specs = fixture["declaration"][-1]
    # Alternate SYNTHETIC producer convention: global acceptance assigned to s1.
    # Public declaration is frozen before private replay; values do not choose it.
    spec = (1, *specs[0][1:])
    fixture["declaration"] = (*fixture["declaration"][:-1], (name, (spec,)))
    fixture["records"][-1]["selection"] = 1
    comparator = _comparator(fixture)
    result, _ = _stream_replay(fixture, comparator)
    comparator.finish(result, final_components=fixture["final"])
    records = deepcopy(fixture["records"])
    records[-1]["selection"] = 0
    comparator = _comparator(fixture, records=records)
    result, _ = _stream_replay(fixture, comparator)
    with pytest.raises(OracleError, match="identity/order"):
        comparator.finish(result, final_components=fixture["final"])


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_wide_multilayer_gqa_has_distinct_groups_and_rejects_wrong_value_group(family):
    cfg, weights, rotary = generated_fixture(family, wide=True)
    authored = FixtureTape()
    expected = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    _assert_fixture_gqa_coverage(cfg, expected.trace, 2, authored.batches)
    replay = run_decoder(cfg, weights, [0, 1], 2, Tape(authored.batches), rotary)
    compare_trace(expected.trace, replay.trace)
    for selection in range(2):
        for layer in range(2):
            p = f"s{selection}/layer.{layer}/"
            assert expected.trace[p+"full_prefix_k"].shape == (2, 2+selection, 18)
            values = expected.trace[p+"full_prefix_v"]
            assert not np.array_equal(values[0], values[1]), "periodic fixture must not hide a GQA routing error"
            name = p+"context/h2/r1"
            wrong = linear(expected.trace[p+"probabilities"][2, 1, :2], values[0, :2].T,
                           Tape({name: authored.batches[name]}), name)
            assert not np.array_equal(wrong, expected.trace[p+"context"][2, 1])
            assert np.all(expected.trace[p+"probabilities"][:, 0, 1:] == 0)
            assert expected.trace[p+"gate_up"].shape == (2+selection, 131)
    assert expected.trace["s1/logits"].shape == (5,)
    degenerate = dict(expected.trace)
    repeated = expected.trace["s0/layer.0/full_prefix_v"].copy()
    repeated[1] = repeated[0]
    degenerate["s0/layer.0/full_prefix_v"] = repeated
    with pytest.raises(OracleError, match="degenerate"):
        _assert_fixture_gqa_coverage(cfg, degenerate, 2)


@pytest.mark.parametrize("head_dim", [18, 32])
def test_complete_and_tail_guard_tiles_preserve_values_and_sticky_late_failure(head_dim):
    cfg, weights, rotary = generated_fixture("qwen3", wide=True, head_dim=head_dim)
    hidden = cfg["hidden_size"]
    weights["lm_head.weight"] = np.zeros((5, hidden), dtype=object)
    weights["lm_head.weight"][0, 0], weights["lm_head.weight"][2, 0] = Q//32, -Q//32
    names = [f"s{s}/layer.{layer}/{kind}/input" for s in range(2)
             for layer in range(2) for kind in ("input_norm", "post_attention_norm", "q_norm", "k_norm")]
    names += [f"s{s}/final_norm/input" for s in range(2)]
    schedule = CheckedRangeSchedule.from_mapping({name: RangeCheck() for name in names})
    authored = FixtureTape()
    ordinary = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    accepted = run_decoder(cfg, weights, [0, 1], 2, Tape(authored.batches), rotary, checked_ranges=schedule)
    assert ordinary.tokens[0] == accepted.tokens[0] == 2
    compare_trace(ordinary.trace, {name: accepted.trace[name] for name in ordinary.trace})
    first = accepted.trace["s0/layer.0/input_norm/input/checked_range/all_valid"]
    assert first.tolist() == [1, 1]  # 144 lanes with tail, or 256 complete lanes
    weights["model.embed_tokens.weight"][2, 0] = CHECKED_RANGE_LIMIT+1
    failed_tape = FixtureTape()
    rejected = run_decoder(cfg, weights, [0, 1], 2, failed_tape, rotary, checked_ranges=schedule)
    assert not rejected.accepted and rejected.tokens == []
    assert rejected.trace["s0/selected"].item() == 2
    at = "s1/layer.0/input_norm/input/checked_range/"
    assert rejected.trace[at+"all_valid"].tolist() == ([1, 0] if head_dim == 18 else [1, 1, 0])
    assert rejected.trace[at+"tile_sticky"].tolist() == rejected.trace[at+"all_valid"].tolist()
    assert rejected.trace["s1/layer.1/q_norm/input/checked_range/sticky"].item() == 0
    strict = run_decoder(cfg, weights, [0, 1], 2, Tape(failed_tape.batches), rotary, checked_ranges=schedule)
    compare_trace(rejected.trace, strict.trace)
    for name in (at+"valid", at+"all_valid", at+"tile_sticky"):
        incomplete = dict(strict.trace)
        incomplete.pop(name)
        with pytest.raises(OracleError, match="missing"):
            compare_trace(rejected.trace, incomplete)


def _public_native_events(stage_lanes, *, node=10, offset=0):
    """Synthetic PUBLIC schedule controls, not native execution evidence."""
    return [dict(binding=f"b{offset+i}", node=node, lanes=lanes, loop_offset=0,
                 invocation=0, tensor_step=i) for i, lanes in enumerate(stage_lanes)]


def test_public_map_multilayer_gqa_norm_rows_are_exact_not_just_same_total():
    cfg, _, _ = generated_fixture("qwen3", wide=True)
    stages = [18, 1, 1, 1, 1, 1, 1, 1, 18, 18]
    events = _public_native_events(stages*8)
    spans = [dict(selection=0, id="layer.1.q_norm", start=10, end=11)]
    layout = generated_native_layout(cfg, 2, 2, events, spans)
    assert layout["s0/layer.1/q_norm/normalize"]["shape"] == [4, 2, 18]
    assert layout["s0/layer.1/q_norm/mean"]["shape"] == [4, 2]
    # Keep identical total vector lanes while corrupting individual row widths.
    broken = [dict(event) for event in events]
    broken[0]["lanes"], broken[10]["lanes"] = 17, 19
    with pytest.raises(OracleError, match="row/tile widths"):
        generated_native_layout(cfg, 2, 2, broken, spans)
    with pytest.raises(OracleError, match="overlapping"):
        generated_native_layout(cfg, 2, 2, events, spans*2)


def test_public_map_full_prefix_private_contractions_retain_every_actual_tail():
    cfg, _, _ = generated_fixture("qwen3", wide=True)
    # Actual compiler matrix rule: min(MAX_LANES/inner, 128) output lanes.
    # At visible key count 71, 18-wide context contracts in 14+4 lane tiles.
    events, count = [], 0
    for head in range(4):
        for row in range(71):
            tile = min(1024//(row+1), 128)
            for start in range(0, 18, tile):
                events.extend(_public_native_events([min(tile, 18-start)], node=10+count, offset=count))
                count += 1
    spans = [dict(selection=7, id="layer.1.attention_values", start=10, end=10+count)]
    layout = generated_native_layout(cfg, 64, 8, events, spans)
    tail = layout["s7/layer.1/context/h3/r70"]
    assert tail["shape"] == [18]
    assert len({binding for binding, _ in tail["lanes"]}) == 2
    assert sum(len(spec["lanes"]) for spec in layout.values()) == 4*71*18
    with pytest.raises(OracleError, match="tile crossing"):
        generated_native_layout(cfg, 64, 8, events[:-1], spans)


def test_public_map_activation_tails_account_for_each_term_and_missing_excess_draw():
    cfg, _, _ = generated_fixture("llama", wide=True)
    events = []
    for start in range(0, 71*131, 128):
        chunk = _public_native_events([min(128, 71*131-start)]*9, offset=len(events))
        for event in chunk:
            event["loop_offset"] = start
        events.extend(chunk)
    spans = [dict(selection=7, id="layer.1.silu", start=10, end=11)]
    layout = generated_native_layout(cfg, 64, 8, events, spans)
    physical = {event["binding"]: {"shift": 18, "masks": [Q-1]*event["lanes"]} for event in events}
    tape = adapt_tapes(physical, layout)
    assert len(tape.batches) == 9
    assert all(batch["residues"].shape == (71, 131) for batch in tape.batches.values())
    for name in layout:
        tape.truncate(name, np.zeros((71, 131), dtype=object))
    tape.finish()
    physical["unused"] = {"shift": 18, "masks": [0]}
    with pytest.raises(OracleError, match="excess"):
        adapt_tapes(physical, layout)
    physical.pop("unused")
    physical.pop(events[-1]["binding"])
    with pytest.raises(OracleError, match="missing"):
        adapt_tapes(physical, layout)


@pytest.mark.parametrize("field,value", [("num_attention_heads", 0), ("num_key_value_heads", 0),
                                        ("head_dim", 129), ("hidden_size", 0),
                                        ("intermediate_size", False), ("num_hidden_layers", 0)])
def test_public_geometry_rejects_before_division_or_any_tape_consumption(field, value):
    cfg, weights, rotary = generated_fixture("qwen3", wide=True)
    cfg[field] = value
    with pytest.raises(OracleError, match="geometry"):
        generated_native_layout(cfg, 2, 2, [], [])
    tape = FixtureTape()
    with pytest.raises(OracleError, match="geometry"):
        run_decoder(cfg, weights, [0, 1], 2, tape, rotary)
    assert tape.batches == {}


def test_complete_full_vocabulary_dense_public_head_bounded_reader_and_scalar_recipe():
    """Complete head-only numeric control; NOT a checkpoint/model conformance claim."""
    vocab, calls = 151936, []

    def coefficients(row):
        if row == 0:
            return Q//32, 0
        if row == vocab-1:
            return -Q//16, 0
        return ((row*7)%13-6)*(Q//8192), ((row*11)%17-8)*(Q//8192)

    def read(start, stop):
        calls.append((start, stop))
        return array([coefficients(row) for row in range(start, stop)])

    matrix = PublicMatrix((vocab, 2), read)
    residues = np.array([(row*104729+7) % Q for row in range(vocab)], dtype=object)
    tape = Tape({"head": {"shift": 18, "residues": residues}})
    logits = linear([-Q//4, Q//16], matrix, tape, "head")
    tape.finish()
    # Separately pinned unwrapped scalar dot + one final stochastic truncation.
    expected = np.array([((-Q//4)*coefficients(row)[0]+(Q//16)*coefficients(row)[1]
                          + int(residues[row]))//Q for row in range(vocab)], dtype=object)
    compare_trace({"full_head": expected}, {"full_head": logits})
    assert stable_argmax(logits) == vocab-1
    assert calls[0][0] == 0 and calls[-1][1] == vocab
    assert all(stop-start <= 128 for start, stop in calls)
    assert all(left[1] == right[0] for left, right in zip(calls, calls[1:]))
    logits[-1] += 1
    with pytest.raises(OracleError, match="mismatch"):
        compare_trace({"full_head": expected}, {"full_head": logits})


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_complete_decoder_gqa_full_prefix_tapes_logits_and_private_feedback(family):
    cfg, weights, rotary = generated_fixture(family)
    authored = FixtureTape()
    expected = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    replay = Tape(authored.batches)
    actual = run_decoder(cfg, weights, [0, 1], 2, replay, rotary)
    compare_trace(expected.trace, actual.trace)
    assert len(actual.tokens) == 2
    assert actual.trace["s0/logits"].shape == (5,)
    assert actual.trace["s1/logits"].shape == (5,)
    assert actual.trace["s1/last_hidden"].shape == (4,)
    assert actual.trace["s1/input.tokens"].tolist() == [0, 1, actual.tokens[0]]
    assert actual.trace["s1/feedback"].tolist() == [0, 1, *actual.tokens]
    for selection in range(2):
        prefix = 2+selection
        for layer in range(2):
            p = f"s{selection}/layer.{layer}/"
            assert actual.trace[p + "full_prefix_k"].shape == (1, prefix, 2)
            assert actual.trace[p + "full_prefix_v"].shape == (1, prefix, 2)
            probs = actual.trace[p + "probabilities"]
            assert probs.shape == (2, prefix, prefix)
            assert all(probs[h, r, k] == 0 for h in range(2) for r in range(prefix) for k in range(r+1, prefix))
            assert any(v != 0 for v in actual.trace[p + "gate_up"].flat)
    assert any(v != 0 for v in actual.trace["s1/logits"].flat)
    assert not np.array_equal(actual.trace["s1/final_norm"][0], actual.trace["s1/final_norm"][1])
    # Single hidden corruption fails even with unchanged final selected IDs.
    corrupted = dict(actual.trace)
    corrupted["s0/layer.0/full_prefix_k"] = corrupted["s0/layer.0/full_prefix_k"].copy()
    corrupted["s0/layer.0/full_prefix_k"][0, 0, 0] += 1
    with pytest.raises(OracleError, match="trace mismatch"):
        compare_trace(expected.trace, corrupted)


def test_causal_prefix_independent_of_future_inputs_and_gqa_head_mapping():
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    first = run_decoder(cfg, weights, [0, 1], 1, FixtureTape("zero"), rotary)
    changed = run_decoder(cfg, weights, [0, 4], 1, FixtureTape("zero"), rotary)
    assert np.array_equal(first.trace["s0/final_norm"][0], changed.trace["s0/final_norm"][0])
    assert not np.array_equal(first.trace["s0/logits"], changed.trace["s0/logits"])
    # Both query heads attend same single KV head at first causal position.
    context = first.trace["s0/layer.0/context"]
    assert np.array_equal(context[0, 0], context[1, 0])


def test_full_decoder_streamed_public_sources_equal_array_fixture():
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    authored = FixtureTape()
    expected = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    streamed = {name: PublicMatrix(value.shape, lambda a, b, value=value: value[a:b])
                if value.ndim == 2 else value for name, value in weights.items()}
    actual = run_decoder(cfg, streamed, [0, 1], 2, Tape(authored.batches), rotary)
    compare_trace(expected.trace, actual.trace)


def test_stable_full_vocabulary_argmax_and_feedback_not_public_input_reuse():
    assert stable_argmax([1, 9, 9, -1]) == 1
    assert stable_argmax([-9, -9, -10]) == 0
    assert stable_argmax([0]*127 + [1]) == 127  # no sampled/head subset
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    weights["lm_head.weight"] = array([[0]*4]*5)
    out = run_decoder(cfg, weights, [3, 4], 2, FixtureTape(), rotary)
    assert out.tokens == [0, 0]
    assert out.trace["s1/input.tokens"].tolist() == [3, 4, 0]
    assert out.trace["s1/feedback"].tolist() == [3, 4, 0, 0]


def test_trace_missing_private_evidence_cannot_fabricate_equality():
    with pytest.raises(OracleError, match="nonempty"):
        compare_trace({}, {})
    for actual in ({}, {"x": None}, {"x": [0], "extra": [1]}, {"x": [[0]]}):
        with pytest.raises(OracleError):
            compare_trace({"x": [0]}, actual)
    assert reconstruct_test_shares([[2**64-1], [7], [2**64-8]]).tolist() == [-2]
    assert ring([2**64, 2**63, -2**63-1]).tolist() == [0, -2**63, 2**63-1]
    with pytest.raises(OracleError):
        reconstruct_test_shares([[1], [2]])


def test_decoder_unsupported_geometry_and_missing_public_inputs_fail_closed():
    cfg, weights, rotary = generated_fixture("llama")
    with pytest.raises(OracleError, match="rotary"):
        run_decoder(cfg, weights, [0, 1], 1, FixtureTape(), {})
    with pytest.raises(OracleError, match="unsupported"):
        run_decoder(dict(cfg, rope_scaling={"type": "linear", "factor": 2}), weights, [0], 1, FixtureTape(), rotary)
    with pytest.raises(OracleError, match="token domain"):
        run_decoder(cfg, weights, [5], 1, FixtureTape(), rotary)
    with pytest.raises(OracleError, match="source"):
        run_decoder(cfg, {}, [0], 1, FixtureTape(), rotary)


def guard_placements(cfg, outputs):
    names = []
    for selection in range(outputs):
        root = f"s{selection}/"
        for layer in range(cfg["num_hidden_layers"]):
            p = root + f"layer.{layer}/"
            names += [p + "input_norm/input", p + "post_attention_norm/input"]
            if cfg["model_type"] == "qwen3":
                names += [p + "q_norm/input", p + "k_norm/input"]
        names += [root + "final_norm/input"]
    return {name: RangeCheck() for name in names}


def test_checked_range_inclusive_endpoints_full_ring_and_exact_schema():
    bound = CHECKED_RANGE_LIMIT
    incoming = [-2**63, -bound-1, -bound, -1, 0, bound, bound+1, 2**63-1]
    safe, valid = RangeCheck().apply(incoming)
    assert valid.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert safe.tolist() == [0, 0, -bound, -1, 0, bound, 0, 0]
    # Core guard semantics are safe even for full signed interval endpoints,
    # without a subtraction that could wrap at MIN/MAX.
    for rule in (RangeCheck(-2**63, 2**63-1), RangeCheck(-2**63, -2**63, -2**63),
                 RangeCheck(2**63-1, 2**63-1, 2**63-1)):
        safe, valid = rule.apply(incoming)
        for original, selected, bit in zip(incoming, safe, valid):
            expected = int(rule.lower <= original <= rule.upper)
            assert bit == expected
            assert selected == (original if expected else rule.fallback)
    for args in ((-1.0, 1, 0), (False, 1, 0), (-2**63-1, 1, 0),
                 (-1, 2**63, 0), (1, -1, 0), (-1, 1, 2)):
        with pytest.raises(OracleError):
            RangeCheck(*args)


def test_checked_range_schedule_is_copied_frozen_and_no_private_auto_placement():
    from dataclasses import FrozenInstanceError

    placements = {"s0/final_norm/input": RangeCheck()}
    schedule = CheckedRangeSchedule.from_mapping(placements)
    placements.clear()
    assert len(schedule.entries) == 1
    with pytest.raises(FrozenInstanceError):
        schedule.entries = ()
    with pytest.raises(FrozenInstanceError):
        schedule.entries[0][1].fallback = 1
    for entries in ([('x/input', RangeCheck())],
                    (("x/input", RangeCheck()), ("x/input", RangeCheck())),
                    (("x", RangeCheck()),)):
        with pytest.raises(OracleError):
            CheckedRangeSchedule(entries)
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    with pytest.raises(OracleError, match="unused guard"):
        run_decoder(cfg, weights, [0, 1], 1, FixtureTape(), rotary,
                    checked_ranges=CheckedRangeSchedule((("s0/unknown/input", RangeCheck()),)))
    with pytest.raises(OracleError, match="immutable"):
        run_decoder(cfg, weights, [0, 1], 1, FixtureTape(), rotary, checked_ranges={})


@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3"])
def test_checked_range_accepted_unchanged_and_no_extra_truncations(family):
    cfg, weights, rotary = generated_fixture(family, layers=1)
    authored = FixtureTape()
    original = run_decoder(cfg, weights, [0, 1], 2, authored, rotary)
    schedule = CheckedRangeSchedule.from_mapping(guard_placements(cfg, 2))
    replay = Tape(authored.batches)
    checked = run_decoder(cfg, weights, [0, 1], 2, replay, rotary, checked_ranges=schedule)
    assert checked.accepted and checked.tokens == original.tokens
    compare_trace(original.trace, {name: checked.trace[name] for name in original.trace})
    assert checked.trace["acceptance.valid"] == 1
    for name, _ in schedule.entries:
        assert np.array_equal(checked.trace[name], checked.trace[name + "/checked_range/safe"])
        assert checked.trace[name + "/checked_range/sticky"] == 1
    # Explicit empty schedule retains EXACT old trace as well as values/tapes.
    empty = run_decoder(cfg, weights, [0, 1], 2, Tape(authored.batches), rotary,
                        checked_ranges=CheckedRangeSchedule())
    compare_trace(original.trace, empty.trace)


@pytest.mark.parametrize("failure_site", ["early", "late"])
def test_checked_range_failure_sticky_full_request_no_returned_ids(failure_site):
    cfg, weights, rotary = generated_fixture("qwen3", layers=2)
    placements = guard_placements(cfg, 2)
    failed = "s0/layer.0/input_norm/input" if failure_site == "early" else "s1/final_norm/input"
    placements[failed] = RangeCheck(0, 0)
    schedule = CheckedRangeSchedule.from_mapping(placements)
    authored = FixtureTape()
    rejected = run_decoder(cfg, weights, [0, 1], 2, authored, rotary, checked_ranges=schedule)
    assert rejected.accepted is False and rejected.tokens == []
    assert rejected.trace["acceptance.valid"] == 0
    assert rejected.trace[failed + "/checked_range/all_valid"] == 0
    assert set(rejected.trace[failed + "/checked_range/safe"].flat) == {0}
    # Continued execution including internal selection/feedback is diagnostic
    # ONLY. No IDs are returned even if the failure occurs after first selection.
    assert "s1/logits" in rejected.trace and "s1/feedback" in rejected.trace
    assert len(rejected.trace["s1/feedback"]) == 4
    guard_keys = [name for name in rejected.trace if name.endswith("/checked_range/sticky")]
    sticky_values = [int(rejected.trace[name]) for name in guard_keys]
    first_failure = sticky_values.index(0)
    assert all(value == 0 for value in sticky_values[first_failure:])
    if failure_site == "early":
        assert rejected.trace["s1/final_norm/input/checked_range/all_valid"] == 1
        assert rejected.trace["s1/final_norm/input/checked_range/sticky"] == 0
    else:
        assert all(value == 1 for value in sticky_values[:first_failure])
    replay = run_decoder(cfg, weights, [0, 1], 2, Tape(authored.batches), rotary,
                         checked_ranges=schedule)
    assert replay.tokens == [] and not replay.accepted
    compare_trace(rejected.trace, replay.trace)


def test_checked_range_guard_precedes_unsafe_square_and_consumes_no_tape():
    cfg, weights, rotary = generated_fixture("llama", layers=1)
    weights["model.embed_tokens.weight"][0, 0] = 2**63-1
    # Residual retains original invalid lane; every subsequent RMS input must
    # have its own public guard. Sticky failure does not authorize unsafe work.
    schedule = CheckedRangeSchedule.from_mapping(guard_placements(cfg, 1))
    t = FixtureTape()
    rejected = run_decoder(cfg, weights, [0], 1, t, rotary, checked_ranges=schedule)
    assert not rejected.accepted and rejected.tokens == []
    first = "s0/layer.0/input_norm/input/checked_range/"
    assert rejected.trace[first + "safe"][0, 0] == 0
    assert rejected.trace[first + "valid"][0, 0] == 0
    assert all("checked_range" not in name for name in t.batches)
    with pytest.raises(OracleError, match="quarter-gap"):
        run_decoder(cfg, weights, [0], 1, FixtureTape(), rotary)


def test_silu_operation_order_against_independent_pinned_scalar_recipe():
    # Pinned source coefficients and literal signed rounding, not oracle mul,
    # activation helpers, production arithmetic, or analytic floating SiLU.
    values = [-8*Q-1, -8*Q, -4*Q-1, -4*Q, -17, 0, 37, 4*Q, 4*Q+1]
    t = FixtureTape()
    actual = silu(values, t, "pinned")
    for lane, original in enumerate(values):
        def tr(raw, suffix):
            residue = int(t.batches["pinned/" + suffix]["residues"][lane])
            return (raw + residue)//Q

        x = original if -8*Q <= original <= 4*Q else 0
        x2 = tr(x*x, "x2")
        x4 = tr(x2*x2, "x4")
        x6 = tr(x2*x4, "x6")
        low = -80413 + tr(-21489*x, "low.x") + tr(-1453*x2, "low.x2")
        high = (2229 + tr(131072*x, "high.x") + tr(59806*x2, "high.x2")
                + tr(-2913*x4, "high.x4") + tr(71*x6, "high.x6"))
        expected = 0 if original < -8*Q else low if original < -4*Q else high if original <= 4*Q else original
        assert actual[lane] == expected


def test_softmax_operation_order_against_independent_pinned_scalar_recipe():
    values = [123, -Q+9, -14*Q+123, -14*Q+124, -100*Q]
    t = FixtureTape()
    actual = softmax(values, t, "pinned")

    def tr(raw, suffix, lane=None, shift=18):
        residues = t.batches["pinned/" + suffix]["residues"]
        residue = int(residues if lane is None else residues[lane])
        return (raw+residue)//(1 << shift)

    exponentials = []
    for lane, value in enumerate(values):
        z = value-max(values)
        keep = z > -14*Q
        e = Q + tr(z if keep else 0, "reduce", lane, 5)
        for square in range(5):
            e = tr(e*e, f"square.{square}", lane)
        exponentials.append(e if keep else 0)
    denominator = sum(exponentials)
    factor = 1 << (36-denominator.bit_length())
    c = tr(denominator*factor, "reciprocal/c")
    w = tr((763940-2*c)*factor, "reciprocal/w")
    error = Q-tr(denominator*w, "reciprocal/xw")
    r = tr(w*(Q+error), "reciprocal/r")
    error2 = tr(error*error, "reciprocal/e2")
    inverse = tr(r*(Q+error2), "reciprocal/out")
    assert actual.tolist() == [tr(e*inverse, "probabilities", lane) for lane, e in enumerate(exponentials)]


def test_rmsnorm_mixed_rows_order_against_independent_pinned_scalar_recipe():
    shared = offline_module("shared_profile_oracle.py")
    values = array([[Q//3, -Q//7, 17], [Q, 23, -Q//2]])
    gamma = [Q//2, Q, 2*Q]
    t = FixtureTape()
    actual = rmsnorm(values, gamma, t, "pinned", partitioned=True)
    for row in range(2):
        # Original independent scalar fixture consumes row lanes followed by
        # scalar denominator/rsqrt, row normalization, then row public gamma.
        masks = list(t.batches["pinned/squares"]["residues"][row])
        masks += [t.batches["pinned/mean"]["residues"][row]]
        for suffix in ("u", "u2", "u3", "u4", "polynomial", "out"):
            masks += [t.batches["pinned/rsqrt/" + suffix]["residues"][row]]
        masks += list(t.batches["pinned/normalize"]["residues"][row])
        masks += list(t.batches["pinned/gamma"]["residues"][row])
        assert actual[row].tolist() == shared.puma_rmsnorm(values[row], gamma, iter(masks))


# Source-present cut: all publications below are MOCK bytes, not credentials.
def _source_commit(domain, raw):
    return hashlib.sha256(domain.encode("ascii") + b"\0" + raw).hexdigest()


class _MockSourceOwner:
    """Synchronous test lease. Observer stores ranges only, never read payloads."""
    def __init__(self, payloads):
        self.payloads = payloads
        self.generation = 7
        self.calls = []
        self.destinations = []
        self.closed = False
        self.closes = 0
        self.mode = None

    def readinto(self, source_id, offset, destination):
        assert not self.closed
        for previous in self.destinations:
            with pytest.raises(ValueError):
                previous.obj  # Previous borrowed frame already released.
        self.destinations[:] = [destination]
        self.calls.append((source_id, offset, len(destination)))
        if self.mode == "error":
            raise RuntimeError("mock read error")
        if self.mode == "epoch":
            self.generation += 1
        data = self.payloads[source_id]
        if callable(data):
            data = data(offset, len(destination))
        else:
            data = data[offset:offset+len(destination)]
        destination[:len(data)] = data
        return len(data)-int(self.mode == "short")

    def close(self):
        self.closes += 1
        self.closed = True
        self.payloads = None


def _source_fixture(*, tied=False, vocab=131, dtype="F32", hidden=2, heads=1, kv=1, dim=2):
    """Small non-two-row Qwen2, independent exact dyadic source coefficients."""
    cfg = dict(model_type="qwen2", hidden_size=hidden, intermediate_size=3,
               num_hidden_layers=1, num_attention_heads=heads, num_key_value_heads=kv,
               head_dim=dim, vocab_size=vocab, attention_bias=True,
               tie_word_embeddings=tied, hidden_act="silu", rms_norm_eps=1e-6)
    geometry = source.Geometry(json.dumps(cfg, sort_keys=True).encode(), 3)
    descriptors, payloads, integers = [], {}, {}
    for index, (name, shape) in enumerate(geometry.roles()):
        if tied and name == "lm_head.weight":
            continue
        count = int(np.prod(shape))
        if "layernorm" in name or name == "model.norm.weight":
            words = [Q]*count
        else:
            words = [((i*3+index*5) % 11-5)*(Q//64) for i in range(count)]
        integers[name] = np.array(words, dtype=object).reshape(shape)
        canonical = b"".join(struct.pack("<f", word/Q) for word in words)
        if dtype == "F16":
            raw = b"".join(struct.pack("<e", word/Q) for word in words)
        elif dtype == "BF16":
            raw = b"".join(canonical[i+2:i+4] for i in range(0, len(canonical), 4))
        else:
            raw = canonical
        encoded = b"".join(struct.pack("<q", word) for word in words)
        sid = f"original:{index}"
        payloads[sid] = raw
        descriptors.append(source.Descriptor(name, sid, f"shard:{index}", dtype, shape,
            0, len(raw), len(raw), _source_commit(source.ORIGINAL, raw),
            f"canonical:{index}", len(canonical), _source_commit(source.CANONICAL, canonical),
            f"encoded:{index}", len(encoded), _source_commit(source.ENCODED, encoded)))
    owner = _MockSourceOwner(payloads)
    aliases = (("lm_head.weight", "model.embed_tokens.weight"),) if tied else ()
    view = source.PublicationView("mock-revision", "mock-original-publication",
        "mock-canonical-publication", geometry.config_bytes, 7, tuple(descriptors), aliases, owner)
    return cfg, geometry, view, integers


def _open_source(view, geometry, limits=None, quantize=source_f32):
    return source.open_weights(view, geometry, limits or source.Limits(),
        source_f32=quantize, matrix_type=PublicMatrix, array=array)


@pytest.mark.parametrize("dtype,raw,bits,words", [
    ("F32", "0000a0bf00000000000000800000803f", "0000a0bf00000000000000800000803f", [-327680, 0, 0, 262144]),
    ("F16", "00bd000000800100ff030004003c", "0000a0bf00000000000000800000803300c07f38000080380000803f", [-327680, 0, 0, 0, 15, 16, 262144]),
    ("BF16", "a0bf0000008001007f008000803f", "0000a0bf00000000000000800000010000007f00000080000000803f", [-327680, 0, 0, 0, 0, 0, 262144]),
    # +/-2^-19: trunc toward zero, not floor or nearest, plus next negative value.
    ("F32", "00000036000000b6000040b6", "00000036000000b6000040b6", [0, 0, 0]),
    ("F32", "0000804a000080ca", "0000804a000080ca", [1099511627776, -1099511627776]),
    ("F32", "010080bf0100803f", "010080bf0100803f", [-262144, 262144]),
])
def test_source_cut_literal_ieee_widening_and_q18(dtype, raw, bits, words):
    canonical = source.widen_le(bytes.fromhex(raw), dtype)
    assert canonical == bytes.fromhex(bits)
    assert source_f32(canonical).tolist() == words


@pytest.mark.parametrize("dtype,raw", [
    ("F32", "0000807f"), ("F32", "0000c07f"), ("F32", "000080ff"),
    ("F16", "007c"), ("F16", "017c"), ("F16", "00fc"),
    ("BF16", "807f"), ("BF16", "c17f"), ("BF16", "80ff"),
])
def test_source_cut_nonfinite_original_refused(dtype, raw):
    with pytest.raises(source.SourceError, match="nonfinite"):
        source.widen_le(bytes.fromhex(raw), dtype)


@pytest.mark.parametrize("raw", ["0100804a", "010080ca"])
def test_source_cut_next_f32_outside_inclusive_q18_boundary(raw):
    with pytest.raises(OracleError, match="source Q18 range"):
        source_f32(source.widen_le(bytes.fromhex(raw), "F32"))


def test_source_cut_endian_is_declared_not_guessed():
    # Reversed 1.0 is a finite subnormal, never silently byte-swapped.
    assert source.widen_le(bytes.fromhex("3f800000"), "F32") == bytes.fromhex("3f800000")
    assert source_f32(bytes.fromhex("3f800000")).tolist() == [0]
    _, geometry, view, _ = _source_fixture()
    bad = replace(view.descriptors[0], endian="BE")
    with pytest.raises(source.SourceError, match="endian"):
        _open_source(replace(view, descriptors=(bad,)+view.descriptors[1:]), geometry)
    assert view.owner.calls == []


@pytest.mark.parametrize("change", ["raw8", "short", "extra", "dtype", "shape", "rank", "order",
    "range", "negative", "bool", "digest", "domain", "identity", "overlap", "duplicate",
    "canonical", "encoded", "missing", "extra_binding", "alias", "config", "profile", "epoch"])
def test_source_cut_preflight_refuses_before_read_or_conversion(change):
    _, geometry, view, _ = _source_fixture()
    descs = list(view.descriptors)
    d = descs[0]
    edits = {
        "raw8": dict(byte_length=d.encoded_length), "short": dict(byte_length=d.byte_length-1),
        "extra": dict(byte_length=d.byte_length+1), "dtype": dict(dtype="F16"),
        "shape": dict(shape=(d.shape[0]+1, d.shape[1])), "rank": dict(shape=(1, 1, 1)),
        "order": dict(order="F"), "range": dict(byte_offset=1), "negative": dict(byte_offset=-1),
        "bool": dict(byte_offset=True), "digest": dict(original_digest="z"*64),
        "domain": dict(original_domain="sha256"), "identity": dict(canonical_id=d.source_id),
        "canonical": dict(canonical_length=d.byte_length+4), "encoded": dict(encoded_length=d.byte_length),
    }
    if change in edits:
        descs[0] = replace(d, **edits[change])
    elif change == "overlap":
        descs[1] = replace(descs[1], shard_id=d.shard_id, shard_extent=d.shard_extent)
    elif change == "duplicate":
        descs[1] = replace(descs[1], source_id=d.source_id)
    elif change == "missing":
        descs.pop()
    elif change == "extra_binding":
        descs.append(d)
    elif change == "alias":
        view = replace(view, aliases=(("lm_head.weight", d.name),))
    elif change == "config":
        view = replace(view, config_bytes=b"{}")
    elif change == "profile":
        view = replace(view, numeric_identity="source-exact-epsilon")
    else:
        view.owner.generation += 1
    view = replace(view, descriptors=tuple(descs))
    def forbidden(raw):
        pytest.fail("preflight refusal reached conversion")
    with pytest.raises(source.SourceError):
        _open_source(view, geometry, quantize=forbidden)
    assert view.owner.calls == []


@pytest.mark.parametrize("field,threshold", [("frame_words", 256), ("scratch_bytes", 8192),
                                            ("vector_words", 2), ("gather_words", 8)])
@pytest.mark.parametrize("short", [False, True])
def test_source_cut_exact_and_one_short_logical_limits(field, threshold, short, monkeypatch):
    _, geometry, view, _ = _source_fixture()
    limits = replace(source.Limits(), **{field: threshold-int(short)})
    if short:
        def forbidden(*args):
            pytest.fail("denied budget reached frame allocation")
        monkeypatch.setattr(source, "bytearray", forbidden, raising=False)
        with pytest.raises(source.SourceError, match="budget"):
            _open_source(view, geometry, limits)
        assert view.owner.calls == []
    else:
        weights = _open_source(view, geometry, limits)
        assert weights._state == "active"
        weights.close()


@pytest.mark.parametrize("domain", ["original_digest", "canonical_digest", "encoded_digest"])
def test_source_cut_commitment_failure_precedes_any_compute(domain):
    _, geometry, view, _ = _source_fixture()
    d = replace(view.descriptors[0], **{domain: "0"*64})
    view = replace(view, descriptors=(d,)+view.descriptors[1:])
    with pytest.raises(source.SourceError, match="commitment"):
        _open_source(view, geometry)
    assert view.owner.closed and all(sid == d.source_id for sid, _, _ in view.owner.calls)


def test_source_cut_signed_zero_original_mutation_not_hidden_by_q18():
    _, geometry, view, _ = _source_fixture()
    d = view.descriptors[0]
    # Pin a positive zero before mutating only its sign: Q18 stays zero.
    raw = b"\0"*d.byte_length
    d = replace(d, original_digest=_source_commit(source.ORIGINAL, raw),
                canonical_digest=_source_commit(source.CANONICAL, raw),
                encoded_digest=_source_commit(source.ENCODED, bytes(d.encoded_length)))
    view.owner.payloads[d.source_id] = b"\0\0\0\x80"+raw[4:]
    with pytest.raises(source.SourceError, match="commitment"):
        _open_source(replace(view, descriptors=(d,)+view.descriptors[1:]), geometry)
    assert source_f32(bytes.fromhex("0000000000000080")).tolist() == [0, 0]


@pytest.mark.parametrize("mode", ["short", "error", "epoch"])
def test_source_cut_read_failure_releases_borrow_and_closes(mode):
    _, geometry, view, _ = _source_fixture()
    view.owner.mode = mode
    with pytest.raises((source.SourceError, RuntimeError)):
        _open_source(view, geometry)
    assert view.owner.closed and view.owner.closes == 1
    for destination in view.owner.destinations:
        with pytest.raises(ValueError):
            destination.obj


def test_source_cut_descriptor_only_lookup_close_and_drop():
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    n = len(view.owner.calls)
    for _ in range(3):
        assert "model.embed_tokens.weight" in weights
        matrix = weights["model.embed_tokens.weight"]
    assert len(view.owner.calls) == n  # No payload cache, vector or lookahead read.
    with pytest.raises(source.SourceError, match="mock only"):
        weights.require_execution_qualification()
    weights.close()
    weights.close()
    assert view.owner.closes == 1
    for action in (lambda: matrix.rows(0, 128), weights.finish,
                   lambda: weights.gather_embedding([0, 1, 2])):
        with pytest.raises(source.SourceError):
            action()
    assert len(view.owner.calls) == n
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    ref = weakref.ref(weights)
    del weights
    gc.collect()
    assert ref() is None and view.owner.closes == 1


def test_source_cut_complete_mixed_sign_contraction_one_negative_truncation():
    # First two products separately floor to -1 and 0; sum floors to 0.
    matrix = PublicMatrix((3, 3), lambda start, stop:
        array([[-1, 1, 0], [1, -1, 1], [-1, -1, 1]][start:stop]))
    tape = Tape({"dot": {"shift": 18, "residues": [0, 0, 0]}})
    assert linear([131073, 131074, -262145], matrix, tape, "dot").tolist() == [0, -2, -3]
    tape.finish()
    assert (-131073)//Q + 131074//Q == -1


def _source_scalar_qwen2(cfg, weights, original_ids, rotary):
    """Independent complete small decoder: lists/scalar dots, no adapter helpers.

    Reuses preexisting standalone scalar HAL norm/reciprocal. Fixed zero low
    residues are an explicit consumer mock, not a native randomness claim.
    Returns every logit, last hidden and feedback for both selections.
    """
    shared = offline_module("shared_profile_oracle.py")
    w = {name: value.tolist() for name, value in weights.items()}
    if cfg["tie_word_embeddings"]:
        w["lm_head.weight"] = w["model.embed_tokens.weight"]
    def dot(x, row):
        return sum(a*b for a, b in zip(x, row))//Q
    def project(rows, name, bias=None):
        return [[dot(x, row)+(0 if bias is None else bias[i])
                 for i, row in enumerate(w[name])] for x in rows]
    def norm(rows, name):
        return [shared.puma_rmsnorm(row, w[name], repeat(0)) for row in rows]
    def activate(x):
        safe = x if -8*Q <= x <= 4*Q else 0
        x2 = safe*safe//Q
        x4, x6 = x2*x2//Q, x2*(x2*x2//Q)//Q
        low = -80413 + (-21489*safe)//Q + (-1453*x2)//Q
        high = 2229 + (131072*safe)//Q + (59806*x2)//Q + (-2913*x4)//Q + (71*x6)//Q
        return 0 if x < -8*Q else low if x < -4*Q else high if x <= 4*Q else x
    ids, result = list(original_ids), []
    for selection in range(2):
        x = [list(w["model.embed_tokens.weight"][i]) for i in ids]
        for l in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{l}."
            normalized = norm(x, p+"input_layernorm.weight")
            projections = []
            for kind in "qkv":
                rows = project(normalized, p+f"self_attn.{kind}_proj.weight", w[p+f"self_attn.{kind}_proj.bias"])
                if kind in "qk":
                    rotated = []
                    for pos, (a, b) in enumerate(rows):
                        (c,), (s,) = rotary[pos]
                        rotated.append([a*c//Q + b*(-s)//Q, b*c//Q + a*s//Q])
                    rows = rotated
                projections.append(rows)
            q, k, v = projections
            contexts = []
            for position, query in enumerate(q):
                # Literal binary32 Q18 1/sqrt(2), independent of adapter.
                scores = [(dot(query, key)*185363)//Q for key in k[:position+1]]
                exps = []
                for score in scores:
                    z = score-max(scores)
                    e = Q + (z if z > -14*Q else 0)//32
                    for _ in range(5):
                        e = e*e//Q
                    exps.append(e if z > -14*Q else 0)
                inv = shared.puma_reciprocal(sum(exps), repeat(0))
                probabilities = [e*inv//Q for e in exps]
                contexts.append([sum(probabilities[j]*v[j][col] for j in range(position+1))//Q for col in range(2)])
            out = project(contexts, p+"self_attn.o_proj.weight")
            x = [[a+b for a, b in zip(row, delta)] for row, delta in zip(x, out)]
            normalized = norm(x, p+"post_attention_layernorm.weight")
            gate = project(normalized, p+"mlp.gate_proj.weight")
            up = project(normalized, p+"mlp.up_proj.weight")
            product = [[activate(a)*b//Q for a, b in zip(g, u)] for g, u in zip(gate, up)]
            delta = project(product, p+"mlp.down_proj.weight")
            x = [[a+b for a, b in zip(row, d)] for row, d in zip(x, delta)]
        last = norm(x, "model.norm.weight")[-1]
        logits = project([last], "lm_head.weight")[0]
        token = max(range(len(logits)), key=lambda i: (logits[i], -i))
        ids.append(token)
        result.append((last, logits, list(ids), token))
    return result


def _source_expected_reads(view, tied):
    # Hand-enumerated consumer grammar, NOT Geometry.roles / adapter cursor.
    p = "model.layers.0."
    names = ["model.embed_tokens.weight", p+"input_layernorm.weight",
        p+"self_attn.q_proj.bias", p+"self_attn.q_proj.weight",
        p+"self_attn.k_proj.bias", p+"self_attn.k_proj.weight",
        p+"self_attn.v_proj.bias", p+"self_attn.v_proj.weight",
        p+"self_attn.o_proj.weight", p+"post_attention_layernorm.weight",
        p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight",
        "model.norm.weight", "lm_head.weight"]
    descs = {d.name: d for d in view.descriptors}
    def frames(name):
        d = descs["model.embed_tokens.weight" if tied and name == "lm_head.weight" else name]
        width = 4 if d.dtype == "F32" else 2
        if len(d.shape) == 1:
            return [(d.source_id, 0, d.shape[0]*width)]
        rows, cols = d.shape
        # This fixture has 2/3 columns: independent expected 128-row public step.
        return [(d.source_id, start*cols*width, min(128, rows-start)*cols*width)
                for start in range(0, rows, 128)]
    verification = [call for name in names if not (tied and name == "lm_head.weight") for call in frames(name)]
    compute = [call for _ in range(2) for name in names for call in frames(name)]
    return verification+compute


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
def test_source_cut_non2row_two_selection_real_decoder_scalar_complete_outputs(tied, dtype):
    cfg, geometry, view, integers = _source_fixture(tied=tied, dtype=dtype)
    rotary = {0: ([Q], [0]), 1: ([141637], [220586]),
              2: ([-109090], [238366]), 3: ([-259520], [36994])}
    ids = [4, 2, 4]
    expected = _source_scalar_qwen2(cfg, integers, ids, rotary)
    expected_reads = _source_expected_reads(view, tied)
    encoded_refs, encoded_sizes = [], []
    def quantize(raw):
        gc.collect()
        # Qwen bias (two words) is legitimately live during projection read.
        # No prior matrix block or vector history may survive the next read.
        alive = [(ref, size) for ref, size in encoded_refs if ref() is not None]
        assert len(alive) <= 1 and all(size == 2 for _, size in alive)
        value = source_f32(raw)
        encoded_sizes.append(value.size)
        encoded_refs.append((weakref.ref(value), value.size))
        return value
    weights = _open_source(view, geometry, quantize=quantize)
    # Full arrays exist only in independent tiny test fixture, never provider.
    declarations, records = [], []
    for s, (last, logits, feedback, token) in enumerate(expected):
        for suffix, native_id, payload in (("last_hidden", "last_hidden", last),
                ("logits", "output_head", logits), ("selected", "token_selection", token),
                ("feedback", "feedback.history", feedback)):
            shape = () if type(payload) is int else (len(payload),)
            name = f"s{s}/{suffix}"
            declarations.append((name, ((s, native_id, shape, False, None),)))
            records.append(dict(selection=s, id=native_id, shape=shape,
                                batch_dimension=False, values=[payload] if not shape else payload))
    comparator = StreamingTraceComparator(iter(records), declaration=tuple(declarations),
        source_provenance=(("kind", "independent-scalar-consumer-mock-not-native"),),
        checked_ranges=None, outputs=2, max_records=8,
        max_total_values=sum(len(r["values"]) for r in records))
    seen = []
    def sink(name, value):
        if name.endswith(("/logits", "/last_hidden", "/feedback", "/selected")):
            comparator(name, value)
            seen.append(name)
    with weights:
        result = run_decoder(cfg, weights, ids, 2, FixtureTape("zero"), rotary, trace_sink=sink)
        weights.finish()
    comparator.finish(result, final_components=[[row[3] for row in expected], [0, 0], [0, 0]])
    assert result.tokens == [row[3] for row in expected] and result.accepted and result.trace == {}
    assert len(seen) == 8 and view.owner.calls == expected_reads
    assert max(encoded_sizes) == 256 and all(ref() is None for ref, _ in encoded_refs)
    assert view.owner.closed and view.owner.closes == 1


def test_source_cut_private_ids_and_rejection_have_identical_public_reads():
    observations = []
    for ids, reject in (([0, 0, 130], False), ([7, 3, 7], False), ([5, 9, 2], True)):
        cfg, geometry, view, _ = _source_fixture()
        rotary = {0: ([Q], [0]), 1: ([141637], [220586]),
                  2: ([-109090], [238366]), 3: ([-259520], [36994])}
        guards = CheckedRangeSchedule.from_mapping({"s0/layer.0/input_norm/input": RangeCheck(0, 0)}) if reject else None
        with _open_source(view, geometry) as weights:
            result = run_decoder(cfg, weights, ids, 2, FixtureTape("zero"), rotary, checked_ranges=guards,
                                 trace_sink=lambda name, value: None)
            weights.finish()
        assert result.accepted is not reject
        observations.append(view.owner.calls)
    assert observations[0] == observations[1] == observations[2]


@pytest.mark.parametrize("violation", ["skip", "wrong_interval", "wrong_input_count", "finish", "generation"])
def test_source_cut_consumer_grammar_and_compute_epoch_fail_closed(violation):
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    n = len(view.owner.calls)
    if violation == "generation":
        view.owner.generation += 1
    actions = {
        "skip": lambda: weights["model.layers.0.input_layernorm.weight"],
        "wrong_interval": lambda: weights["model.embed_tokens.weight"].rows(1, 2),
        "wrong_input_count": lambda: weights.gather_embedding([0, 1]),
        "finish": weights.finish,
        "generation": lambda: weights.gather_embedding([0, 1, 2]),
    }
    with pytest.raises(source.SourceError):
        actions[violation]()
    assert len(view.owner.calls) == n and view.owner.closed


def test_source_cut_coordinate_spy_gather_release_no_decoded_cache():
    _, geometry, view, _ = _source_fixture()
    desc = view.descriptors[0]
    def coordinate_bytes(offset, size):
        assert offset % 8 == size % 8 == 0
        return b"".join(struct.pack("<f", ((word//2)*2+word % 2)/Q)
                        for word in range(offset//4, (offset+size)//4))
    # Freeze coordinate-coded mock bytes/commitments BEFORE opening the scope.
    canonical = coordinate_bytes(0, desc.byte_length)
    encoded = b"".join(struct.pack("<q", word) for word in range(262))
    desc = replace(desc, original_digest=_source_commit(source.ORIGINAL, canonical),
        canonical_digest=_source_commit(source.CANONICAL, canonical),
        encoded_digest=_source_commit(source.ENCODED, encoded))
    view = replace(view, descriptors=(desc,)+view.descriptors[1:])
    view.owner.payloads[desc.source_id] = coordinate_bytes
    weights = _open_source(view, geometry)
    result = weights.gather_embedding([130, 1, 130])
    assert result.tolist() == [[260, 261], [2, 3], [260, 261]]
    ref = weakref.ref(result)
    del result
    gc.collect()
    assert ref() is None  # Provider alive, selected result not cached.
    weights.close()


def test_source_cut_immutable_contracts_and_descriptor_hash_not_authority():
    from dataclasses import FrozenInstanceError
    _, geometry, view, _ = _source_fixture()
    for obj, field, replacement in ((view, "generation", 8), (geometry, "input_rows", 2),
                                     (view.descriptors[0], "dtype", "F16")):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, replacement)
    with _open_source(view, geometry) as weights:
        # Even perfectly matching self-supplied hashes never establish authority.
        with pytest.raises(source.SourceError, match="missing borrowed source authority"):
            weights.require_execution_qualification()


@pytest.mark.parametrize("field,value", [("input_rows", 0), ("input_rows", 33), ("input_rows", True),
    ("hidden_size", 65537), ("head_dim", 3), ("num_attention_heads", 0),
    ("attention_bias", 1), ("tie_word_embeddings", 1), ("rope_scaling", {"type": "linear"})])
def test_source_cut_invalid_geometry_precedes_reads(field, value):
    cfg, geometry, view, _ = _source_fixture()
    if field == "input_rows":
        geometry = replace(geometry, input_rows=value)
    else:
        cfg[field] = value
        raw = json.dumps(cfg, sort_keys=True).encode()
        geometry = replace(geometry, config_bytes=raw)
        view = replace(view, config_bytes=raw)
    with pytest.raises(source.SourceError):
        _open_source(view, geometry)
    assert view.owner.calls == []


def test_source_cut_wide_columns_step_and_last_full_inner_axis():
    d = source.Descriptor("w", "original", "shard", "F32", (129, 1024),
        0, 129*1024*4, 129*1024*4, "0"*64, "canonical", 129*1024*4, "0"*64,
        "encoded", 129*1024*8, "0"*64)
    d.validate()
    assert list(source.SourceWeights._frames(None, d)) == [
        (0, 65536), (65536, 131072), (131072, 132096)]
    # No partial inner axis, despite 1-row final block and step reduced to 64.
    with pytest.raises(source.SourceError):
        replace(d, shape=(1, 65537)).validate()


@pytest.mark.parametrize("site", ["middle", "tail", "bias"])
def test_source_cut_independent_middle_tail_bias_corruption_controls(site):
    cfg, geometry, view, integers = _source_fixture(vocab=7)
    rotary = {0: ([Q], [0]), 1: ([141637], [220586]),
              2: ([-109090], [238366]), 3: ([-259520], [36994])}
    ids = [4, 2, 4]
    original = _source_scalar_qwen2(cfg, integers, ids, rotary)
    name = {"middle": "model.layers.0.mlp.down_proj.weight", "tail": "lm_head.weight",
            "bias": "model.layers.0.self_attn.v_proj.bias"}[site]
    changed = integers[name].copy()
    changed.flat[-1] += Q//4
    integers[name] = changed
    raw = b"".join(struct.pack("<f", int(word)/Q) for word in changed.flat)
    encoded = b"".join(struct.pack("<q", int(word)) for word in changed.flat)
    descs = list(view.descriptors)
    index = next(i for i, d in enumerate(descs) if d.name == name)
    d = descs[index]
    descs[index] = replace(d, original_digest=_source_commit(source.ORIGINAL, raw),
        canonical_digest=_source_commit(source.CANONICAL, raw),
        encoded_digest=_source_commit(source.ENCODED, encoded))
    view.owner.payloads[d.source_id] = raw
    view = replace(view, descriptors=tuple(descs))
    expected = _source_scalar_qwen2(cfg, integers, ids, rotary)
    assert original[0][1] != expected[0][1]  # Complete logits distinguish change.
    with _open_source(view, geometry) as weights:
        actual = run_decoder(cfg, weights, ids, 2, FixtureTape("zero"), rotary)
        weights.finish()
    for s in range(2):
        assert actual.trace[f"s{s}/logits"].tolist() == expected[s][1]
        with pytest.raises(OracleError, match="mismatch"):
            compare_trace({"logits": original[s][1]}, {"logits": actual.trace[f"s{s}/logits"]})


def test_source_cut_consumer_error_scope_closes_and_prior_block_dies_in_traceback():
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    refs = []
    def reader(start, stop):
        gc.collect()
        assert all(ref() is None for ref in refs)
        if start:
            raise RuntimeError("next block failed")
        block = np.zeros((stop-start, 2), dtype=object)
        refs.append(weakref.ref(block))
        return block
    with pytest.raises(RuntimeError, match="next block failed") as caught:
        with weights:
            linear([Q, Q], PublicMatrix((129, 2), reader), Tape({}), "never_truncate")
    gc.collect()
    assert caught.value.__traceback__ is not None and all(ref() is None for ref in refs)
    assert view.owner.closed


@pytest.mark.parametrize("mode", ["short", "error", "epoch"])
def test_source_cut_compute_read_failure_cannot_finish(mode):
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    view.owner.mode = mode
    with pytest.raises((source.SourceError, RuntimeError)):
        weights.gather_embedding([0, 1, 2])
    with pytest.raises(source.SourceError):
        weights.finish()
    assert view.owner.closed and view.owner.closes == 1
    for destination in view.owner.destinations:
        with pytest.raises(ValueError):
            destination.obj


def test_source_cut_conversion_error_traceback_drops_transient_array():
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    refs = []
    def failed_conversion(raw):
        transient = source_f32(raw)
        refs.append(weakref.ref(transient))
        raise RuntimeError("mock conversion failed")
    weights._quantize = failed_conversion
    with pytest.raises(RuntimeError, match="mock conversion failed") as caught:
        weights.gather_embedding([0, 1, 2])
    gc.collect()
    assert caught.value.__traceback__ is not None and all(ref() is None for ref in refs)
    assert view.owner.closed


def test_source_cut_24_layer_full_vocab_count_only_grammar_not_execution():
    """Metadata census ONLY: no byte read, arithmetic, native or authority proof."""
    cfg = dict(model_type="qwen2", hidden_size=896, intermediate_size=4864,
        num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
        head_dim=64, vocab_size=151936, tie_word_embeddings=False, attention_bias=True)
    geometry = source.Geometry(json.dumps(cfg).encode(), 4)
    roles = tuple(geometry.roles())
    # Deliberately no PublicationView or open_weights: not a verified publication.
    descs = {name: SimpleNamespace(shape=shape) for name, shape in roles}
    cursor = SimpleNamespace(_roles=roles, _resolve=lambda name: descs[name],
                             _frames=lambda desc: source.SourceWeights._frames(None, desc))
    count, endpoints, layer_counts = 0, {}, {}
    for selection, name, start, stop in source.SourceWeights._demands(cursor):
        count += 1
        endpoints.setdefault(selection, [name, start, stop])
        endpoints[selection][2] = stop
        if name.startswith("model.layers."):
            layer = int(name.split(".")[2])
            layer_counts[selection, layer] = layer_counts.get((selection, layer), 0)+1
        last = selection, name, start, stop
    # Independent arithmetic: embedding/head each ceil(151936/73)=2082;
    # per layer: 1+1+13+1+2+1+2+13+1+67+67+69 = 238 frames.
    assert set(layer_counts.values()) == {238} and len(layer_counts) == 48
    assert count == 2*(2082+24*238+1+2082)
    assert endpoints[0][0:2] == endpoints[1][0:2] == ["model.embed_tokens.weight", 0]
    assert last == (1, "lm_head.weight", 151913*896, 151936*896)


@pytest.mark.parametrize("family,bias", [("qwen2", False), ("qwen3", False),
                                         ("qwen3", True), ("llama", False)])
def test_source_cut_family_optional_bias_and_qk_norm_demand_roles(family, bias):
    cfg, _, _, _ = _source_fixture(vocab=3)
    cfg.update(model_type=family, attention_bias=bias)
    roles = list(source.Geometry(json.dumps(cfg).encode(), 3).roles())
    names = [name for name, shape in roles]
    p = "model.layers.0."
    expected = ["model.embed_tokens.weight", p+"input_layernorm.weight"]
    for kind in "qkv":
        if bias:
            expected.append(p+f"self_attn.{kind}_proj.bias")
        expected.append(p+f"self_attn.{kind}_proj.weight")
        if family == "qwen3" and kind in "qk":
            expected.append(p+f"self_attn.{kind}_norm.weight")
    expected += [p+"self_attn.o_proj.weight", p+"post_attention_layernorm.weight",
                 p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight",
                 "model.norm.weight", "lm_head.weight"]
    assert names == expected


@pytest.mark.parametrize("field,threshold", [("index_records", 17), ("index_bytes", 544)])
@pytest.mark.parametrize("short", [False, True])
def test_source_cut_repair_fingerprint_index_exact_cardinality_preflight(field, threshold, short, monkeypatch):
    _, geometry, view, _ = _source_fixture()
    limits = replace(source.Limits(), **{field: threshold-int(short)})
    if short:
        def forbidden(*args):
            pytest.fail("metadata refusal reached any index/frame allocation")
        monkeypatch.setattr(source, "bytearray", forbidden, raising=False)
        with pytest.raises(source.SourceError, match="fingerprint index budget"):
            _open_source(view, geometry, limits)
        assert view.owner.calls == []
    else:
        with _open_source(view, geometry, limits) as weights:
            assert len(weights._frame_hashes) == 544  # 17 physical partitions.
            assert len(weights._index_offsets) == 15  # 15 descriptors, no aliases.
            before = bytes(weights._frame_hashes)
            weights.gather_embedding([0, 128, 130])
            assert len(weights._frame_hashes) == 544 and weights._frame_hashes == before
        assert weights._frame_hashes is weights._index_offsets is None


@pytest.mark.parametrize("word", [0, 256])
@pytest.mark.parametrize("mutation", ["value", "encoded_collision"])
def test_source_cut_repair_same_generation_reread_first_and_noninitial_frame(word, mutation):
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    d = view.descriptors[0]
    raw = view.owner.payloads[d.source_id]
    old = raw[4*word:4*(word+1)]
    if mutation == "value":
        changed = bytes.fromhex("0000803f")
    else:
        # One original F32 bit changes; encoded Q18 is independently pinned
        # unchanged. Thus an encoded-only reread hash would miss this fault.
        changed = (int.from_bytes(old, "little") ^ 1).to_bytes(4, "little")
        golden = -20480 if word == 0 else 16384
        assert source_f32(old).tolist() == source_f32(changed).tolist() == [golden]
    view.owner.payloads[d.source_id] = raw[:4*word]+changed+raw[4*(word+1):]
    n = len(view.owner.calls)
    with pytest.raises(source.SourceError, match="reread content mismatch"):
        weights.gather_embedding([128, 0, 130])
    assert len(view.owner.calls) == n+(1 if word == 0 else 2)
    assert view.owner.generation == 7 and view.owner.closes == 1
    assert weights._frame_hashes is None


class _SourceSentinel:
    pass


@pytest.mark.parametrize("fault", ["short", "callback", "reentrant", "close_inside"])
@pytest.mark.parametrize("close_raises", [False, True])
def test_source_cut_repair_borrow_release_poison_and_throwing_close(fault, close_raises, monkeypatch):
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    owner = view.owner
    read, close, empty = owner.readinto, owner.close, np.empty
    borrowed, sentinels, allocation_calls = [], [], []
    callback_active = False
    def allocate(*args, **kwargs):
        allocation_calls.append(args[0])
        return empty(*args, **kwargs)
    monkeypatch.setattr(source.np, "empty", allocate)
    def callback(sid, offset, destination):
        nonlocal callback_active
        callback_active = True
        borrowed.append(destination)
        sentinel = _SourceSentinel()
        sentinels.append(weakref.ref(sentinel))
        try:
            count = read(sid, offset, destination)
            if fault == "callback":
                raise RuntimeError("callback with payload sentinel")
            if fault == "reentrant":
                n = len(allocation_calls)
                with pytest.raises(source.SourceError, match="reentrant"):
                    weights.gather_embedding([0, 1, 2])
                assert len(allocation_calls) == n and not owner.closed
            elif fault == "close_inside":
                weights.close()
                assert not owner.closed
            return count-int(fault == "short")
        finally:
            callback_active = False
    def checked_close():
        assert not callback_active
        for destination in borrowed:
            with pytest.raises(ValueError):
                destination.obj
        sentinel = _SourceSentinel()
        sentinels.append(weakref.ref(sentinel))
        close()
        if close_raises:
            raise LookupError("cleanup with payload sentinel")
    owner.readinto, owner.close = callback, checked_close
    with pytest.raises((source.SourceError, RuntimeError)) as retained:
        weights.gather_embedding([0, 1, 2])
    gc.collect()
    assert retained.value.__traceback__ is not None
    assert all(ref() is None for ref in sentinels)
    assert owner.closes == 1 and owner.closed and len(borrowed) == 1
    if close_raises:
        assert isinstance(retained.value.__cause__, LookupError)
    with pytest.raises(source.SourceError):
        weights.finish()


def test_source_cut_repair_equal_mutable_view_config_refused_before_reads():
    _, geometry, view, _ = _source_fixture()
    with pytest.raises(source.SourceError, match="config/profile"):
        _open_source(replace(view, config_bytes=bytearray(geometry.config_bytes)), geometry)
    assert view.owner.calls == []


@pytest.mark.parametrize("site", ["bias_then_weight", "norm_gamma"])
def test_source_cut_repair_real_decoder_error_retires_bias_and_gamma(site):
    cfg, geometry, view, _ = _source_fixture()
    weights = None
    refs = []
    target = "model.layers.0." + ("self_attn.q_proj.bias" if site == "bias_then_weight" else "input_layernorm.weight")
    def codec(raw):
        result = source_f32(raw)
        if weights is not None and weights._next[1] == target:
            refs.append(weakref.ref(result))
        return result
    weights = _open_source(view, geometry, quantize=codec)
    if site == "bias_then_weight":
        read = view.owner.readinto
        sid = next(d.source_id for d in view.descriptors if d.name == "model.layers.0.self_attn.q_proj.weight")
        def fail_weight(source_id, offset, destination):
            if source_id == sid:
                assert refs and refs[-1]() is not None  # Actual successful bias.
                raise RuntimeError("real projection read failed")
            return read(source_id, offset, destination)
        view.owner.readinto = fail_weight
    class FailingNorm(FixtureTape):
        def truncate(self, name, raw, shift=18):
            if site == "norm_gamma" and name == "s0/layer.0/input_norm/squares":
                assert refs and refs[-1]() is not None  # Actual gamma argument.
                raise RuntimeError("real norm tape failed")
            return super().truncate(name, raw, shift)
    rotary = {0: ([Q], [0]), 1: ([141637], [220586]),
              2: ([-109090], [238366]), 3: ([-259520], [36994])}
    with pytest.raises(RuntimeError, match="real .* failed") as retained:
        with weights:
            run_decoder(cfg, weights, [4, 2, 4], 2, FailingNorm("zero"), rotary)
    gc.collect()
    assert retained.value.__traceback__ is not None and len(refs) == 1
    assert all(ref() is None for ref in refs)
    assert view.owner.closes == 1 and weights._state == "closed"


@pytest.mark.parametrize("violation", ["split_first", "split_second", "prefix", "typed_config", "head_remap"])
def test_source_cut_repair_decoder_entry_rejects_before_compute(violation):
    cfg, geometry, view, _ = _source_fixture(hidden=4, dim=4)
    weights = _open_source(view, geometry)
    n = len(view.owner.calls)
    ids, outputs = [0, 1, 2], 2
    if violation.startswith("split"):
        outputs = 1
        if violation == "split_second":
            ids = [4, 4, 4, 4]
    elif violation == "prefix":
        ids = [0, 1]
    elif violation == "typed_config":
        cfg = dict(cfg, attention_bias=1)  # Equal by ordinary bool/int equality.
    else:
        cfg = dict(cfg, num_attention_heads=2, num_key_value_heads=2, head_dim=2)
    with pytest.raises(source.SourceError, match="decoder entry"):
        run_decoder(cfg, weights, ids, outputs, FixtureTape("zero"), {})
    assert len(view.owner.calls) == n and view.owner.closes == 1
    with pytest.raises(source.SourceError):
        # Cannot turn a rejected first outputs=1 call into a second epoch.
        run_decoder(cfg, weights, [4, 4, 4, 4], 1, FixtureTape("zero"), {})
    assert len(view.owner.calls) == n


@pytest.mark.parametrize("epoch", [0, 1])
def test_source_cut_repair_correct_embedding_matrix_frames_cannot_fake_gather(epoch):
    cfg, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    n = len(view.owner.calls)
    def drain():
        # Correct ascending full frames, NOT the old wrong-interval witness.
        for _, block in weights["model.embed_tokens.weight"].blocks():
            del block
    if epoch == 0:
        with pytest.raises(source.SourceError, match="reserved for semantic gather"):
            drain()
        assert len(view.owner.calls) == n
    else:
        rotary = {0: ([Q], [0]), 1: ([141637], [220586]),
                  2: ([-109090], [238366]), 3: ([-259520], [36994])}
        def sink(name, value):
            if name == "s1/input.tokens":
                drain()
        with pytest.raises(source.SourceError, match="reserved for semantic gather"):
            run_decoder(cfg, weights, [0, 1, 2], 2, FixtureTape("zero"), rotary, trace_sink=sink)
    with pytest.raises(source.SourceError):
        weights.finish()
    assert view.owner.closes == 1


def test_source_cut_repair_second_actual_decoder_invocation_refused():
    cfg, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    rotary = {0: ([Q], [0]), 1: ([141637], [220586]),
              2: ([-109090], [238366]), 3: ([-259520], [36994])}
    run_decoder(cfg, weights, [0, 1, 2], 2, FixtureTape("zero"), rotary, trace_sink=lambda *args: None)
    n = len(view.owner.calls)
    with pytest.raises(source.SourceError, match="decoder entry"):
        run_decoder(cfg, weights, [4, 4, 4], 2, FixtureTape("zero"), rotary)
    assert len(view.owner.calls) == n and view.owner.closes == 1


def test_source_cut_repair_row_transcript_finish_is_not_decoder_execution():
    cfg, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    # Public-method witness only: no private assignments or decoder invocation.
    # Standalone public markers must not authorize a row-only transcript.
    with pytest.raises(AttributeError):
        weights.begin_decoder(cfg, [4, 2, 4], 2)
    for ids in ([4, 2, 4], [7, 7, 7, 7]):
        weights.gather_embedding(ids)
        for name, _ in geometry.roles():
            if name == "model.embed_tokens.weight":
                continue
            value = weights[name]
            if isinstance(value, PublicMatrix):
                for _, block in value.blocks():
                    del block
            del value
        weights.selection_finished()
    with pytest.raises(AttributeError):
        weights.end_decoder()
    with pytest.raises(source.SourceError, match="unfinished mock source grammar"):
        weights.finish()
    assert view.owner.closes == 1


def test_source_cut_repair_qwen3_literal_distinct_norm_bias_and_projection_shapes():
    cfg = dict(model_type="qwen3", hidden_size=10, intermediate_size=7,
        num_hidden_layers=1, num_attention_heads=6, num_key_value_heads=2,
        head_dim=4, vocab_size=17, attention_bias=True, tie_word_embeddings=False)
    # Literal independent widths: H=10, h*d=24, kv*d=8, d=4; h>kv.
    roles = list(source.Geometry(json.dumps(cfg).encode(), 3).roles())
    p = "model.layers.0."
    assert roles == [
        ("model.embed_tokens.weight", (17, 10)),
        (p+"input_layernorm.weight", (10,)),
        (p+"self_attn.q_proj.bias", (24,)),
        (p+"self_attn.q_proj.weight", (24, 10)),
        (p+"self_attn.q_norm.weight", (4,)),
        (p+"self_attn.k_proj.bias", (8,)),
        (p+"self_attn.k_proj.weight", (8, 10)),
        (p+"self_attn.k_norm.weight", (4,)),
        (p+"self_attn.v_proj.bias", (8,)),
        (p+"self_attn.v_proj.weight", (8, 10)),
        (p+"self_attn.o_proj.weight", (10, 24)),
        (p+"post_attention_layernorm.weight", (10,)),
        (p+"mlp.gate_proj.weight", (7, 10)),
        (p+"mlp.up_proj.weight", (7, 10)),
        (p+"mlp.down_proj.weight", (10, 7)),
        ("model.norm.weight", (10,)),
        ("lm_head.weight", (17, 10))]


def test_source_cut_repair_nonzero_residue_complete_negative_contraction_law():
    # Complete raw sum = -262146. Adding residue 3 then flooring gives -1.
    # Per-product flooring would instead produce -3; toward-zero raw trunc gives
    # -1 without the residue and is separately rejected by the existing witness.
    matrix = PublicMatrix((1, 3), lambda start, stop: array([[1, -1, 1]]))
    tape = Tape({"dot": {"shift": 18, "residues": [3]}})
    assert linear([131073, 131074, -262145], matrix, tape, "dot").tolist() == [-1]
    tape.finish()
    assert (131073//Q + (-131074)//Q + (-262145)//Q) == -3
    assert -262146//Q == -2
    assert (-262146+3)//Q == -1


@pytest.mark.parametrize("returned", ["sentinel", "array", "hostile"])
@pytest.mark.parametrize("close_raises", [False, True])
def test_source_cut_repair_invalid_readinto_return_dies_with_retained_failure(returned, close_raises):
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    owner = view.owner
    read, close = owner.readinto, owner.close
    refs, borrowed = [], []
    class HostileReturn(_SourceSentinel):
        def __eq__(self, other):
            raise AssertionError("must not compare callback objects")
        def __bool__(self):
            raise AssertionError("must not coerce callback objects")
        def __int__(self):
            raise AssertionError("must not convert callback objects")
    def callback(sid, offset, destination):
        read(sid, offset, destination)
        borrowed.append(destination)
        if returned == "array":
            # A separate NumPy view owns an independent buffer export; releasing
            # just destination is insufficient while this rejected return lives.
            value = np.frombuffer(destination, dtype=np.uint8)
        elif returned == "hostile":
            value = HostileReturn()
        else:
            value = _SourceSentinel()
        refs.append(weakref.ref(value))
        return value
    def checked_close():
        for destination in borrowed:
            with pytest.raises(ValueError):
                destination.obj
        assert all(ref() is None for ref in refs)
        close()
        if close_raises:
            marker = _SourceSentinel()
            refs.append(weakref.ref(marker))
            raise LookupError("invalid-return cleanup failure")
    owner.readinto, owner.close = callback, checked_close
    with pytest.raises(source.SourceError, match="short/oversized source readinto") as retained:
        weights.gather_embedding([4, 2, 4])
    gc.collect()
    assert retained.value.__traceback__ is not None and all(ref() is None for ref in refs)
    assert owner.closes == 1 and owner.closed and len(borrowed) == 1
    if close_raises:
        assert isinstance(retained.value.__cause__, LookupError)


@pytest.mark.parametrize("kind", ["exception", "base"])
@pytest.mark.parametrize("context_cycle", [False, True])
@pytest.mark.parametrize("close_raises", [False, True])
def test_source_cut_repair_group_children_retire_payloads_preserve_structure(kind, context_cycle, close_raises):
    try:
        group_type = BaseExceptionGroup
    except NameError:
        pytest.skip("exception groups require Python 3.11")
    _, geometry, view, _ = _source_fixture()
    weights = _open_source(view, geometry)
    owner = view.owner
    read, close = owner.readinto, owner.close
    refs, roots, borrowed = [], [], []
    def child_failure(error_type, message):
        marker = _SourceSentinel()
        refs.append(weakref.ref(marker))
        try:
            raise error_type(message)
        except BaseException as error:
            return error
    def callback(sid, offset, destination):
        read(sid, offset, destination)
        borrowed.append(destination)
        first = child_failure(AssertionError, "nested child assertion")
        second = child_failure(KeyboardInterrupt if kind == "base" else ValueError, "second child")
        inner = ExceptionGroup("inner", [first])
        root = group_type("outer", [inner, second])
        if context_cycle:
            first.__context__ = root  # Finite graph with an explicit back-edge.
        roots.append(root)
        # Not raised inside a child except handler: group->child linkage is the
        # exceptions tuple, not cause/context. Keep the root to test preservation.
        raise root
    def checked_close():
        for destination in borrowed:
            with pytest.raises(ValueError):
                destination.obj
        close()
        if close_raises:
            marker = _SourceSentinel()
            refs.append(weakref.ref(marker))
            raise LookupError("group cleanup failure")
    owner.readinto, owner.close = callback, checked_close
    with pytest.raises(group_type) as retained:
        weights.gather_embedding([4, 2, 4])
    gc.collect()
    root = retained.value
    assert root is roots[0] and root.message == "outer"
    assert type(root) is (BaseExceptionGroup if kind == "base" else ExceptionGroup)
    assert len(root.exceptions) == 2
    inner, second = root.exceptions
    assert type(inner) is ExceptionGroup and inner.message == "inner"
    assert len(inner.exceptions) == 1
    first = inner.exceptions[0]
    assert type(first) is AssertionError and str(first) == "nested child assertion"
    assert type(second) is (KeyboardInterrupt if kind == "base" else ValueError)
    assert str(second) == "second child"
    assert first.__traceback__ is second.__traceback__ is None
    if context_cycle:
        assert first.__context__ is root
    assert root.__traceback__ is not None and all(ref() is None for ref in refs)
    assert owner.closes == 1 and owner.closed and len(borrowed) == 1
    if close_raises:
        assert isinstance(root.__cause__, LookupError)

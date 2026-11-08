"""Explicit bounded C parsing; genuine new-wheel backend, never fake owners."""
import _json
import builtins
import gc
import json
import sys
from types import ModuleType

import pytest

from pllm.runtime import source_io
from pllm.runtime.source_io import SourceReadError, SourceReadPolicy


# Keep the genuine interpreter primitive for comparisons across replacement
# tests. Never infer a universal managed-instance size or physical header delta.
_INTERPRETER_GETSIZEOF = sys.getsizeof


class BudgetExceeded(Exception):
    pass


def bounded(**kwargs):
    return SourceReadPolicy(bounded_json=True, **kwargs)


@pytest.mark.parametrize("raw", [
    b'{"a":0,"b":1,"c":2,"d":3}',
    b'{' + b','.join(b'"' + str(n).encode() + b'":0' for n in range(1000)) + b'}',
    b'[{},[],{},[],{},[]]',
    b'[' + b','.join([b'{}', b'[]'] * 500) + b']',
    b'[{"a":[{"b":[]}]}]',
    b'[{"same":0},{"same":1},{"same":2}]',
    b'{"escaped\\\"key":"\\u263a\\ud83d\\ude00\\\\"}',
    '{"é":"你好😀"}'.encode(),
    b'["' + b'aa\\n' * 1000 + b'"]',
    b'["' + b'aa\\u0100' * 1024 + b'"]',
    b'["' + b'\\ud83d\\ude00' * 1024 + b'"]',
    ('["' + 'a' * 4096 + 'Ā😀' + '"]').encode(),
    b'["\\u0100","\\ud800","\\udfff","\\ud83d\\ude00"]',
])
def test_exact_parse_capacity_and_short_refusal_before_decoder(monkeypatch, raw):
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    census = bounded()._json_census(raw)
    events = []
    loads = source_io._bounded_loads
    def parsed(decoded):
        events.append("scanner")
        return loads(decoded)  # Observes the genuine production C path.
    monkeypatch.setattr(source_io, "_bounded_loads", parsed)
    def admit(size):
        events.append(("admit", size))
        if size > census.parse_peak_bytes - 1:
            raise BudgetExceeded
    def forbidden_pairs(*args, **kwargs):
        pytest.fail("short admission reached C pairs hook")
    with monkeypatch.context() as patch:
        patch.setattr(source_io, "_BOUNDED_OBJECT", forbidden_pairs)
        with pytest.raises(BudgetExceeded):
            bounded(metadata_admit=admit).parse_json(raw)
    assert events == [("admit", census.parse_peak_bytes)]
    events.clear()
    def exact(size):
        assert size <= census.parse_peak_bytes
        events.append(("admit", size))
    def retain(size):
        assert size == census.retained_bytes
        events.append("retain")
    value, actual = bounded(metadata_admit=exact)._parse_json(raw, retain=retain)
    assert actual == census
    assert value == json.loads(raw)
    assert events[0] == ("admit", census.parse_peak_bytes)
    assert events[-1] == "retain" and "scanner" in events


def test_census_counts_occurrences_not_memoized_keys():
    raw = b'[{"x":0},{"x":1},{"x":2}]'
    census = bounded()._json_census(raw)
    assert (census.nodes, census.strings, census.tokens, census.containers,
            census.max_depth) == (10, 3, 3, 4, 2)
    assert census.parse_peak_bytes > census.retained_bytes
    assert census.freeze_bytes == 256 * (census.nodes + census.containers) + 32768 * (2 * census.max_depth + 1)


@pytest.mark.parametrize("raw,kwargs", [
    (b"[] ", {"metadata_bytes": 2}),
    (b"[0,1]", {"inventory_nodes": 2}),
    (b"[" + b",".join([b"0"] * 65536) + b"]", {}),
    (b"[" * 18 + b"]" * 18, {}),
    (bytearray(b"{}"), {}),
    (b"", {}),
])
def test_structural_limits_precede_callback_and_loads(monkeypatch, raw, kwargs):
    def forbidden(*args, **kwargs):
        pytest.fail("structural refusal reached allocation callback/decoder")
    monkeypatch.setattr(source_io.json, "loads", forbidden)
    monkeypatch.setattr(source_io, "_bounded_loads", forbidden)
    monkeypatch.setattr(source_io, "_BOUNDED_OBJECT", forbidden)
    with pytest.raises(SourceReadError):
        bounded(metadata_admit=forbidden, **kwargs).parse_json(raw)


def test_exact_lexical_limits():
    assert bounded(metadata_bytes=2, inventory_nodes=1).parse_json(b"[]") == []
    assert bounded(inventory_nodes=3).parse_json(b"[0,1]") == [0, 1]
    assert bounded().parse_json(b"[" * 17 + b"]" * 17)


@pytest.mark.parametrize("raw", [
    b'{"x":0,"x":1}', b'{"x":NaN}', b'{"x":Infinity}',
    b'{"x":-Infinity}', b'{"x":1e999}', b'{"x":"\xff"}',
    b"{} []", b"[1,]", b"{a:1}", b"", b" \t\r\n", b"1e", b"01", b"+1",
])
def test_strict_json_failures_do_not_transfer_retained_charge(raw):
    retained = []
    with pytest.raises(SourceReadError):
        bounded()._parse_json(raw, retain=retained.append)
    assert retained == []


def test_read_json_return_compatibility(tmp_path):
    raw = '{"é":[{},[],true,null,1.25]}'.encode()
    path = tmp_path / "metadata.json"
    path.write_bytes(raw)
    assert SourceReadPolicy().read_json(path) == json.loads(raw)
    assert bounded().read_json(path) == json.loads(raw)


@pytest.mark.parametrize("raw", [b"\xc2\xa0{}", b"[1\xc3\xa9]", "[−1]".encode()])
def test_nonascii_outside_quotes_refused_before_admission_or_scanner(raw, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unquoted non-ASCII reached admission/scanner")
    monkeypatch.setattr(source_io, "_bounded_loads", forbidden)
    with pytest.raises(SourceReadError, match="non-ASCII"):
        bounded(metadata_admit=forbidden).parse_json(raw)


@pytest.mark.parametrize("raw", [
    b"9" * 1024, b"-" + b"9" * 1023,
    b"0." + b"0" * 1021 + b"1", b"1e-" + b"0" * 1020 + b"1",
])
def test_scalar_1024_counts_first_minus_decimal_exponent_and_uses_c_backend(raw):
    census = bounded()._json_census(raw)
    assert census.longest_token == 1024
    assert bounded().parse_json(raw) == json.loads(raw)


@pytest.mark.parametrize("raw", [
    b"9" * 1025, b"-" + b"9" * 1024, b"0." + b"0" * 1023,
    b"1e-" + b"0" * 1022, b"x" * 1025, b"0" * 1023 + b".e",
])
def test_every_scalar_1025_including_malformed_refused_before_scanner(raw, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("oversized scalar reached admission/scanner")
    monkeypatch.setattr(source_io, "_bounded_loads", forbidden)
    with pytest.raises(SourceReadError, match="1024-byte"):
        bounded(metadata_admit=forbidden).parse_json(raw)


@pytest.mark.parametrize("prefix,suffix", [
    (b"[", b"]"), (b'{"n":', b"}"), (b"[0,", b"]"),
    (b"[", b",0]"), (b'{"n":', b',"m":0}'),
])
def test_scalar_first_byte_after_openers_and_delimiters_exact_boundary(prefix, suffix, monkeypatch):
    raw = prefix + b"9" * 1024 + suffix
    assert bounded()._json_census(raw).longest_token == 1024
    assert bounded().parse_json(raw) == json.loads(raw)
    def forbidden(*args, **kwargs):
        pytest.fail("opener/delimiter scalar byte 1025 reached scanner/admission")
    monkeypatch.setattr(source_io, "_bounded_loads", forbidden)
    with pytest.raises(SourceReadError, match="1024-byte"):
        bounded(metadata_admit=forbidden).parse_json(prefix + b"9" * 1025 + suffix)


def test_scalar_counter_resets_at_delimiter_not_document_total():
    raw = b"[" + b"9" * 1024 + b"," + b"8" * 1024 + b"]"
    census = bounded()._json_census(raw)
    assert census.longest_token == 1024 and census.tokens == 2
    assert bounded().parse_json(raw) == json.loads(raw)


def test_malformed_scalar_1024_admitted_then_strict_c_scanner_refuses():
    observed = []
    with pytest.raises(SourceReadError, match="invalid strict"):
        bounded(metadata_admit=observed.append).parse_json(b"x" * 1024)
    assert len(observed) == 1


def test_ascii_json_whitespace_start_end_and_no_suffix_acceptance():
    assert bounded().parse_json(b" \t\r\n{\"a\":1.25}\n\t ") == {"a": 1.25}
    with pytest.raises(SourceReadError):
        bounded().parse_json(b" {}\x0b")


def test_duplicate_errors_never_interpolate_input_key():
    secret = b"private-input-key" * 100
    raw = b'{"' + secret + b'":0,"' + secret + b'":1}'
    with pytest.raises(SourceReadError) as caught:
        bounded().parse_json(raw)
    assert str(caught.value) == "invalid strict source JSON"
    assert str(caught.value.__cause__) == "duplicate source JSON key"
    assert "private-input-key" not in str(caught.value.__cause__)


def test_captured_backend_ignores_later_stdlib_selection_and_loads_replacement(monkeypatch):
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    original = source_io._BOUNDED_BACKEND
    def forbidden(*args, **kwargs):
        pytest.fail("bounded parser used mutable stdlib decoder selection")
    monkeypatch.setattr(json, "loads", forbidden)
    monkeypatch.setattr(json, "JSONDecoder", forbidden)
    monkeypatch.setattr(json.decoder, "JSONDecoder", forbidden)
    monkeypatch.setattr(json.scanner, "make_scanner", json.scanner.py_make_scanner)
    monkeypatch.setattr(_json, "make_scanner", json.scanner.py_make_scanner)
    assert bounded().parse_json(b'{"a":[1,1.25,"\\u0100"]}') == {"a": [1, 1.25, "Ā"]}
    assert source_io._BOUNDED_BACKEND is original
    assert source_io._capture_bounded_backend() is None  # Genuine Python fallback is unsupported.


def test_captured_native_converter_not_later_module_replacement(monkeypatch):
    from pllm import _native
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    def forbidden(*args, **kwargs):
        pytest.fail("bounded parser used later module converter replacement")
    monkeypatch.setattr(_native, "source_json_f64", forbidden)
    assert bounded().parse_json(b"1.25") == 1.25
    assert source_io._capture_bounded_backend() is None


def test_missing_converter_capture_is_unavailable_legacy_still_works(monkeypatch):
    from pllm import _native
    monkeypatch.delattr(_native, "source_json_f64", raising=False)
    assert source_io._capture_bounded_backend() is None
    monkeypatch.setattr(source_io, "_BOUNDED_BACKEND", None)
    def forbidden(*args, **kwargs):
        pytest.fail("unavailable bounded backend reached decode/admission")
    monkeypatch.setattr(source_io, "_bounded_loads", forbidden)
    with pytest.raises(SourceReadError, match="backend unavailable"):
        bounded(metadata_admit=forbidden).parse_json(b'"\xff"')
    assert SourceReadPolicy().parse_json(b'{"a":1.25}') == {"a": 1.25}


def test_legacy_default_keeps_loads_and_unbounded_scalar_behavior(monkeypatch):
    calls = []
    loads = json.loads
    def observed(*args, **kwargs):
        calls.append(True)
        return loads(*args, **kwargs)
    monkeypatch.setattr(json, "loads", observed)
    raw = b"9" * 1025
    assert SourceReadPolicy().parse_json(raw) == int(raw)
    assert calls == [True]
    with pytest.raises(SourceReadError, match="1024-byte"):
        bounded().parse_json(raw)


def test_legacy_callback_amounts_remain_historical_not_new_c_certificate():
    raw = b'{"a":[1,1.25]}'
    census = SourceReadPolicy()._json_census(raw)
    calls, retained = [], []
    value, _ = SourceReadPolicy(metadata_admit=calls.append)._parse_json(raw, retain=retained.append)
    old_retained = 33 + 5 * len(raw) + 256 * census.nodes + 128 * census.containers
    old_peak = old_retained + 80 + 8 * len(raw) + 384 * census.nodes + 128 * census.containers + 8192 * (census.max_depth + 1)
    assert value == {"a": [1, 1.25]}
    assert calls == [old_peak] and retained == [old_retained]
    assert old_peak != census.parse_peak_bytes


def test_exact_named_component_formula_and_quote_envelope():
    census = source_io._JSONCensus(2048, 32, 32, 32, 32, 2, 1024)
    assert census.integer_bytes == 580
    assert census.fixed_control_bytes == 8192
    assert census.retained_bytes == 30753
    assert census.freeze_bytes == 256 * 64 + 32768 * 5
    assert (census.decoded_bytes, census.writer_bytes, census.memo_bytes,
            census.hook_pairs_bytes, census.numeric_bytes, census.parse_frame_bytes) == (
                8272, 24832, 8448, 12288, 5512, 524288)
    assert census.parse_peak_bytes == 622585
    assert census.parse_peak_bytes + census.retained_bytes == 653338 < 8 << 20


def test_actual_fixed_control_inventory_and_actual_code_slots():
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    factory, converter = source_io._BOUNDED_BACKEND
    inventory = source_io._bounded_control_inventory(factory, converter)
    assert sum(size for _, size in inventory) <= 8192, inventory
    assert {"context", "scanner", "census", "converter", "fixed_hooks", "counter_slots_64"} <= {
        name for name, _ in inventory}
    sizes = dict(inventory)
    assert sizes["cached_import_empty_fromlist"] == sys.getsizeof([])
    assert sizes["cached_import_five_args"] == sys.getsizeof((None, None, None, None, 0))
    assert sizes["cached_import_names"] == sum(sys.getsizeof(value)
                                               for value in ("json.decoder", "json", "JSONDecodeError"))
    assert sizes["error_constructor_three_args"] == sys.getsizeof((None, None, 0))
    assert sizes["error_initializer_self_args"] == sys.getsizeof((None, None, None, 0))
    assert {"JSONDecodeError", "JSONDecodeError_args", "JSONDecodeError_attributes",
            "error_bound_initializer", "error_base_initializer", "error_initializer_result",
            "cached_host_anchors"} <= sizes.keys()
    assert len(source_io._ScannerContext.__slots__) == 8
    assert source_io._reported_owner_bytes(converter("0.0")) == 24
    for value in (source_io._ScannerContext(factory, converter), factory(source_io._ScannerContext(factory, converter)),
                  (None, sys.maxsize), SourceReadError("invalid source JSON")):
        assert source_io._reported_owner_bytes(value) == sys.getsizeof(value)
    slots = source_io._code_slot_inventory(source_io._SOURCE_JSON_FRAME_FUNCTIONS)
    assert slots and all(count <= 137 for _, count in slots), slots
    assert source_io._SOURCE_JSON_FRAMES_SUPPORTED


def test_inventory_size_matches_genuine_interpreter_for_concrete_internal_owners():
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    assert source_io._INVENTORY_GETSIZEOF is _INTERPRETER_GETSIZEOF
    factory, converter = source_io._BOUNDED_BACKEND
    context = source_io._ScannerContext(factory, converter)
    error = json.decoder.JSONDecodeError("invalid source JSON", "", 0)
    empty_dict, atomic_dict, empty_tuple = {}, {"a": 1}, ()
    assert not gc.is_tracked(empty_dict)
    assert not gc.is_tracked(atomic_dict)
    assert not gc.is_tracked(empty_tuple)
    values = (
        context, factory(context), source_io._JSONCensus(1, 1, 0, 1, 0, 0, 1),
        empty_dict, atomic_dict, {"a": []}, [], [None], empty_tuple, (None,), ([],),
        source_io._BOUNDED_OBJECT, source_io._BOUNDED_CONSTANT, error.__init__,
        ValueError.__init__, converter, converter("0.0"), "".startswith,
        source_io._CACHED_JSON_ERROR_PATH[14], source_io._CACHED_JSON_ERROR_PATH[17],
        source_io._ERROR_ABSENT, None, sys.maxsize,
    )
    for value in values:
        assert source_io._reported_owner_bytes(value) == _INTERPRETER_GETSIZEOF(value), type(value)


def test_exception_shallow_size_preheaders_and_separate_materialized_dicts():
    errors = (
        SourceReadError("invalid source JSON"), json.decoder.JSONDecodeError("invalid source JSON", "", 0),
        ValueError("invalid source JSON"), UnicodeDecodeError("utf-8", b"", 0, 1, "invalid source JSON"),
    )
    for error in errors:
        before = _INTERPRETER_GETSIZEOF(error)
        assert source_io._reported_owner_bytes(error) == before
        # The removed rule omitted managed preheaders on applicable ABIs.
        # Compare actual physical delta, not an environment-specific 104 value.
        old_gc_only = type(error).__sizeof__(error) + (16 if type(error).__flags__ & (1 << 14) else 0)
        delta = before - old_gc_only
        assert delta >= 0
        if delta:
            assert source_io._reported_owner_bytes(error) > old_gc_only
            assert source_io._reported_owner_bytes(error) - old_gc_only == delta
        attributes = error.__dict__  # May materialize previously absent storage.
        assert source_io._reported_owner_bytes(error) == _INTERPRETER_GETSIZEOF(error)
        assert source_io._reported_owner_bytes(attributes) == _INTERPRETER_GETSIZEOF(attributes)
        assert source_io._reported_owner_bytes(error.args) == _INTERPRETER_GETSIZEOF(error.args)
        attributes["inventory_probe"] = None
        separate = (source_io._reported_owner_bytes(error) + source_io._reported_owner_bytes(attributes)
                    + source_io._reported_owner_bytes(error.args))
        assert separate == (_INTERPRETER_GETSIZEOF(error) + _INTERPRETER_GETSIZEOF(attributes)
                            + _INTERPRETER_GETSIZEOF(error.args))


def test_later_getsizeof_replacement_cannot_redirect_or_disable_captured_inventory(monkeypatch):
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    captured = source_io._INVENTORY_GETSIZEOF
    before = source_io._bounded_control_inventory(*source_io._BOUNDED_BACKEND)
    invoked = []
    def forbidden(*args, **kwargs):
        invoked.append(True)
        raise AssertionError("inventory used later sys.getsizeof replacement")
    with monkeypatch.context() as patch:
        patch.setattr(sys, "getsizeof", forbidden)
        assert source_io._capture_inventory_getsizeof() is None
        assert source_io._INVENTORY_GETSIZEOF is captured
        assert source_io._bounded_control_inventory(*source_io._BOUNDED_BACKEND) == before
        assert source_io._capture_bounded_backend() is not None
        assert bounded().parse_json(b'{"a":1.25}') == {"a": 1.25}
    assert invoked == []


@pytest.mark.parametrize("missing", [False, True])
def test_unsupported_startup_sizing_capture_refuses_before_inventory_legacy_unchanged(monkeypatch, missing):
    invoked = []
    def wrapper(value):
        invoked.append("wrapper")
        return _INTERPRETER_GETSIZEOF(value)
    wrapper.__module__, wrapper.__name__, wrapper.__self__ = "sys", "getsizeof", sys
    def forbidden(*args, **kwargs):
        invoked.append("inventory")
        raise AssertionError("unsupported startup sizing reached inventory")
    with monkeypatch.context() as patch:
        if missing:
            patch.delattr(sys, "getsizeof")
        else:
            patch.setattr(sys, "getsizeof", wrapper)
        captured = source_io._capture_inventory_getsizeof()
        assert captured is None
        # Negative startup configuration only; no fabricated successful backend.
        patch.setattr(source_io, "_INVENTORY_GETSIZEOF", captured)
        patch.setattr(source_io, "_bounded_control_inventory", forbidden)
        backend = source_io._capture_bounded_backend()
        assert backend is None
        patch.setattr(source_io, "_BOUNDED_BACKEND", backend)
        with pytest.raises(SourceReadError, match="backend unavailable"):
            bounded().parse_json(b"{}")
        assert SourceReadPolicy().parse_json(b'{"a":1.25}') == {"a": 1.25}
    assert invoked == []


def test_oversized_code_slot_configuration_fails_closed(monkeypatch):
    # Negative support configuration only, never fake-positive parser authority.
    monkeypatch.setattr(source_io, "_SOURCE_JSON_FRAMES_SUPPORTED", False)
    assert source_io._capture_bounded_backend() is None
    with pytest.raises(SourceReadError, match="backend unavailable"):
        bounded().parse_json(b"{}")


@pytest.mark.parametrize("raw", [b'{"a" 1}', b'{"a":1,}', b'["bad\\q"]'])
def test_genuine_c_syntax_error_uses_supported_cached_error_class(raw):
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    assert source_io._cached_json_error_path_supported()
    with pytest.raises(SourceReadError, match="invalid strict") as caught:
        bounded().parse_json(raw)
    assert type(caught.value.__cause__) is source_io._CACHED_JSON_ERROR_PATH[9]


@pytest.mark.parametrize("change", [
    "error_class", "initializer", "initializer_code", "initializer_defaults", "initializer_global",
    "constructor_store", "builtin_import", "builtin_value_error", "source_builtins", "decoder_builtins",
    "modules_replaced", "decoder_missing", "decoder_replaced", "parent_missing", "parent_replaced",
    "parent_attribute", "decoder_spec_missing", "decoder_spec_replaced", "parent_spec_missing",
    "decoder_initializing", "parent_initializing", "spec_state_replaced", "spec_property", "module_hook",
])
def test_unsupported_cached_error_host_configuration_refuses_before_scanner(change, monkeypatch):
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    assert source_io._cached_json_error_path_supported()
    policy = bounded()
    decoder = json.decoder
    error_class = decoder.JSONDecodeError
    initializer = error_class.__init__
    original_import = builtins.__import__
    entered, hooks = [], []
    def scanner_forbidden(*args, **kwargs):
        entered.append(True)
        raise AssertionError("unsupported configuration reached scanner")
    def hook_forbidden(*args, **kwargs):
        hooks.append(True)
        raise AssertionError("unsupported host callback invoked")
    def other_initializer(self, msg, doc, pos):
        return None  # No closure cells: permits real initializer code replacement.
    failure = None
    with monkeypatch.context() as patch:
        patch.setattr(source_io, "_bounded_loads", scanner_forbidden)
        if change == "error_class":
            patch.setattr(decoder, "JSONDecodeError", ValueError)
        elif change == "initializer":
            patch.setattr(error_class, "__init__", other_initializer)
        elif change == "initializer_code":
            patch.setattr(initializer, "__code__", other_initializer.__code__)
        elif change == "initializer_defaults":
            patch.setattr(initializer, "__defaults__", (None,))
        elif change == "initializer_global":
            patch.setitem(decoder.__dict__, "ValueError", RuntimeError)
        elif change == "constructor_store":
            patch.setattr(error_class, "msg", property(hook_forbidden, hook_forbidden), raising=False)
        elif change == "builtin_import":
            patch.setattr(builtins, "__import__", hook_forbidden)
        elif change == "builtin_value_error":
            patch.setattr(builtins, "ValueError", RuntimeError)
        elif change == "source_builtins":
            patch.setitem(source_io.__dict__, "__builtins__", {"__import__": original_import})
        elif change == "decoder_builtins":
            patch.setitem(decoder.__dict__, "__builtins__", {"__import__": original_import})
        elif change == "modules_replaced":
            patch.setattr(sys, "modules", {})
        elif change == "decoder_missing":
            patch.delitem(sys.modules, "json.decoder")
        elif change == "decoder_replaced":
            patch.setitem(sys.modules, "json.decoder", ModuleType("json.decoder"))
        elif change == "parent_missing":
            patch.delitem(sys.modules, "json")
        elif change == "parent_replaced":
            patch.setitem(sys.modules, "json", ModuleType("json"))
        elif change == "parent_attribute":
            patch.setattr(json, "decoder", ModuleType("json.decoder"))
        elif change == "decoder_spec_missing":
            patch.delitem(decoder.__dict__, "__spec__")
        elif change == "decoder_spec_replaced":
            patch.setattr(decoder, "__spec__", None)
        elif change == "parent_spec_missing":
            patch.delitem(json.__dict__, "__spec__")
        elif change == "decoder_initializing":
            patch.setitem(decoder.__spec__.__dict__, "_initializing", True)
        elif change == "parent_initializing":
            patch.setitem(json.__spec__.__dict__, "_initializing", True)
        elif change == "spec_state_replaced":
            patch.setattr(decoder.__spec__, "__dict__", {"name": "json.decoder", "_initializing": False})
        elif change == "spec_property":
            patch.setattr(type(decoder.__spec__), "_initializing", property(hook_forbidden), raising=False)
        else:
            patch.setattr(decoder, "__getattr__", hook_forbidden, raising=False)
        # No pytest context/import calls while host import/module facts are
        # changed. Restore first, then inspect failure; never prime a good state
        # by capturing a Python import hook as if it were the genuine builtin.
        try:
            policy.parse_json(b'{"a" 1}')
        except SourceReadError as exc:
            failure = str(exc)
    assert failure == "bounded source JSON cached error configuration unavailable"
    assert entered == [] and hooks == []
    assert source_io._cached_json_error_path_supported()


def test_admission_callback_host_change_rechecked_before_decode(monkeypatch):
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    entered = []
    def forbidden(*args, **kwargs):
        entered.append(True)
        raise AssertionError("changed admission host reached scanner")
    def admit(size):
        monkeypatch.setattr(json.decoder, "JSONDecodeError", ValueError)
    monkeypatch.setattr(source_io, "_bounded_loads", forbidden)
    with pytest.raises(SourceReadError, match="cached error configuration"):
        bounded(metadata_admit=admit).parse_json(b'"\xff"')
    assert entered == []  # Configuration refusal, not the invalid UTF-8 error.


def test_scanner_entry_itself_rechecks_cached_error_configuration(monkeypatch):
    assert source_io._BOUNDED_BACKEND is not None, "new native wheel required"
    monkeypatch.setattr(json.decoder, "JSONDecodeError", ValueError)
    with pytest.raises(SourceReadError, match="cached error configuration"):
        source_io._bounded_loads('{"a" 1}')


def test_python_import_wrapper_never_captured_as_supported_startup(monkeypatch):
    original_import = builtins.__import__
    state = backend = "not attempted"
    invoked = []
    def wrapper(*args, **kwargs):
        invoked.append(True)
        return original_import(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", wrapper)
        state = source_io._capture_cached_json_error_path()
        backend = source_io._capture_bounded_backend()
    assert state is None and backend is None and invoked == []


@pytest.mark.parametrize("size", [-1, True, sys.maxsize + 1])
def test_checked_address_arithmetic_precedes_callback(size):
    observed = []
    with pytest.raises(SourceReadError, match="addressable"):
        SourceReadPolicy(metadata_admit=observed.append)._admit(size)
    assert observed == []

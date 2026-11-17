"""Neutral lexical loop, loaded by file without importing the pllm package."""
import builtins
import importlib.util
from pathlib import Path
import sys

import pytest


def load_census():
    path = Path(__file__).resolve().parents[1] / "python/pllm/_json_census.py"
    spec = importlib.util.spec_from_file_location("neutral_census_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scanner = load_census()


def test_neutral_error_borrows_builtin_with_fixed_reason_only():
    assert scanner.JSONCensusError is ValueError
    assert scanner.json_census.__doc__
    with pytest.raises(ValueError) as caught:
        scanner.json_census(b"\xff")
    assert type(caught.value) is ValueError
    assert caught.value.args == ("ascii",)


def test_direct_file_load_has_no_package_native_decoder_or_numpy_dependency(monkeypatch):
    before = set(sys.modules)
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("pllm", "numpy", "json", "_json"):
            pytest.fail("neutral census imported decoder/package")
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert load_census().json_census(b"[0]") == (3, 2, 0, 1, 1, 1, 1)
    assert not any(name == "pllm" or name.startswith("pllm.") for name in set(sys.modules)-before)


@pytest.mark.parametrize("raw,expected", [
    (b"[]", (2, 1, 0, 0, 1, 1, 0)),
    (b'[0,1]', (5, 3, 0, 2, 1, 1, 1)),
    (b'{"x":0,"x":1}', (13, 5, 2, 2, 1, 1, 1)),
    (b'"a\\\"b\\\\c"', (9, 1, 1, 0, 0, 0, 0)),
    ('["é😀",-1.2e+3]'.encode(), (18, 3, 1, 1, 1, 1, 7)),
])
def test_bytes_and_explicit_readonly_view_counter_parity(raw, expected):
    assert scanner.json_census(raw) == expected
    for backing in (raw, bytearray(raw)):
        with memoryview(backing).toreadonly() as view:
            assert scanner.json_census(view) == expected


@pytest.mark.parametrize("raw,kwargs,reason", [
    (b"[] ", {"byte_limit": 2}, "bytes"),
    (b"[0,1]", {"node_limit": 2}, "structure"),
    (b"[[]]", {"depth_limit": 1}, "structure"),
    (b"[123]", {"token_limit": 2}, "scalar"),
    (b"]", {}, "nesting"),
    (b"[\xff]", {}, "ascii"),
    (b"", {}, "bytes"),
    (b"[\xff]", {"node_limit": 1}, "ascii"),
    (b"123", {"node_limit": 1, "token_limit": 2}, "scalar"),
])
def test_refusal_reasons_and_original_order(raw, kwargs, reason):
    with pytest.raises(scanner.JSONCensusError) as caught:
        scanner.json_census(raw, **kwargs)
    assert caught.value.args == (reason,)


@pytest.mark.parametrize("raw,kwargs", [
    (b"[]", {"byte_limit": 2, "node_limit": 1, "depth_limit": 1}),
    (b"[0,1]", {"node_limit": 3}),
    (b"[[]]", {"depth_limit": 2}),
    (b"123", {"token_limit": 3}),
    (b"[" * 17 + b"]" * 17, {}),
    (b"[" + b",".join([b"0"] * 65535) + b"]", {}),
    (b"9" * 1024, {}),
])
def test_exact_ceilings(raw, kwargs):
    assert scanner.json_census(raw, **kwargs)[0] == len(raw)


@pytest.mark.parametrize("raw", [
    b"[" * 18, b"[" + b",".join([b"0"] * 65536) + b"]",
    b"9" * 1025,
])
def test_one_over_hard_ceilings(raw):
    with pytest.raises(scanner.JSONCensusError):
        scanner.json_census(raw)


@pytest.mark.parametrize("prefix,suffix", [(b"[", b"]"), (b'{"a":', b"}"),
                                           (b"[0,", b",0]")])
def test_delimiters_reset_first_scalar_byte(prefix, suffix):
    assert scanner.json_census(prefix + b"9" * 1024 + suffix)[6] == 1024
    with pytest.raises(scanner.JSONCensusError, match="scalar"):
        scanner.json_census(prefix + b"9" * 1025 + suffix)


@pytest.mark.parametrize("raw", [b"[", b'"unterminated\\', b"[}", b"01", b"NaN",
                                 b'{"x":0,"x":1}', b"[1,]", b" "])
def test_not_another_grammar(raw):
    assert len(scanner.json_census(raw)) == 7


def test_legacy_scalar_and_nonascii_are_explicit_options():
    assert scanner.json_census(b"9" * 1025, token_limit=None)[6] == 1025
    assert scanner.json_census(b"\xff", ascii_only=False)[3] == 1
    assert scanner.json_census('"é😀"'.encode())[2] == 1


def test_explicit_contiguous_subview_scans_only_selected_bytes():
    backing = bytearray(b"x[0]x")
    with memoryview(backing).toreadonly()[1:-1] as view:
        assert scanner.json_census(view) == scanner.json_census(b"[0]")


@pytest.mark.parametrize("value", [0, 1, None])
def test_ascii_mode_requires_exact_bool(value):
    with pytest.raises(scanner.JSONCensusError, match="policy"):
        scanner.json_census(b"0", ascii_only=value)


@pytest.mark.parametrize("name,ceiling", [("byte_limit", 16_777_216), ("node_limit", 65_536),
                                         ("depth_limit", 17), ("token_limit", 1024)])
@pytest.mark.parametrize("bad", [True, 0, -1, 1.0, float("inf"), float("nan"), "1"])
def test_policy_requires_positive_exact_ints(name, ceiling, bad):
    with pytest.raises(scanner.JSONCensusError, match="policy"):
        scanner.json_census(b"0", **{name: bad})


@pytest.mark.parametrize("name,ceiling", [("byte_limit", 16_777_216), ("node_limit", 65_536),
                                         ("depth_limit", 17), ("token_limit", 1024)])
def test_policy_cannot_raise_caps(name, ceiling):
    with pytest.raises(scanner.JSONCensusError, match="policy"):
        scanner.json_census(b"0", **{name: ceiling+1})


@pytest.mark.parametrize("kind", ["bytearray", "list", "iterator", "writable", "stride",
                                  "signed", "two_dimensional", "released", "subclass",
                                  "foreign_backing", "subclass_backing"])
def test_unsupported_input_never_becomes_arbitrary_iterator(kind):
    from array import array
    class BytesSubclass(bytes):
        pass

    raw = {"bytearray": lambda: bytearray(b"[]"), "list": lambda: [91, 93],
           "iterator": lambda: iter([91, 93]), "writable": lambda: memoryview(bytearray(b"[]")),
           "stride": lambda: memoryview(b"[ ] ")[::2],
           "signed": lambda: memoryview(b"[]").cast("b"),
           "two_dimensional": lambda: memoryview(b"[]").cast("B", (1, 2)),
           "released": lambda: memoryview(b"[]"), "subclass": lambda: BytesSubclass(b"[]"),
           "foreign_backing": lambda: memoryview(array("B", [91, 93])).toreadonly(),
           "subclass_backing": lambda: memoryview(BytesSubclass(b"[]"))}[kind]()
    if kind == "released":
        raw.release()
    with pytest.raises(scanner.JSONCensusError, match="bytes"):
        scanner.json_census(raw)

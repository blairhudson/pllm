"""Independent byte-language, sampler and native record lifecycle gates."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("finite_language_boundary", ROOT / "scripts/finite_language_boundary.py")
grammar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grammar
SPEC.loader.exec_module(grammar)
SPEC = importlib.util.spec_from_file_location("public_grammar_probe", ROOT / "scripts/probe_public_grammar_boundary.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def tokenizer(pieces=(b"a", b"b", b"ab", b"ba", b"aba", b"x")):
    alphabet = {byte: char for char, byte in grammar.byte_alphabet().items()}
    vocab = {"".join(alphabet[byte] for byte in value): i for i, value in enumerate(pieces)}
    return grammar.canonical({"model": {"type": "BPE", "vocab": vocab}, "decoder": {"type": "ByteLevel"},
                              "added_tokens": [{"id": len(vocab), "special": True, "content": "<eos>"}]})


def test_all_generated_segmentations_not_only_canonical_encodings():
    raw = tokenizer()
    language = grammar.compile_language(raw, ["ab", "aba"], vocabulary=7, eos=6)
    pieces = grammar.tokenizer_pieces(raw, 7, 6)
    accepted = []
    for count in range(1, 5):
        for ids in itertools.product(range(6), repeat=count):
            expected = b"".join(pieces[i] for i in ids) in (b"ab", b"aba")
            state = 0
            try:
                for i, token in enumerate(ids + (6,)):
                    state = language.advance(state, token, 5 - i)
                actual = state == -1
            except ValueError:
                actual = False
            assert actual == expected
            if actual:
                accepted.append(ids)
    assert (2,) in accepted and (0, 1) in accepted and (0, 3) in accepted
    assert set(language.union) == {0, 1, 2, 3, 4, 6}


def test_dead_end_tokens_pruned_and_missing_string_rejected():
    raw = tokenizer((b"a", b"ab"))
    language = grammar.compile_language(raw, ["ab"], vocabulary=3, eos=2)
    assert language.union == (1, 2)  # 'a' alone cannot finish: no 'b' token.
    with pytest.raises(ValueError, match="represented"):
        grammar.compile_language(raw, ["b"], vocabulary=3, eos=2)


def test_short_string_can_stop_or_continue_and_eos_counts_toward_cap():
    language = grammar.compile_language(tokenizer(), ["ab", "aba"], vocabulary=7, eos=6)
    assert language.allowed(0, 2) == (2, 4)
    with pytest.raises(ValueError, match="budget"):
        language.allowed(0, 1)
    state = language.advance(0, 2, 3)
    assert set(language.allowed(state, 2)) == {0, 6}
    assert language.allowed(state, 1) == (6,)
    assert language.advance(state, 6, 1) == -1
    with pytest.raises(ValueError):
        language.advance(-1, 6, 1)


def test_byte_fragments_inside_utf8_are_not_replacement_characters():
    raw = tokenizer((b"\xc3", b"\xa9", "é".encode()))
    language = grammar.compile_language(raw, ["é"], vocabulary=4, eos=3)
    assert language.advance(language.advance(0, 0, 3), 1, 2) == language.advance(0, 2, 3)


def test_nonspecial_added_token_has_original_byte_decoder_semantics():
    value = json.loads(tokenizer())
    value["added_tokens"].append({"id": 7, "content": "<tool>", "special": False})
    language = grammar.compile_language(grammar.canonical(value), ["<tool>"], vocabulary=8, eos=6)
    assert language.union == (6, 7)
    assert language.advance(language.advance(0, 7, 2), 6, 1) == -1


@pytest.mark.parametrize("change", ["decoder", "suffix", "duplicate_id", "empty", "nonbyte", "special", "eos", "large"])
def test_incompatible_tokenizer_rejects(change):
    value = json.loads(tokenizer())
    if change == "decoder": value["decoder"]["type"] = "WordPiece"
    if change == "suffix": value["model"]["end_of_word_suffix"] = "</w>"
    if change == "duplicate_id": value["model"]["vocab"]["b"] = 0
    if change == "empty": value["model"]["vocab"][""] = 10
    if change == "nonbyte": value["model"]["vocab"]["漢"] = 10
    if change == "special": value["added_tokens"][0]["special"] = False
    if change == "eos": value["added_tokens"] = []
    if change == "large": value["model"]["vocab"]["x" * 4097] = 10
    with pytest.raises(ValueError):
        grammar.compile_language(grammar.canonical(value), ["ab"], vocabulary=16, eos=6)


@pytest.mark.parametrize("words,visibility", [([], "public"), ([""], "public"), (["a", "a"], "public"),
                                             (["a" * 257], "public"), (["ab"], "private")])
def test_language_resource_and_public_contract_rejects(words, visibility):
    with pytest.raises(ValueError):
        grammar.compile_language(tokenizer(), words, vocabulary=7, eos=6, visibility=visibility)


def test_dense_original_sampler_preserves_global_tie_order_rng_and_masks():
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime
    language = grammar.compile_language(tokenizer(), ["ab", "aba"], vocabulary=7, eos=6)
    # Include exactly tied logits and a huge excluded value, across greedy/nucleus draws.
    for full in (np.zeros(7, np.float32), np.array([1, 2, 3, 4, 5, 10000, 0], np.float32)):
        for state in range(len(language.prefixes)):
            oracle = np.full(7, -np.inf, np.float32)
            allowed = list(language.allowed(state, 8))
            oracle[allowed] = full[allowed]
            actual = grammar.dense_restricted(full[list(language.union)], language.union, language, state, 8)
            np.testing.assert_array_equal(actual.view("u4"), oracle.view("u4"))
            for seed, (temperature, top_p) in itertools.product(range(8), ((0, 1), (1, 1), (0.7, 0.75))):
                a, b = [SimpleNamespace(rng=np.random.default_rng(seed)) for _ in range(2)]
                assert MaskedTransformerClientRuntime.sample(a, oracle, temperature=temperature, top_p=top_p) == MaskedTransformerClientRuntime.sample(b, actual, temperature=temperature, top_p=top_p)
                assert a.rng.bit_generator.state == b.rng.bit_generator.state
    with pytest.raises(ValueError):
        grammar.dense_restricted(np.zeros(len(language.union), np.float64), language.union, language, 0, 8)
    with pytest.raises(ValueError):
        grammar.dense_restricted(np.zeros(len(language.union), np.float32), language.union[::-1], language, 0, 8)


def test_artifact_actual_byte_counts_and_identity_changes(tmp_path):
    import zlib
    p = tmp_path / "weights"
    p.write_bytes(b"abc" * 10)
    header = {"shape": [3, 10]}
    bill = probe.artifact_lengths(header, [p])
    base = 12 + len(grammar.canonical(header)) + 8
    assert bill["raw_framed_bytes"] == base + 30
    assert bill["zlib_framed_bytes"] == base + len(zlib.compress(b"abc" * 10, 6))
    lock = {"weights": {"bytes": 30, "sha256": probe.digest(p)}}
    probe.verify_files(tmp_path, lock)
    p.write_bytes(b"abd" * 10)
    with pytest.raises(ValueError, match="changed"):
        probe.verify_files(tmp_path, lock)


def test_each_native_worker_receives_one_share_and_retires_queries(tmp_path):
    import hashlib
    import struct
    from pllm import _native
    table = bytes(range(60))  # ten rows: two i8 codes plus four opaque scale bytes.
    (tmp_path / "records.bin").write_bytes(table)
    descriptor = hashlib.sha256(b"pllm/public-page-table/v1\0" + struct.pack("<QQ", 10, 6) + table).hexdigest()
    public = {"shape": [10, 2], "descriptor": descriptor,
              "files": {"records.bin": {"bytes": 60, "sha256": probe.digest(tmp_path / "records.bin")}}}
    query = _native.private_page_issue(bytes.fromhex(descriptor), 10, 6, 7)
    results = [probe.party({"root": str(tmp_path), "public": public, "party": party,
                            "keys": [query[party].hex()]}) for party in (0, 1)]
    assert all(row["replay_rejected"] for row in results)
    assert query[2].decode(*[bytes.fromhex(row["replies"][0]) for row in results]) == table[42:48]
    with pytest.raises(ValueError):
        query[2].decode(b"", b"")
    wrong = copy.deepcopy(public)
    wrong["descriptor"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor"):
        probe.party({"root": str(tmp_path), "public": wrong, "party": 0, "keys": [query[0].hex()]})

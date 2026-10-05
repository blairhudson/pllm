"""Research gates: exact tokenizer semantics, ciphertext binding and cost scope."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from cryptography.exceptions import InvalidTag
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tokenizer_probe = load("client_offload_tokenizer")
sealed_probe = load("client_offload_sealed")


@pytest.fixture
def compiled(tmp_path):
    reference = Tokenizer(models.BPE())
    reference.normalizer = normalizers.NFC()
    reference.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    reference.decoder = decoders.ByteLevel()
    reference.train_from_iterator(["hello hello hello café cafe\u0301 中文 🙂", "  spaces\n\t12345"] * 3,
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=[AddedToken("<special>", special=True, normalized=False)]))
    reference.add_tokens([AddedToken("<added>", normalized=False)])
    path = tmp_path / "source.json"
    reference.save(str(path))
    root = tmp_path / "index"
    tokenizer_probe.compile_public_tokenizer(path, root)
    digest = tokenizer_probe.file_digest(root / "contract.json")
    return reference, root, digest


def test_public_bpe_index_exact_ids_and_decoded_bytes(compiled):
    reference, root, digest = compiled
    indexed = tokenizer_probe.IndexedTokenizer(root, digest)
    try:
        cases = ["", "hello hello", " café cafe\u0301", " 中文 🙂🦀", "  spaces\n\t12345",
                 "\x00\x7f", "<special>hello<special>", "<special><special>",
                 "<special><added>hello<added><special>", "x" * 256]
        for text in cases:
            ids = reference.encode(text, add_special_tokens=False).ids
            assert indexed.encode(text) == ids
            assert indexed.decode(ids) == reference.decode(ids, skip_special_tokens=False)
            assert indexed.decode(ids, skip_special_tokens=True) == reference.decode(ids, skip_special_tokens=True)
        with pytest.raises(ValueError, match="4096"):
            indexed.encode("x" * 4097)
    finally:
        indexed.close()


def test_public_bpe_index_rejects_untrusted_or_corrupted_artifact(compiled):
    _, root, digest = compiled
    with pytest.raises(ValueError, match="untrusted"):
        tokenizer_probe.IndexedTokenizer(root, "0" * 64)
    with (root / "bpe.sqlite").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="identity mismatch"):
        tokenizer_probe.IndexedTokenizer(root, digest)


def test_public_bpe_compiler_rejects_changed_matching_contract(compiled, tmp_path):
    _, root, _ = compiled
    source = tmp_path / "source.json"
    config = json.loads(source.read_bytes())
    config["added_tokens"][0]["lstrip"] = True
    source.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="matching contract"):
        tokenizer_probe.compile_public_tokenizer(source, root.parent / "bad")


def test_public_bpe_compiler_rejects_token_inserting_postprocessor(compiled, tmp_path):
    _, root, _ = compiled
    source = tmp_path / "source.json"
    config = json.loads(source.read_bytes())
    config["post_processor"] = {"type": "TemplateProcessing"}
    source.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="post-processor"):
        tokenizer_probe.compile_public_tokenizer(source, root.parent / "bad")


def test_indexed_client_binds_original_source_and_preserves_sdk_semantics(compiled, tmp_path):
    from pllm.runtime.transformer_client import TokenizersJSONTokenizer

    reference, root, digest = compiled
    bundle = SimpleNamespace(tokenizer_descriptor={"type": "tokenizer_json",
        "model": (tmp_path / "source.json").read_bytes()}, cfg={"bos_token_id": 0, "eos_token_id": 1})
    control = TokenizersJSONTokenizer(reference, 0, 1)
    candidate = tokenizer_probe.IndexedClientTokenizer(bundle, root, digest)
    try:
        assert candidate.vocab_size == control.vocab_size
        for text in ("hello café", "<special>中文 🙂<added>"):
            for add_bos in (False, True):
                ids = candidate.encode(text, add_bos=add_bos)
                assert ids == control.encode(text, add_bos=add_bos)
                assert candidate.decode(ids) == control.decode(ids)
    finally:
        candidate.close()
    bundle.tokenizer_descriptor["model"] += b" "
    with pytest.raises(ValueError, match="source mismatch"):
        tokenizer_probe.IndexedClientTokenizer(bundle, root, digest)


def test_indexed_tokenizer_supports_response_worker_and_owner_cleanup(compiled):
    reference, root, digest = compiled
    with ThreadPoolExecutor(max_workers=1) as worker:
        indexed = worker.submit(tokenizer_probe.IndexedTokenizer, root, digest).result()
        try:
            ids = indexed.encode("hello café")
            assert ids == reference.encode("hello café", add_special_tokens=False).ids
            assert worker.submit(indexed.decode, ids).result() == reference.decode(ids)
        finally:
            indexed.close()


def test_sealed_kv_immutable_reads_and_context_binding():
    owner = sealed_probe.SealedSlots(b"plan-A/session-A/layer-A/shape-A/f32")
    first, second = owner.seal(b"same bytes"), owner.seal(b"same bytes")
    assert first[12:] != second[12:]
    assert owner.open(0, first) == owner.open(0, first) == b"same bytes"
    with pytest.raises(ValueError, match="binding"):
        owner.open(1, first)
    damaged = first[:-1] + bytes([first[-1] ^ 1])
    with pytest.raises(InvalidTag):
        owner.open(0, damaged)
    owner.context = b"another plan/session/layer/shape"
    with pytest.raises(InvalidTag):
        owner.open(0, first)
    owner.close()
    with pytest.raises(ValueError, match="retired"):
        owner.open(0, first)


def test_sealed_mask_tapes_burn_on_use_malformed_frame_and_cancellation():
    owner = sealed_probe.SealedSlots(b"inventory/stage/rows")
    first, second = owner.seal(b"mask-0"), owner.seal(b"mask-1")
    assert owner.open(0, first, consume=True) == b"mask-0"
    with pytest.raises(ValueError, match="consumed"):
        owner.open(0, first, consume=True)
    with pytest.raises(ValueError, match="binding"):
        owner.open(1, second[:-1], consume=True)
    with pytest.raises(ValueError, match="consumed"):
        owner.open(1, second, consume=True)
    last = owner.seal(b"mask-2")
    owner.close()
    with pytest.raises(ValueError, match="retired"):
        owner.open(2, last, consume=True)


def test_sealed_preflight_rejects_before_allocating_nonce(monkeypatch):
    owner = sealed_probe.SealedSlots(b"bounded")
    monkeypatch.setattr(owner, "MAX_BYTES", 4)
    monkeypatch.setattr(owner, "MAX_SLOTS", 1)
    with pytest.raises(ValueError, match="capacity"):
        owner.seal(b"large")
    assert not owner.lengths
    owner.seal(b"fits")
    with pytest.raises(ValueError, match="capacity"):
        owner.seal(b"fits")
    assert len(owner.lengths) == 1


def test_kv_costs_charge_public_tiles_and_every_decode_reload(monkeypatch):
    monkeypatch.setitem(sys.modules, "client_offload_tokenizer", tokenizer_probe)
    probe = load("probe_client_offload")
    geometry = {"layers": 2, "kv_heads": 1, "head_dim": 2, "max_context": 100}
    result = probe.kv_projection(geometry, 3, 3)
    assert result["full_kv_logical_bytes"] == 2 * 5 * 16
    assert result["sealed_upload_body_bytes"] == 2 * (5 * 16 + 3 * 28)
    assert result["sealed_decode_download_body_bytes"] == 2 * ((4 + 5) * 16 + (2 + 3) * 28)
    assert result["slot_count"] == 6
    assert probe.kv_projection(geometry, 3, 1)["sealed_decode_download_body_bytes"] == 0
    wide = geometry | {"head_dim": 1 << 20}
    assert probe.kv_projection(wide, 3, 3)["slot_count"] == 10
    with pytest.raises(ValueError, match="context"):
        probe.kv_projection(geometry, 99, 3)

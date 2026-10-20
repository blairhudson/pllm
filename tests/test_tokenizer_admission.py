"""Shared GKM/FSS tokenizer validation; no native admission or model import."""
import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import Unigram, WordLevel

from pllm.runtime import delegated_source, fss_source
from pllm.runtime.transformer_client import TokenizersJSONTokenizer


def write_tokenizer(root, size=5):
    vocab = {"<bos>": 0, "<eos>": 1, "<unk>": 2, "hello": 3, "world": 4}
    vocab.update({f"t{i}": i for i in range(5, size)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.add_special_tokens(["<bos>", "<eos>", "<unk>"])
    raw = tokenizer.to_str().encode() + b"\n  "
    (root / "tokenizer.json").write_bytes(raw)
    return raw


def admit(method, root, *, capacity="capsule", vocabulary=5, bos=0):
    config = {"bos_token_id": bos, "eos_token_id": 1}
    contract = {"vocabulary": vocabulary, "input_rows": 1, "decode_steps": 1,
                "maximum_sequence": 2}
    if method == "gkm":
        if capacity == "decoder":
            contract["decoder_resources"] = {"profile": "pllm.gkm.decoder-capacity.v1"}
        return delegated_source._tokenizer(
            root, SimpleNamespace(context_length=16, chat_template=None), config, contract)
    if capacity == "decoder":
        contract["capacity"] = "decoder"
    raw, options = fss_source._tokenizer(root, config, contract)
    return {**json.loads(options), "model": raw}


@pytest.mark.parametrize("method", ["gkm", "fss"])
def test_capsule_original_bytes_and_complete_vocabulary_unchanged(tmp_path, method):
    raw = write_tokenizer(tmp_path)
    result = admit(method, tmp_path)
    assert result["model"] == raw
    assert result["vocab_size"] == 5 and result["bos_token_id"] == 0
    with pytest.raises(ValueError, match="complete table/head"):
        admit(method, tmp_path, vocabulary=8)


@pytest.mark.parametrize("method", ["gkm", "fss"])
def test_decoder_model_padding_uses_existing_unmapped_id_text_semantics(tmp_path, method):
    raw = write_tokenizer(tmp_path)
    result = admit(method, tmp_path, capacity="decoder", vocabulary=8)
    assert result["model"] == raw and result["vocab_size"] == 8
    inner = Tokenizer.from_buffer(raw)
    wrapper = TokenizersJSONTokenizer(inner, result["bos_token_id"], result["eos_token_id"])
    assert wrapper.vocab_size == 5 and inner.id_to_token(7) is None
    # Ordinary tokenizers.decode omits unmapped IDs; the input ID sequence and
    # full model domain remain intact. No FSS-specific text replacement exists.
    ids = [3, 7, 4]
    assert wrapper.decode(ids) == inner.decode(ids, skip_special_tokens=True) == "hello world"
    assert ids == [3, 7, 4] and wrapper.decode([7]) == ""
    with pytest.raises(ValueError, match="original tokenizer entries"):
        admit(method, tmp_path, capacity="decoder", vocabulary=8, bos=7)


@pytest.mark.parametrize("method", ["gkm", "fss"])
@pytest.mark.parametrize("change", ["padding", "truncation", "model", "sparse", "bool_id", "added", "added_shape"])
def test_invalid_document_rejects_before_tokenizer_construction(tmp_path, monkeypatch, method, change):
    document = json.loads(write_tokenizer(tmp_path))
    if change in {"padding", "truncation"}:
        document[change] = {"unbounded": 10**9}
    elif change == "model":
        document["model"] = None
    elif change in {"sparse", "bool_id"}:
        document["model"]["vocab"]["world"] = 10**9 if change == "sparse" else True
    elif change == "added":
        document["added_tokens"][0]["id"] = 10**9
    else:
        document["added_tokens"] = {}
    (tmp_path / "tokenizer.json").write_text(json.dumps(document))

    class ForbiddenTokenizer:
        @staticmethod
        def from_buffer(_raw):
            raise AssertionError("invalid document reached tokenizer construction")

    monkeypatch.setattr("tokenizers.Tokenizer", ForbiddenTokenizer)
    with pytest.raises(ValueError, match="tokenizer"):
        admit(method, tmp_path, capacity="decoder", vocabulary=8)


@pytest.mark.parametrize("method", ["gkm", "fss"])
def test_decoder_original_ids_must_still_be_contiguous(tmp_path, method):
    document = json.loads(write_tokenizer(tmp_path))
    del document["model"]["vocab"]["hello"]
    (tmp_path / "tokenizer.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="complete table/head"):
        admit(method, tmp_path, capacity="decoder", vocabulary=8)


@pytest.mark.parametrize("method", ["gkm", "fss"])
def test_list_vocabulary_uses_same_bounded_original_id_space(tmp_path, method):
    raw = Tokenizer(Unigram([("<unk>", 0.0), ("hello", -1.0)], unk_id=0)).to_str().encode()
    (tmp_path / "tokenizer.json").write_bytes(raw)
    result = admit(method, tmp_path, capacity="decoder", vocabulary=8)
    assert result["model"] == raw and result["vocab_size"] == 8


def test_gkm_capsule_entry_limit_is_not_replaced_by_fss_limit(tmp_path):
    raw = write_tokenizer(tmp_path, size=257)
    with pytest.raises(ValueError, match="complete-table limit"):
        admit("gkm", tmp_path, vocabulary=257)
    assert admit("fss", tmp_path, vocabulary=257)["model"] == raw
    assert admit("gkm", tmp_path, capacity="decoder", vocabulary=257)["model"] == raw

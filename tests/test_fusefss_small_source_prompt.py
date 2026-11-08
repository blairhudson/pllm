"""D58 weights-free prompt preconditions, not checkpoint or execution admission.

Only the actual-cache case opts in via PLLM_TEST_D58_SMOLLM2_SNAPSHOT. An unset
variable skips that case; every supplied invalid path/metadata/template fails.
The resolved cache namespace binds a cohort label, not authenticated weights.
Synthetic tests prove neither the actual count nor native/tape/program fit.
Authoring this file does not authorize running it; actual count remains unknown
until a later bounded parent grant permits the ordinary pytest command.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pllm.configuration import Model
from pllm.runtime.responses import (
    ResponsesError,
    _owned_prompt_source,
    _prepare_prompt,
)
from pllm.runtime.tokenizer_admission import (
    _DescriptorPolicy,
    _PromptMetadataError,
    _capture_prompt_metadata,
    reserve_decoder_metadata,
)


REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
PIN = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
LABEL = "fusefss-smollm2-d58"
MAX_SEQUENCE = 16
NATIVE_BYTES = 64 << 20
TAPE_BYTES = 32 << 30
SNAPSHOT_ENV = "PLLM_TEST_D58_SMOLLM2_SNAPSHOT"
NAMED_METADATA = ("config.json", "tokenizer.json", "tokenizer_config.json")

# Synthetic only: not a replacement for any cached source template.
SYNTHETIC_TEMPLATE = (
    "{{ bos_token }}|"
    "{% for message in messages %}{{ message.role }}={{ message.content }}|{% endfor %}"
    "{% if add_generation_prompt %}assistant={% endif %}"
)


def fresh_request():
    return {"model": REPO, "instructions": "Hi", "input": "hello",
            "max_output_tokens": 2}


def _assert_count_bounds(rows):
    assert rows > 0, "prompt must contain positive rows"
    assert rows <= MAX_SEQUENCE, "prompt exceeds sequence bound"
    assert rows + fresh_request()["max_output_tokens"] - 1 <= MAX_SEQUENCE, (
        "prompt plus output feedback exceeds sequence bound")


def _snapshot_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("explicit snapshot path must be nonempty")
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("snapshot path must be a directory")
    expected = ("models--HuggingFaceTB--SmolLM2-135M-Instruct", "snapshots", PIN)
    if root.parts[-3:] != expected:
        raise ValueError("snapshot namespace/pin differs from D58 cohort label")
    return root


def _require_source_template(descriptor):
    template = descriptor.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("D58 requires a present nonempty string source chat_template")


class _EncodeSpy:
    def __init__(self, inner=None):
        self.inner = inner
        self.calls = []

    def encode(self, text, *, add_bos):
        self.calls.append((text, add_bos))
        if self.inner is not None:
            return self.inner.encode(text, add_bos=add_bos)
        return list(range(15))  # Synthetic rows, never actual tokenizer evidence.

    def close(self):
        if self.inner is not None:
            self.inner.close()
            self.inner = None


def _count_local_metadata(root):
    """Only three named metadata files; no resolver, index, body or tensor loader.

    Owned-source-shaped namespace below carries provisional metadata only. It is
    not an admitted source, and consume checks association, not source authority.
    All parsed tokenizer/prepared owners retire inside the metadata reservation.
    """
    with reserve_decoder_metadata():
        snapshot = descriptor = owned = tokenizer = prepared = None
        metadata_association = source_association = returned = ids = None
        try:
            model = Model.path(str(root), model_id=REPO)
            snapshot = _capture_prompt_metadata(
                model, policy=_DescriptorPolicy.FSS, capacity="decoder",
                input_rows=15, outputs=2)
            snapshot.check_unchanged()
            metadata_association = snapshot.association()
            descriptor = snapshot.tokenizer_descriptor
            _require_source_template(descriptor)  # Before tokenizer/encode; no fallback.
            owned = SimpleNamespace(
                model=model, config_source_bytes=snapshot.config_bytes,
                tokenizer_descriptor=descriptor, source_lock_digest=None)
            source_association = _owned_prompt_source(owned)
            tokenizer = _EncodeSpy(snapshot.tokenizer())
            prepared = _prepare_prompt(
                fresh_request(), descriptor=descriptor, tokenizer=tokenizer,
                source=source_association)
            returned, ids = prepared.consume(fresh_request(), _owned_prompt_source(owned))
            assert returned is tokenizer
            assert len(tokenizer.calls) == 1
            assert tokenizer.calls[0][1] is descriptor["add_bos_token"]
            snapshot.check_unchanged()
            assert snapshot.association() == metadata_association
            assert _owned_prompt_source(owned) == source_association
            return len(ids)
        finally:
            if prepared is not None:
                prepared.close()
            elif tokenizer is not None:
                tokenizer.close()
            # Snapshot has no close API; release retained metadata before refund.
            snapshot = descriptor = owned = tokenizer = prepared = None
            metadata_association = source_association = returned = ids = None


def _synthetic_metadata(tmp_path):
    """Hand-authored tiny metadata, never copied from any source cache."""
    root = tmp_path / "models--HuggingFaceTB--SmolLM2-135M-Instruct" / "snapshots" / PIN
    root.mkdir(parents=True)
    documents = {
        "config.json": {"vocab_size": 3, "bos_token_id": 0, "eos_token_id": 1},
        "tokenizer.json": {
            "version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [], "normalizer": None, "pre_tokenizer": None,
            "post_processor": None, "decoder": None,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "end": 1, "hello": 2},
                      "unk_token": "[UNK]"},
        },
        "tokenizer_config.json": {
            "chat_template": SYNTHETIC_TEMPLATE, "bos_token": "source-bos",
            "add_bos_token": False, "model_max_length": MAX_SEQUENCE,
        },
    }
    for name in NAMED_METADATA:
        (root / name).write_text(json.dumps(documents[name]), encoding="utf-8")
    return root


def test_frozen_request_and_unchanged_ceilings():
    first = fresh_request()
    assert first == {"model": REPO, "instructions": "Hi", "input": "hello",
                     "max_output_tokens": 2}
    first["input"] = "changed"
    assert fresh_request()["input"] == "hello"
    assert (MAX_SEQUENCE, NATIVE_BYTES, TAPE_BYTES) == (16, 64 << 20, 32 << 30)
    assert LABEL == "fusefss-smollm2-d58"


@pytest.mark.parametrize("rows, refusal", [
    (15, None),
    (16, "feedback"),
    (17, "prompt exceeds sequence"),
    (0, "positive rows"),
])
def test_synthetic_frozen_request_budget(rows, refusal):
    if refusal is None:
        _assert_count_bounds(rows)
    else:
        with pytest.raises(AssertionError, match=refusal):
            _assert_count_bounds(rows)


@pytest.mark.parametrize("add_bos", [False, True])
def test_synthetic_shared_preparation_preserves_source_semantics(add_bos):
    descriptor = {"chat_template": SYNTHETIC_TEMPLATE, "bos_token": "source-bos",
                  "add_bos_token": add_bos}
    owned = SimpleNamespace(
        model=Model.path("synthetic-metadata-only", model_id=REPO),
        config_source_bytes=b'{"vocab_size": 3}',
        tokenizer_descriptor=descriptor, source_lock_digest=None)
    tokenizer = _EncodeSpy()
    prepared = None
    try:
        _require_source_template(descriptor)
        association = _owned_prompt_source(owned)
        prepared = _prepare_prompt(
            fresh_request(), descriptor=descriptor, tokenizer=tokenizer, source=association)
        actual, ids = prepared.consume(fresh_request(), _owned_prompt_source(owned))
        assert actual is tokenizer
        assert tokenizer.calls == [("source-bos|system=Hi|user=hello|assistant=", add_bos)]
        assert len(ids) == 15
        _assert_count_bounds(len(ids))
        with pytest.raises(ResponsesError, match="already consumed"):
            prepared.consume(fresh_request(), association)
    finally:
        if prepared is not None:
            prepared.close()
        else:
            tokenizer.close()


@pytest.mark.parametrize("suffix", [
    ("models--HuggingFaceTB--SmolLM2-135M-Instruct", "snapshots", "0" * 40),
    ("models--other--model", "snapshots", PIN),
    ("models--HuggingFaceTB--SmolLM2-135M-Instruct", "refs", PIN),
])
def test_synthetic_wrong_snapshot_namespace_refused(tmp_path, suffix):
    root = tmp_path.joinpath(*suffix)
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="namespace/pin"):
        _snapshot_path(str(root))


@pytest.mark.parametrize("kind", ["empty", "missing", "file"])
def test_supplied_bad_snapshot_path_is_not_skip(tmp_path, kind):
    value = str(tmp_path / "absent")
    if kind == "empty":
        value = ""
    elif kind == "file":
        path = tmp_path / "file"
        path.write_text("synthetic", encoding="utf-8")
        value = str(path)
    with pytest.raises((ValueError, OSError)):
        _snapshot_path(value)


@pytest.mark.parametrize("name", NAMED_METADATA)
@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_synthetic_named_metadata_refuses_without_fallback(tmp_path, name, damage):
    root = _synthetic_metadata(tmp_path)
    if damage == "missing":
        (root / name).unlink()
    else:
        (root / name).write_text("{", encoding="utf-8")
    with pytest.raises((_PromptMetadataError, ValueError)):
        _count_local_metadata(_snapshot_path(str(root)))


@pytest.mark.parametrize("template", [None, "", " ", 7, [], {}, "{%"])
def test_synthetic_invalid_source_template_never_falls_back(tmp_path, template, monkeypatch):
    root = _synthetic_metadata(tmp_path)
    options = {"chat_template": template, "add_bos_token": False}
    (root / "tokenizer_config.json").write_text(json.dumps(options), encoding="utf-8")

    def forbidden_encode(*args, **kwargs):
        pytest.fail("invalid source template reached encode")

    monkeypatch.setattr(_EncodeSpy, "encode", forbidden_encode)
    with pytest.raises((ValueError, ResponsesError)):
        _count_local_metadata(_snapshot_path(str(root)))


def test_actual_explicit_cached_small_source_count():
    value = os.environ.get(SNAPSHOT_ENV)
    if value is None:
        pytest.skip(f"explicit cached metadata only: set {SNAPSHOT_ENV}; never fetch")
    root = _snapshot_path(value)
    rows = _count_local_metadata(root)
    print(f"cohort={LABEL} repo={REPO} pin={PIN} prompt_rows={rows} outputs=2")
    _assert_count_bounds(rows)  # Overbudget is a failure, never a skip.

"""Cached-tokenizer-only public workload precondition, not accepted SDK usage.

Opt in with PLLM_TEST_CACHED_QWEN3_TOKENIZER=1. The ordinary offline source
resolver checks the pinned cached snapshot; only config/tokenizer bytes are read.
No tensor store, model engine, preparation, generation or download executes.

The tracked public text, including its trailing newline, is reusable verbatim
through --prompt-file examples/papers/low_memory_qwen3_prompt.txt. Actual accepted
input/output usage must still pass the campaign's ordinary 64/8 checks and
same-cohort source/numeric/trajectory/state and memory contracts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


PUBLIC_PROMPT = (
    "Explain how a prepared private inference client can keep prompt text local while "
    "sending only masked requests to remote inference helpers. Describe why one-use "
    "material must be committed before a response starts, why consumed rows cannot "
    "be reused, and why paging public weights is different from generating fresh "
    "cryptographic material.\n"
)
ASSET_SHA256 = {
    "config.json": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}


@pytest.mark.skipif(os.environ.get("PLLM_TEST_CACHED_QWEN3_TOKENIZER") != "1",
                    reason="explicit cached-only tokenizer check; never fetch model assets")
def test_pinned_cached_sdk_frontend_counts_public_candidate_as_64(monkeypatch):
    from examples.papers.low_memory_qwen3 import MODEL_ID, REVISION, SOURCE, private_inputs
    from pllm.runtime.client import RuntimeClient
    from pllm.runtime.benchmark_context import context_bytes, validate_context
    from pllm.runtime.hf_hub import resolve_huggingface_source
    from pllm.runtime.models import ModelManifest
    from pllm.runtime.responses import normalize_input
    from pllm.runtime.tools import tool_policy
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert MODEL_ID == SOURCE.source == "Qwen/Qwen3-4B"
    assert REVISION == SOURCE.revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert SOURCE.local_files_only is True
    spec_before = SOURCE.to_spec()

    def cached_source():
        resolved = resolve_huggingface_source(SOURCE.source, revision=SOURCE.revision,
                                             local_files_only=True, token=False)
        assert resolved.repo_id == MODEL_ID and resolved.revision == REVISION
        assert resolved.path.name == REVISION
        assert resolved.path.parent.name == "snapshots"
        assert resolved.path.parent.parent.name == "models--Qwen--Qwen3-4B"
        return resolved

    source = cached_source()

    def hashes():
        return {name: hashlib.sha256((source.path / name).read_bytes()).hexdigest()
                for name in ASSET_SHA256}

    before = hashes()
    assert before == ASSET_SHA256
    raw = (Path(__file__).resolve().parents[1] / "examples/papers/low_memory_qwen3_prompt.txt").read_bytes()
    prompt = validate_context(raw.decode("utf-8"))
    assert prompt == PUBLIC_PROMPT and context_bytes(prompt) == raw
    config = json.loads((source.path / "config.json").read_text())
    tokenizer_config = json.loads((source.path / "tokenizer_config.json").read_text())
    manifest = ModelManifest(
        id=MODEL_ID, architecture=config["architectures"][0], source_format="huggingface",
        source=str(source.path), context_length=config["max_position_embeddings"],
        chat_template=tokenizer_config["chat_template"],
        **{key: config[key] for key in ("vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")},
    )
    descriptor = MaskedTransformerEngine._load_tokenizer_descriptor(source.path, manifest, config)
    # Tokenizer-only holder: no tensors/stages or claim of admitted execution.
    bundle = ClientBundle(MODEL_ID, {}, config, descriptor, {}, {}, {})
    assert descriptor["type"] == "tokenizer_json"
    assert descriptor["chat_template"] == tokenizer_config["chat_template"]
    assert hashlib.sha256(descriptor["model"]).hexdigest() == ASSET_SHA256["tokenizer.json"]
    assert descriptor["add_bos_token"] is False
    assert descriptor["bos_token_id"] == 151643 and descriptor["eos_token_id"] == 151645
    body = {"input": prompt, "max_output_tokens": 8, "temperature": 0}
    policy = tool_policy(body)
    assert not policy.enabled and policy.prompt_instruction() is None
    messages = normalize_input(body["input"], instructions=None)
    assert [message.to_prompt_dict() for message in messages] == [{"role": "user", "content": PUBLIC_PROMPT}]
    # Same normalization, template wrapper and tokenizer as cold SDK responses;
    # no alternate HF chat API, enable_thinking override, BOS or token padding.
    rendered = RuntimeClient._render_cached_decoder_prompt(
        SimpleNamespace(bundle=bundle), messages, add_generation_prompt=True, tools=None)
    assert rendered == "<|im_start|>user\n" + PUBLIC_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
    tokenizer = bundle.tokenizer()
    tokens = tokenizer.encode(rendered, add_bos=descriptor["add_bos_token"])
    assert len(tokens) == 64
    assert len(tokenizer.encode(PUBLIC_PROMPT, add_bos=False)) == 56
    assert hashlib.sha256(PUBLIC_PROMPT.encode()).hexdigest() == (
        "34281fc41a22c45475b85498b511a7cd4ca8852a13e7fa39912bb4f7bb6626a7")
    assert hashlib.sha256(rendered.encode()).hexdigest() == (
        "1606559db89fd32ae7c5625c7d1b36f0cce42f99ade5cd5ec7181d8c710bcc15")
    assert hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest() == (
        "c4c3092b07837c52dfc3f8d88522b8ff5811688d1703fbf189d18259016e4fef")
    assert private_inputs(PUBLIC_PROMPT) == {
        "maintained": {"prompt": PUBLIC_PROMPT}, "paged-prepared": {"prompt": PUBLIC_PROMPT}}
    assert hashes() == before and cached_source() == source and SOURCE.to_spec() == spec_before

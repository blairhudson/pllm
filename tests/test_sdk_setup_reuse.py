"""Source/lifetime isolation and real SDK execution for public setup reuse."""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from contextvars import copy_context
import dataclasses
import weakref

import pytest

import pllm
from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu
from pllm.runtime.client import AsyncOpenAI, OpenAI
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


@pytest.fixture
def setup(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1,
                                       model_type="qwen3", qk_norm=True, with_qkv_bias=False)
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(root, model_id="setup-reuse")))
    experiment = pllm.Experiment(
        "setup-reuse", ClientOnlyCpu(pllm.Model.path(str(root), model_id="setup-reuse")),
        pllm.Deployment.local(root=str(tmp_path)),
        pllm.ExecutionBudget(requests=3, max_input_tokens=64, max_new_tokens=2),
    )
    return experiment, engine


@pytest.fixture
def tokenizers(monkeypatch):
    original = ClientBundle.tokenizer
    references = []

    class Tracked:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    def create(bundle):
        value = Tracked(original(bundle))
        references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(ClientBundle, "tokenizer", create)
    return references


@pytest.mark.parametrize("ending", ["completed", "failure", "cancelled"])
def test_sdk_tokenizer_scope_releases_after_response_failure_or_cancel(setup, tokenizers, ending):
    experiment, engine = setup
    with OpenAI(experiment=experiment, local_engine=engine) as client:
        with client.tokenizer_scope():
            assert client._core._response_input_tokens("setup-reuse", "A") > 0
            with client.tokenizer_scope():
                assert client._core._response_input_tokens("setup-reuse", "B") > 0
            assert len(tokenizers) == 1 and tokenizers[0]() is not None
            if ending == "completed":
                actual = client.responses.create(input="A", max_output_tokens=2, temperature=0)
            elif ending == "cancelled":
                stream = client.responses.create(input="A", max_output_tokens=2, stream=True)
                for event in stream:
                    if event.type == "response.output_text.delta":
                        break
                stream.close()
            else:
                with pytest.raises(ValueError):
                    client.responses.create(input="A", max_output_tokens=-1)
            assert len(tokenizers) == 1
        assert tokenizers[0]() is None
        assert client._core._tokenizer_owner.get() is None
        if ending == "completed":
            expected = client.responses.create(input="A", max_output_tokens=2, temperature=0)
            assert (actual.output_text, actual.usage) == (expected.output_text, expected.usage)
            assert len(tokenizers) == 2 and tokenizers[1]() is None
        assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
        assert client.privacy_audit.plaintext_token_ids_sent == 0


def test_tokenizer_scope_binds_bundle_and_closes_copied_context_on_error(setup, tokenizers):
    experiment, engine = setup
    with OpenAI(experiment=experiment, local_engine=engine) as client:
        bundle = client._core._transformer_state("setup-reuse").bundle
        replacement = dataclasses.replace(bundle, tokenizer_descriptor={
            "kind": "alphabet", "alphabet": "ABC",
        })
        with pytest.raises(RuntimeError, match="abort setup"):
            with client.tokenizer_scope():
                client._core._response_tokenizer(bundle)
                client._core._response_tokenizer(replacement)
                assert len(tokenizers) == 2 and tokenizers[0]() is None
                copied = copy_context()
                raise RuntimeError("abort setup")
        assert all(reference() is None for reference in tokenizers)
        with pytest.raises(RuntimeError, match="scope is closed"):
            copied.run(client._core._response_tokenizer, bundle)
        assert client._core._tokenizer_owner.get() is None


@pytest.mark.asyncio
async def test_async_scope_reuses_tokenizer_through_response_worker(setup, tokenizers):
    experiment, engine = setup
    async with AsyncOpenAI(experiment=experiment, local_engine=engine) as client:
        with client.tokenizer_scope():
            await asyncio.to_thread(client.sync._core._response_input_tokens, "setup-reuse", "A")
            stream = await client.responses.create(input="A", max_output_tokens=2, stream=True)
            async for _ in stream:
                pass
            assert len(tokenizers) == 1 and tokenizers[0]() is not None
        assert tokenizers[0]() is None


@pytest.mark.integration
def test_prepared_sdk_reuses_setup_and_preserves_response_and_online_bodies(setup, tokenizers):
    local, _ = setup
    prepared = dataclasses.replace(local, pipeline=MaskedLinearCpu(local.pipeline.model))
    results, bodies = [], []
    with build_roles(prepared, engine_threads=1) as roles:
        for scoped in (False, True):
            with roles.client(background_inventory_refill=False, bundle_cache_mode="off") as client:
                before = len(tokenizers)
                with client.tokenizer_scope() if scoped else nullcontext():
                    count = client._core._response_input_tokens(roles.model_id, "A")
                    rows = client.prepared_rows_for_response("A", 2)
                    assert rows == count + 1
                    client.preprocess(count=rows)
                    response = client.responses.create(input="A", max_output_tokens=2, temperature=0)
                    results.append((response.output_text, response.usage))
                    audit = client.privacy_audit
                    bodies.append((audit.masked_online_upload_bytes, audit.masked_online_download_bytes))
                    assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
                assert len(tokenizers) - before == (1 if scoped else 3)
                assert all(reference() is None for reference in tokenizers[before:])
    assert results[0] == results[1]
    assert bodies[0] == bodies[1] and all(bodies[0])

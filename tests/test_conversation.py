"""Multi-turn parity, single-owner retirement and failed-turn isolation."""

import asyncio
import threading
import weakref

import numpy as np
import pytest

import pllm
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from test_batched_prefix_reuse import prefix_experiment


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_managed_chain_matches_explicit_continuation_and_retires_old_kv(tmp_path, family):
    model, experiment = prefix_experiment(tmp_path, family)
    with build_roles(experiment, engine_threads=1) as roles, roles.client() as client:
        previous = None
        reference = []
        for prompt in ("Explain a tree", "And its roots?", "One detail"):
            result = client.responses.create(
                input=prompt, previous_response_id=previous, max_output_tokens=3, temperature=0
            )
            saved = client._core._transformer_conversations[result.id]
            reference.append(
                (
                    result,
                    saved.next_logits.copy(),
                    [(r.key.copy(), r.value.copy()) for r in saved.snapshot.caches],
                )
            )
            if previous:
                client._core.evict_response(previous)
            previous = result.id
        client._core.evict_response(previous)
        # Prefix cache remains a distinct, independently bounded owner.
        state = client._core._transformer_states[model]
        state.prefill_cache.clear()
        references = []
        with client.conversation(
            max_state_bytes=1 << 20, max_output_tokens=3, temperature=0
        ) as chat:
            for prompt, (expected, logits, kv) in zip(
                ("Explain a tree", "And its roots?", "One detail"), reference, strict=True
            ):
                result = chat.send(prompt)
                assert result.output_text == expected.output_text
                assert result.usage == expected.usage
                assert set(client._core._transformer_conversations) == {result.id}
                assert set(client._core.histories) == {result.id}
                assert all(ref() is None for ref in references)
                saved = client._core._transformer_conversations[result.id]
                np.testing.assert_array_equal(saved.next_logits.view(np.uint32), logits.view(np.uint32))
                for row, (key, value) in zip(saved.snapshot.caches, kv, strict=True):
                    np.testing.assert_array_equal(row.key.view(np.uint32), key.view(np.uint32))
                    np.testing.assert_array_equal(row.value.view(np.uint32), value.view(np.uint32))
                assert 0 < chat.state_bytes <= 1 << 20
                references = [weakref.ref(row.key) for row in saved.snapshot.caches]
                del saved, row
        assert all(ref() is None for ref in references)
        assert not client._core.cache and not client._core.histories
        assert chat.state_bytes == 0
        assert client.privacy_audit.kv_continuation_batched_hits == 4
        assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
        assert client.privacy_audit.preparation_requests_during_online == 0


def test_failed_turn_preserves_prior_owner_and_burns_reserved_rows(tmp_path, monkeypatch):
    model, experiment = prefix_experiment(tmp_path)
    with build_roles(experiment, engine_threads=1) as roles, roles.client() as client:
        with client.conversation(
            max_state_bytes=1 << 20, max_output_tokens=3, temperature=0
        ) as chat:
            first = chat.send("First turn")
            before = client._core._transformer_conversations[first.id]
            state = before.snapshot.caches[0].key.copy()
            original = SemanticDecoderRuntime.sample

            def fail(*args, **kwargs):
                raise RuntimeError("injected sampling failure")

            monkeypatch.setattr(SemanticDecoderRuntime, "sample", fail)
            with pytest.raises(RuntimeError, match="injected"):
                chat.send("failed turn")
            assert chat.response_id == first.id
            assert set(client._core._transformer_conversations) == {first.id}
            np.testing.assert_array_equal(before.snapshot.caches[0].key, state)
            assert client.privacy_audit.prepared_stage_rows_burned > 0
            monkeypatch.setattr(SemanticDecoderRuntime, "sample", original)
            again = chat.send("Retry with a fresh session")
            assert again.id != first.id


def test_geometry_rejects_before_reservation_and_client_close_retires_state(tmp_path):
    model, experiment = prefix_experiment(tmp_path)
    with build_roles(experiment, engine_threads=1) as roles:
        client = roles.client()
        with client.conversation(max_state_bytes=1024, max_output_tokens=1) as chat:
            with pytest.raises(ValueError, match="exceed max_state_bytes"):
                chat.send("must not reserve")
            assert client.privacy_audit.inference_stage_calls == 0
            assert client.privacy_audit.prepared_stage_rows_issued == 0
        with client.conversation(max_state_bytes=1 << 20, max_output_tokens=1) as chat:
            result = chat.send("Now run")
            refs = [
                weakref.ref(c.key)
                for c in client._core._transformer_conversations[result.id].snapshot.caches
            ]
            client.close()
            assert all(ref() is None for ref in refs)
            assert not client._core._transformer_states
            with pytest.raises(RuntimeError, match="closed"):
                chat.send("closed")


@pytest.mark.asyncio
async def test_async_cancellation_drains_worker_and_retires_attempt(tmp_path, monkeypatch):
    model, experiment = prefix_experiment(tmp_path)
    entered, release = threading.Event(), threading.Event()
    with build_roles(experiment, engine_threads=1) as roles:
        async with pllm.AsyncOpenAI(
            experiment=experiment,
            base_url=roles.inference_url,
            api_key=roles._inference_key,
            preparation_base_url=roles.preparation_url,
            preparation_api_key=roles._preparation_key,
            bundle_cache_dir=tmp_path / "async-cache",
        ) as client:
            async with client.conversation(
                max_state_bytes=1 << 20, max_output_tokens=3, temperature=0
            ) as chat:
                await chat.send("first")
                original = SemanticDecoderRuntime.sample

                def blocked(*args, **kwargs):
                    entered.set()
                    assert release.wait(15)
                    return original(*args, **kwargs)

                monkeypatch.setattr(SemanticDecoderRuntime, "sample", blocked)
                task = asyncio.create_task(chat.send("cancel second"))
                assert await asyncio.to_thread(entered.wait, 15)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                with pytest.raises(RuntimeError, match="active turn"):
                    await chat.send("concurrent")
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert not client.sync._core._transformer_conversations
                assert not client.sync._core._pending_response_store
                with pytest.raises(RuntimeError, match="closed"):
                    await chat.send("cannot reuse cancelled state")
            assert not client.sync._core._transformer_conversations

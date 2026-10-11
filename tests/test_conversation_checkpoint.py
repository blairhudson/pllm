"""Exact prepared continuation across fresh clients and interpreter boundaries."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

import pllm
from pllm.preparation import PreparedInventory
from pllm.quantization import SymmetricPerRow
from pllm.runtime.conversation_checkpoint import _Storage
from pllm.runtime.servers import build_roles
from pllm.state import ConversationCheckpoint
from test_batched_prefix_reuse import prefix_experiment


def experiment_at(tmp_path, family="qwen2"):
    model, experiment = prefix_experiment(tmp_path, family)
    return model, experiment.with_params(
        pipeline__quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
        pipeline__inventory=PreparedInventory(allocation="demand", refill="on-demand"))


def commitment(client, response_id):
    saved = client._core._transformer_conversations[response_id]
    values = [saved.next_logits]
    for cache in saved.snapshot.caches:
        values.extend((cache.key[:cache.length], cache.value[:cache.length]))
    return {"position": saved.snapshot.position, "words": sum(v.size for v in values),
        "bits": hashlib.sha256(b"".join(v.view(np.uint32).tobytes() for v in values)).hexdigest(),
        "tokens": hashlib.sha256(json.dumps(saved.token_ids).encode()).hexdigest(),
        "pending": hashlib.sha256(json.dumps(saved.pending_token_ids).encode()).hexdigest()}


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_prepared_checkpoint_exact_restart_and_fresh_inventory(tmp_path, family):
    _, experiment = experiment_at(tmp_path, family)
    root, key = tmp_path / "checkpoint", os.urandom(32)
    with build_roles(experiment, engine_threads=1) as roles:
        with roles.client(background_inventory_refill=False) as client, client.conversation(
            max_state_bytes=1 << 20, temperature=0, max_output_tokens=3
        ) as chat:
            first = chat.send("Explain trees.")
            first_state = commitment(client, first.id)
            head = chat.checkpoint(root, key, page_rows=8)
            assert ConversationCheckpoint.from_spec(head.to_spec()) == head
            assert chat.response_id == first.id
            expected = chat.send("And roots?")
            expected_state = commitment(client, expected.id)
        for path in root.iterdir():
            assert b"Explain trees" not in path.read_bytes()
            assert key not in path.read_bytes()
        with roles.client(background_inventory_refill=False) as client, client.conversation(
            max_state_bytes=1 << 20, temperature=0, max_output_tokens=3
        ) as chat:
            result = chat.restore(root, key, head)
            assert result.id != first.id and result.output_text == first.output_text
            assert commitment(client, result.id) == first_state
            assert client.privacy_audit.prepared_stage_rows_issued == 0
            result = chat.send("And roots?")
            assert result.output_text == expected.output_text and result.usage == expected.usage
            assert commitment(client, result.id) == expected_state
            assert client.privacy_audit.prepared_stage_rows_issued > 0
            assert client.privacy_audit.kv_continuation_batched_hits > 0
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
            assert client.privacy_audit.preparation_requests_during_online == 0
        # A fresh interpreter has a different process-local snapshot seal. Only
        # authenticated storage may mint its new, response-owned provenance.
        code = """
import json, os, sys
import pllm
from pllm.state import ConversationCheckpoint
from test_conversation_checkpoint import commitment
p=json.load(sys.stdin)
with pllm.OpenAI(experiment=pllm.Experiment.from_spec(p['experiment']),
    base_url=os.environ['PLLM_CHECKPOINT_TEST_INFERENCE'], api_key=os.environ['PLLM_CHECKPOINT_TEST_INFERENCE_KEY'],
    preparation_base_url=os.environ['PLLM_CHECKPOINT_TEST_PREPARATION'],
    preparation_api_key=os.environ['PLLM_CHECKPOINT_TEST_PREPARATION_KEY'], background_inventory_refill=False) as client:
    with client.conversation(max_state_bytes=1<<20, temperature=0, max_output_tokens=3) as chat:
        chat.restore(p['directory'], bytes.fromhex(p['key']), ConversationCheckpoint.from_spec(p['head']))
        result=chat.send('And roots?')
        print(json.dumps(commitment(client, result.id)))
"""
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.resolve()),
            "PLLM_CHECKPOINT_TEST_INFERENCE": roles.inference_url,
            "PLLM_CHECKPOINT_TEST_PREPARATION": roles.preparation_url,
            "PLLM_CHECKPOINT_TEST_INFERENCE_KEY": roles._inference_key,
            "PLLM_CHECKPOINT_TEST_PREPARATION_KEY": roles._preparation_key,
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        child = subprocess.run([sys.executable, "-c", code], input=json.dumps({"experiment": experiment.to_spec(),
            "directory": str(root), "key": key.hex(), "head": head.to_spec()}),
            text=True, capture_output=True, timeout=90, env=env, check=True)
        assert json.loads(child.stdout) == expected_state


def test_failed_writes_and_imports_preserve_the_completed_chain(tmp_path, monkeypatch):
    _, experiment = experiment_at(tmp_path)
    root, key = tmp_path / "checkpoint", os.urandom(32)
    with build_roles(experiment, engine_threads=1) as roles, roles.client() as client:
        with client.conversation(max_state_bytes=1 << 20, temperature=0, max_output_tokens=2) as chat:
            first = chat.send("First.")
            head = chat.checkpoint(root, key)
            original = _Storage._write

            def fail_metadata(owner, raw, metadata):
                if "conversation" in metadata:
                    raise OSError("injected publication failure")
                return original(owner, raw, metadata)

            with monkeypatch.context() as patch:
                patch.setattr(_Storage, "_write", fail_metadata)
                with pytest.raises(OSError, match="injected"):
                    chat.checkpoint(tmp_path / "failed", key)
            assert chat.response_id == first.id
            chat.send("Still usable.")
        with client.conversation(max_state_bytes=1 << 20) as chat:
            with pytest.raises(InvalidTag):
                chat.restore(root, os.urandom(32), head)
            assert chat.response_id is None and not client._core._transformer_conversations
            # Corrupt a numeric page after manifest authentication. No partially
            # restored cache or response may escape.
            pages = [p for p in root.iterdir() if p.name not in {".writer", head.name, head.state.name}]
            page = pages[-1]
            raw = page.read_bytes()
            page.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            with pytest.raises(ValueError, match="trusted commitment"):
                chat.restore(root, key, head)
            assert chat.response_id is None and not client._core.cache
            page.write_bytes(raw)
            restored = chat.restore(root, key, head)
            assert restored.output_text == first.output_text
            with pytest.raises(ValueError, match="empty"):
                chat.restore(root, key, head)
        # Old heads remain recoverable after a newer live turn. Choosing the
        # current trusted head remains caller custody, not a rollback oracle.
        with client.conversation(max_state_bytes=1 << 20) as chat:
            chat.restore(root, key, head)


def test_checkpoint_requires_canonical_state(tmp_path):
    _, experiment = prefix_experiment(tmp_path)
    with build_roles(experiment, engine_threads=1) as roles, roles.client() as client:
        with client.conversation(max_state_bytes=1 << 20, temperature=0, max_output_tokens=2) as chat:
            chat.send("First.")
            with pytest.raises(ValueError, match="prefix_f32"):
                chat.checkpoint(tmp_path / "invalid", b"k" * 32)
            assert not (tmp_path / "invalid").exists()


def test_metadata_preflights_nested_fanout_and_nonfinite_values():
    from pllm.runtime.conversation_checkpoint import _encode_metadata
    for value in ([[0] * 1000] * 1000, float("inf"), 1 << 64):
        with pytest.raises(ValueError):
            _encode_metadata(value)


@pytest.mark.asyncio
async def test_async_restore_cancellation_drains_and_retires_state(tmp_path, monkeypatch):
    _, experiment = experiment_at(tmp_path)
    root, key = tmp_path / "checkpoint", os.urandom(32)
    entered, release = threading.Event(), threading.Event()
    with build_roles(experiment, engine_threads=1) as roles, roles.client() as client:
        with client.conversation(max_state_bytes=1 << 20, temperature=0, max_output_tokens=2) as chat:
            chat.send("First.")
            head = chat.checkpoint(root, key)
        original = _Storage.snapshot

        def blocked(owner, *args):
            entered.set()
            assert release.wait(15)
            return original(owner, *args)

        monkeypatch.setattr(_Storage, "snapshot", blocked)
        async with pllm.AsyncOpenAI(experiment=experiment, base_url=roles.inference_url,
            api_key=roles._inference_key, preparation_base_url=roles.preparation_url,
            preparation_api_key=roles._preparation_key) as async_client:
            async with async_client.conversation(max_state_bytes=1 << 20) as chat:
                task = asyncio.create_task(chat.restore(root, key, head))
                assert await asyncio.to_thread(entered.wait, 15)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                with pytest.raises(RuntimeError, match="active turn"):
                    await chat.send("Concurrent")
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert chat.response_id is None
                assert not async_client.sync._core._transformer_conversations
                assert not async_client.sync._core.cache

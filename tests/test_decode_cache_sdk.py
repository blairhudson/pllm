"""Fresh-wheel ordinary prepared SDK decode-cache regressions, not benchmarks.

Never bypass public/native admission to make these pass against an old wheel.
Generated fixture/provider execution is reserved for the parent's central run.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import httpx
import numpy as np
import pytest

import pllm
from pllm.preparation import PreparedInventory
from pllm.quantization import SymmetricPerRow
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.state import ClientPrefixReuse, DecodeCache, DecodeCacheHead
from test_batched_prefix_reuse import prefix_experiment

pytestmark = pytest.mark.integration
CAP = 4


def experiment_at(root, family="qwen2"):
    model_id, experiment = prefix_experiment(root, family, bound=128)
    return model_id, experiment.with_params(
        pipeline__quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        pipeline__inventory=PreparedInventory(allocation="demand", refill="on-demand", decode_epoch_tokens=1),
        pipeline__cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128,
            generated_prefixes=True, decode=True, decode_max_bytes=32 << 20))


def cache_at(root, key=None, head=None):
    return DecodeCache(root, os.urandom(32) if key is None else key, head=head,
                       max_snapshot_bytes=1 << 20, max_transient_bytes=16 << 20)


def numeric(runtime, logits):
    values = [logits, *(v[:row.length] for row in runtime.caches for v in (row.key, row.value))]
    return tuple(v.view(np.uint32).tobytes() for v in values)


def commitment(client, response_id):
    saved = client._core._transformer_conversations[response_id]
    values = [saved.next_logits, *(v[:row.length] for row in saved.snapshot.caches for v in (row.key, row.value))]
    return {"position": saved.snapshot.position, "tokens": saved.token_ids.copy(),
            "pending": saved.pending_token_ids.copy(),
            "bits": hashlib.sha256(b"".join(v.view(np.uint32).tobytes() for v in values)).hexdigest()}


class SamplingRecorder:
    """Seed the request's actual runtime RNG, never NumPy's unrelated global RNG."""
    def __init__(self, monkeypatch):
        self.original = SemanticDecoderRuntime.sample
        self.begin(42)
        def sample(runtime, logits, **kwargs):
            return self.sample(runtime, logits, **kwargs)
        monkeypatch.setattr(SemanticDecoderRuntime, "sample", sample)

    def begin(self, seed, *, diverge_at=None, eos_at=None):
        self.seed, self.records, self.runtime = seed, [], None
        self.diverge_at, self.eos_at = diverge_at, eos_at
        self.store = True

    def sample(self, runtime, logits, **kwargs):
        if not self.records:
            runtime.rng = np.random.default_rng(self.seed)
            self.runtime = runtime
        assert runtime is self.runtime, "rotation/hit replaced request RNG owner"
        before = copy.deepcopy(runtime.rng.bit_generator.state)
        token = self.original(runtime, logits, **kwargs)
        # Controlled fault/branch cases still execute the real sampler and draw.
        if self.diverge_at == len(self.records):
            token = (token + 1) % int(runtime.cfg["vocab_size"])
            while token == runtime.cfg["eos_token_id"]:
                token = (token + 1) % int(runtime.cfg["vocab_size"])
        if self.eos_at == len(self.records):
            token = int(runtime.cfg["eos_token_id"])
        self.records.append({"position": runtime.position, "token": token,
            "numeric": numeric(runtime, logits), "rng_before": before,
            "rng_after": copy.deepcopy(runtime.rng.bit_generator.state)})
        return token


class LifecycleTrace:
    """Observe provider rows and successful ACKs, not inferred cache-hit savings."""
    def __init__(self, monkeypatch, client, cache, sampler):
        self.client, self.cache, self.sampler = client, cache, sampler
        self.events, self.inventories, self.tickets = [], [], {}
        self.accepted = cache.head
        core = client._core
        for method in ("lookup", "lookup_child"):
            original = getattr(cache, method)

            def lookup(*args, _method=method, _original=original):
                runtime = cache._runtime
                rng = None if runtime is None else copy.deepcopy(runtime.rng.bit_generator.state)
                result = _original(*args)
                if runtime is not None:
                    assert runtime.rng.bit_generator.state == rng
                ids = tuple(args[0]) if _method == "lookup" else (*args[0], args[1])
                self.events.append(("lookup", ids, result is not None))
                return result

            monkeypatch.setattr(cache, method, lookup)
        stage = cache.stage

        def staged(ids, snapshot, logits):
            assert len(ids) == snapshot.position
            assert snapshot.state_basis.cache_eligible(generated_prefixes=True)
            assert snapshot.state_basis.owner_response_id is None
            ticket = stage(ids, snapshot, logits)
            self.tickets[ticket] = tuple(ids)
            self.events.append(("stage", tuple(ids)))
            return ticket

        monkeypatch.setattr(cache, "stage", staged)
        commit = cache.commit

        def publish(tickets, *, acknowledged):
            assert acknowledged is True
            recent = self.events[self.last_publication():]
            sessions = {event[1] for event in recent if event[0] == "session"}
            acknowledgements = {event[1] for event in recent if event[0] == "ack"}
            assert sessions <= acknowledgements, "published before authenticated retirement"
            assert len(sampler.records) == CAP or (sampler.records and
                sampler.records[-1]["token"] == sampler.runtime.cfg["eos_token_id"]), "published mid-response"
            result = commit(tickets, acknowledged=acknowledged)
            self.accepted = result
            self.events.append(("publish", tuple(tickets)))
            return result

        monkeypatch.setattr(cache, "commit", publish)
        prepare = core._prepare_inventory_locked

        def prepared(*args, **kwargs):
            lookups = [e for e in self.events[self.last_publication():] if e[0] == "lookup"]
            if sampler.store:
                assert lookups and lookups[-1][2] is False, "material issued before root/child lookup miss"
            inventory = prepare(*args, **kwargs)
            self.inventories.append(inventory)
            self.events.append(("prepare", inventory.id, args[2]))
            return inventory

        monkeypatch.setattr(core, "_prepare_inventory_locked", prepared)
        post = core.http.post

        def request(path, **kwargs):
            if path.endswith("/complete"):
                assert cache.head == self.accepted, "accepted head changed before response ACK"
            result = post(path, **kwargs)
            if path == "/v1/runtime/inventories":
                self.events.append(("inventory", kwargs["json"], result.status_code))
            if path == "/v1/runtime/sessions" and result.is_success:
                self.events.append(("session", result.json()["id"]))
            if path.endswith("/complete") and result.is_success:
                self.events.append(("ack", path.split("/")[-2]))
            if path.endswith("/cancel"):
                self.events.append(("cancel", path))
            return result

        monkeypatch.setattr(core.http, "post", request)

    def last_publication(self):
        return next((i + 1 for i in reversed(range(len(self.events))) if self.events[i][0] == "publish"), 0)

    def settle(self):
        audit = self.client.privacy_audit
        assert audit.prepared_stage_rows_reserved == audit.prepared_stage_rows_claimed + audit.prepared_stage_rows_burned
        assert audit.prepared_stage_rows_issued == audit.prepared_stage_rows_reserved + audit.prepared_stage_rows_discarded
        assert audit.preparation_requests_during_online == 0
        assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
        assert len({i.id for i in self.inventories}) == len(self.inventories)
        assert sum(sum(s.request.rows for s in i.stages.values()) for i in self.inventories) == audit.prepared_stage_rows_issued
        assert all(i.status()["reserved"] == 0 for i in self.inventories)
        assert self.client._core._online_active == self.client._core._preparation_active == 0
        assert self.cache.storage_metrics["staged_count"] == 0


def run_request(client, sampler, seed=42, *, store=True, temperature=0, prompt="abc", diverge_at=None, eos_at=None):
    sampler.begin(seed, diverge_at=diverge_at, eos_at=eos_at)
    sampler.store = store
    response = client.responses.create(input=prompt, max_output_tokens=CAP,
                                       temperature=temperature, top_p=0.9, store=store)
    records = copy.deepcopy(sampler.records)
    if store:
        saved = client._core._transformer_conversations[response.id]
        assert saved.snapshot.position == records[-1]["position"]
        arrays = [saved.next_logits, *(v[:row.length] for row in saved.snapshot.caches for v in (row.key, row.value))]
        assert tuple(v.view(np.uint32).tobytes() for v in arrays) == records[-1]["numeric"]
        assert len(saved.token_ids) == saved.snapshot.position + len(saved.pending_token_ids)
    return response, records


def clear_resident(client, model_id):
    cache = client._core._transformer_states[model_id].prefill_cache
    if cache is not None:
        cache.clear()


def same_response(actual, expected):
    assert actual[0].output_text == expected[0].output_text
    assert actual[0].usage.to_dict() == expected[0].usage.to_dict()
    assert actual[0].status == expected[0].status
    assert actual[1] == expected[1]  # Every state bit, sampled token and RNG draw.


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_greedy_cold_hot_root_and_children_before_any_material(tmp_path, monkeypatch, family):
    model_id, experiment = experiment_at(tmp_path, family)
    cache = cache_at(tmp_path / "decode")
    sampler = SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        trace = LifecycleTrace(monkeypatch, client, cache, sampler)
        control = run_request(client, sampler, store=False)
        assert cache.head is None and cache.storage_metrics.get("node_count", 0) == 0
        clear_resident(client, model_id)
        first = run_request(client, sampler)
        same_response(first, control)
        head = cache.head
        assert head is not None
        assert cache.storage_metrics["node_count"] == first[1][-1]["position"] - first[0].usage.input_tokens + 1
        clear_resident(client, model_id)
        before, cursor = client.privacy_audit.to_dict(), len(trace.events)
        disk_before = cache.storage_metrics
        hot = run_request(client, sampler)
        same_response(hot, first)
        assert cache.head == head
        assert cache.storage_metrics["written_bytes"] == disk_before["written_bytes"]
        assert cache.storage_metrics["publications"] == disk_before["publications"]
        recent = trace.events[cursor:]
        assert [e for e in recent if e[0] == "lookup"]
        assert all(e[2] for e in recent if e[0] == "lookup")
        assert not any(e[0] in {"inventory", "prepare", "session", "ack", "stage", "publish"}
                       for e in recent)
        after = client.privacy_audit.to_dict()
        for name in ("preparation_attempts", "prepared_stage_rows_issued", "prepared_stage_rows_reserved",
                     "prepared_stage_rows_claimed", "prepared_stage_rows_burned", "inference_upload_bytes",
                     "inference_download_bytes"):
            assert after[name] == before[name], name
        saved = client._core._transformer_conversations[hot[0].id]
        assert cache.lookup(saved.token_ids[:saved.snapshot.position]) is not None
        if saved.pending_token_ids:
            assert cache.lookup(saved.token_ids) is None
        trace.settle()
    assert cache._runtime is cache._client is cache._key is cache._store.aead is None
    assert cache.head is not None and not cache._store.staged


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_actual_distinct_stochastic_rng_trajectories_match_cold_oracle(tmp_path, monkeypatch, family):
    model_id, experiment = experiment_at(tmp_path, family)
    cache, sampler = cache_at(tmp_path / "decode"), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        trace = LifecycleTrace(monkeypatch, client, cache, sampler)
        # All independent oracles run while durable trie is empty. Global np.seed
        # would not seed these request-local runtime generators.
        seeds = (7, 17, 29)
        controls = {}
        for seed in seeds:
            clear_resident_if_loaded(client, model_id)
            controls[seed] = run_request(client, sampler, seed, store=False, temperature=0.85)
            assert cache.head is None
        paths = set()
        for seed in seeds:
            clear_resident(client, model_id)
            actual = run_request(client, sampler, seed, temperature=0.85)
            same_response(actual, controls[seed])
            paths.add(tuple(r["token"] for r in actual[1]))
        assert len(paths) > 1, "fixture did not exercise distinct stochastic branches"
        for seed in seeds:
            clear_resident(client, model_id)
            before = client.privacy_audit.prepared_stage_rows_issued
            same_response(run_request(client, sampler, seed, temperature=0.85), controls[seed])
            assert client.privacy_audit.prepared_stage_rows_issued == before
        trace.settle()


def clear_resident_if_loaded(client, model_id):
    if model_id in client._core._transformer_states:
        clear_resident(client, model_id)


def test_root_hit_new_child_uses_only_fresh_one_row_demand(tmp_path, monkeypatch):
    model_id, experiment = experiment_at(tmp_path)
    cache, sampler = cache_at(tmp_path / "decode"), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        trace = LifecycleTrace(monkeypatch, client, cache, sampler)
        oracle = run_request(client, sampler, store=False, diverge_at=0)
        clear_resident(client, model_id)
        run_request(client, sampler)
        clear_resident(client, model_id)
        cursor, before = len(trace.events), client.privacy_audit.to_dict()
        actual = run_request(client, sampler, diverge_at=0)
        same_response(actual, oracle)
        recent = trace.events[cursor:]
        lookups = [e for e in recent if e[0] == "lookup"]
        assert lookups[0][2] and any(not e[2] for e in lookups[1:])
        inventories = [e[1] for e in recent if e[0] == "inventory"]
        assert inventories and all(body["rows"] == 1 for body in inventories)
        state = client._core._transformer_states[model_id]
        compiled = client._core._compiled_public_decoder(state,
            max_input_tokens=actual[0].usage.input_tokens, max_new_tokens=CAP)
        expected = compiled.prepared_stage_rows(1, 0, continuation=True)
        assert all(body["stage_rows"] == expected for body in inventories)
        issued = client.privacy_audit.prepared_stage_rows_issued - before["prepared_stage_rows_issued"]
        assert issued == len(inventories) * sum(expected.values())
        sessions = {e[1] for e in recent if e[0] == "session"}
        assert len(sessions) == len(inventories)
        assert sessions == {e[1] for e in recent if e[0] == "ack"}
        trace.settle()


@pytest.mark.parametrize("mode", ["store_false", "stream_close", "cancel", "retirement_failure", "stream_failure"])
def test_unacknowledged_or_unstored_turn_never_changes_prior_trusted_head(tmp_path, monkeypatch, mode):
    model_id, experiment = experiment_at(tmp_path)
    cache, sampler = cache_at(tmp_path / "decode"), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        trace = LifecycleTrace(monkeypatch, client, cache, sampler)
        warm = run_request(client, sampler)
        head, count = cache.head, cache.storage_metrics["node_count"]
        clear_resident(client, model_id)
        sampler.begin(42)
        cursor = len(trace.events)
        request = dict(input="different", max_output_tokens=CAP, temperature=0)
        if mode == "store_false":
            sampler.store = False
            response = client.responses.create(**request, store=False)
            assert response.id not in client._core._transformer_conversations
        elif mode in {"stream_close", "cancel"}:
            stream = client.responses.create(**request, stream=True)
            created = next(stream)
            # Drain to second delta so at least root/child state was staged.
            deltas = 0
            for event in stream:
                deltas += event.type == "response.output_text.delta"
                if deltas == 2:
                    break
            if mode == "cancel":
                client.responses.cancel(created.response["id"])
                with pytest.raises(Exception, match="cancel"):
                    list(stream)
            stream.close()
        else:
            post = client._core.http.post

            def fail_retirement(path, **kwargs):
                result = post(path, **kwargs)
                if path.endswith("/complete"):
                    return httpx.Response(503, json={"error": {"message": "injected retirement failure"}}, request=result.request)
                return result

            monkeypatch.setattr(client._core.http, "post", fail_retirement)
            with pytest.raises(Exception, match="retirement|503|injected"):
                if mode == "stream_failure":
                    list(client.responses.create(**request, stream=True))
                else:
                    client.responses.create(**request)
        assert cache.head == head and cache.storage_metrics["node_count"] == count
        assert not any(e[0] == "publish" for e in trace.events[cursor:])
        assert cache.storage_metrics["staged_count"] == 0
        trace.settle()
        assert warm[0].id in client._core._transformer_conversations


def test_eos_is_sampled_fresh_but_never_executed_or_indexed(tmp_path, monkeypatch):
    model_id, experiment = experiment_at(tmp_path)
    cache, sampler = cache_at(tmp_path / "decode"), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        trace = LifecycleTrace(monkeypatch, client, cache, sampler)
        first = run_request(client, sampler, eos_at=1)
        assert first[0].status == "completed" and first[0].usage.output_tokens == 1
        clear_resident(client, model_id)
        before = client.privacy_audit.prepared_stage_rows_issued
        hot = run_request(client, sampler, eos_at=1)
        same_response(hot, first)
        assert client.privacy_audit.prepared_stage_rows_issued == before
        saved = client._core._transformer_conversations[hot[0].id]
        assert saved.pending_token_ids == []
        assert cache.lookup_child(saved.token_ids, int(sampler.runtime.cfg["eos_token_id"])) is None
        trace.settle()


def test_corrupt_numeric_page_fails_closed_before_provider_issuance(tmp_path, monkeypatch):
    model_id, experiment = experiment_at(tmp_path)
    cache, sampler = cache_at(tmp_path / "decode"), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        LifecycleTrace(monkeypatch, client, cache, sampler)
        run_request(client, sampler)
        head = cache.head
        clear_resident(client, model_id)
        page = next(iter(cache._store.pages.values()))
        path = tmp_path / "decode" / page["record"]["name"]
        raw = path.read_bytes()
        path.write_bytes(raw[:-1])
        before = client.privacy_audit.to_dict()
        with pytest.raises(Exception, match="commitment|extent|truncat"):
            run_request(client, sampler)
        after = client.privacy_audit.to_dict()
        assert after["prepared_stage_rows_issued"] == before["prepared_stage_rows_issued"]
        assert after["prepared_stage_rows_reserved"] == before["prepared_stage_rows_reserved"]
        assert cache.head == head and cache.storage_metrics["staged_count"] == 0
        path.write_bytes(raw)


def test_one_client_owner_and_one_active_response_exclusion(tmp_path, monkeypatch):
    _, experiment = experiment_at(tmp_path)
    cache, sampler = cache_at(tmp_path / "decode"), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        with pytest.raises(Exception, match="owner|client|belong|share"):
            roles.client(decode_cache=cache, background_inventory_refill=False)
        sampler.begin(42)
        stream = client.responses.create(input="abc", max_output_tokens=CAP, temperature=0, stream=True)
        next(stream)
        before = client.privacy_audit.prepared_stage_rows_issued
        with pytest.raises(Exception, match="active|concurrent|busy|response"):
            client.responses.create(input="other", max_output_tokens=CAP, temperature=0)
        assert client.privacy_audit.prepared_stage_rows_issued == before
        stream.close()
        assert cache.storage_metrics["staged_count"] == 0


def test_async_cancel_drains_native_work_before_discard(tmp_path, monkeypatch):
    model_id, experiment = experiment_at(tmp_path)
    cache, sampler = cache_at(tmp_path / "decode"), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        trace = LifecycleTrace(monkeypatch, client, cache, sampler)
        run_request(client, sampler)
        head = cache.head
        clear_resident(client, model_id)
        entered, release = threading.Event(), threading.Event()
        original = SemanticDecoderRuntime.prepare_ids

        def blocked(runtime, ids):
            entered.set()
            assert release.wait(15)
            return original(runtime, ids)

        monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", blocked)
        from pllm.runtime.client import AsyncResponsesResource

        async def run():
            sampler.begin(42)
            task = asyncio.create_task(AsyncResponsesResource(client.responses).create(
                input="different", max_output_tokens=CAP, temperature=0))
            try:
                assert await asyncio.to_thread(entered.wait, 15)
                task.cancel()
                await asyncio.sleep(0.05)
                assert not task.done(), "cancellation escaped before native work drained"
                assert cache.head == head
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 15)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run())
        assert cache.head == head and cache.storage_metrics["staged_count"] == 0
        trace.settle()


@pytest.mark.parametrize("with_prior", [False, True])
def test_async_cancel_after_final_publication_restores_only_remembered_prior_head(tmp_path, monkeypatch, with_prior):
    _, experiment = experiment_at(tmp_path)
    cache, sampler = cache_at(tmp_path / "decode"), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles, roles.client(
            decode_cache=cache, background_inventory_refill=False) as client:
        trace = LifecycleTrace(monkeypatch, client, cache, sampler)
        if with_prior:
            run_request(client, sampler)
        prior = cache.head
        prior_nodes = cache.storage_metrics.get("node_count", 0)
        entered, release = threading.Event(), threading.Event()
        published, restored = [], []
        commit, restore = cache.commit, cache._restore_head

        def blocked_commit(tickets, *, acknowledged):
            result = commit(tickets, acknowledged=acknowledged)
            published.append((result, cache.storage_metrics["disk_bytes"]))
            entered.set()
            assert release.wait(15)
            return result

        def remembered_restore(previous):
            assert previous == prior
            assert cache.head == published[-1][0], "rollback crossed another response's publication"
            restored.append(previous)
            return restore(previous)

        monkeypatch.setattr(cache, "commit", blocked_commit)
        monkeypatch.setattr(cache, "_restore_head", remembered_restore)
        from pllm.runtime.client import AsyncResponsesResource

        async def run():
            sampler.begin(42)
            task = asyncio.create_task(AsyncResponsesResource(client.responses).create(
                input="different", max_output_tokens=CAP, temperature=0))
            try:
                assert await asyncio.to_thread(entered.wait, 15)
                assert cache.head == published[-1][0] and cache.head != prior
                task.cancel()
                await asyncio.sleep(0.05)
                assert not task.done(), "finalization race escaped before worker drain"
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 15)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run())
        assert restored == [prior]
        assert cache.head == prior and cache.storage_metrics["node_count"] == prior_nodes
        assert cache.storage_metrics["disk_bytes"] >= published[-1][1]
        assert cache.storage_metrics["staged_count"] == 0
        assert len(client._core._transformer_conversations) == int(with_prior)
        # Publication/rollback never refunds issued or already reserved material.
        trace.settle()


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_sdk_fresh_interpreter_trusted_head_full_hit_no_material(tmp_path, monkeypatch, family):
    model_id, experiment = experiment_at(tmp_path, family)
    key = os.urandom(32)
    cache, sampler = cache_at(tmp_path / "decode", key), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles:
        with roles.client(decode_cache=cache, background_inventory_refill=False,
                          bundle_cache_dir=tmp_path / "bundle") as client:
            result = run_request(client, sampler, seed=17, temperature=0.85)[0]
            expected = commitment(client, result.id)
            head = cache.head
        assert cache._store.aead is cache._runtime is None
        code = """
import json, os, sys
import numpy as np
import pllm
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.state import DecodeCache, DecodeCacheHead
from test_decode_cache_sdk import commitment
p = json.load(sys.stdin)
original = SemanticDecoderRuntime.sample
calls = 0
def sample(runtime, logits, **kwargs):
    global calls
    if calls == 0:
        runtime.rng = np.random.default_rng(17)
    calls += 1
    return original(runtime, logits, **kwargs)
SemanticDecoderRuntime.sample = sample
cache = DecodeCache(p['root'], bytes.fromhex(p['key']), head=DecodeCacheHead.from_spec(p['head']),
    max_snapshot_bytes=1<<20, max_transient_bytes=16<<20)
with pllm.OpenAI(experiment=pllm.Experiment.from_spec(p['experiment']), decode_cache=cache,
    base_url=os.environ['PLLM_DECODE_INFERENCE'], api_key=os.environ['PLLM_DECODE_INFERENCE_KEY'],
    preparation_base_url=os.environ['PLLM_DECODE_PREPARATION'],
    preparation_api_key=os.environ['PLLM_DECODE_PREPARATION_KEY'], background_inventory_refill=False,
    bundle_cache_dir=p['bundle']) as client:
    post = client._core.http.post
    def forbid(path, **kwargs):
        assert path not in {'/v1/runtime/inventories', '/v1/runtime/sessions'}, 'full hit issued new material'
        assert not path.endswith(('/complete', '/ready')), 'full hit opened provider session'
        return post(path, **kwargs)
    client._core.http.post = forbid
    response = client.responses.create(model=p['model'], input='abc', max_output_tokens=4, temperature=0.85, top_p=0.9)
    assert client.privacy_audit.prepared_stage_rows_issued == client.privacy_audit.prepared_stage_rows_reserved == 0
    assert client.privacy_audit.inference_upload_bytes == client.privacy_audit.inference_download_bytes == 0
    assert calls == p['calls']
    print(json.dumps(commitment(client, response.id)))
"""
        payload = {"root": str(tmp_path / "decode"), "key": key.hex(), "head": head.to_spec(),
            "bundle": str(tmp_path / "bundle"), "model": model_id, "experiment": experiment.to_spec(),
            "calls": len(sampler.records)}
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.resolve()),
            "PLLM_DECODE_INFERENCE": roles.inference_url, "PLLM_DECODE_INFERENCE_KEY": roles._inference_key,
            "PLLM_DECODE_PREPARATION": roles.preparation_url, "PLLM_DECODE_PREPARATION_KEY": roles._preparation_key,
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        child = subprocess.run([sys.executable, "-c", code], input=json.dumps(payload), text=True,
                               capture_output=True, check=True, timeout=90, env=env)
        assert json.loads(child.stdout) == expected


def test_wrong_head_key_workload_or_source_rejects_before_material(tmp_path, monkeypatch):
    model_id, experiment = experiment_at(tmp_path)
    key = os.urandom(32)
    cache, sampler = cache_at(tmp_path / "decode", key), SamplingRecorder(monkeypatch)
    with build_roles(experiment, engine_threads=1) as roles:
        with roles.client(decode_cache=cache, background_inventory_refill=False) as client:
            run_request(client, sampler)
            head = cache.head
        for wrong_key, wrong_head, max_output in (
                (os.urandom(32), head, CAP),
                (key, DecodeCacheHead(head.revision, "0" * 64, head.name, head.digest, head.bytes), CAP),
                (key, head, CAP - 1)):
            owner = cache_at(tmp_path / "decode", wrong_key, wrong_head)
            with roles.client(decode_cache=owner, background_inventory_refill=False) as client:
                with pytest.raises(Exception):
                    client.responses.create(model=model_id, input="abc", max_output_tokens=max_output, temperature=0)
                assert client.privacy_audit.prepared_stage_rows_issued == client.privacy_audit.prepared_stage_rows_reserved == 0
                assert owner.head == wrong_head
    # A genuinely different generated source, not merely a model name change.
    _, other = experiment_at(tmp_path / "other", "qwen3")
    owner = cache_at(tmp_path / "decode", key, head)
    with build_roles(other, engine_threads=1) as roles, roles.client(
            decode_cache=owner, background_inventory_refill=False) as client:
        with pytest.raises(Exception, match="context|source|binding"):
            client.responses.create(input="abc", max_output_tokens=CAP, temperature=0)
        assert client.privacy_audit.prepared_stage_rows_issued == client.privacy_audit.prepared_stage_rows_reserved == 0

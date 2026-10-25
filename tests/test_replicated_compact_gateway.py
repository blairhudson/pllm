"""UNRUN D49 gateway gates; matching noneditable wheel + explicit slot required.

Trusted LOCAL CLIENT gateway sees plaintext API input. This is NOT an opaque
customer relay/security claim. No trained source, alternative execution or V3
oracle: use the existing V5 active SDK source unchanged.

Native factory/storage/directory ABI is frozen in I source03. Matching wheel
and real lifetime/numerical qualification still required. Missing observation
FAILS, never skips to green.
"""
import asyncio
from contextlib import ExitStack, suppress
from contextvars import ContextVar
import threading

import httpx
import pytest

from pllm import AsyncOpenAI, OpenAI, _native
from pllm.runtime import replicated_worker as worker
from pllm.runtime.replicated_protocol import ROLES
from pllm.runtime.replicated_source import ReplicatedSource
from pllm.runtime.sidecar import create_sidecar_app
from test_network_benchmark import free_port, running_http
from test_replicated_policy_sdk import observe_publishers, assert_publishers_retired
from test_replicated_public_weights import _NativeUseObserver
from test_replicated_sdk import experiment_fixture, active_sessions


def _require_sealed_storage_reader():
    # Actual readonly property in I's frozen source02 binding.
    # Native must derive this from the retained Program/storage discriminant,
    # not config/Manifest echo, Python flags, constructor entry or a digest.
    # Publisher may report None before seal; these checks use SEALED owners.
    # No owner adoption. A stale/missing ABI fails before generating the fixture.
    for native_type in (_native.ReplicatedPublisher, _native.ReplicatedWorker):
        if not hasattr(native_type, "_storage_kind"):
            raise RuntimeError("I/E sealed native publisher+worker _storage_kind ABI unavailable")

    def read(owner):
        kind = owner._storage_kind
        assert kind in ("compact", "expanded"), "actual sealed Program required"
        return kind

    return read


def _experiment(path):
    original = experiment_fixture(path, family="qwen2", weights="public", active=True)
    return original.with_params(pipeline__components__linear__range_policy="checked",
        pipeline__components__linear__max_abs_activation=256,
        pipeline__components__linear__storage="compact")


class _Owners:
    """IO/lifetime observations only; every algorithm/acceptance call delegates."""
    def __init__(self, monkeypatch, read_storage, *, cancel=False):
        self.events, self.created = [], []
        self.entered, self.release = threading.Event(), threading.Event()
        self.current_role = ContextVar("compact_gateway_worker_role")
        self.generations = [0, 0, 0]
        self.lock = threading.Lock()
        self.publishers = observe_publishers(monkeypatch)
        original_publish = ReplicatedSource.publisher
        original_native = _native.ReplicatedWorker
        original_reserve = worker.reserve_incremental_memory
        parent = self

        def publish(source, *args, **kwargs):
            observed, manifest = original_publish(source, *args, **kwargs)
            assert read_storage(observed.owner) == "compact"
            parent.events.append(("publisher_compact_sealed", len(parent.publishers)))
            return observed, manifest

        class Proxy:
            def __init__(self, owner, role):
                self.owner, self.role, self.use = owner, role, _NativeUseObserver()
                with parent.lock:
                    self.key = (role, parent.generations[role])
                    parent.generations[role] += 1
                    parent.created.append(self.key)

            def __getattr__(self, name):
                return getattr(self.owner, name)

            def append_public_source(self, *args):
                with self.use.operation():
                    result = self.owner.append_public_source(*args)
                    parent.events.append(("appended", self.key))
                    return result

            def seal_public_source(self):
                with self.use.operation():
                    result = self.owner.seal_public_source()
                    assert read_storage(self.owner) == "compact"
                    parent.events.append(("worker_compact_sealed", self.key))
                    return result

            def call(self, *args):
                with self.use.operation():
                    result = self.owner.call(*args)
                    if cancel and self.role == 0 and args[0] == "advance" and not parent.entered.is_set():
                        parent.entered.set()
                        assert parent.release.wait(45), "compact execution IO gate timed out"
                        parent.events.append(("drained", self.key))
                    return result

            def close(self):
                self.use.close(self.owner.close)
                parent.events.append(("closed", self.key))

        class Constructors:
            def __new__(cls, *args, **kwargs):
                raise AssertionError("compact worker attempted legacy constructor fallback")

            @staticmethod
            def _from_source_request(manifest, role, *, server_policy):
                # Existing real constructor; admitted immutable policy forwarded
                # byte-for-byte. Invocation alone does NOT prove storage kind.
                return Proxy(original_native._from_source_request(
                    manifest, role, server_policy=server_policy), role)

        def reserve(*args, **kwargs):
            context = original_reserve(*args, **kwargs)
            role = parent.current_role.get()
            with parent.lock:
                key = (role, parent.generations[role])

            class Reservation:
                def __enter__(self):
                    return context.__enter__()

                def __exit__(self, *exc):
                    result = context.__exit__(*exc)
                    parent.events.append(("refunded", key))
                    return result

            return Reservation()

        monkeypatch.setattr(ReplicatedSource, "publisher", publish)
        monkeypatch.setattr(worker._native, "ReplicatedWorker", Constructors)
        monkeypatch.setattr(worker, "reserve_incremental_memory", reserve)

    def middleware(self, role):
        async def identify(request, call_next):
            token = self.current_role.set(role)
            try:
                return await call_next(request)
            finally:
                self.current_role.reset(token)
        return identify

    def assert_retired(self):
        assert {role for role, _ in self.created} == {0, 1, 2}
        for key in self.created:
            assert self.events.index(("appended", key)) < self.events.index(("worker_compact_sealed", key))
            assert self.events.index(("closed", key)) < self.events.index(("refunded", key))
        assert_publishers_retired(self.publishers)


def _worker_http(stack, experiment, observed):
    origins = [f"http://127.0.0.1:{free_port()}" for _ in ROLES]
    keys = [f"compact-local-client-{i}-12345678" for i in range(3)]
    for i, role in enumerate(ROLES):
        peers = {other: (origins[j], f"compact-directed-{i}-{j}-12345678",
                        f"compact-directed-{j}-{i}-12345678")
                 for j, other in enumerate(ROLES) if i != j}
        app = worker.create_replicated_worker_app(experiment, role, keys[i], peers)
        app.middleware("http")(observed.middleware(i))
        origin = stack.enter_context(running_http(app, int(origins[i].rsplit(":", 1)[1])))
        assert origin == origins[i]
    return {role: (origins[i], keys[i]) for i, role in enumerate(ROLES)}


def test_compact_v5_local_gateway_both_private_feedback_branches(tmp_path, monkeypatch):
    read_storage = _require_sealed_storage_reader()  # Missing matching-wheel capability FAILS.
    experiment = _experiment(tmp_path)
    observed = _Owners(monkeypatch, read_storage)
    captured = []
    with ExitStack() as stack:
        connections = _worker_http(stack, experiment, observed)
        sdk = stack.enter_context(OpenAI(experiment=experiment, role_connections=connections, timeout=120))
        sdk._core._result_hook = captured.append
        gateway = create_sidecar_app(client=sdk, local_api_key="trusted-local-client")
        origin = stack.enter_context(running_http(gateway, free_port()))
        with httpx.Client(base_url=origin, timeout=120, trust_env=False) as client:
            # Existing independent V5 oracle, NOT native V3 [1,0]: Q/K=0,
            # V=I, O=-I/2, +/-5/32 embeddings, signed head, tiny MLP.
            # Rendering gives two zero-embedding UNKs then two source tokens;
            # >1/32 margin preserves each branch through private feedback.
            for prompt, expected in [("hello hello", (4, 4)), ("world hello", (1, 1))]:
                result = client.post("/v1/responses", headers={"authorization": "Bearer trusted-local-client"},
                    json={"model": sdk._core.default_model, "input": prompt, "max_output_tokens": 2, "temperature": 0})
                assert result.status_code == 200, (result.status_code, result.text)
                result.raise_for_status()
                response = result.json()
                assert response["usage"]["input_tokens"] == 4
                assert response["usage"]["output_tokens"] == 2
                # delegated_client's fixed-count contract: EOS is selected,
                # never an early stop. Even (1,1) ends at the requested limit.
                assert response["status"] == "incomplete"
                assert response["incomplete_details"]["reason"] == "max_output_tokens"
                assert captured[-1].selected == expected
                assert observed.publishers[-1].finish_calls == 1
                assert active_sessions(sdk._core) == [0, 0, 0]
        assert len(captured) == len(observed.publishers) == 2
        sealed = [key for event, key in observed.events if event == "worker_compact_sealed"]
        assert len(sealed) == 6 and {role for role, _ in sealed} == {0, 1, 2}
        audit = sdk._core.audit.to_dict()
        assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
        observed.assert_retired()


@pytest.mark.asyncio
async def test_compact_v5_execution_cancel_drains_before_owner_refund(tmp_path, monkeypatch):
    read_storage = _require_sealed_storage_reader()  # No stale wheel success/skip.
    experiment = _experiment(tmp_path)
    observed = _Owners(monkeypatch, read_storage, cancel=True)
    captured = []
    with ExitStack() as stack:
        connections = _worker_http(stack, experiment, observed)
        sdk_owner = await asyncio.to_thread(AsyncOpenAI, experiment=experiment,
            role_connections=connections, timeout=120)
        async with asyncio.timeout(300), sdk_owner as sdk:
            core = sdk.sync._core
            core._result_hook = captured.append
            task = asyncio.create_task(sdk.responses.create(input="hello hello", max_output_tokens=2))
            gate = asyncio.create_task(asyncio.to_thread(observed.entered.wait, 60))
            try:
                await asyncio.wait({task, gate}, timeout=60, return_when=asyncio.FIRST_COMPLETED)
                if task.done() and not observed.entered.is_set():
                    await task  # Preserve actual admission/producer exception.
                assert observed.entered.is_set(), "request never reached compact execution"
                # Publisher snapshot/seal returned and all worker source seals
                # completed; observations read retained native Programs, not params.
                assert any(event == "publisher_compact_sealed" for event, _ in observed.events)
                sealed = [key for event, key in observed.events if event == "worker_compact_sealed"]
                assert {role for role, _ in sealed} == {0, 1, 2}
                assert not captured and not core.records
                task.cancel()
                await asyncio.sleep(0)
                assert not captured and not core.records
                observed.release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert core._active is None and not captured and not core.records
                assert await asyncio.to_thread(active_sessions, core) == [0, 0, 0]
                key = next(key for key in observed.created if key[0] == 0)
                assert observed.events.index(("drained", key)) < observed.events.index(("closed", key))
                assert all(owner.finish_calls == 0 for owner in observed.publishers)
                observed.assert_retired()
                audit = core.audit.to_dict()
                assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
            finally:
                observed.release.set()
                # Wake/drain ONLY the test observer waiter on an error path;
                # all success/gate assertions above ran before this cleanup.
                observed.entered.set()
                await gate
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

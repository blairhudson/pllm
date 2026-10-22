"""Python pricing/error routing with native and HTTP doubles, not serving evidence."""

import json
from types import SimpleNamespace

import httpx
import pytest

from pllm import OpenAI, _native
from pllm.runtime import benchmark_delegated, benchmark_memory, delegated_transport
from pllm.runtime.client import ProtocolError
from pllm.runtime.fss_client import FssClient
from pllm.runtime.fss_protocol import ROLES
from test_fss_benchmark import contract, experiment, imported_source, source  # noqa: F401


PUBLIC_REJECTION = (
    "FuseFSS program transport admission rejected: stage=aggregate_reservation error=Resource"
    " observed=67108865 limit=67108864 metadata_bytes=1024 live_array_bytes=2048"
    " source_table_cache_bytes=4096 transport_peak_bytes=8192 fixed_metadata_bytes=65536"
    " batch_native_bytes=0 batch_python_bytes=0"
)


def test_imported_pricing_preserves_native_public_diagnostic(experiment, imported_source, monkeypatch):
    loaded, _, _ = imported_source
    calls = []

    def reject(owner, raw_contract):
        assert owner is loaded.native_program and not loaded._closed
        assert raw_contract == loaded.contract_bytes
        calls.append(owner)
        raise ValueError(PUBLIC_REJECTION)

    monkeypatch.setattr(_native, "fss_transport_admission", reject)
    with pytest.raises(ValueError) as direct:
        benchmark_delegated.inspect_fss_imported(loaded)
    assert str(direct.value) == PUBLIC_REJECTION
    with pytest.raises(benchmark_memory.BenchmarkMemoryError) as wrapped:
        benchmark_memory.benchmark_memory(experiment, fss_imported=loaded)
    assert str(wrapped.value) == "FuseFSS imported source pricing rejected"
    assert str(wrapped.value.__cause__) == PUBLIC_REJECTION
    assert len(calls) == 2
    assert not loaded._closed  # Rejection does not steal the caller's source owner.


@pytest.mark.parametrize("payload", [
    PUBLIC_REJECTION.encode(),  # A role cannot promote even a valid pricing record.
    b'{"role":"worker_a","callback":{"payload":"private-input-share"}}',
    b"Bearer private-credential seed=private-seed tensor=/private/source.safetensors",
], ids=["pricing-record", "callback-payload", "credential-seed-path"])
def test_sdk_role_rejection_never_exposes_remote_diagnostic(experiment, contract, monkeypatch, payload):
    monkeypatch.setattr(type(experiment), "resolve", lambda self: SimpleNamespace(
        client_runtime="compiled_fss_v1", model="fss-diagnostic-double"))
    raw_contract = json.dumps(contract).encode()
    frozen = SimpleNamespace(contract=contract, runtime_contract=lambda *args: raw_contract)
    # Keep ordinary SDK request/retirement and transport error mapping. Only
    # source/tokenizer/native qualification and network sockets are doubled.
    monkeypatch.setattr(FssClient, "_ensure_source", lambda self: frozen)
    monkeypatch.setattr(FssClient, "_encode", lambda *args: (None, (1,)))
    requests = []
    clients = []
    async_client = httpx.AsyncClient

    def reply(request):
        requests.append(request.url.path)
        return httpx.Response(400, stream=httpx.ByteStream(payload), headers={
            "x-pllm-error-code": payload.decode(),
        })

    def client(**kwargs):
        value = async_client(**kwargs, transport=httpx.MockTransport(reply))
        clients.append(value)
        return value

    monkeypatch.setattr(delegated_transport.httpx, "AsyncClient", client)
    monkeypatch.setattr(_native, "FssClientRequest",
                        lambda *args: pytest.fail("role rejection reached native input owner"), raising=False)
    connections = {role: (f"http://127.0.0.1:{18000 + index}", f"private-credential-{index}")
                   for index, role in enumerate(ROLES)}
    with OpenAI(experiment=experiment, role_connections=connections, timeout=5) as sdk:
        with pytest.raises(ProtocolError) as caught:
            sdk.responses.create(input="hello", max_output_tokens=1, store=False)
        assert str(caught.value) == "GKM worker rejected its one-use request"
        assert sdk._core._active is None
        assert not sdk._core.records
        assert sdk._core._completed_program is None
    assert requests == ["/v1/fss/manifest"]
    assert len(clients) == 3 and all(client.is_closed for client in clients)

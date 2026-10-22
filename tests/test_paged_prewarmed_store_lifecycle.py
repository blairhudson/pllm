"""Real public artifact snapshots and seeded row owners under the SDK lease.

Only four-byte public matrices and installed bounded mask primitives execute.
The model dispatcher and two HTTP services are doubles; authority, artifact
authentication/import, paged reads, masked linear codec, row claims/burns and SDK
foreground ownership are production code. No compiled source admission or model
quality is claimed. These are not memory measurements.

This candidate has SHA-256-authenticated *public* weight snapshots, not an AEAD
prepared-row store. Existing encrypted decode/KV stores are different opt-ins;
combining them here would invent a candidate. Seeded rows expand an already
committed deterministic cursor, never mint a new seed/inventory online.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import secrets
from types import SimpleNamespace

import httpx
import msgpack
import numpy as np
import pytest

from pllm.preparation import PreparedInventory as InventoryPolicy
from pllm.protocols.bundle_transport import ClientBundleTransport
from pllm.runtime._native_support import extension
from pllm.runtime.bundle_artifacts import (
    ArtifactError, export_bundle, parse_manifest, reconstruct_document,
)
from pllm.runtime.bundle_storage import FilePayload
from pllm.runtime.client import ModelError, ResponsesResource, RuntimeClient
from pllm.runtime.paged import PagedGEMM
from pllm.runtime.preparation_protocol import (
    PreparationRequest, derive_online_attempt_id, expand_output_mask,
    expand_preparation_mask, seeded_ring_profile,
)
from pllm.runtime.prepared_accounting import material_accounting
from pllm.runtime.prepared_queue import PreparedExecutionQueue, PreparedQueueClosed, prepared_attempt
from pllm.runtime.protocol import ProtocolError, encode_length_prefixed, iter_length_prefixed
from pllm.runtime.stage_protocol import (
    MaskedStageRequest, MaskedStageResponse,
)
from pllm.runtime.transformer_client import (
    ClientBundle, PreparedInventory, PreparedRemoteLinear, PreparedStageRows, TransformerClientError,
)
from pllm.runtime.types import Response, ResponseEvent


MODEL = "public-two-by-two-store-fixture"
PRIVATE = "private prompt stays on caller"
WEIGHTS = np.asarray([[1, -2], [3, 4]], dtype=np.int8)
WEIGHT_DIGEST = hashlib.sha256(WEIGHTS.tobytes()).hexdigest()
PROFILE = seeded_ring_profile(2 * 127 * 127)


def artifact_fixture():
    """An explicitly synthetic schema-2 publication, not a guessed source pin."""
    body = hashlib.sha256(b"two-by-two-test-body:" + WEIGHTS.tobytes()).hexdigest()
    specs = [{"id": name, "op": "linear", "in_features": 2, "out_features": 2}
             for name in ("body", "lm_head")]
    commitment = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
    metadata = {"client_runtime": "masked_transformer_v1", "privacy_mode": "public",
                "body_fingerprint": body, "stage_commitment": commitment,
                "architecture": "test-fixture", "stage_count": 2}
    stages = {}
    scales = np.ones(2, dtype="<f4").tobytes()
    for spec in specs:
        stages[spec["id"]] = {**spec, "weight_bits": 8, "activation_bits": 8,
            "ring": PROFILE.ring, "modulus": PROFILE.modulus, "wire_bits": PROFILE.wire_bits,
            "weight_scales": scales, "weight_digest": WEIGHT_DIGEST,
            "seeded_profile": PROFILE.to_dict(), "source_keys": [f"fixture.{spec['id']}.weight"]}
    stages["lm_head"]["client_weight"] = {"ref": "lm_head", "layout": "linear"}
    document = {"v": 2, "runtime": "masked_transformer", "model": MODEL,
        "manifest": {"id": MODEL, "architecture": "test-fixture", "stages": specs,
                     "metadata": {**metadata, "source_lock_digest": body}},
        "config": {"bos_token_id": 0, "eos_token_id": 1, "hidden_size": 2, "vocab_size": 2},
        "tokenizer": {"kind": "alphabet", "alphabet": "ab"}, "local_tensors": {},
        "client_weights": {"lm_head": {"dtype": "i1", "shape": [2, 2],
                                      "data": WEIGHTS.tobytes(), "scales": scales}},
        "stages": stages,
        "privacy": {"mode": "public", "preprocessed": True, "protocol": "seeded-inventory/v1",
                    "model_privacy_threat_model": "public_weights", "body_fingerprint": body,
                    "stage_commitment": commitment, "weight_bits": 8, "activation_bits": 8}}
    raw = msgpack.packb(document, use_bin_type=True)
    fingerprint = hashlib.sha256(raw).hexdigest()
    exported = export_bundle(raw)
    manifest = parse_manifest(exported.manifest, fingerprint=fingerprint, size=len(raw))
    return SimpleNamespace(raw=raw, fingerprint=fingerprint, exported=exported,
                           manifest=manifest, metadata=metadata, body=body)


def import_bundle(publication, delivery):
    """Use the same authenticated document/file import as artifact delivery."""
    files = []

    def read(row):
        data = publication.exported.objects[row["sha256"]]
        if delivery.get_params()["storage"] == "paged" and row["metadata"]["dtype"] == "i1":
            payload = FilePayload([data], size=row["size"], digest=row["content_sha256"], native=True)
            files.append(payload)
            return payload
        return bytes(data)

    try:
        document = reconstruct_document(publication.manifest, read)
        return ClientBundle._from_document(document, storage=delivery.get_params()["storage"], threads=1)
    finally:
        # The matrix must retain its own immutable snapshot, not this source owner.
        for payload in files:
            payload.close()


class IdleScheduler:
    """Record scheduling, not issuance. Explicitly run only eligibility probes."""
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        future = Future()
        self.calls.append((function, args, future))
        return future

    def shutdown(self, **kwargs):
        for _, _, future in self.calls:
            future.cancel()


@pytest.fixture
def installed_primitives():
    native = extension()
    if native is None or not all(hasattr(native, name) for name in ("PublicFileWriter", "PagedMatrix", "PreparedMaskRows")):
        pytest.skip("installed bounded public-file/paged/mask primitives required; never build in this test")
    return native


@pytest.fixture
def harness(monkeypatch, tmp_path, installed_primitives):
    delivery = ClientBundleTransport(**json.loads(json.dumps(
        ClientBundleTransport("artifacts", storage="paged").get_params())))
    policy = InventoryPolicy(**json.loads(json.dumps(
        InventoryPolicy("prewarm", rows=4, refill="idle").get_params())))
    publication = artifact_fixture()
    bundle = import_bundle(publication, delivery)
    matrix = bundle.stages["lm_head"].client_weight
    assert type(matrix) is PagedGEMM and matrix.weight_bytes == 4
    h = SimpleNamespace(http=[], remote_rows=0, live=True, response_count=0, stage_row=0,
                        leases=[], retired=[], requests=[], fault=None, lost_ack=False,
                        delivery=delivery, policy=policy, publication=publication, bundle=bundle)

    def handler(request):
        h.http.append(request)
        path = request.url.path
        if request.method == "GET" and path == f"/v1/runtime/models/{MODEL}":
            return httpx.Response(200, json={"id": MODEL, "metadata": publication.metadata,
                "client_bundle": {"schema": 2, "sha256": publication.fingerprint,
                                  "size": len(publication.raw), "etag": f'"{publication.fingerprint}"'}})
        if request.method == "GET" and path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL, "runtime": publication.metadata,
                "preparation": {**publication.metadata, "protocol": "seeded-inventory/v1",
                                "weight_bits": 8, "activation_bits": 8}}]})
        if request.method == "GET" and "/inventories/" in path:
            return httpx.Response(200 if h.live else 410, json={"status": "ready", "next_row": h.remote_rows})
        if request.method == "POST" and path == "/v1/runtime/sessions":
            values = json.loads(request.content)
            assert values["execution"] == "seeded-preparation"
            h.stage_row = values["inventory_start"]
            h.remote_rows = h.stage_row + values["inventory_rows"]
            h.response_count += 1
            return httpx.Response(200, json={"id": "session", "response_id": f"response-{h.response_count}"})
        if request.method == "POST" and path == "/v1/runtime/sessions/session/stages/body":
            replies = []
            for frame in iter_length_prefixed(request.content):
                stage_request = MaskedStageRequest.unpack(frame)
                h.requests.append(stage_request)
                row = h.stage_row
                assert stage_request.correlation_id == h.expected_ids[row]
                assert stage_request.body_fingerprint == publication.body
                assert stage_request.weight_digest == WEIGHT_DIGEST
                assert stage_request.weight_bits == stage_request.activation_bits == 8
                # Already committed synthetic correction. Network double never
                # receives the seed, private activation, or an unmasked result.
                value = (stage_request.masked_input.astype(np.int64) @ WEIGHTS.astype(np.int64).T
                         + h.corrections[row:row + 1]) % PROFILE.modulus
                response = MaskedStageResponse(correlation_id=stage_request.correlation_id,
                    stage_id="body", masked_output=value.astype(np.uint32), ring=PROFILE.ring,
                    modulus=PROFILE.modulus, wire_bits=PROFILE.wire_bits)
                payload = response.pack()
                if h.fault == "truncate":
                    payload = payload[:-1]
                elif h.fault == "replay":
                    payload = replace(response, correlation_id="00" * 16).pack()
                h.stage_row += 1
                replies.append(payload)
            return httpx.Response(200, content=encode_length_prefixed(replies))
        if request.method == "POST" and path.startswith("/v1/responses/") and path.endswith("/cancel"):
            return httpx.Response(200, json=Response(id=path.split("/")[-2], model=MODEL,
                output=[], status="cancelled").to_dict())
        if request.method == "POST" and path.endswith("/cancel"):
            if h.lost_ack and "/sessions/" in path:
                raise httpx.ReadTimeout("lost cancel ACK", request=request)
            return httpx.Response(200, json={"status": "cancelled"})
        pytest.fail(f"unexpected request/fresh material issuance: {request.method} {path}")

    inference = httpx.Client(base_url="https://inference.test", transport=httpx.MockTransport(handler))
    preparation = httpx.Client(base_url="https://preparation.test", transport=httpx.MockTransport(handler))
    core = RuntimeClient(base_url=str(inference.base_url), api_key="inference-test-key", default_model=MODEL,
        http_client=inference, preparation_http_client=preparation, preparation_api_key="preparation-test-key",
        prepared_inventory_rows=policy.params["rows"], background_inventory_refill=True,
        bundle_compression=delivery.params["encoding"], bundle_cache_dir=tmp_path / "cache")
    # Only the offline model-loading boundary is doubled. No forged profile,
    # native pointer, preparation_verified flag or authority method replacement.
    monkeypatch.setattr(core, "_load_client_bundle_record", lambda *a, **kw: (bundle, publication.fingerprint))
    bound = core._transformer_state(MODEL)
    request = PreparationRequest("11" * 16, "committed-fixture", MODEL, publication.body,
        "body", WEIGHT_DIGEST, 4, 2, 2, 8, 8, PROFILE.signed_output_bound,
        PROFILE.ring, PROFILE.modulus, PROFILE.wire_bits, b"committed-fixture-seed".ljust(32, b"!"))
    stage = PreparedStageRows(request)
    assert stage._mask_rows is not None and stage.input_mask is stage.output_mask is None
    inventory = PreparedInventory(request.session_id, 4, {"body": stage}, _audit=core.audit.record_inventory)
    bound.prepared_inventory = inventory
    core._provider_executor.shutdown(wait=True)
    core._provider_executor = IdleScheduler()
    core._prepared_queue = PreparedExecutionQueue()
    h.core, h.sdk, h.bound, h.inventory, h.stage, h.request = core, ResponsesResource(core), bound, inventory, stage, request
    h.expected_r, h.expected_s = expand_preparation_mask(request), expand_output_mask(request)
    # Commit toy service fixtures before admitting any response. The online
    # service double only reads these rows; it must not compute fresh A*r-s.
    h.corrections = (h.expected_r.astype(np.int64) @ WEIGHTS.astype(np.int64).T
                     - h.expected_s) % PROFILE.modulus
    h.corrections.flags.writeable = False
    h.expected_ids = tuple(derive_online_attempt_id(request, row) for row in range(request.rows))
    h.http.clear()
    yield h
    core.close()
    matrix.close()
    inference.close()
    preparation.close()


def stream_fixture(h, monkeypatch, *, needed=3, used=1, error=None, callback=None):
    """A model double; storage, linear codec, admission and owners stay real."""
    def execute(body, messages, descriptor, **kwargs):
        state = h.core._transformer_state(MODEL)
        h.core._ensure_prepared_inventory(MODEL, state, needed)
        session, state, lease = h.core._open_transformer_session(MODEL,
            max_output_tokens=2, required_rows=needed)
        h.leases.append(lease)
        attempt = prepared_attempt.get()
        attempt.enter()
        attempt.response_id, attempt.model_id = session["response_id"], MODEL
        completed = False
        retired = False

        def retire():
            nonlocal retired
            if not retired:
                retired = True
                h.core._finish_prepared_response(MODEL, state, lease)
                h.retired.append((h.core._foreground_active, h.core._online_active))

        h.retire_session = retire
        try:
            yield ResponseEvent.from_dict({"type": "response.created", "sequence_number": 0,
                "response": Response(id=session["response_id"], model=MODEL, output=[], status="in_progress").to_dict()})

            def exchange(stage_id, payloads):
                response = h.core.http.post(f"/v1/runtime/sessions/session/stages/{stage_id}",
                    headers={**h.core.headers, "Content-Type": "application/octet-stream"},
                    content=encode_length_prefixed(payloads))
                response.raise_for_status()
                return list(iter_length_prefixed(response.content))

            remote = PreparedRemoteLinear(MODEL, h.publication.body, state.bundle.stages, lease, exchange)
            for _ in range(used):
                attempt.check_cancelled()
                activation = np.asarray([[1, -1]], np.float32)
                # This is the intended public local head, never a remote-stage
                # error fallback. Both execute even with fresh entropy forbidden.
                local = state.bundle.local_linear("lm_head", activation)
                result = remote("body", activation)
                np.testing.assert_array_equal(result, local)
            if callback is not None:
                callback()
            attempt.check_cancelled()
            if error is not None:
                raise error
            completed = True
            yield ResponseEvent.from_dict({"type": "response.completed", "sequence_number": 1,
                "response": Response(id=session["response_id"], model=MODEL, output=[], status="completed").to_dict()})
        finally:
            if not completed:
                h.core._cancel_transformer_session(session["id"])
            retire()

    monkeypatch.setattr(h.core, "_transformer_events", execute)
    return h.sdk.create(model=MODEL, input=PRIVATE, stream=True, max_output_tokens=2)


def forbid_fresh_material(h, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("fresh entropy/material construction during foreground")

    # This fixture uses raw single-row frames (no fresh batch-ticket nonce).
    # This is not a claim that every production wire codec is entropy-free.
    monkeypatch.setattr(secrets, "token_bytes", forbidden)
    monkeypatch.setattr(secrets, "token_hex", forbidden)
    monkeypatch.setattr("pllm.runtime.transformer_client.prepared_mask_rows", forbidden)


def assert_private_controls(h):
    assert all(PRIVATE.encode() not in request.content and h.request.seed not in request.content for request in h.http)
    assert all(not (request.method == "POST" and "/preparation/" in request.url.path
                    and not request.url.path.endswith("/cancel")) for request in h.http)
    assert all(request.masked_input.shape == (1, 2) for request in h.requests)
    clear = np.asarray([[127, -127]], dtype=np.int64) % PROFILE.modulus
    for request in h.requests:
        assert not np.array_equal(request.masked_input, clear)
        np.testing.assert_array_equal(request.activation_scales, np.ones(1, np.float32))


@pytest.mark.parametrize("used", [1, 3])
def test_real_paged_reads_and_masked_rows_retire_before_idle_refill(harness, monkeypatch, used):
    h = harness
    forbid_fresh_material(h, monkeypatch)
    with stream_fixture(h, monkeypatch, used=used) as stream:
        next(stream)
        assert h.core._foreground_active == h.core._online_active == 1
        assert h.core._provider_executor.calls == []
        assert next(stream).type == "response.completed"
        assert h.core._foreground_active == 1  # terminal delivery is still owned
        assert h.core._provider_executor.calls == []
    assert h.retired == [(1, 0)]
    assert h.core._foreground_active == h.core._online_active == 0
    assert len(h.core._provider_executor.calls) == 1
    status = h.inventory.status()
    assert status["stage_rows_claimed"] == used
    assert status["stage_rows_burned"] == 3 - used
    assert status["available"] == 1 and status["reserved"] == 0
    with pytest.raises(ValueError, match="consumed"):
        h.stage.take_masks(0, 1)
    if used < 3:
        with pytest.raises(ValueError, match="consumed"):
            h.stage.take_masks(used, 1)
    assert_private_controls(h)


@pytest.mark.parametrize("fault", ["underflow", "stale", "replay"])
def test_unavailable_committed_rows_never_issue_or_read_a_substitute(harness, monkeypatch, fault):
    h = harness
    needed = 5 if fault == "underflow" else 3
    if fault == "stale":
        h.live = False
    if fault == "replay":
        h.remote_rows = 1
    before = h.inventory.status().copy()
    forbid_fresh_material(h, monkeypatch)
    with stream_fixture(h, monkeypatch, needed=needed) as stream:
        with pytest.raises(ModelError, match="prepared inventory unavailable"):
            next(stream)
    assert h.inventory.status() == before
    assert not h.requests and not h.leases and not h.core._provider_executor.calls
    assert not any(request.url.path == "/v1/runtime/sessions" for request in h.http)
    assert h.core._foreground_active == h.core._preparation_active == 0
    assert_private_controls(h)


@pytest.mark.parametrize("fault", ["truncate", "replay"])
def test_bad_stage_reply_burns_original_rows_without_retry_or_local_fallback(harness, monkeypatch, fault):
    h = harness
    h.fault = fault
    forbid_fresh_material(h, monkeypatch)
    with stream_fixture(h, monkeypatch) as stream:
        next(stream)
        error_type = ProtocolError if fault == "truncate" else TransformerClientError
        with pytest.raises(error_type, match="(?i)(invalid|mismatch|unpack|incomplete)"):
            next(stream)
    assert len(h.requests) == 1 and h.response_count == 1
    assert h.inventory.status()["stage_rows_claimed"] == 1
    assert h.inventory.status()["stage_rows_burned"] == 2
    with pytest.raises(ValueError, match="consumed"):
        h.stage.take_masks(1, 1)
    assert_private_controls(h)


def test_callback_thread_and_idle_session_gap_keep_global_owner(harness, monkeypatch):
    h = harness

    def callback():
        # Retire the online session while model delivery remains active.
        h.retire_session()
        assert h.core._online_active == 0 and h.core._foreground_active == 1
        before = len(h.http)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(h.core.preprocess, MODEL, 4)
            with pytest.raises(ModelError, match="foreground response"):
                future.result(timeout=2)
        assert len(h.http) == before
        h.core._background_refill(MODEL, h.bound)
        assert not h.bound.refill_in_progress and len(h.http) == before
        assert not h.core._provider_executor.calls

    with stream_fixture(h, monkeypatch, callback=callback) as stream:
        list(stream)
    assert h.inventory.status()["stage_rows_burned"] == 2
    assert h.core._foreground_active == 0
    assert_private_controls(h)


@pytest.mark.parametrize("lost_ack", [False, True])
def test_original_error_wins_and_lost_ack_prevents_refill(harness, monkeypatch, lost_ack):
    h = harness
    h.lost_ack = lost_ack
    primary = RuntimeError("original model fixture failure")
    with stream_fixture(h, monkeypatch, error=primary) as stream:
        next(stream)
        with pytest.raises(RuntimeError) as raised:
            next(stream)
    assert raised.value is primary
    assert h.inventory.status()["stage_rows_claimed"] == 1
    assert h.inventory.status()["stage_rows_burned"] == 2
    assert h.core._closing is lost_ack
    assert bool(h.core._provider_executor.calls) is not lost_ack
    assert_private_controls(h)


def test_callback_cancellation_retires_native_cursor_and_charges_unused_inventory(harness, monkeypatch):
    h = harness
    with stream_fixture(h, monkeypatch) as stream:
        first = next(stream)
        h.sdk.cancel(first.response["id"])
    status = h.inventory.status()
    assert status["stage_rows_claimed"] == 0 and status["stage_rows_burned"] == 3
    assert status["available"] == 1  # still retained until client retirement
    assert h.stage.retained_mask_bytes > 0
    with pytest.raises(ValueError, match="consumed"):
        h.stage.take_masks(0, 1)
    h.core.close()
    accounting = material_accounting(h.core.audit.to_dict())
    assert accounting["conserved"] is True
    assert accounting["counts"] == {"issued": 4, "reserved": 3, "claimed": 0, "burned": 3, "discarded": 1}
    with pytest.raises(ValueError, match="closed"):
        h.stage.take_masks(3, 1)
    assert_private_controls(h)


def test_reentrant_callback_cancel_after_real_row_use_keeps_original_owner(harness, monkeypatch):
    h = harness

    def callback():
        h.sdk.cancel("response-1")
        assert h.core._foreground_active == h.core._online_active == 1
        before = len(h.http)
        with pytest.raises(ModelError, match="inference is online"):
            h.core.preprocess(MODEL, 4)
        assert len(h.http) == before and not h.core._provider_executor.calls

    forbid_fresh_material(h, monkeypatch)
    with stream_fixture(h, monkeypatch, callback=callback) as stream:
        next(stream)
        with pytest.raises(PreparedQueueClosed, match="cancelled"):
            next(stream)
    assert h.inventory.status()["stage_rows_claimed"] == 1
    assert h.inventory.status()["stage_rows_burned"] == 2
    assert h.retired == [(1, 0)]
    assert h.core._foreground_active == 0
    assert_private_controls(h)


def test_disjoint_next_response_consumes_original_context_not_a_new_seed(harness, monkeypatch):
    h = harness
    forbid_fresh_material(h, monkeypatch)
    with stream_fixture(h, monkeypatch, needed=3, used=1) as stream:
        list(stream)
    with stream_fixture(h, monkeypatch, needed=1, used=1) as stream:
        list(stream)
    assert [lease.start for lease in h.leases] == [0, 3]
    assert [request.correlation_id for request in h.requests] == [
        derive_online_attempt_id(h.request, row) for row in (0, 3)]
    assert h.bound.prepared_inventory is h.inventory and h.stage.request is h.request
    assert h.inventory.status()["available"] == 0
    assert h.inventory.status()["stage_rows_claimed"] == 2
    assert h.inventory.status()["stage_rows_burned"] == 2
    assert_private_controls(h)


@pytest.mark.parametrize("fault", ["tamper", "truncate", "stale"])
def test_real_public_artifact_authentication_rejects_bad_source_before_row_claim(installed_primitives, fault):
    publication = artifact_fixture()

    def read(row):
        data = bytes(publication.exported.objects[row["sha256"]])
        if row["metadata"]["dtype"] == "i1" and fault != "stale":
            data = data[:-1] if fault == "truncate" else bytes(len(data))
        return data

    if fault == "stale":
        publication.manifest["raw"]["sha256"] = "0" * 64
    with pytest.raises(ArtifactError, match="digest"):
        reconstruct_document(publication.manifest, read)


def test_paged_file_owner_independent_of_source_and_foreground_retirement(harness, monkeypatch):
    h = harness
    matrix = h.bundle.stages["lm_head"].client_weight
    # import_bundle already closed every FilePayload. Reads use the retained
    # production immutable snapshot; file ownership is not foreground ownership.
    with stream_fixture(h, monkeypatch) as stream:
        next(stream)
        np.testing.assert_array_equal(matrix.gather_rows(np.asarray([1])), WEIGHTS[[1]])
        assert matrix.weight_bytes == 4 and h.core._foreground_active == 1
    h.core.close()
    # This test intentionally retains an external bundle/matrix alias. Its four
    # logical bytes are still owned; client close cannot claim them deallocated.
    np.testing.assert_array_equal(matrix.gather_rows(np.asarray([0])), WEIGHTS[[0]])
    assert matrix.weight_bytes == 4 and h.core._foreground_active == 0
    matrix.close()
    with pytest.raises(ValueError, match="closed"):
        matrix.gather_rows(np.asarray([0]))


def test_maintained_and_paged_public_bundle_share_actual_toy_numeric_contract(harness):
    h = harness
    baseline = ClientBundle.unpack(h.publication.raw, threads=1)
    activation = np.asarray([[1, -1]], np.float32)
    try:
        assert baseline.cfg == h.bundle.cfg
        assert baseline.manifest == h.bundle.manifest
        assert baseline.privacy == h.bundle.privacy
        assert baseline.weight_storage == "memory" and h.bundle.weight_storage == "paged"
        assert baseline.stages["body"].weight_bits == h.bundle.stages["body"].weight_bits == 8
        assert baseline.stages["body"].activation_bits == h.bundle.stages["body"].activation_bits == 8
        np.testing.assert_array_equal(baseline.local_linear("lm_head", activation),
                                      h.bundle.local_linear("lm_head", activation))
    finally:
        baseline.close()


@pytest.mark.parametrize("fault", ["tamper", "truncate"])
def test_committed_paged_snapshot_survives_upstream_file_change_online(harness, monkeypatch, tmp_path, fault):
    h = harness
    source = tmp_path / "public-four-byte-weights.i8"
    source.write_bytes(WEIGHTS.tobytes())
    # Real direct import makes an independent immutable production snapshot.
    with PagedGEMM.from_raw(source, WEIGHT_DIGEST, (2, 2), threads=1) as matrix:
        forbid_fresh_material(h, monkeypatch)
        with stream_fixture(h, monkeypatch) as stream:
            next(stream)
            source.write_bytes(b"" if fault == "truncate" else bytes(4))
            assert h.core._foreground_active == h.core._online_active == 1
            np.testing.assert_array_equal(matrix.gather_rows(np.asarray([0, 1])), WEIGHTS)
            assert not h.core._provider_executor.calls
        # Reimporting changed bytes under the old pin fails, rather than silently
        # replacing the committed snapshot or initiating any source fallback.
        with pytest.raises(ValueError):
            PagedGEMM.from_raw(source, WEIGHT_DIGEST, (2, 2), threads=1)
    assert_private_controls(h)

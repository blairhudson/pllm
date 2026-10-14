"""Actual native head/table validation, fail-closed admission and lifecycle.

Context-enabled serving tests skip until the parent's native API is installed;
legacy native table tests still validate actual source identities and rejection.
"""

import asyncio
import hashlib
from dataclasses import replace

import httpx
import numpy as np
import pytest

from pllm import _native
from pllm.runtime.native import MaskedGEMM
from pllm.runtime.private_boundary_protocol import (
    BoundaryAdmission, BoundaryDescriptor, BoundaryError, ROLES, SCHEMA, canonical,
    context_bytes, contextual_lookup_digest, pack_frame, unpack_frame,
)
from pllm.runtime.private_boundary_worker import create_private_boundary_worker_app


KEYS = {ROLES[0]: "private-boundary-A-secret-92837465", ROLES[1]: "private-boundary-B-secret-19283746"}
KEY_DIGESTS = {role: hashlib.sha256(key.encode()).hexdigest() for role, key in KEYS.items()}


def fixture():
    weights = np.asarray([[2, -3, 4], [-7, 1, 0], [0, 0, 0], [127, -127, 3]], np.int8)
    scales = np.asarray([0.5, 0.25, 1.0, 0.125], np.float32)
    rows = b"".join(w.tobytes() + scales[i:i + 1].astype("<f4").tobytes() for i, w in enumerate(weights))
    roots = [_native.PrivatePageServer(rows, 7, p) for p in (0, 1)]
    descriptor = BoundaryDescriptor(
        **{k: "a" * 64 for k in ("source_digest", "checkpoint_digest", "model_config_digest",
            "composition_digest", "schedule_digest")},
        head_weight_digest=hashlib.sha256(weights.tobytes()).hexdigest(),
        head_scales_digest=hashlib.sha256(scales.astype("<f4").tobytes()).hexdigest(),
        lookup_digest=roots[0].descriptor()[0].hex(), vocabulary=4, hidden=3,
        signed_output_bound=257 * 127,
    )
    return descriptor, MaskedGEMM(threads=1).compile(weights), scales, roots


def app_for(d, head, scales, root, admit, role=ROLES[0], **kwargs):
    return create_private_boundary_worker_app(descriptor=d, head=head, head_scales=scales,
        role_id=role, api_key=KEYS[role],
        peer_credential_digest=KEY_DIGESTS[ROLES[1 - ROLES.index(role)]],
        admit=admit, lookup_root=root, **kwargs)


def test_factory_requires_trusted_callback_and_actual_weights_scales_table():
    d, head, scales, roots = fixture()
    def admit(actual, requested):
        return requested
    for changed in (replace(d, head_weight_digest="f" * 64),
                    replace(d, head_scales_digest="f" * 64),
                    replace(d, lookup_digest="f" * 64),
                    replace(d, signed_output_bound=1)):
        with pytest.raises(BoundaryError):
            app_for(changed, head, scales, roots[0], admit)
    with pytest.raises(BoundaryError):
        app_for(d, head, scales, roots[0], None)
    with pytest.raises(BoundaryError):
        create_private_boundary_worker_app(descriptor=d, head=head, head_scales=scales,
            role_id=ROLES[0], api_key=KEYS[ROLES[0]], peer_credential_digest=KEY_DIGESTS[ROLES[0]],
            admit=admit, lookup_root=roots[0])
    with pytest.raises(BoundaryError):
        app_for(d, head, scales.astype(np.float64), roots[0], admit)


def test_auth_actual_admission_and_missing_native_context_fail_closed():
    d, head, scales, roots = fixture()
    admitted = BoundaryAdmission(d.digest, "c" * 64, 4, 2)
    calls = []
    def admit(actual, requested):
        calls.append((actual, requested))
        if requested != admitted:
            raise BoundaryError("compiled plan mismatch")
        return admitted
    app = app_for(d, head, scales, roots[0], admit)
    row = {"schema": SCHEMA + ".open", "role": ROLES[0], "session": "d" * 32,
           "epoch": 0, "admission": admitted.to_dict(), "credential_digests": KEY_DIGESTS}
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://localhost") as client:
            assert (await client.post("/v1/private-boundary/sessions", content=canonical(row))).status_code == 401
            assert calls == []
            headers = {"authorization": "Bearer " + KEYS[ROLES[0]]}
            mismatched_peer = {**row, "credential_digests": {**KEY_DIGESTS, ROLES[1]: "f" * 64}}
            assert (await client.post("/v1/private-boundary/sessions", content=canonical(mismatched_peer), headers=headers)).status_code == 400
            assert calls == []
            assert KEYS[ROLES[1]].encode() not in canonical(row)
            forged = {**row, "admission": replace(admitted, plan_digest="f" * 64).to_dict()}
            assert (await client.post("/v1/private-boundary/sessions", content=canonical(forged), headers=headers)).status_code == 400
            reply = await client.post("/v1/private-boundary/sessions", content=canonical(row), headers=headers)
            assert reply.status_code == (200 if hasattr(roots[0], "bind_context") else 400)
            assert calls[-1] == (d, admitted)
    asyncio.run(scenario())


def test_factory_credential_api_never_accepts_peer_plaintext():
    import inspect
    signature = inspect.signature(create_private_boundary_worker_app)
    assert "api_key" in signature.parameters and "peer_credential_digest" in signature.parameters
    assert "credentials" not in signature.parameters
    d, head, scales, roots = fixture()
    with pytest.raises(BoundaryError):
        create_private_boundary_worker_app(descriptor=d, head=head, head_scales=scales,
            role_id=ROLES[0], api_key=KEYS[ROLES[0]], peer_credential_digest=KEYS[ROLES[1]],
            admit=lambda actual, requested: requested, lookup_root=roots[0])


def test_context_enabled_worker_exact_head_lookup_sequence_and_burn():
    d, head, scales, roots = fixture()
    if not hasattr(roots[0], "bind_context"):
        pytest.skip("parent's native context API not installed; no serving fallback")
    admitted = BoundaryAdmission(d.digest, "c" * 64, 4, 2)
    sid = "d" * 32
    def admit(actual, requested):
        assert actual == d
        if requested != admitted:
            raise BoundaryError("compiled plan mismatch")
        return admitted
    apps = [app_for(d, head, scales, roots[p], admit, ROLES[p]) for p in (0, 1)]
    async def scenario():
        clients = [httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://localhost") for app in apps]
        try:
            for p, client in enumerate(clients):
                opened = await client.post("/v1/private-boundary/sessions", content=canonical({
                    "schema": SCHEMA + ".open", "role": ROLES[p], "session": sid, "epoch": 0,
                    "admission": admitted.to_dict(), "credential_digests": KEY_DIGESTS}),
                    headers={"authorization": "Bearer " + KEYS[ROLES[p]]})
                assert opened.status_code == 200, opened.text
            cd = contextual_lookup_digest(d, context_bytes(d, admitted, sid, 0))
            query = _native.private_page_issue(cd, 4, 7, 1, leaf_bits=128)
            replies = []
            for p, client in enumerate(clients):
                payload = pack_frame(role=ROLES[p], paired_session=sid, epoch=0, sequence=0,
                                     operation="lookup", frames=[query[p]])
                response = await client.post(f"/v1/private-boundary/sessions/{sid}/execute", content=payload,
                    headers={"authorization": "Bearer " + KEYS[ROLES[p]]})
                assert response.status_code == 200, response.text
                replies.append(unpack_frame(response.content, response=True)["frames"][0])
            assert query[2].decode(*replies) == head.weight_view()[1].tobytes() + scales[1:2].tobytes()
            share = np.asarray([[0, (1 << 32) - 1, 127]], "<u4")
            payload = pack_frame(role=ROLES[0], paired_session=sid, epoch=0, sequence=1,
                                 operation="head", frames=[share.tobytes()])
            headers = {"authorization": "Bearer " + KEYS[ROLES[0]]}
            route = f"/v1/private-boundary/sessions/{sid}/execute"
            response = await clients[0].post(route, content=payload, headers=headers)
            assert response.status_code == 200
            raw = unpack_frame(response.content, response=True)["frames"][0]
            np.testing.assert_array_equal(np.frombuffer(raw, "<u4"), head.wrap32(share)[0])
            assert (await clients[0].post(route, content=payload, headers=headers)).status_code == 400
            assert (await clients[0].post(route, content=payload, headers=headers)).status_code == 400
        finally:
            for client in clients:
                await client.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["role", "epoch", "sequence", "scalar", "head_width", "budget", "old_context"])
def test_context_enabled_rejections_burn_owner(fault):
    d, head, scales, roots = fixture()
    if not hasattr(roots[0], "bind_context"):
        pytest.skip("parent's native context API not installed")
    admitted = BoundaryAdmission(d.digest, "c" * 64, 1, 1)
    sid = "d" * 32
    def admit(actual, requested):
        if actual != d or requested != admitted:
            raise BoundaryError("fixture compiled contract differs")
        return admitted
    app = app_for(d, head, scales, roots[0], admit)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://localhost") as client:
            headers = {"authorization": "Bearer " + KEYS[ROLES[0]]}
            opened = await client.post("/v1/private-boundary/sessions", content=canonical({
                "schema": SCHEMA + ".open", "role": ROLES[0], "session": sid, "epoch": 0,
                "admission": admitted.to_dict(), "credential_digests": KEY_DIGESTS}), headers=headers)
            assert opened.status_code == 200
            params = dict(role=ROLES[0], paired_session=sid, epoch=0, sequence=0,
                          operation="head", frames=[bytes(12)])
            if fault in ("role", "epoch", "sequence"):
                params.update({fault: {"role": ROLES[1], "epoch": 1, "sequence": 1}[fault]})
            elif fault == "head_width":
                params["frames"] = [bytes(8)]
            elif fault in ("scalar", "old_context"):
                context = context_bytes(d, admitted, "e" * 32 if fault == "old_context" else sid, 0)
                query = _native.private_page_issue(contextual_lookup_digest(d, context), 4, 7, 0,
                                                  leaf_bits=1 if fault == "scalar" else 128)
                query[2].cancel()
                params.update(operation="lookup", frames=[query[0]])
            elif fault == "budget":
                first = await client.post(f"/v1/private-boundary/sessions/{sid}/execute",
                                          content=pack_frame(**params), headers=headers)
                assert first.status_code == 200
                params["sequence"] = 1
            response = await client.post(f"/v1/private-boundary/sessions/{sid}/execute",
                                         content=pack_frame(**params), headers=headers)
            assert response.status_code == 400
            # Even a correctly sequenced fresh operation cannot recover a burned owner.
            assert (await client.post(f"/v1/private-boundary/sessions/{sid}/execute",
                content=pack_frame(role=ROLES[0], paired_session=sid, epoch=0, sequence=1,
                                   operation="head", frames=[bytes(12)]), headers=headers)).status_code == 400
    asyncio.run(scenario())

"""Optional authenticated expiring membership service, containing public offers only."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from pllm.deployment.network import (
    NetworkError, NetworkSpec, checked_offer, control_request, credential, exact, identity,
)
from pllm.runtime.party import authenticate, bounded_json, now_ms


def create_directory_app(network: NetworkSpec, *, credentials=None, clock=now_ms):
    """Registration pulls authenticated installed offers from approved roots.

    A caller cannot submit self-asserted descriptors. A directory observation
    grants membership only; execution always rechecks the actual party.
    """
    if network.backend != "http" or network.directory_origin is None:
        raise NetworkError("directory requires configured http trust roots")
    key = credential(network.directory_credential_env, credentials)
    roots = {item.party_id: item for item in network.parties}
    records = {}
    lock = asyncio.Lock()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def purge():
        for party, offer in list(records.items()):
            if offer.expires_at_ms <= clock():
                del records[party]

    @app.exception_handler(NetworkError)
    async def rejected(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.get("/healthz")
    async def health():
        return {"role": "directory"}

    @app.post("/v1/directory/{verb}")
    async def membership(verb: str, request: Request):
        body = await bounded_json(request)
        exact(body, {"party_id"})
        identity(body["party_id"], "party_id")
        trust = roots.get(body["party_id"])
        if trust is None:
            raise HTTPException(403, detail="party not approved")
        authenticate(request, credential(trust.credential_env, credentials))
        if verb == "withdraw":
            async with lock:
                records.pop(trust.party_id, None)
            return {"status": "withdrawn"}
        if verb not in {"register", "renew"}:
            raise HTTPException(404)
        value = await asyncio.to_thread(control_request, trust.origin, trust.credential_env,
                                        "/v1/party/offer", credentials=credentials)
        offer = checked_offer(value, trust, clock())
        async with lock:
            purge()
            if not offer.accepting:
                records.pop(trust.party_id, None)
                raise NetworkError("PARTY_DRAINING")
            records[trust.party_id] = offer
        return {"status": "registered", "expires_at_ms": offer.expires_at_ms}

    @app.get("/v1/directory/offers")
    async def offers(request: Request):
        authenticate(request, key)
        async with lock:
            purge()
            return {"schema": "pllm.directory_offers.v1", "network_id": network.network_id,
                    "offers": [records[name].to_spec() for name in sorted(records)]}

    return app

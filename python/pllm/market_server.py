from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any
from fastapi import FastAPI, Header, HTTPException
from .market import CentralMarket, ProviderOffer, RouteRequest

def create_market_app(database: str|Path=":memory:",*,admin_key:str="market-admin",fee_rate:float=.10)->FastAPI:
    market=CentralMarket(database,fee_rate=fee_rate);app=FastAPI(title="PLLM Private Inference Market",version="0.14.0");app.state.market=market
    def admin(a):
        if a!=f"Bearer {admin_key}":raise HTTPException(401,"invalid market key")
    @app.get("/healthz")
    async def health():return {"status":"ok"}
    @app.post("/v1/market/providers")
    async def providers(body:dict[str,Any],authorization:str|None=Header(None)):
        admin(authorization)
        try:o=ProviderOffer(**body);market.put_offer(o)
        except Exception as e:raise HTTPException(400,str(e)) from e
        return {"object":"market.provider","provider":asdict(o)}
    @app.post("/v1/market/providers/{provider_id}/heartbeat")
    async def heartbeat(provider_id:str,body:dict[str,Any],authorization:str|None=Header(None)):
        admin(authorization);market.heartbeat(provider_id,active_requests=int(body.get("active_requests",0)),success_rate=float(body.get("success_rate",1)),latency_ms=body.get("latency_ms"));return {"status":"ok"}
    @app.post("/v1/market/accounts/{account_id}/credit")
    async def credit(account_id:str,body:dict[str,Any],authorization:str|None=Header(None)):
        admin(authorization);market.credit(account_id,float(body["amount"]));return {"status":"ok"}
    @app.post("/v1/market/routes")
    async def route(body:dict[str,Any],authorization:str|None=Header(None)):
        admin(authorization);payload=dict(body);account=str(payload.pop("account_id"))
        try:d=market.route(account,RouteRequest(**payload))
        except Exception as e:raise HTTPException(409,str(e)) from e
        return {"object":"market.route",**asdict(d)}
    @app.post("/v1/market/reservations/{reservation_id}/settle")
    async def settle(reservation_id:str,body:dict[str,Any],authorization:str|None=Header(None)):
        admin(authorization);return market.settle(reservation_id,actual_provider_cost=float(body["actual_provider_cost"]))
    return app

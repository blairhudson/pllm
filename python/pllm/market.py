from __future__ import annotations

import base64
import json
import math
import secrets
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")

def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

@dataclass(slots=True)
class ProviderOffer:
    provider_id: str; endpoint: str; model: str; input_privacy: bool; model_privacy: bool
    price_input_million: float; price_output_million: float; setup_price: float
    capacity_tps: float; latency_ms: float; max_context: int; region: str; expires_at: float
    public_key: str; signature: str = ""
    def unsigned(self) -> dict[str, Any]:
        value = asdict(self); value.pop("signature", None); return value
    def sign(self, key: Ed25519PrivateKey) -> "ProviderOffer":
        self.signature = _b64(key.sign(_canonical(self.unsigned()))); return self
    def verify(self) -> None:
        Ed25519PublicKey.from_public_bytes(_unb64(self.public_key)).verify(_unb64(self.signature), _canonical(self.unsigned()))
        if self.expires_at <= time.time(): raise ValueError("provider offer has expired")
        if not self.input_privacy: raise ValueError("PLLM market providers must protect request content")

@dataclass(slots=True)
class RouteRequest:
    model: str; input_tokens: int; max_output_tokens: int; require_model_privacy: bool = False
    max_input_price: float | None = None; max_output_price: float | None = None
    preferred_region: str | None = None; sort: str = "balanced"

@dataclass(slots=True)
class RouteDecision:
    provider_id: str; endpoint: str; model: str; estimated_cost: float; operator_fee: float
    score: float; expires_at: float; ticket: str

class CentralMarket:
    def __init__(self, database: str | Path = ":memory:", *, fee_rate: float = 0.10,
                 signing_key: Ed25519PrivateKey | None = None, heartbeat_ttl_seconds: float = 30.0) -> None:
        self.db = sqlite3.connect(str(database), check_same_thread=False); self.db.row_factory = sqlite3.Row
        self.fee_rate = float(fee_rate); self.signing_key = signing_key or Ed25519PrivateKey.generate()
        self.heartbeat_ttl_seconds = float(heartbeat_ttl_seconds); self.offer_cache: dict[str, ProviderOffer] = {}
        self.db.executescript("""
        create table if not exists offers(provider_id text primary key,payload text not null,last_seen real not null,active_requests integer not null default 0,success_rate real not null default 1.0);
        create table if not exists accounts(account_id text primary key,balance real not null);
        create table if not exists reservations(id text primary key,account_id text not null,provider_id text not null,amount real not null,status text not null,created_at real not null);
        create table if not exists ledger(id integer primary key autoincrement,account_id text,provider_id text,amount real not null,kind text not null,created_at real not null);
        """); self.db.commit(); self._load_offer_cache()
    def _load_offer_cache(self) -> None:
        for row in self.db.execute("select payload from offers").fetchall():
            try: offer=ProviderOffer(**json.loads(row["payload"])); offer.verify(); self.offer_cache[offer.provider_id]=offer
            except Exception: continue
    def put_offer(self, offer: ProviderOffer) -> None:
        offer.verify(); now=time.time()
        self.db.execute("insert into offers(provider_id,payload,last_seen) values(?,?,?) on conflict(provider_id) do update set payload=excluded.payload,last_seen=excluded.last_seen",(offer.provider_id,json.dumps(asdict(offer)),now)); self.db.commit(); self.offer_cache[offer.provider_id]=offer
    def heartbeat(self, provider_id: str, *, active_requests: int, success_rate: float, latency_ms: float | None = None) -> None:
        row=self.db.execute("select payload from offers where provider_id=?",(provider_id,)).fetchone()
        if row is None: raise KeyError(provider_id)
        payload=json.loads(row["payload"])
        if latency_ms is not None: payload["latency_ms"]=float(latency_ms)
        self.db.execute("update offers set payload=?,last_seen=?,active_requests=?,success_rate=? where provider_id=?",(json.dumps(payload),time.time(),int(active_requests),float(success_rate),provider_id)); self.db.commit(); self.offer_cache[provider_id]=ProviderOffer(**payload)
    def credit(self, account_id: str, amount: float) -> None:
        self.db.execute("insert into accounts(account_id,balance) values(?,?) on conflict(account_id) do update set balance=balance+excluded.balance",(account_id,float(amount)))
        self.db.execute("insert into ledger(account_id,amount,kind,created_at) values(?,?,?,?)",(account_id,float(amount),"credit",time.time())); self.db.commit()
    def _candidates(self, request: RouteRequest) -> list[tuple[ProviderOffer, sqlite3.Row]]:
        now=time.time(); rows=self.db.execute("select provider_id,last_seen,active_requests,success_rate from offers where last_seen>?",(now-self.heartbeat_ttl_seconds,)).fetchall(); out=[]
        for row in rows:
            offer=self.offer_cache.get(str(row["provider_id"]))
            if offer is None or offer.expires_at<=now or offer.model!=request.model or offer.max_context<request.input_tokens+request.max_output_tokens: continue
            if request.require_model_privacy and not offer.model_privacy: continue
            if request.max_input_price is not None and offer.price_input_million>request.max_input_price: continue
            if request.max_output_price is not None and offer.price_output_million>request.max_output_price: continue
            out.append((offer,row))
        return out
    def route(self, account_id: str, request: RouteRequest) -> RouteDecision:
        candidates=self._candidates(request)
        if not candidates: raise LookupError("no provider satisfies the request")
        scored=[]
        for offer,row in candidates:
            cost=offer.setup_price+request.input_tokens*offer.price_input_million/1e6+request.max_output_tokens*offer.price_output_million/1e6
            load=int(row["active_requests"])/max(offer.capacity_tps,.001); failure=1-float(row["success_rate"]); region=0 if not request.preferred_region or offer.region==request.preferred_region else .2
            if request.sort=="price": score=cost
            elif request.sort=="latency": score=offer.latency_ms
            elif request.sort=="throughput": score=-offer.capacity_tps
            else: score=math.log1p(cost*1e6)*.35+offer.latency_ms*.01+load*.25+failure*2+region
            scored.append((score,cost,offer))
        score,cost,offer=min(scored,key=lambda x:x[0]); fee=cost*self.fee_rate; total=cost+fee
        row=self.db.execute("select balance from accounts where account_id=?",(account_id,)).fetchone()
        if row is None or float(row["balance"])<total: raise PermissionError("insufficient account balance")
        rid=secrets.token_urlsafe(18); self.db.execute("update accounts set balance=balance-? where account_id=?",(total,account_id)); self.db.execute("insert into reservations values(?,?,?,?,?,?)",(rid,account_id,offer.provider_id,total,"reserved",time.time())); self.db.commit()
        payload={"reservation_id":rid,"provider_id":offer.provider_id,"endpoint":offer.endpoint,"model":offer.model,"max_amount":total,"expires_at":time.time()+60}
        ticket=_b64(_canonical(payload))+"."+_b64(self.signing_key.sign(_canonical(payload)))
        return RouteDecision(offer.provider_id,offer.endpoint,offer.model,total,fee,float(score),payload["expires_at"],ticket)
    def settle(self, reservation_id: str, *, actual_provider_cost: float) -> dict[str,float]:
        row=self.db.execute("select * from reservations where id=?",(reservation_id,)).fetchone()
        if row is None or row["status"]!="reserved": raise ValueError("invalid reservation")
        reserved=float(row["amount"]); provider=min(float(actual_provider_cost),reserved/(1+self.fee_rate)); fee=provider*self.fee_rate; refund=reserved-provider-fee
        self.db.execute("update reservations set status='settled' where id=?",(reservation_id,))
        if refund>0: self.db.execute("update accounts set balance=balance+? where account_id=?",(refund,row["account_id"]))
        self.db.execute("insert into ledger(account_id,provider_id,amount,kind,created_at) values(?,?,?,?,?)",(row["account_id"],row["provider_id"],provider,"provider_payment",time.time()))
        self.db.execute("insert into ledger(account_id,provider_id,amount,kind,created_at) values(?,?,?,?,?)",(row["account_id"],row["provider_id"],fee,"operator_fee",time.time())); self.db.commit()
        return {"provider_payment":provider,"operator_fee":fee,"refund":refund}

class DecentralizedQuoteBook:
    def __init__(self)->None: self.offers:dict[str,ProviderOffer]={}
    def publish(self,offer:ProviderOffer)->None: offer.verify(); self.offers[offer.provider_id]=offer
    def route(self,request:RouteRequest)->ProviderOffer:
        now=time.time(); c=[o for o in self.offers.values() if o.expires_at>now and o.model==request.model and (not request.require_model_privacy or o.model_privacy)]
        if not c: raise LookupError("no signed offer satisfies the request")
        def score(o:ProviderOffer)->float:
            cost=o.setup_price+request.input_tokens*o.price_input_million/1e6+request.max_output_tokens*o.price_output_million/1e6
            if request.sort=="latency": return o.latency_ms
            if request.sort=="throughput": return -o.capacity_tps
            if request.sort=="price": return cost
            return math.log1p(cost*1e6)*.5+o.latency_ms*.01
        return min(c,key=score)

def generate_provider_offer(*,provider_id:str,endpoint:str,model:str,key:Ed25519PrivateKey,model_privacy:bool=False,price_input:float=.25,price_output:float=.75,capacity_tps:float=20,latency_ms:float=20,region:str="global")->ProviderOffer:
    return ProviderOffer(provider_id,endpoint,model,True,model_privacy,price_input,price_output,0.0,capacity_tps,latency_ms,131072,region,time.time()+300,_b64(key.public_key().public_bytes_raw())).sign(key)

def _percentile(v:list[float],p:float)->float|None:
    if not v:return None
    o=sorted(v);return float(o[min(len(o)-1,max(0,round((len(o)-1)*p)))])

def simulate_market(*,providers:int=100,requests:int=5000,seed:int=7)->dict[str,Any]:
    import random
    rng=random.Random(seed); central=CentralMarket(fee_rate=.10,heartbeat_ttl_seconds=600); central.credit("benchmark",max(1000,requests*.01)); decentralized=DecentralizedQuoteBook()
    for i in range(providers):
        k=Ed25519PrivateKey.generate(); o=generate_provider_offer(provider_id=f"provider-{i}",endpoint=f"https://p{i}.example/v1",model="example/model",key=k,model_privacy=i%4==0,price_input=rng.uniform(.08,.6),price_output=rng.uniform(.2,1.8),capacity_tps=rng.uniform(4,80),latency_ms=rng.uniform(5,120),region=rng.choice(["sydney","singapore","us-west","eu-west"])); central.put_offer(o); central.heartbeat(o.provider_id,active_requests=rng.randrange(0,8),success_rate=rng.uniform(.94,1)); decentralized.publish(o)
    cm=[];dm=[];cc=[];dc=[];cpc={};dpc={};revenue=0.;fail=0; started=time.perf_counter_ns()
    for _ in range(requests):
        req=RouteRequest("example/model",rng.randrange(64,4096),rng.randrange(32,512),rng.random()<.2,preferred_region="sydney")
        t=time.perf_counter_ns()
        try:
            d=central.route("benchmark",req);cm.append((time.perf_counter_ns()-t)/1e6);cc.append(d.estimated_cost);cpc[d.provider_id]=cpc.get(d.provider_id,0)+1;rid=json.loads(_unb64(d.ticket.split('.',1)[0]))["reservation_id"];revenue+=central.settle(rid,actual_provider_cost=d.estimated_cost/(1+central.fee_rate))["operator_fee"]
        except Exception:fail+=1
        t=time.perf_counter_ns();o=decentralized.route(req);dm.append((time.perf_counter_ns()-t)/1e6);dpc[o.provider_id]=dpc.get(o.provider_id,0)+1;dc.append(o.setup_price+req.input_tokens*o.price_input_million/1e6+req.max_output_tokens*o.price_output_million/1e6)
    elapsed=(time.perf_counter_ns()-started)/1e9
    def hhi(c):
        total=sum(c.values()) or 1;return sum((n/total)**2 for n in c.values())
    return {"providers":providers,"requests":requests,"elapsed_s":elapsed,"central":{"route_ops_s":requests/elapsed,"route_ms_mean":sum(cm)/len(cm),"route_ms_p50":_percentile(cm,.5),"route_ms_p95":_percentile(cm,.95),"mean_customer_cost":sum(cc)/len(cc),"operator_revenue":revenue,"failed":fail,"provider_concentration_hhi":hhi(cpc),"trusted_components":["router","account ledger","payment settlement","provider admission","refunds and disputes"]},"decentralized":{"route_ms_mean":sum(dm)/len(dm),"route_ms_p50":_percentile(dm,.5),"route_ms_p95":_percentile(dm,.95),"mean_provider_cost":sum(dc)/len(dc),"provider_concentration_hhi":hhi(dpc),"on_chain_transactions_per_request_without_channels":2,"on_chain_transactions_per_request_with_channels":0,"trusted_components":["client route policy","payment chain","offer availability layer","reputation and challenge mechanism"]},"interpretation":{"central_strengths":["accounts and refunds","provider admission","enterprise support","global batch formation"],"decentralized_strengths":["portable signed offers","client controlled routing","less dependence on one account operator"],"recommended_product":"central"}}

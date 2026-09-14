"""Definitions over native-collected public measurements; no inference/timing hot path."""
from __future__ import annotations
from collections.abc import Iterable
from math import isfinite

def decode_tps(token_times_ns: list[int]) -> float | None:
    if any(type(t) is not int or t<0 for t in token_times_ns): raise ValueError("monotonic timestamps must be nonnegative integers")
    if any(b<a for a,b in zip(token_times_ns,token_times_ns[1:])):raise ValueError("timestamps are not monotonic")
    if len(token_times_ns)<2:return None
    elapsed=token_times_ns[-1]-token_times_ns[0]
    return None if elapsed<=0 else (len(token_times_ns)-1)*1e9/elapsed

def directed_bytes(events: Iterable[dict]) -> dict[str,int]:
    totals={}; seen=set()
    for e in events:
        if e["observer"]!=e["sender"]:continue
        key=(e["sender"],e["receiver"],e["phase"],e["transfer_id"])
        if key in seen:raise ValueError("duplicate sender observation; label retransmission with a distinct transfer ID")
        if type(e["bytes"]) is not int or e["bytes"]<0:raise ValueError("invalid bytes")
        seen.add(key); label=f'{e["sender"]}->{e["receiver"]}/{e["phase"]}'
        totals[label]=totals.get(label,0)+e["bytes"]
    return totals

def sustainable_bound(online_tps: float, materials: list[dict]) -> float:
    if not isfinite(online_tps) or online_tps<0:raise ValueError("invalid online TPS")
    rates=[online_tps]
    for m in materials:
        r=float(m["production_per_second"]);u=float(m["consumption_per_useful_token"])
        if not isfinite(r) or not isfinite(u) or r<0 or u<=0:raise ValueError("invalid material rate")
        rates.append(r/u)
    return min(rates)

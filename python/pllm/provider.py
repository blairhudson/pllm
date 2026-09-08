from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .market import ProviderOffer, generate_provider_offer


def load_or_create_signing_key(path: str | Path) -> Ed25519PrivateKey:
    target = Path(path).expanduser()
    if target.exists():
        raw = bytes.fromhex(target.read_text(encoding="utf-8").strip())
        return Ed25519PrivateKey.from_private_bytes(raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    target.write_text(key.private_bytes_raw().hex() + "\n", encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return key


def register_offer(
    *,
    market_url: str,
    market_key: str,
    offer: ProviderOffer,
    timeout: float = 20.0,
) -> dict[str, Any]:
    with httpx.Client(base_url=market_url.rstrip("/"), timeout=timeout) as client:
        response = client.post(
            "/v1/market/providers",
            headers={"Authorization": f"Bearer {market_key}"},
            json=asdict(offer),
        )
        response.raise_for_status()
        return response.json()


def send_heartbeat(
    *,
    market_url: str,
    market_key: str,
    provider_id: str,
    active_requests: int,
    success_rate: float,
    latency_ms: float,
    timeout: float = 20.0,
) -> dict[str, Any]:
    with httpx.Client(base_url=market_url.rstrip("/"), timeout=timeout) as client:
        response = client.post(
            f"/v1/market/providers/{provider_id}/heartbeat",
            headers={"Authorization": f"Bearer {market_key}"},
            json={
                "active_requests": active_requests,
                "success_rate": success_rate,
                "latency_ms": latency_ms,
            },
        )
        response.raise_for_status()
        return response.json()


def run_provider_agent(
    *,
    market_url: str,
    market_key: str,
    provider_id: str,
    endpoint: str,
    model: str,
    key_path: str | Path,
    model_privacy: bool,
    price_input: float,
    price_output: float,
    capacity_tps: float,
    latency_ms: float,
    region: str,
    heartbeat_seconds: float = 10.0,
    once: bool = False,
) -> None:
    key = load_or_create_signing_key(key_path)
    while True:
        offer = generate_provider_offer(
            provider_id=provider_id,
            endpoint=endpoint,
            model=model,
            key=key,
            model_privacy=model_privacy,
            price_input=price_input,
            price_output=price_output,
            capacity_tps=capacity_tps,
            latency_ms=latency_ms,
            region=region,
        )
        register_offer(market_url=market_url, market_key=market_key, offer=offer)
        send_heartbeat(
            market_url=market_url,
            market_key=market_key,
            provider_id=provider_id,
            active_requests=0,
            success_rate=1.0,
            latency_ms=latency_ms,
        )
        print(
            json.dumps(
                {
                    "status": "registered",
                    "provider_id": provider_id,
                    "market": market_url,
                    "endpoint": endpoint,
                    "model": model,
                    "offer_expires_at": offer.expires_at,
                }
            )
        )
        if once:
            return
        time.sleep(max(1.0, heartbeat_seconds))

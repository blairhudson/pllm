from __future__ import annotations

import hashlib
import hmac


def derive_session_key(api_key: str, session_id: str) -> bytes:
    if not api_key:
        raise ValueError("api_key must not be empty")
    if not session_id:
        raise ValueError("session_id must not be empty")
    return hmac.new(api_key.encode("utf-8"), b"pllm-session-v1\0" + session_id.encode("utf-8"), hashlib.sha256).digest()


def bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    prefix = "Bearer "
    return value[len(prefix):] if value.startswith(prefix) else None

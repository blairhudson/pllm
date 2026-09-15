from __future__ import annotations

import os

import importlib.util

import pytest

from conftest import start_gateway
from pllm.runtime import OpenAI


TENSEAL_PATH = os.environ.get("PLLM_TENSEAL_PATH", "")


def _tenseal_available() -> bool:
    import sys
    if TENSEAL_PATH not in sys.path:
        sys.path.insert(0, TENSEAL_PATH)
    return importlib.util.find_spec("tenseal") is not None


@pytest.mark.he
@pytest.mark.skipif(not _tenseal_available(), reason="TenSEAL test wheel unavailable")
def test_actual_bfv_preprocessing_and_masked_online_inference():
    gateway = start_gateway(bfv=True)
    try:
        with OpenAI(
            base_url=gateway.base_url,
            api_key=gateway.api_key,
            correlation_mode="bfv",
            correlation_prefetch=1,
            tenseal_path=TENSEAL_PATH,
            session_transport="websocket",
        ) as client:
            response = client.responses.create(
                model="pllm-bigram-demo",
                input="BFV_CANARY_MUST_STAY_LOCAL",
                max_output_tokens=32,
            )
            assert response.output_text == "private\n"
            audit = client.privacy_audit
            assert audit.public_context_bytes > 0
            assert audit.encrypted_correlation_upload_bytes > 0
            assert audit.encrypted_correlation_download_bytes > 0
            assert audit.correlation_count == audit.online_steps
            assert audit.plaintext_prompt_bytes_sent == 0
        context_payload = next(payload for kind, payload in gateway.audit if kind == "bfv_context")
        assert len(context_payload) > 1_000_000
        assert b"BFV_CANARY_MUST_STAY_LOCAL" not in b"".join(payload for _, payload in gateway.audit)
    finally:
        gateway.close()

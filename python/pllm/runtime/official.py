from __future__ import annotations

from typing import Any

from .transport import HEAsyncTransport, HETransport, _httpx


def create_openai_client(
    *,
    gateway_url: str | None = None,
    gateway_api_key: str | None = None,
    preparation_url: str | None = None,
    preparation_api_key: str | None = None,
    api_key: str = "he-local-transport",
    base_url: str = "https://he.local/v1",
    he_transport: str | None = None,
    correlation_mode: str | None = None,
    correlation_prefetch: int | None = None,
    prepared_inventory_rows: int | None = None,
    token_cache_size: int | None = None,
    bundle_cache_mode: str | None = None,
    bundle_cache_dir: str | None = None,
    tenseal_path: str | None = None,
    **client_kwargs: Any,
):
    """Create the installed official ``openai.OpenAI`` client over HE.

    OpenAI Python SDK 3.x uses ``httpx2`` while older releases use ``httpx``.
    :mod:`pllm.runtime.transport` selects whichever transport package is installed,
    so the same factory supports both generations.
    """
    from pllm.settings import ClientSettings
    settings = ClientSettings.load().merged(
        base_url=gateway_url, api_key=gateway_api_key, transport=he_transport,
        correlation_mode=correlation_mode, correlation_prefetch=correlation_prefetch,
        prepared_inventory_rows=prepared_inventory_rows,
        token_cache_size=token_cache_size,
        bundle_cache_mode=bundle_cache_mode,
        bundle_cache_dir=bundle_cache_dir,
        preparation_base_url=preparation_url,
        preparation_api_key=preparation_api_key,
    )
    gateway_url, gateway_api_key = settings.base_url, settings.api_key
    preparation_url, preparation_api_key = (
        settings.preparation_base_url,
        settings.preparation_api_key,
    )
    he_transport, correlation_mode = settings.transport, settings.correlation_mode
    correlation_prefetch, token_cache_size = settings.correlation_prefetch, settings.token_cache_size
    try:
        from openai import OpenAI as OfficialOpenAI
    except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the official OpenAI Python SDK to use this factory") from exc

    transport = HETransport(
        gateway_url=gateway_url,
        api_key=gateway_api_key,
        preparation_url=preparation_url,
        preparation_api_key=preparation_api_key,
        he_transport=he_transport,
        correlation_mode=correlation_mode,
        correlation_prefetch=correlation_prefetch,
        prepared_inventory_rows=settings.prepared_inventory_rows,
        token_cache_size=token_cache_size,
        bundle_cache_mode=settings.bundle_cache_mode,
        bundle_cache_dir=settings.bundle_cache_dir,
        tenseal_path=tenseal_path,
    )
    try:
        http_client = _httpx.Client(transport=transport)
        client = OfficialOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            **client_kwargs,
        )
    except BaseException:
        transport.close()
        raise
    if hasattr(transport, "preprocess"):
        client.pllm_preprocess = transport.preprocess
    return client


def create_async_openai_client(
    *,
    gateway_url: str | None = None,
    gateway_api_key: str | None = None,
    preparation_url: str | None = None,
    preparation_api_key: str | None = None,
    api_key: str = "he-local-transport",
    base_url: str = "https://he.local/v1",
    he_transport: str | None = None,
    correlation_mode: str | None = None,
    correlation_prefetch: int | None = None,
    prepared_inventory_rows: int | None = None,
    token_cache_size: int | None = None,
    bundle_cache_mode: str | None = None,
    bundle_cache_dir: str | None = None,
    tenseal_path: str | None = None,
    **client_kwargs: Any,
):
    from pllm.settings import ClientSettings
    settings = ClientSettings.load().merged(
        base_url=gateway_url, api_key=gateway_api_key, transport=he_transport,
        correlation_mode=correlation_mode, correlation_prefetch=correlation_prefetch,
        prepared_inventory_rows=prepared_inventory_rows,
        token_cache_size=token_cache_size,
        bundle_cache_mode=bundle_cache_mode,
        bundle_cache_dir=bundle_cache_dir,
        preparation_base_url=preparation_url,
        preparation_api_key=preparation_api_key,
    )
    gateway_url, gateway_api_key = settings.base_url, settings.api_key
    preparation_url, preparation_api_key = (
        settings.preparation_base_url,
        settings.preparation_api_key,
    )
    he_transport, correlation_mode = settings.transport, settings.correlation_mode
    correlation_prefetch, token_cache_size = settings.correlation_prefetch, settings.token_cache_size
    try:
        from openai import AsyncOpenAI as OfficialAsyncOpenAI
    except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the official OpenAI Python SDK to use this factory") from exc

    transport = HEAsyncTransport(
        gateway_url=gateway_url,
        api_key=gateway_api_key,
        preparation_url=preparation_url,
        preparation_api_key=preparation_api_key,
        he_transport=he_transport,
        correlation_mode=correlation_mode,
        correlation_prefetch=correlation_prefetch,
        prepared_inventory_rows=settings.prepared_inventory_rows,
        token_cache_size=token_cache_size,
        bundle_cache_mode=settings.bundle_cache_mode,
        bundle_cache_dir=settings.bundle_cache_dir,
        tenseal_path=tenseal_path,
    )
    try:
        http_client = _httpx.AsyncClient(transport=transport)
        client = OfficialAsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            **client_kwargs,
        )
    except BaseException:
        transport.sync.close()
        raise
    if hasattr(transport, "preprocess"):
        client.pllm_preprocess = transport.preprocess
    return client

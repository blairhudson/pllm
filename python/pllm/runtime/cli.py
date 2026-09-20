from __future__ import annotations

import argparse
import ipaddress
import os
import socket
from pathlib import Path
from typing import Any

import uvicorn

from pllm.configuration import Model
from pllm.model_loader import resolve_model
from pllm.sources import TinyModel
from pllm.settings import ClientSettings

from .blinded_engine import BlindedTransformerEngine
from .config import GatewayConfig
from .guarded_engine import GuardPolicy, GuardedBlindedTransformerEngine
from .preparation_server import create_preparation_app
from .privacy import PrivacyMode, ProprietaryProtocol
from .proprietary_engine import DirectFHETransformerEngine
from .server import create_app
from .servers import build_roles
from .sidecar import create_sidecar_app
from .transformer_engine import MaskedTransformerEngine


class RuntimeCLIError(ValueError):
    """Configuration error suitable for either runtime or public CLI rendering."""


def _env(name: str, default: Any = None) -> Any:
    value = os.getenv(name)
    return default if value is None else value


def _apply_server_defaults(args: argparse.Namespace) -> None:
    defaults: dict[str, Any] = {
        "config": None,
        "privacy_mode": _env("PLLM_PRIVACY_MODE"),
        "proprietary_protocol": _env("PLLM_PROPRIETARY_PROTOCOL"),
        "guard_max_rows_per_request": int(_env("PLLM_GUARD_MAX_ROWS", "4096")),
        "guard_max_rows_per_stage": int(_env("PLLM_GUARD_STAGE_BUDGET", "16384")),
        "guard_max_requests_per_minute": int(_env("PLLM_GUARD_RATE", "4096")),
        "guard_output_dither": int(_env("PLLM_GUARD_DITHER", "0")),
        "host": "127.0.0.1",
        "port": 8000,
        "api_key": _env("PLLM_API_KEY"),
        "provider_push_api_key": _env("PLLM_PROVIDER_PUSH_API_KEY"),
        "rendezvous_timeout": (
            float(value) if (value := _env("PLLM_RENDEZVOUS_TIMEOUT")) else None
        ),
        "rendezvous_capacity": (
            int(value) if (value := _env("PLLM_RENDEZVOUS_CAPACITY")) else None
        ),
        "rendezvous_max_bytes": (
            int(value) if (value := _env("PLLM_RENDEZVOUS_MAX_BYTES")) else None
        ),
        "prepared_session_capacity": (
            int(value) if (value := _env("PLLM_PREPARED_SESSION_CAPACITY")) else None
        ),
        "prepared_session_idle": (
            float(value) if (value := _env("PLLM_PREPARED_SESSION_IDLE")) else None
        ),
        "inference_url": _env("PLLM_INFERENCE_URL"),
        "push_api_key": _env("PLLM_PUSH_API_KEY"),
        "push_timeout": float(value) if (value := _env("PLLM_PUSH_TIMEOUT")) else None,
        "tenseal_path": _env("PLLM_PYDEPS"),
        "max_batch_size": None,
        "max_wait_ms": None,
        "fixed_batch_wait": False,
        "allow_insecure_local_correlations": False,
        "engine_threads": int(_env("PLLM_ENGINE_THREADS", "0") or 0),
        "native_library": _env("PLLM_NATIVE_LIBRARY"),
        "compiled_cache_dir": _env("PLLM_COMPILED_CACHE"),
        "streaming_threshold_elements": 50_000_000,
        "quantization_chunk_rows": 64,
        "verification_component": _env("PLLM_VERIFICATION_COMPONENT", "none"),
        "verification_target_failure_bits": int(_env("PLLM_VERIFICATION_TARGET_FAILURE_BITS", "0")),
        "weight_bits": 8,
        "activation_bits": 8,
        "model": [],
        "model_id": [],
        "model_kind": "huggingface",
        "revision": _env("PLLM_HF_REVISION"),
        "hf_token": _env("HF_TOKEN"),
        "hf_cache_dir": _env("HF_HUB_CACHE"),
        "local_files_only": False,
    }
    for name, value in defaults.items():
        if not hasattr(args, name) or getattr(args, name) is None:
            setattr(args, name, value)


def run_server(args: argparse.Namespace, *, preparation: bool = False) -> None:
    """Run one role from already-parsed options."""
    _apply_server_defaults(args)
    config = GatewayConfig.load(Path(args.config).expanduser()) if args.config else GatewayConfig()
    mode = PrivacyMode.parse(getattr(args, "privacy_mode", None) or config.privacy_mode)
    proprietary_protocol = ProprietaryProtocol.parse(
        getattr(args, "proprietary_protocol", None) or config.proprietary_protocol
    )
    if config.engine_models and not getattr(args, "model", None):
        requested_mode = getattr(args, "privacy_mode", None)
        if (
            requested_mode is not None
            and PrivacyMode.parse(requested_mode) is not config.privacy_mode
        ):
            raise RuntimeCLIError(
                "--privacy-mode cannot override configured models; update the role config or pass --model"
            )
        requested_protocol = getattr(args, "proprietary_protocol", None)
        if (
            requested_protocol is not None
            and ProprietaryProtocol.parse(requested_protocol) is not config.proprietary_protocol
        ):
            raise RuntimeCLIError(
                "--proprietary-protocol cannot override configured models; update the role config or pass --model"
            )
    if preparation and mode is not PrivacyMode.PUBLIC:
        raise RuntimeCLIError("the preparation service supports public-weight models only")
    if (
        not preparation
        and mode is PrivacyMode.PUBLIC
        and not (getattr(args, "provider_push_api_key", None) or config.provider_push_api_key)
    ):
        raise RuntimeCLIError("public serving requires --provider-push-api-key")
    if proprietary_protocol is ProprietaryProtocol.SECURE:
        raise RuntimeCLIError(
            "secure mode currently runs the complete protocol preview through `pllm secure`. "
            "Arbitrary Hugging Face Transformer serving fails closed until the secure graph "
            "compiler is complete"
        )

    value = config.to_dict()
    value["privacy_mode"] = mode.value
    value["proprietary_protocol"] = proprietary_protocol.value
    if getattr(args, "api_key", None):
        value["api_keys"] = [args.api_key]
    if getattr(args, "provider_push_api_key", None):
        value["provider_push_api_key"] = args.provider_push_api_key
    for argument, field in (
        ("rendezvous_timeout", "rendezvous_timeout_seconds"),
        ("rendezvous_capacity", "rendezvous_capacity"),
        ("rendezvous_max_bytes", "rendezvous_max_bytes"),
        ("prepared_session_capacity", "prepared_session_capacity"),
        ("prepared_session_idle", "prepared_session_idle_seconds"),
    ):
        option = getattr(args, argument, None)
        if option is not None:
            value[field] = option
    if preparation:
        for argument, field in (
            ("inference_url", "preparation_inference_url"),
            ("push_api_key", "preparation_push_api_key"),
            ("push_timeout", "preparation_push_timeout_seconds"),
        ):
            option = getattr(args, argument, None)
            if option is not None:
                value[field] = option
    if args.tenseal_path:
        value["tenseal_path"] = args.tenseal_path
    if args.max_batch_size is not None:
        value["max_batch_size"] = args.max_batch_size
    if args.max_wait_ms is not None:
        value["max_batch_wait_ms"] = args.max_wait_ms
    if args.fixed_batch_wait:
        value["adaptive_batching"] = False
    if args.allow_insecure_local_correlations:
        value["allow_insecure_local_correlations"] = True

    host = str(args.host).strip("[]").lower()
    loopback = host in {"127.0.0.1", "::1", "localhost"}
    if not loopback and value.get("api_keys") == ["pllm-local"]:
        raise RuntimeCLIError(
            "Refusing non-loopback bind with the default API key; configure explicit non-default credentials"
        )

    if args.model:
        if args.model_id and len(args.model_id) != len(args.model):
            raise RuntimeCLIError("--model-id must be supplied once per --model")
        ids = args.model_id or [None] * len(args.model)
        resolved_models = []
        for source, requested_id in zip(args.model, ids, strict=True):
            model = Model(
                source,
                kind=args.model_kind,
                model_id=requested_id,
                revision=args.revision,
                local_files_only=args.local_files_only,
            )
            resolved = resolve_model(
                model,
                token=args.hf_token,
                cache_dir=args.hf_cache_dir,
            )
            if resolved.path is None:
                raise RuntimeCLIError(f"model kind {model.kind!r} has no local runtime source")
            runtime_model = Model(
                str(resolved.path),
                kind=model.kind,
                model_id=resolved.manifest.id,
                local_files_only=model.kind in {"huggingface", "safetensors", "vllm"},
            )
            if mode is PrivacyMode.PUBLIC:
                engine_name = "masked-transformer-w4a4"
            elif proprietary_protocol is ProprietaryProtocol.GUARDED:
                engine_name = "guarded-transformer-proprietary"
            elif proprietary_protocol is ProprietaryProtocol.BLINDED:
                engine_name = "blinded-transformer-proprietary"
            elif proprietary_protocol is ProprietaryProtocol.DIRECT:
                engine_name = "direct-bfv-transformer-proprietary"
            else:
                raise AssertionError("secure mode must fail before model loading")
            resolved_models.append({"engine": engine_name, **runtime_model.to_runtime_spec()})
        value["engine_models"] = resolved_models
    config = GatewayConfig.from_dict(value)
    loopback = _is_loopback_host(args.host)
    if not loopback and (not config.api_keys or "pllm-local" in config.api_keys):
        raise RuntimeCLIError(
            "non-loopback serving requires an explicit --api-key or config api_keys"
        )

    if mode is PrivacyMode.PUBLIC:
        engine_type = MaskedTransformerEngine
    elif proprietary_protocol is ProprietaryProtocol.GUARDED:
        engine_type = GuardedBlindedTransformerEngine
    elif proprietary_protocol is ProprietaryProtocol.BLINDED:
        engine_type = BlindedTransformerEngine
    elif proprietary_protocol is ProprietaryProtocol.DIRECT:
        engine_type = DirectFHETransformerEngine
    else:
        raise AssertionError("secure mode must fail before engine creation")

    engine_kwargs: dict[str, Any] = {
        "tenseal_path": config.tenseal_path,
        "threads": args.engine_threads or None,
        "native_library": args.native_library,
        "compiled_cache_dir": args.compiled_cache_dir,
        "streaming_threshold_elements": args.streaming_threshold_elements,
        "quantization_chunk_rows": args.quantization_chunk_rows,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "verification_component": args.verification_component,
        "verification_target_failure_bits": args.verification_target_failure_bits,
    }
    if engine_type is GuardedBlindedTransformerEngine:
        engine_kwargs["guard_policy"] = GuardPolicy(
            max_rows_per_request=args.guard_max_rows_per_request,
            max_rows_per_owner_stage=args.guard_max_rows_per_stage,
            max_requests_per_owner_minute=args.guard_max_requests_per_minute,
            output_dither_bound=args.guard_output_dither,
        )
    engine = engine_type(**engine_kwargs)
    from .telemetry import configure_telemetry

    configure_telemetry("pllm-preparation" if preparation else "pllm-inference")
    app = (
        create_preparation_app(config, engine)
        if preparation
        else create_app(config, engines={engine.capabilities.name: engine})
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=False,
        ws_max_size=config.prepared_payload_max_bytes,
    )


def run_sidecar(args: argparse.Namespace) -> None:
    """Run gateway from already-parsed options."""
    config = getattr(args, "config", None)
    try:
        settings = ClientSettings.load(Path(config).expanduser() if config else None)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeCLIError(f"Cannot load client config: {exc}") from exc
    settings = settings.merged(
        base_url=getattr(args, "inference_url", None),
        api_key=getattr(args, "inference_key", None),
        model=getattr(args, "model", None) or getattr(args, "model_id", None),
        transport=getattr(args, "transport", None),
        correlation_mode=getattr(args, "correlation_mode", None),
        preparation_base_url=getattr(args, "preparation_url", None),
        preparation_api_key=getattr(args, "preparation_key", None),
        correlation_prefetch=getattr(args, "correlation_prefetch", None),
        prepared_inventory_rows=getattr(args, "prepared_inventory_rows", None),
        token_cache_size=getattr(args, "token_cache_size", None),
        bundle_cache_mode=getattr(args, "bundle_cache_mode", None),
        bundle_cache_dir=getattr(args, "bundle_cache_dir", None),
        timeout=getattr(args, "timeout", None),
    )
    app = create_sidecar_app(
        remote_base_url=settings.base_url,
        remote_api_key=settings.api_key,
        local_api_key=getattr(args, "api_key", None) or _env("PLLM_GATEWAY_API_KEY", "local"),
        tenseal_path=getattr(args, "tenseal_path", None) or _env("PLLM_PYDEPS"),
        correlation_mode=settings.correlation_mode,
        preparation_base_url=settings.preparation_base_url,
        preparation_api_key=settings.preparation_api_key,
        session_transport=settings.transport,
        correlation_prefetch=settings.correlation_prefetch,
        prepared_inventory_rows=settings.prepared_inventory_rows,
        default_model=settings.model,
        token_cache_size=settings.token_cache_size,
        bundle_cache_mode=settings.bundle_cache_mode,
        bundle_cache_dir=settings.bundle_cache_dir,
        timeout=settings.timeout,
    )
    uvicorn.run(
        app,
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", 8080),
        access_log=False,
    )


def _is_loopback_host(host: str) -> bool:
    value = str(host).strip("[]")
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        try:
            addresses = {
                item[4][0] for item in socket.getaddrinfo(value, None, type=socket.SOCK_STREAM)
            }
        except socket.gaierror:
            return False
        return bool(addresses) and all(ipaddress.ip_address(item).is_loopback for item in addresses)


def run_local_gateway(args: argparse.Namespace) -> None:
    """Run local inference/preparation process groups and gateway foreground."""
    source = str(args.model)
    model_id = getattr(args, "model_id", None) or source
    if getattr(args, "tiny", False):
        model_id = getattr(args, "model_id", None) or "pllm-gateway-tiny"
        model = TinyModel(model_id=model_id)
    else:
        model = Model.hf(
            source,
            model_id=model_id,
            revision=getattr(args, "revision", None),
            local_files_only=getattr(args, "local_files_only", False),
        )
    with build_roles(
        model,
        model_id=model_id,
        weight_bits=getattr(args, "weight_bits", 8),
        activation_bits=getattr(args, "activation_bits", 8),
        correlation_mode=args.correlation_mode,
        tenseal_path=getattr(args, "tenseal_path", None),
        hf_cache_dir=getattr(args, "hf_cache_dir", None),
        reserved_ports=(args.port,),
    ) as topology:
        app = topology.gateway_app(
            local_api_key=getattr(args, "api_key", None) or _env("PLLM_GATEWAY_API_KEY", "local"),
            tenseal_path=getattr(args, "tenseal_path", None) or _env("PLLM_PYDEPS"),
            session_transport=getattr(args, "transport", "websocket"),
            correlation_prefetch=getattr(args, "correlation_prefetch", 4),
            prepared_inventory_rows=getattr(args, "prepared_inventory_rows", 64),
            token_cache_size=getattr(args, "token_cache_size", 512),
            bundle_cache_mode=getattr(args, "bundle_cache_mode", "read-write"),
            bundle_cache_dir=getattr(args, "bundle_cache_dir", None),
            timeout=getattr(args, "timeout", 300.0),
        )
        uvicorn.run(
            app,
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 8080),
            access_log=False,
        )

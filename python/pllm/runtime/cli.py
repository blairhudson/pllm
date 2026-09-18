from __future__ import annotations

import argparse
import ipaddress
import ipaddress
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from pllm.configuration import Model
from pllm.model_loader import resolve_model
from pllm.settings import ClientSettings

from .blinded_engine import BlindedTransformerEngine
from .config import GatewayConfig
from .guarded_engine import GuardPolicy, GuardedBlindedTransformerEngine
from .preparation_server import create_preparation_app
from .privacy import PrivacyMode, ProprietaryProtocol
from .proprietary_engine import DirectFHETransformerEngine
from .server import create_app
from .sidecar import create_sidecar_app
from .transformer_engine import MaskedTransformerEngine


class RuntimeCLIError(ValueError):
    """Configuration error suitable for either runtime or public CLI rendering."""


def _env(name: str, default: Any = None) -> Any:
    value = os.getenv(name)
    return default if value is None else value


def _server_parser(*, preparation: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PLLM trusted preparation service"
            if preparation
            else "Run the PLLM remote private-inference service"
        )
    )
    parser.add_argument("--config", help="JSON GatewayConfig file")
    parser.add_argument(
        "--privacy-mode",
        "--mode",
        dest="privacy_mode",
        choices=[item.value for item in PrivacyMode],
        default=_env("PLLM_PRIVACY_MODE"),
        help="public: open weights; proprietary: protect server-owned weights",
    )
    parser.add_argument(
        "--protocol",
        "--proprietary-protocol",
        dest="proprietary_protocol",
        choices=[item.value for item in ProprietaryProtocol],
        default=_env("PLLM_PROPRIETARY_PROTOCOL"),
    )
    parser.add_argument(
        "--guard-max-rows-per-request",
        type=int,
        default=int(_env("PLLM_GUARD_MAX_ROWS", "4096")),
    )
    parser.add_argument(
        "--guard-max-rows-per-stage",
        type=int,
        default=int(_env("PLLM_GUARD_STAGE_BUDGET", "16384")),
    )
    parser.add_argument(
        "--guard-max-requests-per-minute",
        type=int,
        default=int(_env("PLLM_GUARD_RATE", "4096")),
    )
    parser.add_argument(
        "--guard-output-dither", type=int, default=int(_env("PLLM_GUARD_DITHER", "0"))
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=_env("PLLM_API_KEY"))
    parser.add_argument("--provider-push-api-key", default=_env("PLLM_PROVIDER_PUSH_API_KEY"))
    parser.add_argument(
        "--rendezvous-timeout",
        type=float,
        default=float(value) if (value := _env("PLLM_RENDEZVOUS_TIMEOUT")) else None,
    )
    parser.add_argument(
        "--rendezvous-capacity",
        type=int,
        default=int(value) if (value := _env("PLLM_RENDEZVOUS_CAPACITY")) else None,
    )
    parser.add_argument(
        "--rendezvous-max-bytes",
        type=int,
        default=int(value) if (value := _env("PLLM_RENDEZVOUS_MAX_BYTES")) else None,
    )
    parser.add_argument(
        "--prepared-session-capacity",
        type=int,
        default=int(value) if (value := _env("PLLM_PREPARED_SESSION_CAPACITY")) else None,
    )
    parser.add_argument(
        "--prepared-session-idle",
        type=float,
        default=float(value) if (value := _env("PLLM_PREPARED_SESSION_IDLE")) else None,
    )
    parser.add_argument("--inference-url", default=_env("PLLM_INFERENCE_URL"))
    parser.add_argument("--push-api-key", default=_env("PLLM_PUSH_API_KEY"))
    parser.add_argument(
        "--push-timeout",
        type=float,
        default=float(value) if (value := _env("PLLM_PUSH_TIMEOUT")) else None,
    )
    parser.add_argument("--tenseal-path", default=_env("PLLM_PYDEPS"))
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--max-wait-ms", type=float)
    parser.add_argument("--fixed-batch-wait", action="store_true")
    parser.add_argument("--allow-insecure-local-correlations", action="store_true")
    parser.add_argument(
        "--engine-threads", type=int, default=int(_env("PLLM_ENGINE_THREADS", "0") or 0)
    )
    parser.add_argument("--native-library", default=_env("PLLM_NATIVE_LIBRARY"))
    parser.add_argument("--compiled-cache-dir", default=_env("PLLM_COMPILED_CACHE"))
    parser.add_argument("--streaming-threshold-elements", type=int, default=50_000_000)
    parser.add_argument("--quantization-chunk-rows", type=int, default=64)
    parser.add_argument("--weight-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--activation-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Hugging Face repository ID or local HF/Safetensors/MLX model directory",
    )
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument(
        "--model-kind",
        default="huggingface",
        choices=["huggingface", "safetensors", "vllm", "mlx", "mlx-lm"],
    )
    parser.add_argument("--revision", default=_env("PLLM_HF_REVISION"))
    parser.add_argument("--hf-token", default=_env("HF_TOKEN"))
    parser.add_argument("--hf-cache-dir", default=_env("HF_HUB_CACHE"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser


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
            resolved_models.append(
                {"engine": engine_name, **runtime_model.to_runtime_spec()}
            )
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
    }
    if engine_type is GuardedBlindedTransformerEngine:
        engine_kwargs["guard_policy"] = GuardPolicy(
            max_rows_per_request=args.guard_max_rows_per_request,
            max_rows_per_owner_stage=args.guard_max_rows_per_stage,
            max_requests_per_owner_minute=args.guard_max_requests_per_minute,
            output_dither_bound=args.guard_output_dither,
        )
    engine = engine_type(**engine_kwargs)
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


def _server_main(argv: list[str] | None = None, *, preparation: bool = False) -> None:
    parser = _server_parser(preparation=preparation)
    args = parser.parse_args(argv)
    try:
        run_server(args, preparation=preparation)
    except RuntimeCLIError as exc:
        parser.error(str(exc))


def server_main(argv: list[str] | None = None) -> None:
    _server_main(argv)


def preparation_main(argv: list[str] | None = None) -> None:
    _server_main(argv, preparation=True)


def _sidecar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local PLLM OpenAI Responses sidecar")
    parser.add_argument("--config", help="client TOML file")
    parser.add_argument(
        "--inference-url",
        "--remote-base-url",
        dest="inference_url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--inference-key",
        "--remote-api-key",
        dest="inference_key",
        default=_env("PLLM_API_KEY", "pllm-local"),
    )
    parser.add_argument("--api-key", "--local-api-key", dest="api_key", default="local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tenseal-path", default=_env("PLLM_PYDEPS"))
    parser.add_argument("--correlation-mode", choices=["bfv", "local-test"], default="bfv")
    parser.add_argument("--preparation-url", "--preparation-base-url", dest="preparation_url")
    parser.add_argument("--preparation-key", "--preparation-api-key", dest="preparation_key")
    parser.add_argument("--transport", choices=["auto", "http", "websocket"], default="websocket")
    parser.add_argument("--correlation-prefetch", type=int, default=4)
    parser.add_argument("--prepared-inventory-rows", type=int, default=64)
    parser.add_argument("--model", "--default-model", dest="model")
    parser.add_argument("--token-cache-size", type=int, default=512)
    parser.add_argument(
        "--bundle-cache-mode",
        choices=["read-write", "read-only", "refresh", "off"],
        default="read-write",
    )
    parser.add_argument("--bundle-cache-dir")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


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


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


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


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=5)


def _wait_for_service(
    process: subprocess.Popen[Any], base_url: str, role: str, *, timeout: float = 300.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"local {role} service exited during startup ({returncode})")
        try:
            response = httpx.get(f"{base_url}/healthz", timeout=0.5)
            if response.status_code == 200 and response.json().get("role") == role:
                return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.1)
    raise RuntimeError(f"local {role} service did not become ready before timeout")


def _random_credential(used: set[str]) -> str:
    while True:
        value = "local_" + secrets.token_urlsafe(24)
        if value not in used:
            used.add(value)
            return value


def run_local_gateway(args: argparse.Namespace) -> None:
    """Run local inference/preparation process groups and gateway foreground."""
    processes: list[subprocess.Popen[Any]] = []
    temporary: tempfile.TemporaryDirectory[str] | None = None
    previous_sigterm: Any = None
    signal_installed = False

    def terminate(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    try:
        if threading.current_thread() is threading.main_thread():
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, terminate)
            signal_installed = True
        model = args.model
        model_id = getattr(args, "model_id", None) or model
        if getattr(args, "tiny", False):
            from .dashboard import _create_demo_checkpoint

            temporary = tempfile.TemporaryDirectory(prefix="pllm-gateway-")
            model = str(_create_demo_checkpoint(Path(temporary.name) / "model"))
            model_id = getattr(args, "model_id", None) or "pllm-gateway-tiny"

        used: set[str] = set()
        inference_key = _random_credential(used)
        preparation_key = _random_credential(used)
        push_key = _random_credential(used)
        reserved = {args.port}
        inference_port = _free_port()
        while inference_port in reserved:
            inference_port = _free_port()
        reserved.add(inference_port)
        preparation_port = _free_port()
        while preparation_port in reserved:
            preparation_port = _free_port()
        inference_url = f"http://127.0.0.1:{inference_port}"
        preparation_url = f"http://127.0.0.1:{preparation_port}"

        common = [sys.executable, "-m", "pllm.runtime.cli"]
        model_options = ["--model", model, "--model-id", model_id]
        if getattr(args, "revision", None):
            model_options.extend(["--revision", args.revision])
        if getattr(args, "hf_cache_dir", None):
            model_options.extend(["--hf-cache-dir", args.hf_cache_dir])
        if getattr(args, "tenseal_path", None):
            model_options.extend(["--tenseal-path", args.tenseal_path])
        if getattr(args, "local_files_only", False) or getattr(args, "tiny", False):
            model_options.append("--local-files-only")
        model_options.extend(
            [
                "--weight-bits",
                str(getattr(args, "weight_bits", 8)),
                "--activation-bits",
                str(getattr(args, "activation_bits", 8)),
            ]
        )
        inference = [
            *common,
            "inference",
            "--privacy-mode",
            "public",
            "--protocol",
            "guarded",
            "--host",
            "127.0.0.1",
            "--port",
            str(inference_port),
            "--rendezvous-capacity",
            "262144",
            "--rendezvous-max-bytes",
            "2147483648",
            *model_options,
        ]
        if args.correlation_mode == "local-test":
            inference.append("--allow-insecure-local-correlations")
        preparation = [
            *common,
            "preparation",
            "--privacy-mode",
            "public",
            "--protocol",
            "guarded",
            "--host",
            "127.0.0.1",
            "--port",
            str(preparation_port),
            *model_options,
        ]
        inference_env = os.environ.copy()
        inference_env.update(
            {"PLLM_API_KEY": inference_key, "PLLM_PROVIDER_PUSH_API_KEY": push_key}
        )
        preparation_env = os.environ.copy()
        preparation_env.update(
            {
                "PLLM_API_KEY": preparation_key,
                "PLLM_INFERENCE_URL": inference_url,
                "PLLM_PUSH_API_KEY": push_key,
            }
        )
        for command, environment in (
            (inference, inference_env),
            (preparation, preparation_env),
        ):
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            )
        _wait_for_service(processes[0], inference_url, "inference")
        _wait_for_service(processes[1], preparation_url, "trusted-preparation")

        gateway_args = argparse.Namespace(**vars(args))
        gateway_args.inference_url = inference_url
        gateway_args.inference_key = inference_key
        gateway_args.preparation_url = preparation_url
        gateway_args.preparation_key = preparation_key
        gateway_args.model = model_id
        run_sidecar(gateway_args)
    finally:
        if signal_installed:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for process in reversed(processes):
            _stop_process(process)
        if temporary is not None:
            temporary.cleanup()
        if signal_installed:
            signal.signal(signal.SIGTERM, previous_sigterm)


def sidecar_main(argv: list[str] | None = None) -> None:
    run_sidecar(_sidecar_parser().parse_args(argv))


def main(argv: list[str] | None = None) -> None:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if not arguments or arguments[0] not in {"inference", "preparation"}:
        raise SystemExit("usage: python -m pllm.runtime.cli {inference,preparation} [options]")
    role = arguments.pop(0)
    _server_main(arguments, preparation=role == "preparation")


if __name__ == "__main__":
    main()

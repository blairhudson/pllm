from __future__ import annotations

import argparse
import os

import uvicorn

from .blinded_engine import BlindedTransformerEngine
from .config import GatewayConfig
from .guarded_engine import GuardPolicy, GuardedBlindedTransformerEngine
from .hf_hub import resolve_huggingface_source
from .privacy import PrivacyMode, ProprietaryProtocol
from .proprietary_engine import DirectFHETransformerEngine
from .server import create_app
from .sidecar import create_sidecar_app
from .transformer_engine import MaskedTransformerEngine


def _env(name: str, default=None):
    value = os.getenv(name)
    return default if value is None else value


def server_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the PLLM remote private-inference service")
    parser.add_argument("--config", help="JSON GatewayConfig file")
    parser.add_argument(
        "--privacy-mode", "--mode",
        dest="privacy_mode",
        choices=[item.value for item in PrivacyMode],
        default=_env("PLLM_PRIVACY_MODE", "public"),
        help="public: open weights; proprietary: protect server-owned weights",
    )
    parser.add_argument(
        "--protocol", "--proprietary-protocol",
        dest="proprietary_protocol",
        choices=[item.value for item in ProprietaryProtocol],
        default=_env("PLLM_PROPRIETARY_PROTOCOL", "guarded"),
        help=(
            "guarded: fast default with query limits; blinded: honest client mode; "
            "secure: protected intermediate values in the research preview; direct: slow reference"
        ),
    )
    parser.add_argument("--guard-max-rows-per-request", type=int, default=int(_env("PLLM_GUARD_MAX_ROWS", default="4096")))
    parser.add_argument("--guard-max-rows-per-stage", type=int, default=int(_env("PLLM_GUARD_STAGE_BUDGET", default="16384")))
    parser.add_argument("--guard-max-requests-per-minute", type=int, default=int(_env("PLLM_GUARD_RATE", default="4096")))
    parser.add_argument("--guard-output-dither", type=int, default=int(_env("PLLM_GUARD_DITHER", default="0")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=_env("PLLM_API_KEY"))
    parser.add_argument("--tenseal-path", default=_env("PLLM_PYDEPS"))
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--max-wait-ms", type=float)
    parser.add_argument("--fixed-batch-wait", action="store_true", help="wait on every batch instead of only under contention")
    parser.add_argument("--allow-insecure-local-correlations", action="store_true")
    parser.add_argument(
        "--engine-threads",
        type=int,
        default=int(_env("PLLM_ENGINE_THREADS", "0") or 0),
    )
    parser.add_argument("--native-library", default=_env("PLLM_NATIVE_LIBRARY"))
    parser.add_argument("--compiled-cache-dir", default=_env("PLLM_COMPILED_CACHE"))
    parser.add_argument("--streaming-threshold-elements", type=int, default=50_000_000)
    parser.add_argument("--quantization-chunk-rows", type=int, default=64)
    parser.add_argument("--weight-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--activation-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument(
        "--model", action="append", default=[],
        help="Hugging Face repository ID or local HF/Safetensors/MLX model directory",
    )
    parser.add_argument("--model-id", action="append", default=[], help="public ID for the corresponding --model")
    parser.add_argument("--model-kind", default="huggingface", choices=["huggingface", "safetensors", "vllm", "mlx", "mlx-lm"])
    parser.add_argument("--revision", default=_env("PLLM_HF_REVISION"))
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token; defaults to HF_TOKEN or the stored login",
    )
    parser.add_argument(
        "--hf-cache-dir",
        help="Hugging Face Hub cache override; defaults to HF_HUB_CACHE or HF_HOME/hub",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    config = GatewayConfig.load(args.config) if args.config else GatewayConfig()
    mode = PrivacyMode.parse(args.privacy_mode or config.privacy_mode)
    proprietary_protocol = ProprietaryProtocol.parse(
        args.proprietary_protocol or config.proprietary_protocol
    )
    if proprietary_protocol is ProprietaryProtocol.SECURE:
        parser.error(
            "secure mode currently runs the complete protocol preview through `pllm secure`. "
            "Arbitrary Hugging Face Transformer serving fails closed until the secure graph compiler is complete"
        )

    value = config.to_dict()
    value["privacy_mode"] = mode.value
    value["proprietary_protocol"] = proprietary_protocol.value
    if args.api_key:
        value["api_keys"] = [args.api_key]
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

    if args.model:
        if args.model_id and len(args.model_id) != len(args.model):
            parser.error("--model-id must be supplied once per --model")
        ids = args.model_id or [None] * len(args.model)
        resolved_models = []
        for source, requested_id in zip(args.model, ids, strict=True):
            if args.model_kind in {"huggingface", "safetensors", "vllm"}:
                resolved = resolve_huggingface_source(
                    source,
                    model_id=requested_id,
                    revision=args.revision,
                    token=args.hf_token,
                    cache_dir=args.hf_cache_dir,
                    local_files_only=args.local_files_only,
                )
                path = str(resolved.path)
                public_id = resolved.model_id
            else:
                path = source
                public_id = requested_id
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
                {
                    "engine": engine_name,
                    "kind": args.model_kind,
                    "path": path,
                    **({"model_id": public_id} if public_id else {}),
                }
            )
        value["engine_models"] = resolved_models
    config = GatewayConfig.from_dict(value)

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

    engine_kwargs = dict(
        tenseal_path=config.tenseal_path,
        threads=args.engine_threads or None,
        native_library=args.native_library,
        compiled_cache_dir=args.compiled_cache_dir,
        streaming_threshold_elements=args.streaming_threshold_elements,
        quantization_chunk_rows=args.quantization_chunk_rows,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
    )
    if engine_type is GuardedBlindedTransformerEngine:
        engine_kwargs["guard_policy"] = GuardPolicy(
            max_rows_per_request=args.guard_max_rows_per_request,
            max_rows_per_owner_stage=args.guard_max_rows_per_stage,
            max_requests_per_owner_minute=args.guard_max_requests_per_minute,
            output_dither_bound=args.guard_output_dither,
        )
    engine = engine_type(**engine_kwargs)
    uvicorn.run(
        create_app(config, engines={engine.capabilities.name: engine}),
        host=args.host,
        port=args.port,
        access_log=False,
    )


def sidecar_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local PLLM OpenAI Responses sidecar")
    parser.add_argument("--remote-base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--remote-api-key",
        default=_env("PLLM_API_KEY", "pllm-local"),
    )
    parser.add_argument("--local-api-key", default="local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tenseal-path", default=_env("PLLM_PYDEPS"))
    parser.add_argument("--correlation-mode", choices=["bfv", "local-test"], default="bfv")
    parser.add_argument("--execution-strategy", choices=["bfv", "two-provider"], default="bfv")
    parser.add_argument("--secondary-base-url")
    parser.add_argument("--secondary-api-key")
    parser.add_argument("--transport", choices=["http", "websocket"], default="websocket")
    parser.add_argument("--correlation-prefetch", type=int, default=4)
    parser.add_argument("--token-cache-size", type=int, default=512)
    args = parser.parse_args(argv)
    app = create_sidecar_app(
        remote_base_url=args.remote_base_url,
        remote_api_key=args.remote_api_key,
        local_api_key=args.local_api_key,
        tenseal_path=args.tenseal_path,
            correlation_mode=args.correlation_mode,
            execution_strategy=args.execution_strategy,
            secondary_base_url=args.secondary_base_url,
            secondary_api_key=args.secondary_api_key,
        he_transport=args.transport,
        correlation_prefetch=args.correlation_prefetch,
        token_cache_size=args.token_cache_size,
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)

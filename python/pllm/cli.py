from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import uvicorn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pllm.runtime.authenticated_mpc import TrustedPreprocessor
from pllm.runtime.cli import preparation_main, server_main, sidecar_main
from pllm.runtime.formal_security import PROFILES
from pllm.runtime.he_authenticated_preprocessing import HEAuthenticatedPreprocessor
from pllm.runtime.native import main as build_native_main
from pllm.runtime.preprocessing_inventory import PreparedInventory, RecordingPreprocessor
from pllm.runtime.secure_transformer import SecureDecoder, SecureDecoderConfig, SecureDecoderWeights

from .chat import run_chat
from .market import generate_provider_offer, simulate_market
from .market_server import create_market_app
from .provider import run_provider_agent
from .settings import ClientSettings

from pllm._version import __version__ as VERSION


def _secure_preview(arguments: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="pllm secure",
        description="Run an authenticated arithmetic simulator, not a distributed security protocol",
        allow_abbrev=False,
    )
    parser.add_argument("--output", choices=("token", "logits"), default="token")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--he", action="store_true", help="generate setup material with BFV")
    parser.add_argument("--tenseal-path", default=os.getenv("PLLM_PYDEPS"))
    args = parser.parse_args(list(arguments))
    if not 1 <= args.batch <= 32:
        parser.error("batch must be between 1 and 32")

    config = SecureDecoderConfig()
    weights = SecureDecoderWeights.random(config, seed=41)
    recorder = RecordingPreprocessor(TrustedPreprocessor(seed=42))
    planner = SecureDecoder(config, weights, recorder)
    tokens = np.arange(args.batch, dtype=np.int64) % config.vocabulary_size
    planner.forward_token(tokens, output_disclosure=args.output)

    if args.he:
        generator = HEAuthenticatedPreprocessor(
            seed=43,
            threads=1,
            tenseal_path=args.tenseal_path,
            enable_linear_correlations=True,
        )
    else:
        generator = TrustedPreprocessor(seed=43)

    started = time.perf_counter_ns()
    inventory = PreparedInventory.generate(recorder.plan, generator)
    preparation_ms = (time.perf_counter_ns() - started) / 1e6

    decoder = SecureDecoder(config, weights, inventory)
    started = time.perf_counter_ns()
    result = decoder.forward_token(tokens, output_disclosure=args.output)
    online_ms = (time.perf_counter_ns() - started) / 1e6
    clear_tokens, _ = decoder.clear_forward_token(tokens)

    output = {
        "exact": bool(np.array_equal(result.token_ids, clear_tokens)),
        "output_policy": args.output,
        "batch": args.batch,
        "prepared_with_he": bool(args.he),
        "preparation_ms": preparation_ms,
        "online_ms": online_ms,
        "online_rounds": result.online_rounds,
        "uploaded_bytes": result.uploaded_bytes,
        "downloaded_bytes": result.downloaded_bytes,
        "token_ids": result.token_ids.tolist(),
        "plan": recorder.plan.counts(),
    }
    if args.he:
        output["he"] = generator.summary()
    print(json.dumps(output, indent=2))


def _add_client_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", "--base-url", dest="base_url", help="remote PLLM server URL")
    parser.add_argument("--api-key", help="remote server API key")
    parser.add_argument(
        "--preparation-url",
        "--preparation-server",
        dest="preparation_base_url",
        help="trusted preparation service URL",
    )
    parser.add_argument("--preparation-api-key", help="trusted preparation service API key")
    parser.add_argument(
        "--model",
        help="model ID; automatically selected when the server exposes one private model",
    )


def _protection_to_legacy(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[str, str]:
    weights = args.weights
    client_trust = args.client_trust
    activation_protection = args.activation_protection

    if args.legacy_mode:
        weights = "public" if args.legacy_mode == "public" else "confidential"
    if weights is None:
        parser.error(
            "declare who may learn the checkpoint weights with "
            "`--weights public` or `--weights confidential`"
        )

    if args.legacy_protocol:
        activation_protection = {
            "guarded": "blinded-masks",
            "blinded": "blinded-masks",
            "secure": "authenticated-shares",
            "direct": "encrypted-activations",
        }[args.legacy_protocol]

    if activation_protection == "automatic":
        if weights == "public":
            activation_protection = "seeded-preparation"
        elif client_trust == "untrusted":
            activation_protection = "authenticated-shares"
        elif client_trust == "honest":
            activation_protection = "blinded-masks"
        else:
            activation_protection = "guarded-blinded-masks"

    if weights == "public":
        if activation_protection != "seeded-preparation":
            parser.error(
                "public weights require `--activation-protection seeded-preparation`"
            )
        return "public", "guarded"

    if activation_protection == "authenticated-shares" or client_trust == "untrusted":
        parser.error(
            "protection from a modified client is not implemented for the Hugging Face runtime. "
            "`pllm secure` is an arithmetic simulator, not a distributed security protocol"
        )
    if activation_protection == "encrypted-activations":
        return "proprietary", "direct"
    if activation_protection == "blinded-masks":
        return "proprietary", "blinded"
    if activation_protection in {"guarded-blinded-masks", "precomputed-masks"}:
        return "proprietary", "guarded"
    parser.error(f"unsupported activation protection: {activation_protection}")


def _serve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    mode, protocol = _protection_to_legacy(args, parser)
    models = list(args.models or []) + list(args.model_compat or [])
    if not models and not args.config:
        parser.error(
            "supply a Hugging Face model, for example "
            "`pllm serve google/gemma-4-e2b-it --weights public`"
        )

    api_key = args.api_key or os.getenv("PLLM_API_KEY")
    generated = False
    if not api_key:
        api_key = secrets.token_urlsafe(24)
        generated = True

    translated = [
        "--privacy-mode",
        mode,
        "--proprietary-protocol",
        protocol,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        api_key,
        "--model-kind",
        args.model_kind,
        "--streaming-threshold-elements",
        str(args.streaming_threshold_elements),
        "--quantization-chunk-rows",
        str(args.quantization_chunk_rows),
        "--weight-bits",
        str(args.weight_bits),
        "--activation-bits",
        str(args.activation_bits),
    ]
    if args.config:
        translated.extend(["--config", args.config])
    for model in models:
        translated.extend(["--model", model])
    for model_id in args.model_id:
        translated.extend(["--model-id", model_id])

    string_options = (
        ("--revision", args.revision),
        ("--hf-token", args.hf_token),
        ("--hf-cache-dir", args.hf_cache_dir),
        ("--tenseal-path", args.tenseal_path),
        ("--native-library", args.native_library),
        ("--compiled-cache-dir", args.compiled_cache_dir),
        ("--provider-push-api-key", args.provider_push_api_key),
    )
    for name, value in string_options:
        if value:
            translated.extend([name, str(value)])

    numeric_options = (
        ("--max-batch-size", args.max_batch_size),
        ("--max-wait-ms", args.max_wait_ms),
        ("--engine-threads", args.engine_threads),
        ("--guard-max-rows-per-request", args.guard_max_rows_per_request),
        ("--guard-max-rows-per-stage", args.guard_max_rows_per_stage),
        ("--guard-max-requests-per-minute", args.guard_max_requests_per_minute),
        ("--guard-output-dither", args.guard_output_dither),
        ("--rendezvous-timeout", args.rendezvous_timeout),
        ("--rendezvous-capacity", args.rendezvous_capacity),
        ("--rendezvous-max-bytes", args.rendezvous_max_bytes),
        ("--prepared-session-capacity", args.prepared_session_capacity),
        ("--prepared-session-idle", args.prepared_session_idle),
    )
    for name, value in numeric_options:
        if value is not None:
            translated.extend([name, str(value)])

    if args.fixed_batch_wait:
        translated.append("--fixed-batch-wait")
    if args.local_files_only:
        translated.append("--local-files-only")
    if args.allow_test_correlations:
        translated.append("--allow-insecure-local-correlations")

    protection = {
        "weights": args.weights,
        "client_trust": args.client_trust,
        "activation_protection": args.activation_protection,
        "runtime_mode": mode,
        "runtime_protocol": protocol,
    }
    print("PLLM server")
    print(
        json.dumps(
            {
                "listen": f"http://{args.host}:{args.port}",
                "models": models,
                "protection": protection,
                "api_key": api_key if generated else "configured",
            },
            indent=2,
        )
    )
    if generated:
        print("The generated API key is shown once. Save it with `pllm configure --api-key ...`.")
    server_main(translated)


def _preparation(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.preparation_command != "serve":
        parser.error("choose `pllm preparation serve`")
    api_key = args.api_key or os.getenv("PLLM_API_KEY")
    generated = False
    if not api_key:
        api_key = secrets.token_urlsafe(24)
        generated = True
    inference_url = args.inference_url or os.getenv("PLLM_INFERENCE_URL")
    push_api_key = args.push_api_key or os.getenv("PLLM_PUSH_API_KEY")
    if not inference_url or not push_api_key:
        parser.error("preparation requires --inference-url and --push-api-key")
    translated = [
        "--privacy-mode", "public",
        "--host", args.host,
        "--port", str(args.port),
        "--api-key", api_key,
        "--model-kind", args.model_kind,
        "--weight-bits", str(args.weight_bits),
        "--activation-bits", str(args.activation_bits),
        "--inference-url", inference_url,
        "--push-api-key", push_api_key,
        "--push-timeout", str(args.push_timeout),
    ]
    for model in args.models:
        translated.extend(["--model", model])
    for model_id in args.model_id:
        translated.extend(["--model-id", model_id])
    for name, value in (
        ("--revision", args.revision),
        ("--hf-token", args.hf_token),
        ("--hf-cache-dir", args.hf_cache_dir),
        ("--native-library", args.native_library),
        ("--compiled-cache-dir", args.compiled_cache_dir),
        ("--engine-threads", args.engine_threads),
    ):
        if value is not None:
            translated.extend([name, str(value)])
    if args.local_files_only:
        translated.append("--local-files-only")
    print(
        json.dumps(
            {
                "service": "trusted-preparation",
                "listen": f"http://{args.host}:{args.port}",
                "models": args.models,
                "api_key": api_key if generated else "configured",
            },
            indent=2,
        )
    )
    preparation_main(translated)


def _configure(args: argparse.Namespace) -> None:
    current = ClientSettings.load()
    updated = current.merged(
        base_url=args.base_url,
        api_key=args.api_key,
        preparation_base_url=args.preparation_base_url,
        preparation_api_key=args.preparation_api_key,
        model=args.model,
        transport=args.transport,
        correlation_mode=args.correlation_mode,
        correlation_prefetch=args.correlation_prefetch,
        token_cache_size=args.token_cache_size,
        bundle_cache_mode=args.bundle_cache_mode,
        bundle_cache_dir=args.bundle_cache_dir,
        timeout=args.timeout,
    )
    path = updated.save()
    print(
        json.dumps(
            {
                "saved": str(path),
                "base_url": updated.base_url,
                "model": updated.model,
                "transport": updated.transport,
                "correlation_mode": updated.correlation_mode,
                "preparation_base_url": updated.preparation_base_url,
                "bundle_cache_mode": updated.bundle_cache_mode,
                "bundle_cache_dir": updated.bundle_cache_dir,
            },
            indent=2,
        )
    )


def _run_sidecar(args: argparse.Namespace) -> None:
    settings = ClientSettings.load().merged(
        base_url=args.base_url,
        api_key=args.api_key,
        preparation_base_url=args.preparation_base_url,
        preparation_api_key=args.preparation_api_key,
        model=args.model,
        transport=args.transport,
        correlation_mode=args.correlation_mode,
        correlation_prefetch=args.correlation_prefetch,
        token_cache_size=args.token_cache_size,
        bundle_cache_mode=args.bundle_cache_mode,
        bundle_cache_dir=args.bundle_cache_dir,
    )
    translated = [
        "--remote-base-url",
        settings.base_url,
        "--remote-api-key",
        settings.api_key,
        "--local-api-key",
        args.local_api_key,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--transport",
        "websocket" if settings.transport == "auto" else settings.transport,
        "--correlation-mode",
        settings.correlation_mode,
        "--correlation-prefetch",
        str(settings.correlation_prefetch),
        "--token-cache-size",
        str(settings.token_cache_size),
        "--bundle-cache-mode",
        settings.bundle_cache_mode,
    ]
    if settings.bundle_cache_dir:
        translated.extend(["--bundle-cache-dir", settings.bundle_cache_dir])
    if settings.preparation_base_url:
        translated.extend(["--preparation-base-url", settings.preparation_base_url])
    if settings.preparation_api_key:
        translated.extend(["--preparation-api-key", settings.preparation_api_key])
    if args.tenseal_path:
        translated.extend(["--tenseal-path", args.tenseal_path])
    sidecar_main(translated)


def _market(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.market_command == "serve":
        key = args.admin_key or os.getenv("PLLM_MARKET_ADMIN_KEY") or secrets.token_urlsafe(24)
        if not args.admin_key and not os.getenv("PLLM_MARKET_ADMIN_KEY"):
            print(f"Generated market admin key: {key}")
        uvicorn.run(
            create_market_app(args.database, admin_key=key, fee_rate=args.fee_rate),
            host=args.host,
            port=args.port,
            access_log=False,
        )
        return
    if args.market_command == "simulate":
        result = simulate_market(providers=args.providers, requests=args.requests, seed=args.seed)
        text = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(text)
        return
    parser.error("choose `pllm market serve` or `pllm market simulate`")


def _provider(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.provider_command == "join":
        run_provider_agent(
            market_url=args.market,
            market_key=args.market_key,
            provider_id=args.provider_id,
            endpoint=args.endpoint,
            model=args.model,
            key_path=args.key_file,
            model_privacy=args.model_privacy,
            price_input=args.price_input,
            price_output=args.price_output,
            capacity_tps=args.capacity_tps,
            latency_ms=args.latency_ms,
            region=args.region,
            heartbeat_seconds=args.heartbeat_seconds,
            once=args.once,
        )
        return
    if args.provider_command != "offer":
        parser.error("choose `pllm provider offer` or `pllm provider join`")
    key = Ed25519PrivateKey.generate()
    offer = generate_provider_offer(
        provider_id=args.provider_id,
        endpoint=args.endpoint,
        model=args.model,
        key=key,
        model_privacy=args.model_privacy,
        price_input=args.price_input,
        price_output=args.price_output,
        capacity_tps=args.capacity_tps,
        latency_ms=args.latency_ms,
        region=args.region,
    )
    output = {"offer": asdict(offer), "private_key": key.private_bytes_raw().hex()}
    text = json.dumps(output, indent=2)
    if args.output:
        path = Path(args.output)
        path.write_text(text + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    print(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pllm",
        allow_abbrev=False,
        description="Private language model inference with an OpenAI Responses compatible client and server",
    )
    parser.add_argument("--version", action="version", version=f"pllm {VERSION}")
    commands = parser.add_subparsers(dest="command")

    serve = commands.add_parser(
        "serve",
        help="load Hugging Face models and run the private inference server",
        allow_abbrev=False,
    )
    serve.add_argument("models", nargs="*", help="Hugging Face repository IDs or local model directories")
    serve.add_argument("--model", dest="model_compat", action="append", default=[], help=argparse.SUPPRESS)
    serve.add_argument("--model-id", action="append", default=[], help="public model ID, one per model")
    serve.add_argument(
        "--weights",
        choices=("public", "confidential"),
        help="whether clients are allowed to learn the checkpoint weights; this declaration is required",
    )
    serve.add_argument(
        "--client-trust",
        "--client-security",
        dest="client_trust",
        choices=("honest", "guarded", "untrusted"),
        default="guarded",
        help=(
            "honest trusts the published client; guarded adds budgets and rate limits; "
            "untrusted requires the authenticated shares compiler and currently fails closed"
        ),
    )
    serve.add_argument(
        "--activation-protection",
        "--linear-protocol",
        dest="activation_protection",
        choices=(
            "automatic",
            "seeded-preparation",
            "precomputed-masks",
            "guarded-blinded-masks",
            "blinded-masks",
            "encrypted-activations",
            "authenticated-shares",
        ),
        default="automatic",
        help="advanced choice for protecting each remote model stage",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--api-key")
    serve.add_argument("--provider-push-api-key", default=os.getenv("PLLM_PROVIDER_PUSH_API_KEY"))
    serve.add_argument("--rendezvous-timeout", type=float, default=30.0)
    serve.add_argument("--rendezvous-capacity", type=int, default=4096)
    serve.add_argument("--rendezvous-max-bytes", type=int, default=268_435_456)
    serve.add_argument("--prepared-session-capacity", type=int, default=4096)
    serve.add_argument("--prepared-session-idle", type=float, default=300.0)
    serve.add_argument("--config")
    serve.add_argument(
        "--model-kind",
        default="huggingface",
        choices=("huggingface", "safetensors", "vllm", "mlx", "mlx-lm"),
    )
    serve.add_argument("--revision")
    serve.add_argument(
        "--hf-token",
        help="Hugging Face token; defaults to HF_TOKEN or the stored login",
    )
    serve.add_argument(
        "--hf-cache-dir",
        help="PLLM model cache base; defaults to HF_HUB_CACHE, HF_HOME, or ~/.cache/pllm/models",
    )
    serve.add_argument("--local-files-only", action="store_true")
    serve.add_argument("--tenseal-path", default=os.getenv("PLLM_PYDEPS"))
    serve.add_argument("--max-batch-size", type=int)
    serve.add_argument("--max-wait-ms", type=float)
    serve.add_argument("--fixed-batch-wait", action="store_true")
    serve.add_argument("--engine-threads", type=int)
    serve.add_argument("--native-library")
    serve.add_argument("--compiled-cache-dir")
    serve.add_argument("--streaming-threshold-elements", type=int, default=50_000_000)
    serve.add_argument("--quantization-chunk-rows", type=int, default=64)
    serve.add_argument("--weight-bits", type=int, choices=(4, 8), default=8)
    serve.add_argument("--activation-bits", type=int, choices=(4, 8), default=8)
    serve.add_argument("--allow-test-correlations", action="store_true", help="unsafe local testing only")
    serve.add_argument("--guard-max-rows-per-request", type=int, default=4096)
    serve.add_argument("--guard-max-rows-per-stage", type=int, default=16384)
    serve.add_argument("--guard-max-requests-per-minute", type=int, default=4096)
    serve.add_argument("--guard-output-dither", type=int, default=0)
    serve.add_argument(
        "--privacy-mode",
        "--mode",
        dest="legacy_mode",
        choices=("public", "proprietary"),
        help=argparse.SUPPRESS,
    )

    preparation = commands.add_parser(
        "preparation", help="run the trusted public-weight preparation service"
    )
    preparation_commands = preparation.add_subparsers(dest="preparation_command")
    preparation_serve = preparation_commands.add_parser(
        "serve", help="load public models and serve seeded preparation"
    )
    preparation_serve.add_argument("models", nargs="+", help="Hugging Face repository IDs or local model directories")
    preparation_serve.add_argument("--model-id", action="append", default=[])
    preparation_serve.add_argument("--host", default="127.0.0.1")
    preparation_serve.add_argument("--port", type=int, default=8001)
    preparation_serve.add_argument("--api-key")
    preparation_serve.add_argument("--inference-url")
    preparation_serve.add_argument("--push-api-key")
    preparation_serve.add_argument("--push-timeout", type=float, default=10.0)
    preparation_serve.add_argument(
        "--model-kind",
        default="huggingface",
        choices=("huggingface", "safetensors", "vllm", "mlx", "mlx-lm"),
    )
    preparation_serve.add_argument("--revision")
    preparation_serve.add_argument("--hf-token")
    preparation_serve.add_argument("--hf-cache-dir")
    preparation_serve.add_argument("--local-files-only", action="store_true")
    preparation_serve.add_argument("--engine-threads", type=int)
    preparation_serve.add_argument("--native-library")
    preparation_serve.add_argument("--compiled-cache-dir")
    preparation_serve.add_argument("--weight-bits", type=int, choices=(4, 8), default=8)
    preparation_serve.add_argument("--activation-bits", type=int, choices=(4, 8), default=8)
    serve.add_argument(
        "--proprietary-protocol",
        "--protocol",
        dest="legacy_protocol",
        choices=("guarded", "blinded", "secure", "direct"),
        help=argparse.SUPPRESS,
    )

    configure = commands.add_parser(
        "configure",
        help="save defaults used automatically by the SDK and chat client",
    )
    _add_client_options(configure)
    configure.add_argument("--transport", choices=("auto", "websocket", "http"))
    configure.add_argument("--correlation-mode", choices=("bfv", "local-test"))
    configure.add_argument("--correlation-prefetch", type=int)
    configure.add_argument("--token-cache-size", type=int)
    configure.add_argument(
        "--bundle-cache-mode", choices=("read-write", "read-only", "refresh", "off")
    )
    configure.add_argument("--bundle-cache-dir")
    configure.add_argument("--timeout", type=float)

    chat = commands.add_parser("chat", help="open an interactive private chat session")
    _add_client_options(chat)
    chat.add_argument("--no-stream", action="store_true")
    chat.add_argument("--max-output-tokens", type=int, default=64)

    sidecar = commands.add_parser("sidecar", help="run a localhost OpenAI compatible proxy for any SDK")
    _add_client_options(sidecar)
    sidecar.add_argument("--local-api-key", default="local")
    sidecar.add_argument("--host", default="127.0.0.1")
    sidecar.add_argument("--port", type=int, default=8080)
    sidecar.add_argument("--transport", choices=("auto", "websocket", "http"))
    sidecar.add_argument("--correlation-mode", choices=("bfv", "local-test"))
    sidecar.add_argument("--correlation-prefetch", type=int)
    sidecar.add_argument("--token-cache-size", type=int)
    sidecar.add_argument(
        "--bundle-cache-mode", choices=("read-write", "read-only", "refresh", "off")
    )
    sidecar.add_argument("--bundle-cache-dir")
    sidecar.add_argument("--tenseal-path", default=os.getenv("PLLM_PYDEPS"))

    commands.add_parser("build", help="build the native modular arithmetic kernels")
    commands.add_parser("security", help="show the precise security claim for every protocol")

    secure = commands.add_parser(
        "secure",
        help="run the complete authenticated decoder research preview",
        allow_abbrev=False,
    )
    secure.add_argument("--output", choices=("token", "logits"), default=None)
    secure.add_argument("--batch", type=int, default=None)
    secure.add_argument("--he", action="store_true")
    secure.add_argument("--tenseal-path")

    market = commands.add_parser("market", help="run or benchmark the private inference provider market")
    market_commands = market.add_subparsers(dest="market_command")
    market_serve = market_commands.add_parser("serve", help="run the central router and settlement service")
    market_serve.add_argument("--database", default="pllm-market.sqlite3")
    market_serve.add_argument("--admin-key")
    market_serve.add_argument("--fee-rate", type=float, default=0.10)
    market_serve.add_argument("--host", default="127.0.0.1")
    market_serve.add_argument("--port", type=int, default=8090)
    market_simulate = market_commands.add_parser("simulate", help="compare central and decentralized routing")
    market_simulate.add_argument("--providers", type=int, default=100)
    market_simulate.add_argument("--requests", type=int, default=5000)
    market_simulate.add_argument("--seed", type=int, default=7)
    market_simulate.add_argument("--output")

    provider = commands.add_parser("provider", help="create signed offers for the inference market")
    provider_commands = provider.add_subparsers(dest="provider_command")
    offer = provider_commands.add_parser("offer", help="generate a signed provider offer and key")
    offer.add_argument("--provider-id", required=True)
    offer.add_argument("--endpoint", required=True)
    offer.add_argument("--model", required=True)
    offer.add_argument("--model-privacy", action="store_true")
    offer.add_argument("--price-input", type=float, default=0.25)
    offer.add_argument("--price-output", type=float, default=0.75)
    offer.add_argument("--capacity-tps", type=float, default=20)
    offer.add_argument("--latency-ms", type=float, default=20)
    offer.add_argument("--region", default="global")
    offer.add_argument("--output")

    join = provider_commands.add_parser("join", help="register an offer and maintain provider heartbeats")
    join.add_argument("--market", required=True, help="central market base URL")
    join.add_argument("--market-key", required=True, help="provider admission key")
    join.add_argument("--provider-id", required=True)
    join.add_argument("--endpoint", required=True)
    join.add_argument("--model", required=True)
    join.add_argument("--model-privacy", action="store_true")
    join.add_argument("--price-input", type=float, default=0.25)
    join.add_argument("--price-output", type=float, default=0.75)
    join.add_argument("--capacity-tps", type=float, default=20)
    join.add_argument("--latency-ms", type=float, default=20)
    join.add_argument("--region", default="global")
    join.add_argument("--key-file", default="~/.config/pllm/provider.key")
    join.add_argument("--heartbeat-seconds", type=float, default=10.0)
    join.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command is None:
        parser.print_help()
        return
    if args.command == "serve":
        _serve(args, parser)
    elif args.command == "preparation":
        _preparation(args, parser)
    elif args.command == "configure":
        _configure(args)
    elif args.command == "chat":
        run_chat(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            preparation_base_url=args.preparation_base_url,
            preparation_api_key=args.preparation_api_key,
            stream=not args.no_stream,
            max_output_tokens=max(1, args.max_output_tokens),
        )
    elif args.command == "sidecar":
        _run_sidecar(args)
    elif args.command == "build":
        build_native_main()
    elif args.command == "security":
        print(json.dumps({name: claim.to_dict() for name, claim in PROFILES.items()}, indent=2))
    elif args.command == "secure":
        forwarded: list[str] = []
        if args.output is not None:
            forwarded.extend(["--output", args.output])
        if args.batch is not None:
            forwarded.extend(["--batch", str(args.batch)])
        if args.he:
            forwarded.append("--he")
        if args.tenseal_path:
            forwarded.extend(["--tenseal-path", args.tenseal_path])
        _secure_preview(forwarded)
    elif args.command == "market":
        _market(args, parser)
    elif args.command == "provider":
        _provider(args, parser)


if __name__ == "__main__":
    main()

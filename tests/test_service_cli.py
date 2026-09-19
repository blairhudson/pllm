from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import pytest

from pllm._cli.app import build_parser


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"


def test_public_parser_owns_gateway_and_serve_syntax() -> None:
    parser = build_parser()

    gateway = parser.parse_args(
        [
            "gateway",
            "--config",
            "client.toml",
            "--inference-url",
            "https://inference.example",
            "--preparation-url",
            "https://preparation.example",
            "--transport",
            "websocket",
            "--prepared-inventory-rows",
            "128",
        ]
    )
    inference = parser.parse_args(
        [
            "serve",
            "inference",
            "--model",
            "org/model",
            "--model-id",
            "public-model",
            "--privacy-mode",
            "proprietary",
            "--weight-bits",
            "4",
        ]
    )
    preparation = parser.parse_args(
        ["serve", "preparation", "--inference-url", "http://127.0.0.1:9000"]
    )

    assert gateway.command == "gateway"
    assert gateway.prepared_inventory_rows == 128
    assert inference.serve_role == "inference"
    assert inference.model == ["org/model"]
    assert inference.model_id == ["public-model"]
    assert inference.weight_bits == 4
    assert preparation.serve_role == "preparation"
    assert "gateway" in parser.format_help()
    assert "serve" in parser.format_help()


def test_serve_accepts_python_experiment_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pllm.cli import main

    main(
        [
            "--format",
            "json",
            "--no-input",
            "serve",
            "inference",
            "--experiment",
            "examples/benchmarks/qwen_prepared.py:cpu_4",
            "--trust-python",
            "--dry-run",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    data = result["data"]
    assert data["models"] == ["Qwen/Qwen2.5-0.5B-Instruct"]
    assert data["experiment"]["name"] == "qwen-prepared-cpu-4"
    assert len(data["experiment"]["configuration_digest"]) == 64


@pytest.mark.parametrize(
    "arguments",
    [
        ["gateway", "--local", "--model", "org/model", "--dry-run"],
        ["serve", "inference", "--model", "org/model", "--dry-run"],
        ["serve", "preparation", "--inference-url", "http://127.0.0.1:9000", "--dry-run"],
    ],
)
def test_service_dry_run_does_not_import_runtime(arguments: list[str]) -> None:
    code = f"""
import sys
from pllm.cli import main
main({arguments!r})
assert 'pllm.runtime.cli' not in sys.modules
assert 'uvicorn' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(PYTHON)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_service_parser_preserves_role_config_precedence() -> None:
    from pllm._cli.app import build_parser

    args = build_parser().parse_args(["serve", "inference", "--config", "role.json"])
    assert args.privacy_mode is None
    assert args.proprietary_protocol is None
    assert args.model_kind is None
    assert args.weight_bits is None
    assert args.activation_bits is None


def test_dry_runs_resolve_configuration_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = tmp_path / "client.toml"
    client.write_text(
        '[client]\nbase_url = "https://inference.example"\napi_key = "secret"\n'
        'model = "configured-model"\ntransport = "websocket"\n',
        encoding="utf-8",
    )
    role = tmp_path / "role.json"
    role.write_text(
        json.dumps(
            {
                "privacy_mode": "proprietary",
                "proprietary_protocol": "direct",
                "api_keys": ["secret"],
                "engine_models": [{"name": "configured-engine"}],
            }
        ),
        encoding="utf-8",
    )
    from pllm.cli import main

    main(["--format", "json", "gateway", "--config", str(client), "--dry-run"])
    gateway = json.loads(capsys.readouterr().out)
    assert gateway["data"]["model"] == "configured-model"
    assert gateway["data"]["transport"] == "websocket"

    main(["--format", "json", "serve", "inference", "--config", str(role), "--dry-run"])
    serve = json.loads(capsys.readouterr().out)
    assert serve["data"]["models"] == ["configured-engine"]
    assert serve["data"]["privacy_mode"] == "proprietary"
    assert serve["data"]["protocol"] == "direct"


def test_service_config_values_survive_public_parser_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pllm._cli.app import build_parser
    from pllm.runtime import cli

    config = tmp_path / "role.json"
    config.write_text(
        json.dumps(
            {
                "privacy_mode": "proprietary",
                "proprietary_protocol": "direct",
                "api_keys": ["configured-secret"],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        cli, "create_app", lambda value, **kwargs: captured.update(value=value) or object()
    )
    monkeypatch.setattr(
        cli.uvicorn, "run", lambda app, **kwargs: captured.update(app=app, run=kwargs)
    )

    args = build_parser().parse_args(["serve", "inference", "--config", str(config)])
    cli.run_server(args)

    assert captured["value"].privacy_mode == "proprietary"
    assert captured["value"].proprietary_protocol == "direct"
    assert captured["value"].api_keys == ("configured-secret",)


def test_gateway_model_id_reaches_live_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from pllm._cli.app import build_parser
    from pllm.runtime import cli

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        cli,
        "create_sidecar_app",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(cli.uvicorn, "run", lambda *_args, **_kwargs: None)
    args = build_parser().parse_args(["gateway", "--model-id", "configured-alias"])
    cli.run_sidecar(args)
    assert captured["default_model"] == "configured-alias"


def test_gateway_and_serve_quiet_dry_runs_emit_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    from pllm.cli import main

    main(["--quiet", "gateway", "--local", "--model", "org/model", "--dry-run"])
    main(["--quiet", "serve", "inference", "--model", "org/model", "--dry-run"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_non_loopback_serve_rejects_default_credentials() -> None:
    from pllm._cli.app import build_parser
    from pllm.runtime.cli import RuntimeCLIError, run_server

    args = build_parser().parse_args(
        [
            "serve",
            "inference",
            "--host",
            "0.0.0.0",
            "--model",
            "org/model",
            "--provider-push-api-key",
            "push-secret",
        ]
    )
    with pytest.raises(RuntimeCLIError, match="explicit non-default credentials"):
        run_server(args)


def test_malformed_service_env_does_not_break_unrelated_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pllm.cli import main

    monkeypatch.setenv("PLLM_RENDEZVOUS_CAPACITY", "not-an-integer")
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    assert capsys.readouterr().out.strip()


def test_serve_rejects_mode_override_for_configured_models(tmp_path: Path) -> None:
    from pllm._cli.app import build_parser
    from pllm.runtime.cli import RuntimeCLIError, run_server

    config = tmp_path / "role.json"
    config.write_text(
        json.dumps(
            {
                "api_keys": ["secret"],
                "privacy_mode": "proprietary",
                "proprietary_protocol": "direct",
                "engine_models": [{"model_id": "configured-model"}],
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["serve", "inference", "--config", str(config), "--privacy-mode", "public"]
    )
    with pytest.raises(RuntimeCLIError, match="cannot override configured models"):
        run_server(args)


def test_gateway_dry_run_reports_effective_client_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from pllm.cli import main

    config = tmp_path / "client.toml"
    config.write_text(
        "\n".join(
            [
                "[client]",
                'base_url = "http://127.0.0.1:9101"',
                'api_key = "server"',
                'preparation_base_url = "http://127.0.0.1:9102"',
                'preparation_api_key = "preparation"',
                'model = "org/config-model"',
                'transport = "websocket"',
                'bundle_cache_mode = "read-only"',
            ]
        ),
        encoding="utf-8",
    )
    main(["--format", "json", "gateway", "--config", str(config), "--dry-run"])
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["model"] == "org/config-model"
    assert data["transport"] == "websocket"
    assert data["bundle_cache_mode"] == "read-only"


def test_serve_dry_run_reports_effective_role_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from pllm.cli import main

    config = tmp_path / "inference.json"
    config.write_text(
        json.dumps(
            {
                "privacy_mode": "proprietary",
                "proprietary_protocol": "direct",
                "engine_models": [{"name": "org/config-model", "model_kind": "huggingface"}],
            }
        ),
        encoding="utf-8",
    )
    main(
        [
            "--format",
            "json",
            "serve",
            "inference",
            "--config",
            str(config),
            "--dry-run",
        ]
    )
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["models"] == ["org/config-model"]
    assert data["privacy_mode"] == "proprietary"
    assert data["protocol"] == "direct"


def test_gateway_dry_run_rejects_public_bind_and_never_prints_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pllm._cli.app as app

    app.main(
        [
            "gateway",
            "--dry-run",
            "--format",
            "json",
            "--api-key",
            "gateway-secret",
            "--inference-key",
            "inference-secret",
            "--preparation-key",
            "preparation-secret",
        ]
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["command"] == "gateway"
    assert "secret" not in captured.out
    assert captured.err == ""

    with pytest.raises(SystemExit, match="3"):
        app.main(["gateway", "--dry-run", "--host", "0.0.0.0"])
    assert "GATEWAY_HOST" in capsys.readouterr().err


def test_runtime_runners_do_not_define_competing_cli_parsers() -> None:
    from pllm.runtime import cli

    for name in (
        "_server_parser",
        "_sidecar_parser",
        "_server_main",
        "server_main",
        "preparation_main",
        "sidecar_main",
        "main",
    ):
        assert not hasattr(cli, name)


def test_public_dispatch_passes_parsed_namespace_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pllm._cli.app as app
    from pllm.runtime import cli

    calls: list[tuple[argparse.Namespace, bool]] = []
    monkeypatch.setattr(
        cli,
        "run_server",
        lambda args, *, preparation=False: calls.append((args, preparation)),
    )
    app.main(["serve", "preparation", "--port", "8123", "--api-key", "secret"])

    assert len(calls) == 1
    args, preparation = calls[0]
    assert isinstance(args, argparse.Namespace)
    assert args.port == 8123
    assert args.api_key == "secret"
    assert preparation is True

    app.main(
        [
            "--no-input",
            "serve",
            "inference",
            "--experiment",
            "examples/benchmarks/qwen_prepared.py:cpu_4",
            "--trust-python",
        ]
    )
    args, preparation = calls[1]
    assert args.model == ["Qwen/Qwen2.5-0.5B-Instruct"]
    assert args.engine_threads == 4
    assert preparation is False

    gateway_calls: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "run_sidecar", gateway_calls.append)

    app.main(["gateway", "--port", "8124", "--inference-url", "http://127.0.0.1:9000"])

    assert len(gateway_calls) == 1
    assert gateway_calls[0].port == 8124
    assert gateway_calls[0].inference_url == "http://127.0.0.1:9000"


def test_gateway_runner_loads_explicit_client_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pllm.runtime import cli

    for name in (
        "PLLM_API_KEY",
        "PLLM_BASE_URL",
        "PLLM_MODEL",
        "PLLM_PREPARATION_API_KEY",
        "PLLM_PREPARATION_BASE_URL",
        "PLLM_TRANSPORT",
    ):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "client.toml"
    config.write_text(
        "[client]\n"
        'base_url = "https://inference.example"\n'
        'api_key = "inference-secret"\n'
        'model = "model-a"\n'
        'transport = "http"\n'
        'preparation_base_url = "https://preparation.example"\n'
        'preparation_api_key = "preparation-secret"\n',
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        cli,
        "create_sidecar_app",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        cli.uvicorn, "run", lambda app, **kwargs: captured.update(app=app, run=kwargs)
    )

    cli.run_sidecar(
        argparse.Namespace(config=str(config), host="127.0.0.1", port=8080, api_key="gateway")
    )

    assert captured["remote_base_url"] == "https://inference.example"
    assert captured["remote_api_key"] == "inference-secret"
    assert captured["preparation_base_url"] == "https://preparation.example"
    assert captured["preparation_api_key"] == "preparation-secret"
    assert captured["default_model"] == "model-a"
    assert captured["session_transport"] == "http"


def test_local_gateway_uses_shared_topology_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pllm.runtime import cli

    args = build_parser().parse_args(["gateway", "--local", "--model", "org/model"])
    captured: dict[str, Any] = {}

    class Topology:
        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, *_args):
            captured["closed"] = True

        @staticmethod
        def gateway_app(**kwargs):
            captured["gateway"] = kwargs
            return object()

    def roles(model, **kwargs):
        captured["model"] = model
        captured["roles"] = kwargs
        return Topology()

    monkeypatch.setattr(cli, "build_roles", roles)
    monkeypatch.setattr(
        cli.uvicorn,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gateway startup failed")),
    )

    with pytest.raises(RuntimeError, match="gateway startup failed"):
        cli.run_local_gateway(args)

    assert captured["entered"] is True
    assert captured["closed"] is True
    assert captured["model"].source == "org/model"
    assert captured["roles"]["reserved_ports"] == (8080,)
    assert captured["gateway"]["local_api_key"] == "local"


def test_local_gateway_tiny_flag_selects_typed_tiny_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pllm.runtime import cli
    from pllm.sources import TinyModel

    args = build_parser().parse_args(["gateway", "--local", "--tiny"])
    captured: dict[str, Any] = {}

    class Topology:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def gateway_app(**_kwargs):
            return object()

    def roles(model, **kwargs):
        captured["model"] = model
        captured["model_id"] = kwargs["model_id"]
        return Topology()

    monkeypatch.setattr(cli, "build_roles", roles)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *_args, **_kwargs: None)
    cli.run_local_gateway(args)

    assert isinstance(captured["model"], TinyModel)
    assert captured["model_id"] == "pllm-gateway-tiny"


def test_local_gateway_propagates_bfv_and_local_correlation_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pllm.runtime import cli

    args = build_parser().parse_args(
        [
            "gateway",
            "--local",
            "--model",
            "org/model",
            "--correlation-mode",
            "local-test",
            "--tenseal-path",
            "/tmp/tenseal",
        ]
    )
    captured: dict[str, Any] = {}

    class Topology:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def gateway_app(**kwargs):
            captured["gateway"] = kwargs
            return object()

    def roles(_model, **kwargs):
        captured["roles"] = kwargs
        return Topology()

    monkeypatch.setattr(cli, "build_roles", roles)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *_args, **_kwargs: None)

    cli.run_local_gateway(args)

    assert captured["roles"]["correlation_mode"] == "local-test"
    assert captured["roles"]["tenseal_path"] == "/tmp/tenseal"
    assert captured["gateway"]["tenseal_path"] == "/tmp/tenseal"


def test_local_cleanup_targets_child_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    from pllm.runtime import servers

    calls: list[tuple[int, signal.Signals]] = []

    class Process:
        pid = 4321

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(*, timeout: float) -> None:
            assert timeout == 10

    monkeypatch.setattr(servers.os, "name", "posix")
    monkeypatch.setattr(servers.os, "killpg", lambda pid, sig: calls.append((pid, sig)))

    servers.LocalTopology._stop(Process())

    assert calls == [(4321, signal.SIGTERM)]

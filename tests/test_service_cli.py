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
    monkeypatch.setattr(cli, "_server_main", lambda *_args, **_kwargs: pytest.fail("reparsed argv"))

    app.main(["serve", "preparation", "--port", "8123", "--api-key", "secret"])

    assert len(calls) == 1
    args, preparation = calls[0]
    assert isinstance(args, argparse.Namespace)
    assert args.port == 8123
    assert args.api_key == "secret"
    assert preparation is True

    gateway_calls: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "run_sidecar", gateway_calls.append)
    monkeypatch.setattr(cli, "sidecar_main", lambda *_args: pytest.fail("reparsed argv"))

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


def test_local_gateway_uses_env_credentials_and_cleans_up_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pllm.runtime import cli

    args = build_parser().parse_args(["gateway", "--local", "--model", "org/model"])
    ports = iter((9101, 9102))
    spawned: list[tuple[list[str], dict[str, str], bool]] = []
    processes: list[object] = []
    stopped: list[object] = []

    class Process:
        pid = 1234

        def poll(self) -> None:
            return None

    def popen(command: list[str], **kwargs: Any) -> Process:
        process = Process()
        processes.append(process)
        spawned.append((command, kwargs["env"], bool(kwargs["start_new_session"])))
        return process

    monkeypatch.setattr(cli, "_free_port", lambda: next(ports))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli, "_wait_for_service", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_stop_process", stopped.append)
    monkeypatch.setattr(
        cli,
        "run_sidecar",
        lambda _args: (_ for _ in ()).throw(RuntimeError("gateway startup failed")),
    )

    with pytest.raises(RuntimeError, match="gateway startup failed"):
        cli.run_local_gateway(args)

    assert len(spawned) == 2
    assert all(start_new_session for _, _, start_new_session in spawned)
    assert stopped == list(reversed(processes))
    all_arguments = [item for command, _, _ in spawned for item in command]
    inference_env = spawned[0][1]
    preparation_env = spawned[1][1]
    credentials = {
        inference_env["PLLM_API_KEY"],
        inference_env["PLLM_PROVIDER_PUSH_API_KEY"],
        preparation_env["PLLM_API_KEY"],
    }
    assert len(credentials) == 3
    assert credentials.isdisjoint(all_arguments)


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
    ports = iter((9201, 9202))
    commands: list[list[str]] = []
    sidecar_args: list[argparse.Namespace] = []

    class Process:
        pid = 1234

        def poll(self) -> None:
            return None

    monkeypatch.setattr(cli, "_free_port", lambda: next(ports))
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command) or Process(),
    )
    monkeypatch.setattr(cli, "_wait_for_service", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_stop_process", lambda _process: None)
    monkeypatch.setattr(cli, "run_sidecar", sidecar_args.append)

    cli.run_local_gateway(args)

    assert "--allow-insecure-local-correlations" in commands[0]
    assert "--correlation-mode" not in commands[0]
    assert "--correlation-mode" not in commands[1]
    assert commands[0][commands[0].index("--tenseal-path") + 1] == "/tmp/tenseal"
    assert commands[1][commands[1].index("--tenseal-path") + 1] == "/tmp/tenseal"
    assert sidecar_args[0].correlation_mode == "local-test"


def test_local_cleanup_targets_child_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    from pllm.runtime import cli

    calls: list[tuple[int, signal.Signals]] = []

    class Process:
        pid = 4321

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(*, timeout: float) -> None:
            assert timeout == 10

    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: calls.append((pid, sig)))

    cli._stop_process(Process())

    assert calls == [(4321, signal.SIGTERM)]

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pllm
from pllm.market import CentralMarket, DecentralizedQuoteBook, RouteRequest, generate_provider_offer
from pllm.provider import load_or_create_signing_key
from pllm.settings import ClientSettings, config_path


def test_package_name_version_and_uv_metadata() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "pllm"' in text
    assert 'dynamic = ["version"]' in text
    assert 'pllm = "pllm.cli:main"' in text
    assert "[tool.uv]" in text
    assert pllm.__version__ == __import__("pllm._version", fromlist=["__version__"]).__version__


def test_client_defaults_select_transport_automatically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLLM_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.delenv("PLLM_TRANSPORT", raising=False)
    assert ClientSettings.load().transport == "auto"


def test_settings_round_trip_and_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    monkeypatch.setenv("PLLM_CONFIG", str(path))
    settings = ClientSettings(
        base_url="https://private.example",
        api_key="secret",
        model="model-a",
        transport="auto",
        correlation_mode="bfv",
        correlation_prefetch=9,
        token_cache_size=700,
        bundle_cache_mode="read-only",
        bundle_cache_dir=str(tmp_path / "bundles"),
        timeout=45.0,
        preparation_base_url="https://preparation.example",
        preparation_api_key="preparation-secret",
    )
    assert settings.save() == path
    loaded = ClientSettings.load()
    assert loaded == settings
    assert config_path() == path
    assert path.stat().st_mode & 0o777 == 0o600


def test_settings_environment_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    monkeypatch.setenv("PLLM_CONFIG", str(path))
    ClientSettings(base_url="http://old", api_key="old").save()
    monkeypatch.setenv("PLLM_BASE_URL", "https://new")
    monkeypatch.setenv("PLLM_API_KEY", "new-key")
    monkeypatch.setenv("PLLM_PREPARATION_BASE_URL", "https://preparation.example")
    monkeypatch.setenv("PLLM_PREPARATION_API_KEY", "preparation-key")
    monkeypatch.setenv("PLLM_BUNDLE_CACHE_MODE", "refresh")
    monkeypatch.setenv("PLLM_BUNDLE_CACHE_DIR", str(tmp_path / "bundle-cache"))
    assert ClientSettings.load().base_url == "https://new"
    assert ClientSettings.load().api_key == "new-key"
    assert ClientSettings.load().preparation_base_url == "https://preparation.example"
    assert ClientSettings.load().preparation_api_key == "preparation-key"
    assert ClientSettings.load().bundle_cache_mode == "refresh"
    assert ClientSettings.load().bundle_cache_dir == str(tmp_path / "bundle-cache")


def test_signed_provider_offer_and_central_settlement() -> None:
    key = Ed25519PrivateKey.generate()
    offer = generate_provider_offer(
        provider_id="provider-a",
        endpoint="https://provider.example/v1",
        model="org/model",
        key=key,
        model_privacy=True,
        price_input=0.2,
        price_output=0.7,
        capacity_tps=25,
        latency_ms=15,
        region="sydney",
    )
    offer.verify()

    market = CentralMarket(fee_rate=0.10, heartbeat_ttl_seconds=600)
    market.put_offer(offer)
    market.heartbeat("provider-a", active_requests=0, success_rate=1.0)
    market.credit("account-a", 10.0)
    decision = market.route(
        "account-a",
        RouteRequest(
            model="org/model",
            input_tokens=1000,
            max_output_tokens=200,
            require_model_privacy=True,
            preferred_region="sydney",
        ),
    )
    assert decision.provider_id == "provider-a"
    reservation_id = json.loads(
        __import__("base64").urlsafe_b64decode(
            decision.ticket.split(".", 1)[0] + "=="
        )
    )["reservation_id"]
    result = market.settle(reservation_id, actual_provider_cost=decision.estimated_cost / 1.1)
    assert result["provider_payment"] > 0
    assert result["operator_fee"] > 0


def test_decentralized_quote_book_verifies_offer() -> None:
    key = Ed25519PrivateKey.generate()
    offer = generate_provider_offer(
        provider_id="p",
        endpoint="https://p.example/v1",
        model="m",
        key=key,
    )
    book = DecentralizedQuoteBook()
    book.publish(offer)
    assert book.route(RouteRequest("m", 10, 10)).provider_id == "p"


def test_provider_key_persists(tmp_path: Path) -> None:
    path = tmp_path / "provider.key"
    first = load_or_create_signing_key(path)
    second = load_or_create_signing_key(path)
    assert first.private_bytes_raw() == second.private_bytes_raw()
    assert path.stat().st_mode & 0o777 == 0o600


def test_docs_and_manuscript_exist() -> None:
    assert Path("docs/package.json").exists()
    assert Path("docs/content/docs/start/first-private-request.mdx").exists()
    assert Path("docs/content/research/paper.mdx").exists()
    manuscript = Path("paper/manuscript.md").read_text(encoding="utf-8")
    assert "abstract: |" in manuscript
    assert "bibliography: paper/references.bib" in manuscript
    assert len(manuscript) > 5000

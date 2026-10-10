from __future__ import annotations

import os
import json
from dataclasses import dataclass, fields
from pathlib import Path
import tomllib


def config_path() -> Path:
    if value := os.getenv("PLLM_CONFIG"):
        return Path(value).expanduser()
    base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "pllm" / "config.toml"


@dataclass(slots=True)
class ClientSettings:
    base_url: str = "http://127.0.0.1:8000"
    api_key: str = "pllm-local"
    model: str | None = None
    transport: str = "auto"
    correlation_mode: str = "bfv"
    preparation_base_url: str | None = None
    preparation_api_key: str | None = None
    correlation_prefetch: int = 4
    prepared_inventory_rows: int = 64
    token_cache_size: int = 512
    bundle_cache_mode: str = "read-write"
    bundle_cache_dir: str | None = None
    timeout: float = 300.0
    network: str | None = None
    experiment: str | None = None
    planning: dict[str, object] | None = None
    performance_prior: str | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "ClientSettings":
        values: dict[str, object] = {}
        source = path or config_path()
        if path is not None and not source.is_file():
            raise FileNotFoundError(source)
        if source.exists():
            raw = tomllib.loads(source.read_text(encoding="utf-8"))
            client = raw.get("client", {})
            if not isinstance(client, dict):
                raise ValueError("[client] must be a TOML table")
            known = {item.name for item in fields(cls)}
            unknown = sorted(set(client) - known)
            if unknown:
                raise ValueError(f"Unknown client configuration field: {unknown[0]}")
            values.update(client)
        env = {
            "base_url": os.getenv("PLLM_BASE_URL"),
            "api_key": os.getenv("PLLM_API_KEY"),
            "model": os.getenv("PLLM_MODEL"),
            "transport": os.getenv("PLLM_TRANSPORT"),
            "correlation_mode": os.getenv("PLLM_CORRELATION_MODE"),
            "preparation_base_url": os.getenv("PLLM_PREPARATION_BASE_URL"),
            "preparation_api_key": os.getenv("PLLM_PREPARATION_API_KEY"),
            "correlation_prefetch": os.getenv("PLLM_CORRELATION_PREFETCH"),
            "prepared_inventory_rows": os.getenv("PLLM_PREPARED_INVENTORY_ROWS"),
            "token_cache_size": os.getenv("PLLM_TOKEN_CACHE_SIZE"),
            "bundle_cache_mode": os.getenv("PLLM_BUNDLE_CACHE_MODE"),
            "bundle_cache_dir": os.getenv("PLLM_BUNDLE_CACHE_DIR"),
            "timeout": os.getenv("PLLM_TIMEOUT"),
            "network": os.getenv("PLLM_NETWORK"),
            "experiment": os.getenv("PLLM_EXPERIMENT"),
            "performance_prior": os.getenv("PLLM_PERFORMANCE_PRIOR"),
        }
        for key, value in env.items():
            if value not in (None, ""):
                values[key] = value
        values["correlation_prefetch"] = int(values.get("correlation_prefetch", 4))
        values["prepared_inventory_rows"] = int(values.get("prepared_inventory_rows", 64))
        values["token_cache_size"] = int(values.get("token_cache_size", 512))
        values["timeout"] = float(values.get("timeout", 300.0))
        for name in ("network", "experiment", "performance_prior"):
            if values.get(name) is not None:
                value = values[name]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"client {name} must be a nonempty JSON document path")
                location = Path(value).expanduser()
                if not location.is_absolute() and not env.get(name):
                    location = source.parent / location
                values[name] = str(location)
        if values.get("planning") is not None:
            from pllm.search.placement import PlanningPolicy

            options = values["planning"]
            if not isinstance(options, dict):
                raise ValueError("[client.planning] must be a TOML table")
            options = dict(options)
            for name in ("objectives", "prior_objectives"):
                if name in options:
                    options[name] = tuple(options[name])
            PlanningPolicy(**options)
            values["planning"] = options
        return cls(**{f.name: values[f.name] for f in fields(cls) if f.name in values})

    def save(self, path: Path | None = None) -> Path:
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = ["[client]"]
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None or item.name == "planning":
                continue
            if isinstance(value, str):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                rows.append(f'{item.name} = "{escaped}"')
            elif isinstance(value, bool):
                rows.append(f"{item.name} = {'true' if value else 'false'}")
            else:
                rows.append(f"{item.name} = {value}")
        if self.planning is not None:
            rows.append("\n[client.planning]")
            for name, value in self.planning.items():
                if value is not None:
                    rows.append(f"{name} = {json.dumps(value, allow_nan=False)}")
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
        try:
            target.chmod(0o600)
        except OSError:
            pass
        return target

    def merged(self, **overrides: object) -> "ClientSettings":
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return ClientSettings(**values)

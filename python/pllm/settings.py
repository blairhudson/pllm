from __future__ import annotations

import os
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
    token_cache_size: int = 512
    bundle_cache_mode: str = "read-write"
    bundle_cache_dir: str | None = None
    timeout: float = 300.0

    @classmethod
    def load(cls) -> "ClientSettings":
        values: dict[str, object] = {}
        path = config_path()
        if path.exists():
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            values.update(raw.get("client", {}))
        env = {
            "base_url": os.getenv("PLLM_BASE_URL"),
            "api_key": os.getenv("PLLM_API_KEY"),
            "model": os.getenv("PLLM_MODEL"),
            "transport": os.getenv("PLLM_TRANSPORT"),
            "correlation_mode": os.getenv("PLLM_CORRELATION_MODE"),
            "preparation_base_url": os.getenv("PLLM_PREPARATION_BASE_URL"),
            "preparation_api_key": os.getenv("PLLM_PREPARATION_API_KEY"),
            "correlation_prefetch": os.getenv("PLLM_CORRELATION_PREFETCH"),
            "token_cache_size": os.getenv("PLLM_TOKEN_CACHE_SIZE"),
            "bundle_cache_mode": os.getenv("PLLM_BUNDLE_CACHE_MODE"),
            "bundle_cache_dir": os.getenv("PLLM_BUNDLE_CACHE_DIR"),
            "timeout": os.getenv("PLLM_TIMEOUT"),
        }
        for key, value in env.items():
            if value not in (None, ""):
                values[key] = value
        values["correlation_prefetch"] = int(values.get("correlation_prefetch", 4))
        values["token_cache_size"] = int(values.get("token_cache_size", 512))
        values["timeout"] = float(values.get("timeout", 300.0))
        return cls(**{f.name: values[f.name] for f in fields(cls) if f.name in values})

    def save(self, path: Path | None = None) -> Path:
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = ["[client]"]
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None:
                continue
            if isinstance(value, str):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                rows.append(f'{item.name} = "{escaped}"')
            elif isinstance(value, bool):
                rows.append(f"{item.name} = {'true' if value else 'false'}")
            else:
                rows.append(f"{item.name} = {value}")
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

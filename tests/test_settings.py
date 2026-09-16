from __future__ import annotations

from pathlib import Path

import pytest

from pllm.settings import ClientSettings


def test_client_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "client.toml"
    path.write_text('[client]\nmodel = "org/model"\nserver_url = "http://127.0.0.1:8081"\n')
    with pytest.raises(ValueError, match="Unknown client configuration field: server_url"):
        ClientSettings.load(path)

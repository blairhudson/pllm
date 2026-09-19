"""Ensure wheels contain the compiled native extension and that source/wheel versions agree."""
from __future__ import annotations
import argparse
import email
import tarfile
import zipfile
from pathlib import Path
from release import version

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheels = list(args.directory.glob("*.whl"))
    sources = list(args.directory.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sources) == 1, "Build exactly one wheel and one sdist"
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert wheel.testzip() is None
        names = wheel.namelist()
        assert "pllm/cli.py" in names
        assert any(n.startswith("pllm/_native") and n.endswith((".so", ".pyd")) for n in names), "Compiled Rust module missing"
        assert not any(n.startswith(("he_openai/", "python/", "docs/", "paper/")) for n in names)
        assert "pllm/runtime/client.py" in names
        assert not {"pllm/market.py", "pllm/market_server.py", "pllm/provider.py"} & set(names)
        assert "pllm/__main__.py" in names
        assert "pllm/py.typed" in names
        assert "pllm/_native.pyi" in names
        assert "pllm/config/__init__.py" in names
        assert "pllm/config/__init__.pyi" in names
        assert "pllm/providers/__init__.py" in names
        assert "pllm/providers/__init__.pyi" in names
        assert "pllm/providers/provider-manifest.schema.json" in names
        assert "pllm/include/pllm_plugin.h" in names
        assert wheel.read("pllm/include/pllm_plugin.h") == Path(
            "crates/pllm-plugin-api/include/pllm_plugin.h"
        ).read_bytes()
        assert wheel.read("pllm/providers/provider-manifest.schema.json") == Path(
            "schemas/provider-manifest.schema.json"
        ).read_bytes()
        assert "pllm/models.py" in names
        assert "pllm/models.pyi" in names
        assert "pllm/plan.py" in names
        assert "pllm/plan.pyi" in names
        assert "pllm/dashboard/index.html" in names
        assert "pllm/dashboard/app.js" in names
        assert "pllm/dashboard/style.css" in names
        wheel_metadata = email.message_from_bytes(wheel.read(next(n for n in names if n.endswith(".dist-info/WHEEL"))))
        assert wheel_metadata["Root-Is-Purelib"] == "false"
        metadata = email.message_from_bytes(wheel.read(next(n for n in names if n.endswith(".dist-info/METADATA"))))
        assert metadata["Name"] == "pllm" and metadata["Version"] == version()
    with tarfile.open(sources[0]) as archive:
        names = archive.getnames()
        assert any(n.endswith("/pyproject.toml") for n in names)
        assert any(n.endswith("/crates/pllm-python/Cargo.toml") for n in names)
        assert any(n.endswith("/crates/pllm-core/Cargo.toml") for n in names)
        assert any(n.endswith("/crates/pllm-plugin-api/Cargo.toml") for n in names)
        assert any(
            n.endswith("/crates/pllm-plugin-api/tests/fixtures/plugin/Cargo.lock")
            for n in names
        )
        assert any(
            n.endswith("/crates/pllm-plugin-api/tests/fixtures/plugin/src/lib.rs")
            for n in names
        )
        plugin_header = next(
            n for n in names if n.endswith("/crates/pllm-plugin-api/include/pllm_plugin.h")
        )
        package_header = next(
            n for n in names if n.endswith("/python/pllm/include/pllm_plugin.h")
        )
        assert archive.extractfile(plugin_header).read() == archive.extractfile(package_header).read()
        assert any(n.endswith("/python/pllm/__init__.py") for n in names)
        assert any(n.endswith("/schemas/provider-manifest.schema.json") for n in names)
        assert any(n.endswith("/schemas/benchmark-result.schema.json") for n in names)
        source_schema = next(
            n for n in names if n.endswith("/schemas/provider-manifest.schema.json")
        )
        package_schema = next(
            n
            for n in names
            if n.endswith("/python/pllm/providers/provider-manifest.schema.json")
        )
        assert archive.extractfile(source_schema).read() == archive.extractfile(package_schema).read()
        assert not any(
            name.endswith((
                "/python/pllm/market.py",
                "/python/pllm/market_server.py",
                "/python/pllm/provider.py",
            ))
            for name in names
        )
        assert any(n.endswith("/crates/pllm-core/src/kernels.rs") for n in names)
        assert any(n.endswith("/tests/test_protocol.py") for n in names)
        assert any(n.endswith("/python/pllm/dashboard/index.html") for n in names)
        assert any(n.endswith("/python/pllm/dashboard/app.js") for n in names)
        assert any(n.endswith("/python/pllm/dashboard/style.css") for n in names)
        assert not any("node_modules" in n or n.endswith(".key") for n in names)
    print("Wheel and source archive contain the required source and matching version")

if __name__ == "__main__":
    main()

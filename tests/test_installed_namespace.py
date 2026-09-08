"""These checks also run against each installed platform wheel in an empty directory."""
import importlib
import os
from pathlib import Path


def test_runtime_belongs_to_pllm():
    import pllm
    from pllm import OpenAI
    assert OpenAI.__module__ == 'pllm.runtime.client'
    assert Path(importlib.import_module('pllm.runtime.client').__file__).is_relative_to(
        Path(pllm.__file__).parent
    )


def test_client_server_exports_resolve_to_same_implementation():
    import pllm
    from pllm.client import OpenAI, AsyncOpenAI
    from pllm.server import create_app
    from pllm.official import create_openai_client
    assert pllm.OpenAI is OpenAI
    assert pllm.AsyncOpenAI is AsyncOpenAI
    assert pllm.create_app is create_app
    assert pllm.create_openai_client is create_openai_client


def test_explicit_native_requirement_cannot_fall_back(monkeypatch):
    from pllm.runtime._native_support import extension
    monkeypatch.setenv('PLLM_REQUIRE_RUST', '1')
    monkeypatch.setenv('PLLM_KERNEL_BACKEND', 'python')
    import pytest
    with pytest.raises(RuntimeError, match='Rust validation'):
        extension()

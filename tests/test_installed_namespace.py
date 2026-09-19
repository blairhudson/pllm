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


def test_public_domain_facades_share_root_identities():
    import pllm
    from pllm.config import ExecutionBudget, Experiment
    from pllm.models import (
        DecoderCoverageReport,
        DecoderRuntimeSchedule,
        ModelLoadError,
        ModelManifest,
        ModelPlan,
        load_model,
        lower_model,
    )
    from pllm.pipeline import MaskedLinearCpu
    from pllm.plan import CompiledPlan
    from pllm.runtime import serve_local

    assert pllm.ExecutionBudget is ExecutionBudget
    assert pllm.Experiment is Experiment
    assert pllm.DecoderCoverageReport is DecoderCoverageReport
    assert pllm.DecoderRuntimeSchedule is DecoderRuntimeSchedule
    assert pllm.ModelLoadError is ModelLoadError
    assert pllm.MaskedLinearCpu is MaskedLinearCpu
    assert pllm.ModelManifest is ModelManifest
    assert pllm.ModelPlan is ModelPlan
    assert pllm.load_model is load_model
    assert pllm.lower_model is lower_model
    assert pllm.serve_local is serve_local
    assert pllm.CompiledPlan is CompiledPlan


def test_runtime_facade_is_intentionally_narrow():
    import pllm.runtime as runtime

    assert set(runtime.__all__) == {
        'AsyncOpenAI', 'AsyncPLLMTransport', 'CompiledRuntimeModel', 'CompiledRuntimeSession',
        'ExecutionBudget', 'GatewayConfig', 'LocalTopology', 'OpenAI', 'PLLMTransport',
        'PrivacyMode', 'ProprietaryProtocol', 'RoleStatus', 'RuntimeBindingError',
        'RuntimeExecutionError', 'RuntimeStageBinding', 'TopologyError', 'build_roles',
        'compile_runtime_model', 'create_app', 'create_sidecar_app', 'serve_local', '__version__',
    }

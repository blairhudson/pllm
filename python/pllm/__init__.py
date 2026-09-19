"""Private LLM Inference: client API and native execution primitives."""

from __future__ import annotations
from importlib import import_module
from ._version import __version__

_EXPORTS = {
    "AsyncOpenAI": ("pllm.runtime.client", "AsyncOpenAI"),
    "AuthenticatedMPC": ("pllm.runtime.authenticated_mpc", "AuthenticatedMPC"),
    "AuthenticationError": ("pllm.runtime.authenticated_mpc", "AuthenticationError"),
    "BenchmarkResult": ("pllm.evidence", "BenchmarkResult"),
    "BlindedTransformerEngine": ("pllm.runtime.blinded_engine", "BlindedTransformerEngine"),
    "BundleModel": ("pllm.sources", "BundleModel"),
    "ClientBundle": ("pllm.runtime.transformer_client", "ClientBundle"),
    "CompilationError": ("pllm.compiler", "CompilationError"),
    "CompiledPlan": ("pllm.plan", "CompiledPlan"),
    "DirectFHEProfile": ("pllm.profiles", "DirectFHEProfile"),
    "DirectFHETransformerEngine": ("pllm.runtime.proprietary_engine", "DirectFHETransformerEngine"),
    "Constraint": ("pllm.search", "Constraint"),
    "GatewayConfig": ("pllm.runtime.config", "GatewayConfig"),
    "GridSearch": ("pllm.search", "GridSearch"),
    "GuardPolicy": ("pllm.runtime.guarded_engine", "GuardPolicy"),
    "GuardedBlindedTransformerEngine": (
        "pllm.runtime.guarded_engine",
        "GuardedBlindedTransformerEngine",
    ),
    "AsyncPLLMTransport": ("pllm.runtime.transport", "AsyncPLLMTransport"),
    "HEAuthenticatedPreprocessor": (
        "pllm.runtime.he_authenticated_preprocessing",
        "HEAuthenticatedPreprocessor",
    ),
    "PLLMTransport": ("pllm.runtime.transport", "PLLMTransport"),
    "MaskedTransformerClientRuntime": (
        "pllm.runtime.transformer_client",
        "MaskedTransformerClientRuntime",
    ),
    "MaskedTransformerEngine": ("pllm.runtime.transformer_engine", "MaskedTransformerEngine"),
    "OpenAI": ("pllm.runtime.client", "OpenAI"),
    "ParetoFrontier": ("pllm.search", "ParetoFrontier"),
    "PROFILES": ("pllm.runtime.formal_security", "PROFILES"),
    "PreparedInventory": ("pllm.runtime.preprocessing_inventory", "PreparedInventory"),
    "ProprietaryBlinded": ("pllm.profiles", "ProprietaryBlinded"),
    "ProprietaryGuarded": ("pllm.profiles", "ProprietaryGuarded"),
    "ProviderDescriptor": ("pllm.providers", "ProviderDescriptor"),
    "ProviderDiscoveryError": ("pllm.providers", "ProviderDiscoveryError"),
    "ProviderResource": ("pllm.providers", "ProviderResource"),
    "PreprocessingPlan": ("pllm.runtime.preprocessing_inventory", "PreprocessingPlan"),
    "PrivacyMode": ("pllm.runtime.privacy", "PrivacyMode"),
    "ProprietaryProtocol": ("pllm.runtime.privacy", "ProprietaryProtocol"),
    "RandomSearch": ("pllm.search", "RandomSearch"),
    "RecordingPreprocessor": ("pllm.runtime.preprocessing_inventory", "RecordingPreprocessor"),
    "Response": ("pllm.runtime.types", "Response"),
    "ResponseEvent": ("pllm.runtime.types", "ResponseEvent"),
    "ResponseStream": ("pllm.runtime.client", "ResponseStream"),
    "SECURE_PREVIEW": ("pllm.runtime.formal_security", "SECURE_PREVIEW"),
    "SecureDecoder": ("pllm.runtime.secure_transformer", "SecureDecoder"),
    "SecureDecoderConfig": ("pllm.runtime.secure_transformer", "SecureDecoderConfig"),
    "SecureDecoderWeights": ("pllm.runtime.secure_transformer", "SecureDecoderWeights"),
    "SearchCandidate": ("pllm.search", "SearchCandidate"),
    "SearchError": ("pllm.search", "SearchError"),
    "SearchEvaluation": ("pllm.search", "SearchEvaluation"),
    "SearchSpace": ("pllm.search", "SearchSpace"),
    "SecurityClaim": ("pllm.runtime.formal_security", "SecurityClaim"),
    "TinyModel": ("pllm.sources", "TinyModel"),
    "TrustedPreprocessor": ("pllm.runtime.authenticated_mpc", "TrustedPreprocessor"),
    "check_linear_result": ("pllm.verification", "check_linear_result"),
    "ComponentRef": ("pllm.configuration", "ComponentRef"),
    "ConfigurationError": ("pllm.configuration", "ConfigurationError"),
    "Cpu": ("pllm.kernels", "Cpu"),
    "Deployment": ("pllm.configuration", "Deployment"),
    "ExecutionBudget": ("pllm.configuration", "ExecutionBudget"),
    "EvidenceRegistry": ("pllm.evidence", "EvidenceRegistry"),
    "EvidenceReport": ("pllm.evidence", "EvidenceReport"),
    "Experiment": ("pllm.configuration", "Experiment"),
    "ExperimentProfile": ("pllm.configuration", "ExperimentProfile"),
    "KvCacheEviction": ("pllm.passes", "KvCacheEviction"),
    "MaskedLinear": ("pllm.protocols", "MaskedLinear"),
    "MaskedLinearCpu": ("pllm.profiles", "MaskedLinearCpu"),
    "Model": ("pllm.configuration", "Model"),
    "ModelLoadError": ("pllm.model_loader", "ModelLoadError"),
    "ModelAwareCorrections": ("pllm.preparation", "ModelAwareCorrections"),
    "ModelManifest": ("pllm.model_loader", "ModelManifest"),
    "ModelPlan": ("pllm.modeling", "ModelPlan"),
    "DecoderCoverageReport": ("pllm.modeling", "DecoderCoverageReport"),
    "DecoderRuntimeSchedule": ("pllm.modeling", "DecoderRuntimeSchedule"),
    "Pipeline": ("pllm.configuration", "Pipeline"),
    "assure": ("pllm.evidence", "assure"),
    "benchmark": ("pllm.evidence", "benchmark"),
    "deployment_benchmark": ("pllm.evidence", "deployment_benchmark"),
    "discover_providers": ("pllm.providers", "discover_providers"),
    "environment_digest": ("pllm.evidence", "environment_digest"),
    "evaluate_search": ("pllm.search", "evaluate_search"),
    "load_component_factory": ("pllm.providers", "load_component_factory"),
    "canonical_bytes": ("pllm.configuration", "canonical_bytes"),
    "compile": ("pllm.compiler", "compile"),
    "configuration_digest": ("pllm.configuration", "configuration_digest"),
    "create_app": ("pllm.runtime.server", "create_app"),
    "create_async_openai_client": ("pllm.runtime.official", "create_async_openai_client"),
    "create_linear_check_key": ("pllm.verification", "create_linear_check_key"),
    "create_openai_client": ("pllm.runtime.official", "create_openai_client"),
    "create_preparation_app": ("pllm.runtime.preparation_server", "create_preparation_app"),
    "create_sidecar_app": ("pllm.runtime.sidecar", "create_sidecar_app"),
    "load_configuration": ("pllm.configuration", "load_configuration"),
    "load_model": ("pllm.model_loader", "load_model"),
    "lower_model": ("pllm.modeling", "lower_model"),
    "loads_configuration": ("pllm.configuration", "loads_configuration"),
    "secure_argmax": ("pllm.runtime.secure_selection", "secure_argmax"),
    "serve_local": ("pllm.runtime.servers", "serve_local"),
}

__all__ = [*sorted(_EXPORTS), "__version__"]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

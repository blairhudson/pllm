"""Private LLM Inference: client API and native execution primitives."""
from __future__ import annotations
from importlib import import_module
from ._version import __version__

_EXPORTS = {'AsyncOpenAI': ('pllm.runtime.client', 'AsyncOpenAI'),
 'AuthenticatedMPC': ('pllm.runtime.authenticated_mpc', 'AuthenticatedMPC'),
 'AuthenticationError': ('pllm.runtime.authenticated_mpc', 'AuthenticationError'),
 'BlindedTransformerEngine': ('pllm.runtime.blinded_engine', 'BlindedTransformerEngine'),
 'ClientBundle': ('pllm.runtime.transformer_client', 'ClientBundle'),
 'DirectFHETransformerEngine': ('pllm.runtime.proprietary_engine',
                                'DirectFHETransformerEngine'),
 'GatewayConfig': ('pllm.runtime.config', 'GatewayConfig'),
 'GuardPolicy': ('pllm.runtime.guarded_engine', 'GuardPolicy'),
 'GuardedBlindedTransformerEngine': ('pllm.runtime.guarded_engine',
                                     'GuardedBlindedTransformerEngine'),
 'HEAsyncTransport': ('pllm.runtime.transport', 'HEAsyncTransport'),
 'HEAuthenticatedPreprocessor': ('pllm.runtime.he_authenticated_preprocessing',
                                 'HEAuthenticatedPreprocessor'),
 'HETransport': ('pllm.runtime.transport', 'HETransport'),
 'MaskedTransformerClientRuntime': ('pllm.runtime.transformer_client',
                                    'MaskedTransformerClientRuntime'),
 'MaskedTransformerEngine': ('pllm.runtime.transformer_engine', 'MaskedTransformerEngine'),
 'OpenAI': ('pllm.runtime.client', 'OpenAI'),
 'PROFILES': ('pllm.runtime.formal_security', 'PROFILES'),
 'PreparedInventory': ('pllm.runtime.preprocessing_inventory', 'PreparedInventory'),
 'PreprocessingPlan': ('pllm.runtime.preprocessing_inventory', 'PreprocessingPlan'),
 'PrivacyMode': ('pllm.runtime.privacy', 'PrivacyMode'),
 'ProprietaryProtocol': ('pllm.runtime.privacy', 'ProprietaryProtocol'),
 'RecordingPreprocessor': ('pllm.runtime.preprocessing_inventory', 'RecordingPreprocessor'),
 'Response': ('pllm.runtime.types', 'Response'),
 'ResponseEvent': ('pllm.runtime.types', 'ResponseEvent'),
 'ResponseStream': ('pllm.runtime.client', 'ResponseStream'),
 'SECURE_PREVIEW': ('pllm.runtime.formal_security', 'SECURE_PREVIEW'),
 'SecureDecoder': ('pllm.runtime.secure_transformer', 'SecureDecoder'),
 'SecureDecoderConfig': ('pllm.runtime.secure_transformer', 'SecureDecoderConfig'),
 'SecureDecoderWeights': ('pllm.runtime.secure_transformer', 'SecureDecoderWeights'),
 'SecurityClaim': ('pllm.runtime.formal_security', 'SecurityClaim'),
 'TrustedPreprocessor': ('pllm.runtime.authenticated_mpc', 'TrustedPreprocessor'),
 'check_linear_result': ('pllm.runtime.linear_integrity', 'check_linear_result'),
 'create_app': ('pllm.runtime.server', 'create_app'),
 'create_async_openai_client': ('pllm.runtime.official', 'create_async_openai_client'),
 'create_linear_check_key': ('pllm.runtime.linear_integrity', 'create_linear_check_key'),
 'create_openai_client': ('pllm.runtime.official', 'create_openai_client'),
 'create_sidecar_app': ('pllm.runtime.sidecar', 'create_sidecar_app'),
 'secure_argmax': ('pllm.runtime.secure_selection', 'secure_argmax')}

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

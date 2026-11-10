"""Dependency-leaf implementation codes and transient client bridge estimate.

Codes mirror compiler/component_binding.rs. They identify implementations, not
matrix indices, permissions, claims, or completion. Python frames remain caller
owned; this estimate is not a physical RSS bound or native arena reservation.
"""

# 16 KiB fixed frames/bridge stack + 8 KiB borrowed ASCII/scalars + 24 KiB
# fixed error/traceback + 16 KiB overlap/rounding. Add before allocation margin.
CLIENT_BRIDGE_ESTIMATE_BYTES = 65_536
LOOKUP = 1
LINEAR = 4
OUTPUT_HEAD = 15


def checked_operator(code, operation):
    """Fixed adapters check exact SemanticIR operator before existing arithmetic."""
    if type(code) is not int:
        raise ValueError("invalid bound decoder step")
    match code:
        case 1: expected = "token_lookup"
        case 2: expected = "reshape"
        case 3: expected = "rms_norm"
        case 4: expected = "linear"
        case 5: expected = "rotary_embedding"
        case 6: expected = "kv_cache_append"
        case 7: expected = "attention_scores"
        case 8: expected = "attention_scale"
        case 9: expected = "causal_mask"
        case 10: expected = "softmax"
        case 11: expected = "attention_values"
        case 12: expected = "residual_add"
        case 13: expected = "silu"
        case 14: expected = "multiply"
        case 15: expected = "output_head"
        case 16: expected = "last_token"
        case 17: expected = "greedy_token_selection"
        case 18: expected = "token_feedback"
        case 19: expected = "cache_suffix"
        case _: raise ValueError("invalid bound decoder step")
    if operation["operator"] != expected:
        raise ValueError("invalid bound decoder step")
    return code


def has_decoder_bridge(contract):
    """Recognize existing native-validated decoder structure, not caller capacity.

    Read-only constant-size shape check. Never supplies execution/admission
    authority: callers already obtained this contract from native admission.
    """
    return (type(contract) is dict
            and contract.get("schema_version") == "pllm.gkm.decoder-reference.v1"
            and contract.get("numeric_profile") ==
                "gkm.nearest-even.2048.i16.ring64.signed.f32-before-div22.le-finite-f32.sha256.secp256k1.v1"
            and contract.get("local_numeric_profile") ==
                "semantic.float32.dense-full-kv.client-bias-after-div22.v1"
            and contract.get("acceptance") == "fixed-length-greedy.complete-window.v1"
            and contract.get("input_bound") == 1 << 31
            and type(contract.get("plan")) is dict
            and type(contract["plan"].get("prefill")) is dict
            and type(contract["plan"].get("decode")) is dict
            and type(contract.get("prefill")) is dict
            and type(contract.get("decode")) is dict
            and type(contract.get("matrices")) is list
            and bool(contract["matrices"])
            and type(contract.get("source_shapes")) is dict)

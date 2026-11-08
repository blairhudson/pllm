"""Shared descriptive admission for the explicit, uncompiled flat legacy route.

This is not native lowering, checkpoint authentication, or a new execution graph.
The source snapshot and census are consistency commitments within a bundle.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass

from .bundle_document import bounded_metadata, MAX_OBJECTS, MAX_RAW_BYTES
from .models import gemma4_stage_plan

CONTRACT_FIELD = "legacy_source_contract"
PRESENCE_REQUIRED = "legacy source presence contract required; rebuild from original checkpoint"
_ARCHITECTURE = "Gemma4ForCausalLM"


def _identity_marker(source):
    if type(source) is not dict:
        return False
    architectures = source.get("architectures")
    model_type = source.get("model_type")
    return (type(model_type) is str and model_type == "gemma4_text"
            or type(architectures) is list and len(architectures) > 0
            and type(architectures[0]) is str and architectures[0] == _ARCHITECTURE)


def legacy_source_candidate(value):
    """Fixed-key, exact-builtin inspection only; not eligibility or authority."""
    if type(value) is not dict:
        return False
    config = value.get("config")
    if type(config) is not dict:
        config = value.get("runtime")
    if type(config) is dict:
        if CONTRACT_FIELD in config:
            return True
        # The preserved outer source, not its flattened runtime text view, owns
        # routing. Canonical Gemma 4 sources contain gemma4_text text configs.
        if "semantic_source_config" in config:
            return _legacy_envelope_marker(config["semantic_source_config"])
        if _legacy_envelope_marker(config):
            return True
        if type(config.get("model_type")) is str and config["model_type"] == "gemma4":
            return False
    manifest = value.get("manifest")
    metadata = manifest.get("metadata") if type(manifest) is dict else None
    return (type(manifest) is dict and type(manifest.get("architecture")) is str
                and manifest["architecture"] == _ARCHITECTURE
            or type(metadata) is dict and (_legacy_envelope_marker(metadata.get("text_config"))
                or type(metadata.get("model_type")) is str and metadata["model_type"] == "gemma4_text"))


def _legacy_envelope_marker(source):
    if type(source) is not dict:
        return False
    if (type(source.get("model_type")) is str and source["model_type"] == "gemma4"
            and type(source.get("text_config")) is dict):
        return False
    return _identity_marker(source) or _identity_marker(source.get("text_config"))


def _integer(source, key, default=None, *, zero=False):
    value = source.get(key, default)
    if type(value) is not int or not (0 if zero else 1) <= value <= MAX_RAW_BYTES:
        raise ValueError(f"legacy source field {key!r} requires a bounded integer")
    return value


def _product(*values):
    size = math.prod(values)
    if size > min(MAX_RAW_BYTES, sys.maxsize):
        raise ValueError("legacy source geometry exceeds raw/addressable bound")
    return size


@dataclass(frozen=True)
class LegacyGeometry:
    source: dict
    config: dict
    dimensions: dict
    layer_types: tuple[str, ...]
    head_dims: tuple[int, ...]
    kv_heads: tuple[int, ...]
    first_shared: int
    ple: int


def legacy_source_geometry(source):
    """Classify before lowering; matching malformed/contradictory sources raise."""
    if type(source) is not dict:
        return None
    if not _legacy_envelope_marker(source):
        return None
    bounded_metadata(source, json_mode=True)
    if (source.get("model_type") != "gemma4_text"
            or source.get("architectures") != [_ARCHITECTURE]
            or "text_config" in source):
        raise ValueError("legacy source requires flat gemma4_text / Gemma4ForCausalLM identity")
    if CONTRACT_FIELD in source or "semantic_source_config" in source:
        raise ValueError("source cannot supply reserved captured source contract")
    layers = _integer(source, "num_hidden_layers")
    # Both conservative bounds precede any range/default-list/stage expansion.
    if 8 + 20 * layers > MAX_OBJECTS or 21 + 44 * layers > MAX_OBJECTS:
        raise ValueError("legacy source aggregate object pre-bound exceeded")
    hidden = _integer(source, "hidden_size")
    heads = _integer(source, "num_attention_heads")
    if hidden % heads:
        raise ValueError("legacy hidden size must divide attention heads")
    intermediate = _integer(source, "intermediate_size", hidden * 4)
    kv = _integer(source, "num_key_value_heads", heads)
    if heads % kv:
        raise ValueError("legacy attention heads must divide KV heads")
    dim = _integer(source, "head_dim", hidden // heads)
    vocab = _integer(source, "vocab_size")
    context = _integer(source, "max_position_embeddings", 2048)
    for alias, expected in (("d_model", hidden), ("n_embd", hidden),
            ("ffn_dim", intermediate), ("n_inner", intermediate),
            ("n_layer", layers), ("num_layers", layers), ("n_head", heads),
            ("num_heads", heads), ("n_kv_head", kv),
            ("max_sequence_length", context), ("seq_length", context), ("n_positions", context)):
        if alias in source and _integer(source, alias) != expected:
            raise ValueError("legacy source geometry aliases disagree")
    ple = _integer(source, "hidden_size_per_layer_input", 0, zero=True)
    shared = _integer(source, "num_kv_shared_layers", 0, zero=True)
    if shared >= layers:
        raise ValueError("legacy shared KV requires an earlier producer")
    for key in ("tie_word_embeddings", "attention_k_eq_v", "use_double_wide_mlp", "enable_moe_block"):
        if key in source and type(source[key]) is not bool:
            raise ValueError(f"legacy source field {key!r} requires a boolean")
    if source.get("enable_moe_block") or source.get("num_experts", 0):
        raise ValueError("legacy source does not support sparse MoE")
    if "num_experts" in source:
        _integer(source, "num_experts", zero=True)
    if "sliding_window" in source and source["sliding_window"] is not None:
        _integer(source, "sliding_window")
    eps = source.get("rms_norm_eps", 1e-6)
    if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("legacy RMS epsilon must be positive and finite")
    if "vocab_size_per_layer_input" in source and _integer(source, "vocab_size_per_layer_input") != vocab:
        raise ValueError("legacy PLE vocabulary must match token vocabulary")
    types = source.get("layer_types", ["full_attention"] * layers)
    if (type(types) is not list or len(types) != layers
            or any(type(item) is not str or item not in {"full_attention", "sliding_attention"} for item in types)):
        raise ValueError("legacy layer_types must match source layers")
    per_layer = source.get("per_layer_config", {})
    if type(per_layer) is not dict:
        raise ValueError("legacy per_layer_config must be a dictionary")
    index_keys = {str(i) for i in range(layers)}
    for key, row in per_layer.items():
        if (type(key) is not str or key not in types and key not in index_keys
                or type(row) is not dict or any(name not in {"head_dim", "num_key_value_heads"} for name in row)):
            raise ValueError("invalid legacy per-layer override")
        for name in row:
            _integer(row, name)
    dims, kvs = [], []
    producers = {}
    first_shared = layers - shared
    for index in range(layers):
        def layer_value(name, default):
            for candidate in (str(index), types[index]):
                row = per_layer.get(candidate, {})
                if name in row:
                    return row[name]
            return default
        d, k = layer_value("head_dim", dim), layer_value("num_key_value_heads", kv)
        if heads % k:
            raise ValueError("legacy attention heads must divide KV heads")
        if index >= first_shared:
            if types[index] not in producers or producers[types[index]] != (d, k):
                raise ValueError("legacy shared KV producer geometry unavailable")
        else:
            producers[types[index]] = (d, k)
        _product(hidden, (heads + 2 * k) * d)
        _product(4, d)
        dims.append(d)
        kvs.append(k)
    _product(hidden, intermediate, 4 if source.get("use_double_wide_mlp") else 2)
    _product(vocab, hidden + layers * ple)
    _product(hidden, layers * ple)
    _product(hidden, 4)
    _product(ple, 4)
    _product(context, heads, max(dims))
    dimensions = dict(hidden_size=hidden, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv,
        head_dim=dim, vocab_size=vocab, max_position_embeddings=context)
    config = {**source, **dimensions}
    geometry = LegacyGeometry(source, config, dimensions, tuple(types), tuple(dims),
                              tuple(kvs), first_shared, ple)
    _validate_profile_fields(source, geometry, source_only=True)
    return geometry


def legacy_local_requirements(geometry):
    """Permitted source-optional norms, including unused shared-layer K."""
    h = geometry.dimensions["hidden_size"]
    result = {"model.norm.weight": (h,)}
    if geometry.ple:
        result["model.per_layer_projection_norm.weight"] = (geometry.ple,)
    for index, dim in enumerate(geometry.head_dims):
        prefix = f"model.layers.{index}."
        for name in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            result[prefix + name + ".weight"] = (h,)
        for name in ("q_norm", "k_norm"):
            result[prefix + "self_attn." + name + ".weight"] = (dim,)
        if geometry.ple:
            result[prefix + "post_per_layer_input_norm.weight"] = (h,)
    return result


def capture_legacy_presence(geometry, source_inventory):
    """Consume genuine header identities/shapes, never payload rows or suffix lookup."""
    allowed = legacy_local_requirements(geometry)
    present = []
    for key, shape in source_inventory.items():
        if type(key) is not str:
            raise ValueError("invalid legacy source tensor identity")
        if key in allowed:
            if (type(shape) not in (tuple, list) or any(type(d) is not int for d in shape)
                    or tuple(shape) != allowed[key]):
                raise ValueError(f"legacy source norm {key!r} differs from source geometry")
            present.append(key)
        elif "norm" in key.lower():
            raise ValueError(f"unknown or noncanonical legacy source norm {key!r}")
    present.sort()
    if any(left >= right for left, right in zip(present, present[1:])):
        raise ValueError("legacy source presence must be sorted unique permitted names")
    return present


def legacy_record_counts(stages, weights, locals_count, *, tokenizer=True):
    if (stages + weights + locals_count > MAX_OBJECTS
            or 4 * stages + 2 * weights + locals_count + int(tokenizer) > MAX_OBJECTS):
        raise ValueError("legacy bundle aggregate object bound exceeded")


def legacy_bundle_record_counts(manifest, rows, *, stages, weights):
    """Actual records precede JSON hashing and source-derived expansion."""
    if any(type(value) is not dict for value in (rows, stages, weights)):
        raise ValueError("invalid legacy bundle record containers")
    legacy_record_counts(len(stages), len(weights), len(rows))
    if type(manifest) is not dict or type(manifest.get("stages")) is not list:
        raise ValueError("legacy manifest stages must be a list")
    legacy_record_counts(len(manifest["stages"]), 0, 0)


def _profile(geometry):
    source = geometry.source
    return dict(model_family="gemma4", block_style="gemma4", norm_offset=0.0,
        embedding_multiplier=math.sqrt(geometry.dimensions["hidden_size"]),
        attention_scaling=1.0, rms_norm_centered=False, qk_norm=True, v_norm=True,
        hidden_activation=source.get("hidden_activation") or source.get("hidden_act") or "silu")


def _validate_profile_fields(config, geometry, *, source_only=False):
    for key, expected in _profile(geometry).items():
        matching_type = (type(config.get(key)) in (int, float) if type(expected) is float
                         else type(config.get(key)) is type(expected))
        if key in config and (not matching_type or config[key] != expected):
            raise ValueError(f"legacy runtime field {key!r} disagrees with source profile")
        if not source_only and key not in config:
            raise ValueError(f"legacy runtime field {key!r} is required")


def validate_legacy_runtime(config, geometry):
    _validate_profile_fields(config, geometry)
    additions = {CONTRACT_FIELD, *_profile(geometry), *geometry.dimensions,
                 "prime_modulus", "plain_moduli", "bos_token_id", "bos_token_policy", "token_lookup_batch"}
    if any(key not in geometry.source and key not in additions for key in config):
        raise ValueError("legacy runtime has unknown source-independent options")
    for key, expected in geometry.source.items():
        if key == "bos_token_id" and expected is None and config.get("bos_token_policy") == "nonempty_only":
            actual = config.get(key)
            if type(actual) is not int or not 0 <= actual < geometry.dimensions["vocab_size"]:
                raise ValueError("legacy runtime BOS fallback differs")
        elif key in _profile(geometry):
            # The existing exporter normalizes profile numerics to floats.
            # Profile checks above still fix their exact numerical semantics.
            if config.get(key) != expected:
                raise ValueError(f"legacy runtime field {key!r} disagrees with original source")
        elif key not in config or type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f"legacy runtime field {key!r} disagrees with original source")
    for key, expected in geometry.dimensions.items():
        if type(config.get(key)) is not int or config[key] != expected:
            raise ValueError(f"legacy runtime geometry {key!r} disagrees with source")
    if geometry.source.get("bos_token_id") is None:
        bos = config["bos_token_id"]
        if (config.get("bos_token_policy") != "nonempty_only" or type(bos) is not int
                or not 0 <= bos < geometry.dimensions["vocab_size"]):
            raise ValueError("invalid legacy runtime BOS fallback")
    elif "bos_token_policy" in config and "bos_token_policy" not in geometry.source:
        raise ValueError("legacy runtime BOS policy is not source-owned")
    if "token_lookup_batch" in config:
        _integer(config, "token_lookup_batch")


def validate_legacy_manifest(geometry, manifest):
    if type(manifest) is not dict:
        raise ValueError("legacy manifest must be a dictionary")
    if manifest.get("architecture") != _ARCHITECTURE:
        raise ValueError("legacy manifest identity disagrees with source")
    for key, expected in geometry.dimensions.items():
        field = "context_length" if key == "max_position_embeddings" else key
        if type(manifest.get(field)) is not int or manifest[field] != expected:
            raise ValueError(f"legacy manifest geometry {field!r} disagrees with source")
    if (type(manifest.get("tied_embeddings")) is not bool
            or manifest["tied_embeddings"] != geometry.source.get("tie_word_embeddings", False)):
        raise ValueError("legacy manifest tied embeddings disagree with source")
    metadata = manifest.get("metadata")
    if type(metadata) is not dict or metadata.get("model_type") != "gemma4_text":
        raise ValueError("legacy manifest source identity differs")
    if metadata.get("architectures") != [_ARCHITECTURE] or metadata.get("text_config") != geometry.source:
        raise ValueError("legacy manifest original source differs")


def validate_legacy_stages(geometry, manifest_stages, rows=None):
    # Reuse the sole existing builder. The engine's existing lookup fusion is
    # described here only by its geometry; no alternate execution builder exists.
    specs = gemma4_stage_plan(geometry.config, include_lm_head=True)
    expected = {spec.id: spec for spec in specs if spec.id != "embed_tokens_per_layer"}
    if type(manifest_stages) is not list or len(manifest_stages) > MAX_OBJECTS:
        raise ValueError("legacy manifest stages must be a list")
    actual = {}
    for row in manifest_stages:
        if type(row) is not dict or type(row.get("id")) is not str or row["id"] in actual:
            raise ValueError("legacy manifest stage identities differ")
        actual[row["id"]] = row
    if rows is not None:
        expected_ids = {*expected, "token_lookup"}
        if type(rows) is not dict or set(rows) != expected_ids or set(actual) != expected_ids:
            raise ValueError("legacy bundle stage inventory differs from source")
    elif set(actual) != set(expected) | ({"embed_tokens_per_layer"} if geometry.ple else set()):
        raise ValueError("legacy source stage inventory differs")
    checks = [(spec.id, spec.op, spec.in_features, spec.out_features, spec.role, spec.layer_index)
              for spec in specs if rows is None or spec.id != "embed_tokens_per_layer"]
    if rows is not None:
        checks.append(("token_lookup", "embedding", geometry.dimensions["vocab_size"],
            geometry.dimensions["hidden_size"] + len(geometry.head_dims) * geometry.ple, "token_lookup", None))
    for sid, op, width, out, role, layer in checks:
        for row in (actual[sid],) if rows is None else (actual[sid], rows[sid]):
            if (type(row) is not dict or row.get("id") != sid or row.get("op") != op
                    or type(row.get("in_features")) is not int or row["in_features"] != width
                    or type(row.get("out_features")) is not int or row["out_features"] != out
                    or row.get("role") != role or row.get("layer_index") != layer
                    or layer is not None and type(row.get("layer_index")) is not int):
                raise ValueError("legacy stage geometry disagrees with source")


def legacy_contract_geometry(config, manifest):
    if type(config) is not dict or type(manifest) is not dict:
        raise ValueError("legacy config/manifest must be dictionaries")
    contract = config.get(CONTRACT_FIELD)
    if contract is None:
        raise ValueError(PRESENCE_REQUIRED)
    if (type(contract) is not dict or set(contract) != {"v", "source", "present_local_tensors"}
            or type(contract.get("v")) is not int or contract["v"] != 1):
        raise ValueError("invalid legacy source presence contract")
    # No serialization, sorting, or decimal conversion precedes bounded JSON.
    bounded_metadata(config, json_mode=True)
    metadata = manifest.get("metadata")
    expected = metadata.get("runtime_config_digest") if type(metadata) is dict else None
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    if type(expected) is not str or expected != digest:
        raise ValueError("legacy runtime config digest mismatch")
    geometry = legacy_source_geometry(contract["source"])
    if geometry is None:
        raise ValueError("legacy source contract identity is not eligible")
    validate_legacy_runtime(config, geometry)
    validate_legacy_manifest(geometry, manifest)
    if metadata.get("stage_origin") != "legacy_stage_plan_v1":
        raise ValueError("legacy source cannot claim native semantic stages")
    if (metadata.get("decoder_execution") not in {"legacy_runtime_graph_v1", "legacy_verified_runtime_v1"}
            or metadata.get("model_family") != "gemma4" or metadata.get("block_style") != "gemma4"):
        raise ValueError("legacy runtime metadata disagrees with source profile")
    return geometry


def legacy_presence_requirements(config, geometry, rows):
    """Validate the committed census before sets; usable for artifact metadata."""
    if type(rows) is not dict:
        raise ValueError("invalid legacy local tensor inventory")
    allowed = legacy_local_requirements(geometry)
    presence = config[CONTRACT_FIELD]["present_local_tensors"]
    if type(presence) is not list or len(presence) > len(allowed):
        raise ValueError("invalid legacy source presence census")
    previous = None
    for key in presence:
        if type(key) is not str or previous is not None and key <= previous or key not in allowed:
            raise ValueError("legacy source presence must be sorted unique permitted names")
        previous = key
    if any(type(key) is not str for key in rows) or set(rows) != set(presence):
        raise ValueError("legacy local tensor inventory differs from source presence")
    return {key: allowed[key] for key in presence}


def validate_legacy_bundle_contract(config, manifest, rows, *, stages, weights):
    legacy_bundle_record_counts(manifest, rows, stages=stages, weights=weights)
    geometry = legacy_contract_geometry(config, manifest)
    validate_legacy_stages(geometry, manifest.get("stages"), stages)
    required = legacy_presence_requirements(config, geometry, rows)
    for key, shape in required.items():
        row = rows[key]
        if (type(row) is not dict or set(row) != {"dtype", "shape", "data"}
                or row.get("dtype") != "f4" or type(row.get("shape")) not in (list, tuple)
                or any(type(d) is not int for d in row["shape"])
                or tuple(row["shape"]) != shape
                or type(row.get("data")) is not bytes
                or len(row["data"]) != _product(*shape, 4)):
            raise ValueError(f"legacy local tensor {key!r} differs from source geometry/storage")
    return geometry

"""Independent, test-only NumPy FP32 forward oracle for a bounded BERT subset.

Public API: ``evaluate(config, tensors, input_ids, *, token_type_ids=None,
position_ids=None, attention_mask=None) -> dict[str, np.ndarray]``. ``config``
is the original source configuration (a mapping or an attribute-bearing object),
not an execution plan. ``tensors`` contains original Hugging Face state-dict
names and raw, finite float32 NumPy arrays; weights use [out, in] layout.
No production kernel, plan, imported stage value, cache, fixture generator, or
actual runtime output participates in this calculation.

Scope: explicitly declared BertModel or BertForSequenceClassification, absolute
positions, exact GELU, encoder-only post-norm attention, evaluation mode, batch
one, and 1 <= sequence length <= max_position_embeddings. Dimensions and all
parameter cardinalities come from source config, never output vocabulary or a
proposed test cohort. A possible *test cohort*, not a production constraint, is
H=8, heads=2, FFN=13, layers=2, vocab=17, positions=8, segments=2,
labels=3, sequence lengths 1..6. All-zero key masks are explicitly outside this
bounded subset; rejecting them is not a claim of Hugging Face parity.

Historical formula references: Transformers commit
47c29ccfaf56947d845971a439cbe75a764b63d7 / 4.43.3 and PyTorch 2.2.1.
These are historical math pins, NOT an executed reference stack. Repository lock
versions Torch 2.14.0 and Transformers 5.16.1 are not assertions about installed
packages. This module requires only Python's standard library and NumPy, not
Torch, Transformers, or SciPy. No runtime integration test exists here: runtime
API is pending. These captures carry no production-authority claim.

Capture keys (all float32; metadata is documented here, not returned):
  embeddings.word, embeddings.token_type, embeddings.position: [1, S, H]
  embeddings.word_plus_token_type, embeddings.sum, embeddings.layer_norm:
      [1, S, H]; addition is (word + token_type) + position.
  attention.additive_mask: [1, 1, 1, S]
  encoder.layer.{i}.attention.{query,key,value}: biased projections [1, S, H]
  encoder.layer.{i}.attention.{query,key,value}_heads: [1, heads, S, D]
  encoder.layer.{i}.attention.{scaled_scores,masked_scores,probabilities}:
      [1, heads, S, S]
  encoder.layer.{i}.attention.context_heads: [1, heads, S, D]
  encoder.layer.{i}.attention.{context,projection,residual,layer_norm}:
      [1, S, H]
  encoder.layer.{i}.ffn.{pre_activation,post_activation}: [1, S, FFN]
  encoder.layer.{i}.ffn.{projection,residual,layer_norm}: [1, S, H]
  last_hidden_state: [1, S, H]
  pooler.pre_activation, pooler_output: [1, H], only if pooler exists
  logits: [1, labels], only for the declared classification architecture

Each returned capture owns a copy. Base-model keys are unprefixed; classification
encoder/pooler keys start with 'bert.' and classifier keys do not. Base-model
pooler is optional (both weight and bias or neither). Classification requires
pooler and classifier. Unknown/missing state keys are rejected, except original
optional embeddings.position_ids/token_type_ids buffers, which are validated
but not used. No head is inferred from tensor dimensions.
Classification label count comes from source num_labels or its serialized
id2label mapping (standard HF JSON may omit the num_labels property). If both
are declared, their cardinalities must agree.

Numerics: separate biased linears, centered population-variance LayerNorm with
source epsilon, noncausal QK^T / float32(sqrt(D)), binary key-only masking by
(1 - mask) * finite float32 minimum, and stable float32 softmax. Neither causal
masking, -infinity, query zeroing, nor padding-token inference is used. Softmax
subtracts float32 max, evaluates float32 exp, sums in float32, then divides in
float32. Exact GELU uses math.erf on a float32 x * float32(1/sqrt(2)) argument,
rounds erf to float32, then computes (x * float32(0.5)) * (1 + erf) in float32.
Pooling is first-token affine then float32 NumPy tanh; dropout is disabled.
FP32 operation order is explicit but does not promise bitwise vendor-kernel
equivalence. Comparison tolerances below are fixed contracts, not tunable knobs;
future failures must not be resolved by widening them.
"""

from collections.abc import Mapping
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


ATOL = 1e-5
RTOL = 1e-4
ATTENTION_PROB_ATOL = 2e-6
ATTENTION_PROB_RTOL = 2e-5

_MISSING = object()
_F32 = np.float32


def _field(config: Any, name: str, default: Any = _MISSING) -> Any:
    value = config.get(name, default) if isinstance(config, Mapping) else getattr(
        config, name, default
    )
    if value is _MISSING:
        raise ValueError(f"Source config must declare {name}")
    return value


def _positive_int(config: Any, name: str) -> int:
    value = _field(config, name)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"Source config {name} must be a positive integer")
    return int(value)


def _label_count(config: Any) -> int:
    declared = _field(config, "num_labels", None)
    id2label = _field(config, "id2label", None)
    mapped_count = None
    if id2label is not None:
        if not isinstance(id2label, Mapping) or not id2label:
            raise ValueError("Source id2label must be a nonempty mapping")
        mapped_count = len(id2label)
        # JSON encodes integer label IDs as strings; do not accept arbitrary
        # label names, duplicate integer/string IDs, or noncontiguous IDs.
        keys = set(id2label)
        integer_keys = set(range(mapped_count))
        string_keys = {str(index) for index in range(mapped_count)}
        if any(isinstance(key, bool) for key in keys) or not (
            keys == integer_keys or keys == string_keys
        ):
            raise ValueError("Source id2label IDs must be contiguous integers or JSON integer strings")
    if declared is None:
        if mapped_count is None:
            raise ValueError("Classification source must declare num_labels or id2label")
        return mapped_count
    count = _positive_int(config, "num_labels")
    if mapped_count is not None and mapped_count != count:
        raise ValueError("Source num_labels and id2label cardinalities disagree")
    return count


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # Separate matmul and bias addition: no fused or production affine path.
    return np.add(np.matmul(x, weight.T), bias, dtype=np.float32)


def _layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: np.float32
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = np.subtract(x, mean, dtype=np.float32)
    variance = np.mean(
        np.multiply(centered, centered, dtype=np.float32),
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )
    denominator = np.sqrt(np.add(variance, eps, dtype=np.float32))
    normalized = np.divide(centered, denominator, dtype=np.float32)
    return np.add(
        np.multiply(normalized, weight, dtype=np.float32), bias, dtype=np.float32
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = np.subtract(scores, np.max(scores, axis=-1, keepdims=True), dtype=np.float32)
    numerator = np.exp(shifted, dtype=np.float32)
    denominator = np.sum(numerator, axis=-1, keepdims=True, dtype=np.float32)
    return np.divide(numerator, denominator, dtype=np.float32)


def _gelu(x: np.ndarray) -> np.ndarray:
    argument = np.multiply(x, _F32(1.0 / math.sqrt(2.0)), dtype=np.float32)
    erf = np.fromiter(
        (math.erf(float(value)) for value in argument.flat),
        dtype=np.float32,
        count=argument.size,
    ).reshape(argument.shape)
    return np.multiply(
        np.multiply(x, _F32(0.5), dtype=np.float32),
        np.add(_F32(1.0), erf, dtype=np.float32),
        dtype=np.float32,
    )


def _indices(value: Any, name: str, shape: tuple[int, int], bound: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array of shape {shape}")
    if np.any(array < 0) or np.any(array >= bound):
        raise ValueError(f"{name} entries must lie in [0, {bound})")
    return array.astype(np.int64, copy=False)


def evaluate(
    config: Any,
    tensors: Mapping[str, np.ndarray],
    input_ids: Any,
    *,
    token_type_ids: Any = None,
    position_ids: Any = None,
    attention_mask: Any = None,
) -> dict[str, np.ndarray]:
    """Calculate source BERT FP32 outputs and test-only intermediate captures.

    Inputs must have shape [1, S], including explicitly supplied positions and
    masks; broadcasting and extended/additive masks are intentionally unsupported.
    Defaults: zero token types, range(S) positions, all-one attention mask.
    Unsupported architecture, malformed source, or invalid input raises ValueError.
    No supplied config, array, or mapping is mutated.
    """
    if _field(config, "model_type") != "bert":
        raise ValueError("Only explicitly declared source model_type='bert' is supported")
    architectures = _field(config, "architectures")
    if not isinstance(architectures, (list, tuple)) or len(architectures) != 1:
        raise ValueError("Source config must declare exactly one BERT architecture")
    architecture = architectures[0]
    if architecture not in ("BertModel", "BertForSequenceClassification"):
        raise ValueError("Only BertModel and BertForSequenceClassification are supported")
    classification = architecture == "BertForSequenceClassification"
    prefix = "bert." if classification else ""
    if _field(config, "hidden_act") != "gelu":
        raise ValueError("Only exact hidden_act='gelu' is supported")
    if _field(config, "position_embedding_type", "absolute") != "absolute":
        raise ValueError("Only absolute position embeddings are supported")
    if _field(config, "is_decoder", False) or _field(config, "add_cross_attention", False):
        raise ValueError("Decoder and cross-attention configurations are unsupported")
    hidden = _positive_int(config, "hidden_size")
    heads = _positive_int(config, "num_attention_heads")
    intermediate = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    vocabulary = _positive_int(config, "vocab_size")
    positions = _positive_int(config, "max_position_embeddings")
    segments = _positive_int(config, "type_vocab_size")
    if hidden % heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if _field(config, "pruned_heads", {}):
        raise ValueError("Pruned attention heads are outside the supported subset")
    eps_source = _field(config, "layer_norm_eps")
    if (
        isinstance(eps_source, bool)
        or not isinstance(eps_source, Real)
        or not math.isfinite(eps_source)
        or not 0 < eps_source <= np.finfo(np.float32).max
    ):
        raise ValueError("layer_norm_eps must be positive and finite in float32")
    eps = _F32(eps_source)
    if eps == 0:
        raise ValueError("layer_norm_eps must not underflow float32")
    if not isinstance(tensors, Mapping) or any(not isinstance(key, str) for key in tensors):
        raise ValueError("tensors must map original source names to float32 NumPy arrays")

    expected: dict[str, tuple[int, ...]] = {}

    def parameter(name: str, shape: tuple[int, ...]) -> None:
        expected[prefix + name] = shape

    def affine(name: str, output_size: int, input_size: int) -> None:
        parameter(name + ".weight", (output_size, input_size))
        parameter(name + ".bias", (output_size,))

    def norm(name: str) -> None:
        parameter(name + ".weight", (hidden,))
        parameter(name + ".bias", (hidden,))

    parameter("embeddings.word_embeddings.weight", (vocabulary, hidden))
    parameter("embeddings.position_embeddings.weight", (positions, hidden))
    parameter("embeddings.token_type_embeddings.weight", (segments, hidden))
    norm("embeddings.LayerNorm")
    for index in range(layers):
        layer = f"encoder.layer.{index}"
        for projection in ("query", "key", "value"):
            affine(f"{layer}.attention.self.{projection}", hidden, hidden)
        affine(f"{layer}.attention.output.dense", hidden, hidden)
        norm(f"{layer}.attention.output.LayerNorm")
        affine(f"{layer}.intermediate.dense", intermediate, hidden)
        affine(f"{layer}.output.dense", hidden, intermediate)
        norm(f"{layer}.output.LayerNorm")
    pooler_names = {prefix + "pooler.dense.weight", prefix + "pooler.dense.bias"}
    present_pooler = pooler_names.intersection(tensors)
    if present_pooler and present_pooler != pooler_names:
        raise ValueError("Source pooler must contain both weight and bias")
    has_pooler = present_pooler == pooler_names
    if classification and not has_pooler:
        raise ValueError("BertForSequenceClassification requires a source pooler")
    if has_pooler:
        affine("pooler.dense", hidden, hidden)
    if classification:
        labels = _label_count(config)
        expected["classifier.weight"] = (labels, hidden)
        expected["classifier.bias"] = (labels,)

    buffer_names = {
        prefix + "embeddings.position_ids",
        prefix + "embeddings.token_type_ids",
    }
    missing = set(expected).difference(tensors)
    unexpected = set(tensors).difference(expected).difference(buffer_names)
    if missing or unexpected:
        raise ValueError(
            f"Source parameter keys mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for name, shape in expected.items():
        value = tensors[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
            raise ValueError(f"{name} must be a raw float32 NumPy array")
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must have shape {shape} and finite values")
    for name in buffer_names.intersection(tensors):
        value = tensors[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype.kind not in "iu"
            or value.shape != (1, positions)
        ):
            raise ValueError(f"{name} must be an integer source buffer of shape {(1, positions)}")
        reference = np.arange(positions)[None, :] if name.endswith("position_ids") else np.zeros(
            (1, positions), dtype=np.int64
        )
        if not np.array_equal(value, reference):
            raise ValueError(f"{name} does not match the standard BERT source buffer")

    ids = np.asarray(input_ids)
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= positions:
        raise ValueError("input_ids must have shape [1, S], with 1 <= S <= source positions")
    shape = (1, ids.shape[1])
    ids = _indices(ids, "input_ids", shape, vocabulary)
    types = _indices(
        np.zeros(shape, dtype=np.int64) if token_type_ids is None else token_type_ids,
        "token_type_ids", shape, segments,
    )
    position = _indices(
        np.arange(shape[1], dtype=np.int64)[None, :] if position_ids is None else position_ids,
        "position_ids", shape, positions,
    )
    mask = np.ones(shape, dtype=np.float32) if attention_mask is None else np.asarray(attention_mask)
    if mask.shape != shape or mask.dtype.kind not in "biuf" or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask must be binary with shape [1, S]")
    if not np.any(mask):
        raise ValueError("All-zero key mask is outside the bounded oracle subset (not HF parity)")

    result: dict[str, np.ndarray] = {}

    def capture(name: str, value: np.ndarray) -> np.ndarray:
        if value.dtype != np.dtype(np.float32):
            raise AssertionError(f"Oracle stage {name} escaped float32")
        result[name] = value.copy()
        return value

    def tensor(name: str) -> np.ndarray:
        return tensors[prefix + name]

    def linear(x: np.ndarray, name: str) -> np.ndarray:
        return _linear(x, tensor(name + ".weight"), tensor(name + ".bias"))

    def layer_norm(x: np.ndarray, name: str) -> np.ndarray:
        return _layer_norm(x, tensor(name + ".weight"), tensor(name + ".bias"), eps)

    word = capture("embeddings.word", tensor("embeddings.word_embeddings.weight")[ids])
    token_type = capture("embeddings.token_type", tensor("embeddings.token_type_embeddings.weight")[types])
    position_value = capture("embeddings.position", tensor("embeddings.position_embeddings.weight")[position])
    embedding_pair = capture("embeddings.word_plus_token_type", np.add(word, token_type, dtype=np.float32))
    embedding_sum = capture("embeddings.sum", np.add(embedding_pair, position_value, dtype=np.float32))
    hidden_state = capture("embeddings.layer_norm", layer_norm(embedding_sum, "embeddings.LayerNorm"))
    additive_mask = capture(
        "attention.additive_mask",
        np.multiply(
            np.subtract(_F32(1.0), mask.astype(np.float32)[:, None, None, :], dtype=np.float32),
            _F32(np.finfo(np.float32).min),
            dtype=np.float32,
        ),
    )
    head_size = hidden // heads
    divisor = _F32(math.sqrt(head_size))
    for index in range(layers):
        layer = f"encoder.layer.{index}"
        attention = layer + ".attention"
        split = {}
        for name in ("query", "key", "value"):
            projected = capture(attention + "." + name, linear(hidden_state, attention + ".self." + name))
            split[name] = capture(
                attention + "." + name + "_heads",
                projected.reshape(1, shape[1], heads, head_size).transpose(0, 2, 1, 3),
            )
        scores = capture(
            attention + ".scaled_scores",
            np.divide(
                np.matmul(split["query"], split["key"].transpose(0, 1, 3, 2)),
                divisor, dtype=np.float32,
            ),
        )
        masked_scores = capture(attention + ".masked_scores", np.add(scores, additive_mask, dtype=np.float32))
        probabilities = capture(attention + ".probabilities", _softmax(masked_scores))
        context_heads = capture(attention + ".context_heads", np.matmul(probabilities, split["value"]))
        context = capture(
            attention + ".context",
            context_heads.transpose(0, 2, 1, 3).reshape(1, shape[1], hidden),
        )
        projected = capture(attention + ".projection", linear(context, attention + ".output.dense"))
        residual = capture(attention + ".residual", np.add(projected, hidden_state, dtype=np.float32))
        attended = capture(attention + ".layer_norm", layer_norm(residual, attention + ".output.LayerNorm"))
        pre_activation = capture(layer + ".ffn.pre_activation", linear(attended, layer + ".intermediate.dense"))
        post_activation = capture(layer + ".ffn.post_activation", _gelu(pre_activation))
        projected = capture(layer + ".ffn.projection", linear(post_activation, layer + ".output.dense"))
        residual = capture(layer + ".ffn.residual", np.add(projected, attended, dtype=np.float32))
        hidden_state = capture(layer + ".ffn.layer_norm", layer_norm(residual, layer + ".output.LayerNorm"))
    capture("last_hidden_state", hidden_state)
    if has_pooler:
        pooled_affine = capture("pooler.pre_activation", linear(hidden_state[:, 0, :], "pooler.dense"))
        pooled = capture("pooler_output", np.tanh(pooled_affine, dtype=np.float32))
        if classification:
            capture("logits", _linear(pooled, tensors["classifier.weight"], tensors["classifier.bias"]))
    return result

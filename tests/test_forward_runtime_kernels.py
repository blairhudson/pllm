"""Shared physical-layout/F32 kernels and genuine native workspace quotes."""
import math

import numpy as np
import pytest

from pllm.configuration import Deployment, Experiment, ForwardBudget, Model
from pllm.numeric import SourceFloat32
from pllm.profiles import ClientOnlyCpu
from pllm.runtime.bundle_storage import immutable_f32_array
from pllm.runtime.forward_binding import ForwardBindingError
from pllm.runtime.forward_inputs import ForwardInputs
from pllm.runtime.semantic_attention import (
    mask_key_padding_scores, semantic_attention_scores, semantic_attention_values,
)
from pllm.runtime.semantic_numeric import (
    source_float32_activation, source_float32_layer_norm, source_float32_lookup,
    source_float32_softmax,
)
from pllm.runtime.semantic_tensors import attention_layout
from pllm.sources.generated_encoder import write_checkpoint


# Positive functional setup: reuse genuine JSON tests' validated 128-MiB
# headroom, not a production cap or a change to any admission boundary.
FUNCTIONAL_SETUP_BYTES = 128 << 20


def immutable(value):
    return immutable_f32_array(value.tobytes(), value.shape)


def test_layout_roundtrip_preserves_batch_and_decoder_adapter():
    source = np.arange(48, dtype=np.float32).reshape(1, 3, 16)
    heads = attention_layout(source, input_layout="batch_sequence_hidden",
                             output_layout="batch_heads_sequence_feature", output_shape=(1, 4, 3, 4))
    assert heads.shape == (1, 4, 3, 4)
    np.testing.assert_array_equal(heads[0, 2, 1], source[0, 1, 8:12])
    for shape in ((1, 3, 16), (3, 16)):
        restored = attention_layout(heads, input_layout="batch_heads_sequence_feature",
                                    output_layout="batch_sequence_hidden", output_shape=shape)
        np.testing.assert_array_equal(restored.reshape(1, 3, 16), source)


def test_f32_lookup_preserves_physical_dimensions_and_immutable_source():
    table = immutable(np.arange(35, dtype=np.float32).reshape(7, 5))
    indices = np.array([[6, 1, 6]], dtype=np.int64)
    result = source_float32_lookup(indices, table)
    assert result.shape == (1, 3, 5) and result.dtype == np.float32
    np.testing.assert_array_equal(result[0, 0], table[6])
    with pytest.raises(ValueError):
        source_float32_lookup(np.array([[7]], dtype=np.int64), table)


def test_centered_population_layer_norm_order():
    source = np.array([[[1001, 1002, 1004], [-0.5, 1.25, 0.125]]], dtype=np.float32)
    gamma = immutable(np.array([0.5, -1, 2], dtype=np.float32))
    beta = immutable(np.array([0.25, 1, -0.5], dtype=np.float32))
    epsilon = np.float32(1e-5)
    output = source_float32_layer_norm(source, gamma, beta, epsilon)
    centered = source - source.mean(axis=-1, keepdims=True, dtype=np.float32)
    expected = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True,
                                                          dtype=np.float32) + epsilon)
    expected = expected * gamma + beta
    np.testing.assert_array_equal(output, expected)


def test_shared_full_attention_mask_is_key_only_noncausal():
    query = np.array([[[[1, 0], [0, 1], [1, 1]]]], dtype=np.float32)
    scores = semantic_attention_scores(query, query, group=1,
                                       layout="batch_kv_heads_sequence_feature")
    scores /= np.sqrt(np.float32(2))
    attrs = {"kind": "key_padding", "mask_input": "input.attention_mask",
             "broadcast": "batch_heads_queries_keys", "masked_value": "float32_min",
             "application": "additive", "compute_dtype": "float32"}
    masked = mask_key_padding_scores(scores, np.array([[True, False, True]]), attrs)
    probabilities = source_float32_softmax(masked)
    assert probabilities.dtype == np.float32
    assert np.all(probabilities[..., 1] == 0)
    assert probabilities[0, 0, 0, 2] > 0  # Future key visible: no causal mask.
    assert probabilities[0, 0, 1, 0] > 0  # Masked query not zeroed.
    output = semantic_attention_values(probabilities, query, group=1,
                                       layout="batch_kv_heads_sequence_feature")
    np.testing.assert_allclose(output, np.matmul(probabilities, query), atol=1e-5, rtol=1e-4)


def test_native_activation_chunks_and_noncontiguous_input():
    value = np.linspace(-5, 5, (1 << 19) + 14, dtype=np.float32)[::2]
    for kind in ("gelu_erf", "tanh"):
        output = source_float32_activation(value, kind)
        if kind == "gelu_erf":
            reference = np.array([float(x) / 2 * (1 + math.erf(float(x) / math.sqrt(2)))
                                  for x in value], dtype=np.float32)
        else:
            reference = np.tanh(value)
        assert output.shape == value.shape and output.dtype == np.float32
        np.testing.assert_allclose(output, reference, atol=1e-5, rtol=1e-4)


def test_real_native_quote_and_python_memory_admission(tmp_path):
    import json
    root = write_checkpoint(tmp_path / "encoder")
    experiment = Experiment("workspace", ClientOnlyCpu(Model(str(root)), numeric=SourceFloat32()),
                            Deployment.local(root=str(tmp_path / "deployment")), ForwardBudget(2, 8))
    with experiment.compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        small = json.loads(model.contract.quote(1))
        large = json.loads(model.contract.quote(8))
        assert small["sequence_length"] == 1 and large["sequence_length"] == 8
        assert large["required_bytes"] >= small["required_bytes"] > 0
        assert large["workspace_bytes"] > 0
        assert model.forward(ForwardInputs([1])).outputs["last_hidden_state"].shape == (1, 1, 8)
    with pytest.raises((ForwardBindingError, ValueError)):
        experiment.compile_forward(memory_bytes=1024)

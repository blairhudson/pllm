from __future__ import annotations

import copy
import json

import numpy as np
from safetensors.numpy import load_file

from pllm.sources.generated_checkpoint import write_checkpoint as write_source
from pllm.sources.generated_encoder import tiny_checkpoint, write_checkpoint


def test_tiny_encoder_source_config():
    config, _ = tiny_checkpoint()
    assert config == {
        "model_type": "bert", "architectures": ["BertForSequenceClassification"],
        "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
        "intermediate_size": 13, "vocab_size": 17, "max_position_embeddings": 8,
        "type_vocab_size": 2, "position_embedding_type": "absolute",
        "hidden_act": "gelu", "layer_norm_eps": 1e-12,
        "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": 0.0, "is_decoder": False, "add_cross_attention": False,
        "pad_token_id": 0, "torch_dtype": "float32", "initializer_range": 0.02,
        "num_labels": 3, "problem_type": "single_label_classification",
        "id2label": {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        "label2id": {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
    }
    assert config["hidden_size"] // config["num_attention_heads"] == 4
    assert "bos_token_id" not in config and "eos_token_id" not in config


def test_tiny_encoder_source_keys_shapes_and_determinism():
    config, tensors = tiny_checkpoint()
    other_config, other = tiny_checkpoint()
    assert config == other_config
    shapes = {
        "bert.embeddings.word_embeddings.weight": (17, 8),
        "bert.embeddings.position_embeddings.weight": (8, 8),
        "bert.embeddings.token_type_embeddings.weight": (2, 8),
        "bert.embeddings.LayerNorm.weight": (8,),
        "bert.embeddings.LayerNorm.bias": (8,),
        "bert.pooler.dense.weight": (8, 8), "bert.pooler.dense.bias": (8,),
        "classifier.weight": (3, 8), "classifier.bias": (3,),
    }
    for layer in range(2):
        prefix = f"bert.encoder.layer.{layer}"
        for name, outputs, inputs in (
            ("attention.self.query", 8, 8), ("attention.self.key", 8, 8),
            ("attention.self.value", 8, 8), ("attention.output.dense", 8, 8),
            ("intermediate.dense", 13, 8), ("output.dense", 8, 13),
        ):
            shapes[f"{prefix}.{name}.weight"] = (outputs, inputs)
            shapes[f"{prefix}.{name}.bias"] = (outputs,)
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            shapes[f"{prefix}.{name}.weight"] = (8,)
            shapes[f"{prefix}.{name}.bias"] = (8,)
    assert tensors.keys() == shapes.keys() == other.keys()
    assert len(tensors) == 41
    for key, tensor in tensors.items():
        assert tensor.shape == shapes[key]
        assert tensor.dtype == np.dtype("float32")
        assert np.isfinite(tensor).all()
        assert tensor.tobytes() == other[key].tobytes()
        assert tensor is not other[key]
        assert np.array_equal(tensor * 256, np.rint(tensor * 256))
        if key.endswith(".bias"):
            assert np.all(tensor != 0)
        if tensor.ndim == 2:
            assert np.unique(tensor).size > 1
            if tensor.shape[0] == tensor.shape[1]:
                assert not np.array_equal(tensor, tensor.T)
    attention = [
        tensors[f"bert.encoder.layer.{layer}.attention.self.{name}.weight"].tobytes()
        for layer in range(2) for name in ("query", "key", "value")
    ]
    assert len(set(attention)) == 6


def test_encoder_checkpoint_standard_roundtrip(tmp_path):
    config, tensors = tiny_checkpoint()
    before = copy.deepcopy(config)
    original_bytes = {key: value.tobytes() for key, value in tensors.items()}
    root = write_checkpoint(tmp_path / "encoder", config, tensors)
    assert root == tmp_path / "encoder"
    assert json.loads((root / "config.json").read_text()) == json.loads(json.dumps(config))
    assert config == before
    assert {path.name for path in root.iterdir()} == {"config.json", "model.safetensors"}
    restored = load_file(root / "model.safetensors")
    assert restored.keys() == tensors.keys()
    for key, value in restored.items():
        assert value.shape == tensors[key].shape
        assert value.dtype == tensors[key].dtype
        assert value.tobytes() == tensors[key].tobytes() == original_bytes[key]
    default_root = write_checkpoint(tmp_path / "default")
    assert (default_root / "config.json").read_bytes() == (root / "config.json").read_bytes()
    defaults = load_file(default_root / "model.safetensors")
    assert defaults.keys() == tensors.keys()
    for key, value in defaults.items():
        assert value.shape == tensors[key].shape
        assert value.dtype == tensors[key].dtype
        assert value.tobytes() == original_bytes[key]


def test_neutral_writer_preserves_explicit_source_and_optional_tokenizer(tmp_path):
    config = {"model_type": "custom", "vocab_size": 17, "source_only": {"unchanged": True}}
    tensors = {
        "original.fp16": np.array([[0.25, -0.5], [1.5, -2.0]], dtype=np.float16),
        "original.int64": np.array([7, -11], dtype=np.int64),
    }
    descriptor = {"type": "explicit", "vocab_size": 17, "pad_token_id": 0}
    before = copy.deepcopy(config)
    descriptor_before = copy.deepcopy(descriptor)
    root = write_source(tmp_path / "explicit", config, tensors, tokenizer_descriptor=descriptor)
    assert json.loads((root / "config.json").read_text()) == before
    assert json.loads((root / "pllm_tokenizer.json").read_text()) == descriptor_before
    assert config == before and descriptor == descriptor_before
    restored = load_file(root / "model.safetensors")
    assert restored.keys() == tensors.keys()
    for key, value in restored.items():
        assert value.dtype == tensors[key].dtype
        assert value.shape == tensors[key].shape
        assert value.tobytes() == tensors[key].tobytes()
    encoder_root = write_checkpoint(tmp_path / "supplied", config, tensors)
    assert json.loads((encoder_root / "config.json").read_text()) == before
    assert not (encoder_root / "pllm_tokenizer.json").exists()
    supplied = load_file(encoder_root / "model.safetensors")
    assert supplied.keys() == tensors.keys()
    for key, value in supplied.items():
        assert value.shape == tensors[key].shape
        assert value.dtype == tensors[key].dtype
        assert value.tobytes() == tensors[key].tobytes()

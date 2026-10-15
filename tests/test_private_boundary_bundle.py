"""Generated public boundary descriptors never import private matrix payloads."""
import asyncio
import copy
import json

import msgpack
import pytest

from pllm import Model, _native, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.roles import PrivateLookupAndFullHead, PreparedBoundaryRoles
from pllm.model_loader import resolve_model
from pllm.runtime.model_binding import RuntimeBindingError, compile_runtime_model
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, TransformerClientError
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def generated_boundary(tmp_path, family="llama", reduction=None, *, boundary_options=None):
    if not hasattr(_native.PrivatePageServer, "from_matrix"):
        pytest.skip("shared native private table API not installed")
    from pllm.quantization import SymmetricPerRow
    path = create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    source = resolve_model(Model.path(str(path), model_id="private-boundary")).manifest
    ordinary = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(ordinary.load(source))
    original = ordinary.client_bundle("private-boundary")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8, private_boundary=True)
    asyncio.run(engine.load(source))
    document = msgpack.unpackb(engine.client_bundle("private-boundary"), raw=False)
    config = json.loads((path / "config.json").read_text())
    plan = lower_model(config, batch=1, max_input_tokens=8, max_new_tokens=4)
    composition = MaskedLinearCpu(Model("private-boundary"), boundary=PrivateLookupAndFullHead(**(boundary_options or {})),
        topology=PreparedBoundaryRoles(), quantization=SymmetricPerRow(causal_reduction=reduction))
    return engine, original, document, plan, composition


def test_public_bundle_and_compiled_boundary_identity(tmp_path):
    _, _, document, plan, composition = generated_boundary(tmp_path)
    bundle = ClientBundle.unpack(msgpack.packb(document, use_bin_type=True))
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    assert all(stage.client_weight is None and stage.client_aux_weight is None
               for stage in bundle.stages.values())
    assert not {"token_lookup", "lm_head"} & compiled.prepared_stage_rows(5, 3).keys()
    descriptor = compiled.private_boundary_descriptor()
    assert descriptor.lookup_digest == document["privacy"]["private_boundary"]["lookup"]["digest"]
    assert descriptor.head_scales_digest == document["privacy"]["private_boundary"]["head_scales_digest"]
    bundle.privacy["private_boundary"]["lookup"]["digest"] = "f" * 64
    with pytest.raises(RuntimeBindingError, match="changed"):
        compiled.validate()


def test_publisher_borrows_native_head_and_exports_after_weight_retirement(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np

    engine, _, document, _, _ = generated_boundary(tmp_path)
    model = engine._model("private-boundary")
    head = model.stages["lm_head"]
    native_matrix = head.compiled_weight._matrix
    server = _native.PrivatePageServer
    calls = []

    class SharedOnly:
        def __init__(self, *args, **kwargs):
            raise AssertionError("publisher must not serialize an interleaved table")

        @staticmethod
        def from_matrix(matrix, scales, party):
            assert matrix is native_matrix
            assert len(scales) == 4 * head.spec.out_features
            calls.append(party)
            return server.from_matrix(matrix, scales, party)

    class NoWeightExport:
        dtype, ndim = np.dtype(np.int8), 2

        def __getattr__(self, name):
            raise AssertionError(f"publisher must not copy Python weight storage: {name}")

    monkeypatch.setattr(_native, "PrivatePageServer", SharedOnly)
    head.quantized_weight = SimpleNamespace(values=NoWeightExport(), scales=head.weight_scales, bits=8)
    model.private_boundary = None
    assert engine.private_boundary_bundle(model.manifest.id) == document["privacy"]["private_boundary"]
    assert calls == [0]
    head.release_weight()
    thin = msgpack.unpackb(engine.client_bundle(model.manifest.id), raw=False)
    assert not thin["client_weights"]
    assert thin["privacy"]["private_boundary"] == document["privacy"]["private_boundary"]
    engine.client_bundle_artifacts(model.manifest.id)
    assert calls == [0]  # Descriptor reuse neither reloads nor serializes weights.


@pytest.mark.parametrize("tamper", ["extra_weight", "stage_weight", "aux", "extra_array", "scales", "field", "untied", "tensor_alias", "tensor_shape"])
def test_import_rejects_unsolicited_matrix_and_descriptor_tamper(tmp_path, tamper):
    _, original, document, _, _ = generated_boundary(tmp_path)
    local = msgpack.unpackb(original, raw=False)
    document = copy.deepcopy(document)
    if tamper == "extra_weight":
        document["client_weights"]["unreferenced"] = next(iter(local["client_weights"].values()))
    elif tamper == "stage_weight":
        document["stages"]["lm_head"]["client_weight"] = local["stages"]["lm_head"]["client_weight"]
    elif tamper == "aux":
        document["stages"]["token_lookup"]["client_aux_weight"] = {"ref": "unsolicited"}
    elif tamper == "extra_array":
        document["local_tensors"]["full_matrix"] = {"dtype": "f4", "shape": [1], "data": b"\0" * 4}
    elif tamper == "scales":
        document["privacy"]["private_boundary"]["head_scales_digest"] = "f" * 64
    elif tamper == "field":
        document["privacy"]["private_boundary"]["matrix"] = b"unsolicited"
    elif tamper == "tensor_alias":
        key, row = next(iter(document["local_tensors"].items()))
        alias = key[len("model."):] if key.startswith("model.") else "model." + key
        document["local_tensors"][alias] = row
    elif tamper == "tensor_shape":
        row = next(iter(document["local_tensors"].values()))
        row["shape"] = [1, row["shape"][0]]
    else:
        document["manifest"]["tied_embeddings"] = False
    with pytest.raises(TransformerClientError):
        ClientBundle.unpack(msgpack.packb(document, use_bin_type=True))

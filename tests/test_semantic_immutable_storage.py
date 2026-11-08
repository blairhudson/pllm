"""Immutable local tensor import and retained-binding storage contracts."""

import dataclasses
import gc
import hashlib
import sys

import numpy as np
import pytest

import pllm
from pllm.runtime.bundle_storage import (
    CHUNK_BYTES, f32_array_nbytes, f32_chunks, immutable_f32_array,
    immutable_f32_backing, immutable_f32_retained_bytes,
)
from pllm.runtime.model_binding import (
    RuntimeBindingError, _compile_runtime_model, _finite_f32_digest, compile_runtime_model,
)
from pllm.runtime.semantic_tensors import required_client_tensors
from pllm.runtime.transformer_client import ClientBundle, TransformerClientError

from absolute_position_decoder_oracle import tiny_checkpoint
from test_semantic_local_routing import source_binding


def test_exact_bytes_import_is_zero_copy_and_irreversibly_readonly():
    data = np.arange(12, dtype="<f4").tobytes()
    value = immutable_f32_array(data, [3, 4])
    assert immutable_f32_backing(value) is data
    assert np.shares_memory(value, np.frombuffer(data, dtype="<f4"))
    with pytest.raises(ValueError):
        value.setflags(write=True)
    with pytest.raises(ValueError):
        value[0, 0] = 1


@pytest.mark.parametrize("shape", [[], [True], [0], [-1], [1.0], ["1"],
                                   [sys.maxsize], [sys.maxsize // 4 + 1], [2, sys.maxsize]])
def test_geometry_rejects_before_buffer_construction(shape):
    with pytest.raises(ValueError):
        immutable_f32_array(b"", shape)


@pytest.mark.parametrize("data", [b"", bytes(12), bytearray(16), memoryview(bytes(16))])
def test_only_exact_bytes_of_exact_length_supported(data):
    with pytest.raises(ValueError):
        immutable_f32_array(data, [2, 2])


@pytest.mark.parametrize("kind", ["owned", "owned_readonly", "bytearray", "memoryview"])
def test_readonly_flags_do_not_authorize_mutable_or_wrapped_backing(kind):
    if kind.startswith("owned"):
        value = np.zeros((2, 2), dtype=np.float32)
    elif kind == "bytearray":
        value = np.frombuffer(bytearray(16), dtype="<f4").reshape(2, 2)
    else:
        value = np.frombuffer(memoryview(bytes(16)), dtype="<f4").reshape(2, 2)
    if kind != "owned":
        value.setflags(write=False)
    with pytest.raises(ValueError, match="immutable|backing|owner"):
        immutable_f32_backing(value)


def test_distinct_retained_bytes_owners_count_once_even_for_slices():
    first = immutable_f32_array(bytes(32), [2, 4])
    # Force a second bytes identity with equal contents.
    second = immutable_f32_array(bytes(bytearray(32)), [2, 4])
    assert immutable_f32_backing(first) is not immutable_f32_backing(second)
    assert immutable_f32_retained_bytes([first, first[1:], first.view(), second]) == 64


def test_chunk_scratch_bound_and_no_serialized_table_copy():
    elements = 2 * (CHUNK_BYTES // 4) + 1
    value = immutable_f32_array(bytes(4 * elements), [elements])
    chunks = list(f32_chunks(value))
    assert [row.nbytes for row in chunks] == [CHUNK_BYTES, CHUNK_BYTES, 4]
    assert all(np.shares_memory(row, value) for row in chunks)
    assert _finite_f32_digest(value) == hashlib.sha256(memoryview(value).cast("B")).hexdigest()


@pytest.mark.parametrize("position", [CHUNK_BYTES // 4 - 1, CHUNK_BYTES // 4, CHUNK_BYTES // 4 + 1])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_on_both_sides_of_import_chunk_edge(position, bad):
    data = np.zeros(CHUNK_BYTES // 4 + 2, dtype="<f4")
    data[position] = bad
    value = immutable_f32_array(data.tobytes(), data.shape)
    with pytest.raises(RuntimeBindingError, match="finite"):
        _finite_f32_digest(value)


def test_normal_import_retains_immutable_roots_and_binding_survives_source_replacement(source_binding):
    plan, bundle, composition, weights = source_binding
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    weight_id = "transformer.wpe.weight"
    first = compiled.local_tensor(weight_id)
    assert type(immutable_f32_backing(first)) is bytes
    assert np.array_equal(first, weights[weight_id])
    key = next(row["key"] for row in compiled.to_spec()["local_tensors"] if row["weight_id"] == weight_id)
    assert immutable_f32_backing(first) is immutable_f32_backing(bundle.arrays[key])
    bundle.arrays[key] = immutable_f32_array(bytes(first.nbytes), first.shape)
    assert np.array_equal(compiled.local_tensor(weight_id), weights[weight_id])
    with pytest.raises(RuntimeBindingError, match="changed since compilation"):
        compiled.validate()
    del first
    bundle.arrays.clear()
    gc.collect()
    assert np.array_equal(compiled.local_tensor(weight_id), weights[weight_id])


def test_accessor_exposes_no_retained_ndarray_metadata(source_binding):
    plan, bundle, composition, _ = source_binding
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    first = compiled.local_tensor("transformer.wpe.weight")
    first.shape = (first.size,)
    second = compiled.local_tensor("transformer.wpe.weight")
    assert second.shape == (8, 4) and type(second.base) is bytes
    with pytest.raises(ValueError):
        second.setflags(write=True)
    with pytest.raises(RuntimeBindingError, match="unknown bound"):
        compiled.local_tensor("wrong.position.weight")


def test_schedule_revalidation_reuses_validated_digest_record_without_table_rescan(source_binding):
    plan, bundle, composition, _ = source_binding
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    refreshed, _ = _compile_runtime_model(plan, bundle, composition=composition, _previous_binding=compiled)
    for key, record in compiled._local_tensors.items():
        assert refreshed._local_tensors[key][0] is record[0]
        assert refreshed._local_tensors[key][1] is record[1]
    assert compiled._local_tensor_bytes == immutable_f32_retained_bytes(bundle.arrays.values())
    with pytest.raises(RuntimeBindingError, match="compiled binding owner"):
        _compile_runtime_model(plan, bundle, composition=composition, _previous_binding={})


def test_manual_mutable_bundle_refused_without_snapshot_copy(source_binding):
    plan, bundle, composition, _ = source_binding
    arrays = {key: value.copy() for key, value in bundle.arrays.items()}
    for value in arrays.values():
        value.setflags(write=False)
    with pytest.raises(RuntimeBindingError, match="immutable bytes backing"):
        compile_runtime_model(plan, dataclasses.replace(bundle, arrays=arrays), composition=composition)


def test_nonfinite_source_bytes_rejected_at_binding_import(source_binding):
    plan, bundle, composition, _ = source_binding
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    key = next(row["key"] for row in compiled.to_spec()["local_tensors"]
               if row["weight_id"] == "transformer.wpe.weight")
    data = np.zeros((8, 4), dtype="<f4")
    data[-1, -1] = np.nan
    arrays = dict(bundle.arrays, **{key: immutable_f32_array(data.tobytes(), data.shape)})
    with pytest.raises(RuntimeBindingError, match="finite"):
        compile_runtime_model(plan, dataclasses.replace(bundle, arrays=arrays), composition=composition)


class _UnreachableStages(dict):
    def items(self):
        raise AssertionError("stage materialization preceded semantic geometry validation")


@pytest.mark.parametrize("damage", ["wrong_shape", "bool_dimension", "huge", "short", "mutable", "missing"])
def test_source_geometry_preflight_precedes_stage_materialization(damage):
    config, weights = tiny_checkpoint()
    plan = pllm.lower_model(config, batch=1, max_input_tokens=1, max_new_tokens=1)
    required = required_client_tensors(plan)
    rows = {key: {"dtype": "f4", "shape": list(shape), "data": weights[key].astype("<f4").tobytes()}
            for key, shape in required.items()}
    row = rows["transformer.wpe.weight"]
    if damage == "wrong_shape":
        row["shape"] = [11, 4]
    elif damage == "bool_dimension":
        row["shape"] = [True, 4]
    elif damage == "huge":
        row["shape"] = [sys.maxsize, 4]
    elif damage == "short":
        row["data"] = row["data"][:-4]
    elif damage == "mutable":
        row["data"] = bytearray(row["data"])
    else:
        del rows["transformer.wpe.weight"]
    document = {"v": 1, "runtime": "client_local_transformer", "model": "geometry",
                "manifest": {"id": "geometry"}, "config": config, "tokenizer": {},
                "stages": _UnreachableStages(), "local_tensors": rows, "privacy": {}}
    with pytest.raises(TransformerClientError, match="semantic source"):
        ClientBundle._from_document(document)

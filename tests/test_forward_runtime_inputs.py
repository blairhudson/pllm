"""Strict input envelope tests, separate from whole-encoder oracle integration."""
import json

import numpy as np
import pytest

from pllm.runtime.forward_inputs import (
    ForwardInputError, ForwardInputs, inspect_forward_inputs, parse_forward_inputs,
    read_forward_inputs, snapshot_forward_inputs,
)


DOMAINS = {"tokens": 17, "positions": 8, "segments": 2}
# Positive functional setup: reuse genuine JSON tests' validated 128-MiB
# headroom, not a production cap or a change to any admission boundary.
FUNCTIONAL_SETUP_BYTES = 128 << 20


def inspect(inputs):
    return inspect_forward_inputs(inputs, maximum=8, domains=DOMAINS)


def test_constructor_borrows_without_coercion():
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError("coercion hook called")
    poison = Poison()
    inputs = ForwardInputs(poison)
    assert inputs.tokens is poison
    with pytest.raises(ForwardInputError):
        inspect(inputs)


@pytest.mark.parametrize("tokens", [
    [True], [1.0], [[[1]]], [object()], [], np.array([1], dtype=np.int32),
    np.array([1.0]), np.array([1], dtype=object), np.ones((2, 1), dtype=np.int64),
    np.ones((1, 1, 1), dtype=np.int64), np.array([1], dtype=">i8"),
])
def test_ids_reject_before_coercion(tokens):
    with pytest.raises(ForwardInputError):
        inspect(ForwardInputs(tokens))


def test_subclasses_rejected_without_hooks():
    class ListSubclass(list):
        def __iter__(self):
            raise AssertionError("subclass hook called")
    class ArraySubclass(np.ndarray):
        def __array__(self, *args, **kwargs):
            raise AssertionError("subclass hook called")
    for value in (ListSubclass([1]), [ListSubclass([1])],
                  np.array([1], dtype=np.int64).view(ArraySubclass)):
        with pytest.raises(ForwardInputError):
            inspect(ForwardInputs(value))


def test_domains_are_independent_and_positions_arbitrary():
    assert inspect(ForwardInputs([16, 0, 4], positions=[7, 7, 0], segments=[1, 0, 1],
                                 attention_mask=[True, False, True])) == 3
    for inputs in (ForwardInputs([17]), ForwardInputs([1], positions=[8]),
                   ForwardInputs([1], segments=[2]), ForwardInputs([-1]),
                   ForwardInputs([1], attention_mask=[False]),
                   ForwardInputs([1], attention_mask=[1]),
                   ForwardInputs([1, 2], positions=[0])):
        with pytest.raises(ForwardInputError):
            inspect(inputs)


@pytest.mark.parametrize("raw", [
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[1],"tokens":[2]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[NaN]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[1],"extra":0}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[true]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[true]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[1.0]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[1e999]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[1]],"attention_mask":[[false]]}',
    b'{"schema_version":"pllm.forward_inputs.v1","tokens":[[1]],"attention_mask":[[1]]}',
    b'{"schema":"pllm.forward_inputs.v1","tokens":[[1]]}',
    b" " * ((1 << 20) + 1),
])
def test_json_refuses_duplicates_unknown_nonfinite_and_oversize(raw):
    with pytest.raises(ForwardInputError):
        parse_forward_inputs(raw)


def test_production_bounded_reader(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"schema_version": "pllm.forward_inputs.v1", "tokens": [1, 2],
                               "positions": [6, 1], "attention_mask": [True, False]}))
    assert inspect(read_forward_inputs(path)) == 2
    path.write_bytes(b" " * ((1 << 20) + 1))
    with pytest.raises(ForwardInputError):
        read_forward_inputs(path)


def test_nested_json_roundtrip_parse_read_and_genuine_inputs(tmp_path):
    document = {"schema_version": "pllm.forward_inputs.v1", "tokens": [[1, 2, 3]],
                "positions": [[0, 1, 2]], "segments": [[0, 0, 1]],
                "attention_mask": [[True, True, False]]}
    raw = json.dumps(document).encode("utf-8")
    path = tmp_path / "batch-one.json"
    path.write_bytes(raw)
    for inputs in (parse_forward_inputs(raw), read_forward_inputs(path),
                   ForwardInputs(document["tokens"], positions=document["positions"],
                                 segments=document["segments"],
                                 attention_mask=document["attention_mask"])):
        assert type(inputs) is ForwardInputs
        assert inspect(inputs) == 3
        assert inputs.tokens == [[1, 2, 3]]
        assert inputs.attention_mask == [[True, True, False]]
    borrowed = ForwardInputs(document["tokens"])
    assert borrowed.tokens is document["tokens"]


@pytest.mark.parametrize("tokens", [[1, 2, 3], [[1, 2, 3]]])
def test_optional_defaults_remain_unallocated_after_parse(tokens):
    inputs = parse_forward_inputs(json.dumps({"schema_version": "pllm.forward_inputs.v1",
                                             "tokens": tokens}).encode())
    assert inspect(inputs) == 3
    assert inputs.positions is inputs.segments is inputs.attention_mask is None


def test_exact_native_arrays_accept_batch_one_and_mixed_sdk_shorthand():
    inputs = ForwardInputs(np.array([[1, 2, 3]], dtype=np.int64), positions=[[7, 0, 7]],
                           segments=[1, 0, 1],
                           attention_mask=np.array([[True, True, False]], dtype=np.bool_))
    assert inspect(inputs) == 3


@pytest.mark.parametrize("tokens", [
    [[1], [2]], [[1, 2], [3]], [[[1, 2]]], [[]], [[1], 2], [1, [2]],
])
def test_multibatch_ragged_and_rank_three_refused_in_sdk_and_json(tokens):
    with pytest.raises(ForwardInputError):
        inspect(ForwardInputs(tokens))
    raw = json.dumps({"schema_version": "pllm.forward_inputs.v1", "tokens": tokens}).encode()
    with pytest.raises(ForwardInputError):
        parse_forward_inputs(raw)


@pytest.mark.parametrize("field,value", [
    ("positions", [[0, 1]]), ("segments", [[0]]),
    ("attention_mask", [[True, False]]), ("positions", [[0], [1], [2]]),
])
def test_nested_fields_require_common_sequence_and_batch_one(field, value):
    with pytest.raises(ForwardInputError):
        inspect(ForwardInputs([[1, 2, 3]], **{field: value}))
    raw = json.dumps({"schema_version": "pllm.forward_inputs.v1", "tokens": [[1, 2, 3]],
                      field: value}).encode()
    with pytest.raises(ForwardInputError):
        parse_forward_inputs(raw)


def test_nested_domains_and_no_preflight_numpy_coercion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy coercion/allocation during preflight")
    monkeypatch.setattr(np, "array", forbidden)
    monkeypatch.setattr(np, "asarray", forbidden)
    raw = json.dumps({"schema_version": "pllm.forward_inputs.v1", "tokens": [[16, 0, 4]],
                      "positions": [[7, 7, 0]], "segments": [[1, 0, 1]],
                      "attention_mask": [[True, False, True]]}).encode()
    assert inspect(parse_forward_inputs(raw)) == 3
    for inputs in (ForwardInputs([[17]]), ForwardInputs([[1]], positions=[[8]]),
                   ForwardInputs([[1]], segments=[[2]]),
                   ForwardInputs([[1]], attention_mask=[[False]])):
        with pytest.raises(ForwardInputError):
            inspect(inputs)


def test_nested_and_flat_sdk_execute_same_native_geometry_and_defaults(tmp_path):
    from pllm.configuration import Deployment, Experiment, ForwardBudget, Model
    from pllm.numeric import SourceFloat32
    from pllm.profiles import ClientOnlyCpu
    from pllm.sources.generated_encoder import write_checkpoint

    root = write_checkpoint(tmp_path / "encoder")
    experiment = Experiment("nested-inputs", ClientOnlyCpu(Model(str(root)), numeric=SourceFloat32()),
                            Deployment.local(root=str(tmp_path / "deployment")), ForwardBudget(4, 8))
    with experiment.compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        nested = model.forward(ForwardInputs([[1, 2, 3]], positions=[[7, 0, 7]],
                                            segments=[[1, 0, 1]],
                                            attention_mask=[[True, True, False]]))
        flat = model.forward(ForwardInputs([1, 2, 3], positions=[7, 0, 7], segments=[1, 0, 1],
                                          attention_mask=[True, True, False]))
        default = model.forward(ForwardInputs([[1, 2, 3]]))
        explicit = model.forward(ForwardInputs([[1, 2, 3]], positions=[[0, 1, 2]],
                                              segments=[[0, 0, 0]],
                                              attention_mask=[[True, True, True]]))
    assert nested.outputs["last_hidden_state"].shape == (1, 3, 8)
    for name in nested.outputs:
        np.testing.assert_array_equal(nested.outputs[name], flat.outputs[name])
        np.testing.assert_array_equal(default.outputs[name], explicit.outputs[name])


@pytest.mark.parametrize("inputs", [
    ForwardInputs([object()] * 9),
    ForwardInputs([[object()] * 9]),
    ForwardInputs([object(), object()], positions=[0]),
    ForwardInputs([object(), object()], segments=[[0] * 9]),
    ForwardInputs([object(), object()], attention_mask=[True]),
])
def test_geometry_rejects_before_any_scalar_validation(monkeypatch, inputs):
    from pllm.runtime import forward_inputs as module

    def forbidden(*args, **kwargs):
        raise AssertionError("scalar validation preceded geometric admission")

    monkeypatch.setattr(module, "_scalar", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    with pytest.raises(ForwardInputError):
        inspect(inputs)
    with pytest.raises(ForwardInputError):
        snapshot_forward_inputs(inputs, maximum=8, domains=DOMAINS, sequence=2)


def test_parser_checks_all_geometry_before_scalar_types(monkeypatch):
    from pllm.runtime import forward_inputs as module

    def forbidden(*args, **kwargs):
        raise AssertionError("parser checked values before mismatched extent")

    monkeypatch.setattr(module, "_scalar", forbidden)
    raw = json.dumps({"schema_version": "pllm.forward_inputs.v1",
                      "tokens": [["not-an-id", "not-an-id"]],
                      "attention_mask": [[True]]}).encode()
    with pytest.raises(ForwardInputError):
        parse_forward_inputs(raw)


def _mutable_inputs(kind):
    fields = {"tokens": [16, 0, 4], "positions": [7, 7, 0],
              "segments": [1, 0, 1], "attention_mask": [True, False, True]}
    if kind == "list":
        fields = {name: [row] for name, row in fields.items()}
    else:
        fields = {name: np.array([row], dtype=np.bool_ if name == "attention_mask" else np.int64)
                  for name, row in fields.items()}
    return ForwardInputs(**fields)


@pytest.mark.parametrize("kind", ["list", "array"])
@pytest.mark.parametrize("field", ["tokens", "positions", "segments", "attention_mask"])
@pytest.mark.parametrize("stage", ["between_inspect_and_snapshot", "after_reinspect",
                                   "during_first_allocation"])
def test_mutable_extent_never_controls_snapshot_allocation(monkeypatch, kind, field, stage):
    from pllm.runtime import forward_inputs as module

    inputs = _mutable_inputs(kind)
    sequence = inspect(inputs)
    original_empty = np.empty
    original_inspect = module.inspect_forward_inputs
    allocations = []
    grown = False

    def grow():
        nonlocal grown
        assert not grown
        grown = True
        value = getattr(inputs, field)
        if kind == "list":
            value[0].extend([True if field == "attention_mask" else 0] * 64)
        else:
            # Deterministic seam: no concurrent native reader or outstanding
            # element view. This does not promise safety for hostile concurrent
            # resize(refcheck=False), which NumPy itself cannot guarantee.
            value.resize((1, 67), refcheck=False)

    def reinspect(*args, **kwargs):
        size = original_inspect(*args, **kwargs)
        if stage == "after_reinspect":
            grow()
        return size

    def fixed_empty(shape, *args, **kwargs):
        allocations.append(shape)
        assert shape == (1, sequence), "allocation followed mutable caller extent"
        if stage == "during_first_allocation" and not grown:
            grow()
        return original_empty(shape, *args, **kwargs)

    def forbidden_copy(*args, **kwargs):
        raise AssertionError("snapshot used variable-sized caller conversion/copy")

    if stage == "between_inspect_and_snapshot":
        grow()
    monkeypatch.setattr(module, "inspect_forward_inputs", reinspect)
    monkeypatch.setattr(np, "empty", fixed_empty)
    monkeypatch.setattr(np, "array", forbidden_copy)
    monkeypatch.setattr(np, "asarray", forbidden_copy)
    monkeypatch.setattr(np, "copy", forbidden_copy)
    with pytest.raises(ForwardInputError):
        snapshot_forward_inputs(inputs, maximum=8, domains=DOMAINS, sequence=sequence)
    assert grown
    if stage == "during_first_allocation":
        assert allocations == [(1, sequence)]
    else:
        assert allocations == []


def test_growing_list_validation_uses_fixed_index_count(monkeypatch):
    from pllm.runtime import forward_inputs as module

    tokens = [1, 2, 3]
    original = module._scalar
    visited = []

    def growing_scalar(value, index, **kwargs):
        visited.append(index)
        item = original(value, index, **kwargs)
        tokens.append(1)
        return item

    monkeypatch.setattr(module, "_scalar", growing_scalar)
    with pytest.raises(ForwardInputError):
        inspect(ForwardInputs(tokens))
    assert visited == [0, 1, 2]


def test_replaced_scalar_rejected_without_coercion_or_comparison(monkeypatch):
    class Poison:
        def __int__(self):
            raise AssertionError("coercion before exact scalar type check")

        def __index__(self):
            raise AssertionError("index conversion before exact scalar type check")

        def __lt__(self, other):
            raise AssertionError("comparison before exact scalar type check")

        def __ge__(self, other):
            raise AssertionError("comparison before exact scalar type check")

    inputs = _mutable_inputs("list")
    original_empty = np.empty

    def replace_scalar(shape, *args, **kwargs):
        inputs.tokens[0][0] = Poison()
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", replace_scalar)
    with pytest.raises(ForwardInputError):
        snapshot_forward_inputs(inputs, maximum=8, domains=DOMAINS, sequence=3)


@pytest.mark.parametrize("kind", ["list", "array"])
@pytest.mark.parametrize("field,bad", [("tokens", 17), ("positions", 8), ("segments", 2),
                                      ("attention_mask", False)])
def test_same_extent_caller_mutation_after_reinspect_is_revalidated(monkeypatch, kind, field, bad):
    inputs = _mutable_inputs(kind)
    original_empty = np.empty
    changed = False

    def mutate_at_allocation(shape, *args, **kwargs):
        nonlocal changed
        assert shape == (1, 3)
        if not changed:
            changed = True
            value = getattr(inputs, field)
            if kind == "list":
                value[0][:] = [bad] * 3
            else:
                value[:] = bad
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", mutate_at_allocation)
    with pytest.raises(ForwardInputError):
        snapshot_forward_inputs(inputs, maximum=8, domains=DOMAINS, sequence=3)
    assert changed


def test_array_dtype_change_after_reinspect_cannot_cast_into_snapshot(monkeypatch):
    inputs = _mutable_inputs("array")
    original_empty = np.empty

    def change_dtype(shape, *args, **kwargs):
        inputs.tokens.dtype = np.float64
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", change_dtype)
    with pytest.raises(ForwardInputError):
        snapshot_forward_inputs(inputs, maximum=8, domains=DOMAINS, sequence=3)


@pytest.mark.parametrize("field,bad", [("tokens", 17), ("positions", 8), ("segments", 2),
                                      ("attention_mask", False)])
def test_same_extent_mutation_still_validates_private_snapshot(monkeypatch, field, bad):
    inputs = _mutable_inputs("list")
    original_empty = np.empty
    allocated = []

    def corrupt_previous_private_destination(shape, *args, **kwargs):
        # Corrupt a previously completed private copy, not caller geometry. The
        # final validation must catch this even though caller inputs stay valid.
        if len(allocated) == 3:
            target = ("tokens", "positions", "segments").index(field) if field != "attention_mask" else None
            if target is not None:
                allocated[target][0, 0] = bad
        output = original_empty(shape, *args, **kwargs)
        allocated.append(output)
        if field == "attention_mask" and len(allocated) == 4:
            # All-false caller mask is copied into the fixed mask destination;
            # its unsupported semantics must be checked on the owned snapshot.
            inputs.attention_mask[0][:] = [False, False, False]
        return output

    monkeypatch.setattr(np, "empty", corrupt_previous_private_destination)
    with pytest.raises(ForwardInputError):
        snapshot_forward_inputs(inputs, maximum=8, domains=DOMAINS, sequence=3)


@pytest.mark.parametrize("kind", ["list", "array"])
def test_snapshot_has_owned_readonly_batch_one_geometry_and_caller_isolation(kind):
    inputs = _mutable_inputs(kind)
    result = snapshot_forward_inputs(inputs, maximum=8, domains=DOMAINS, sequence=3)
    assert set(result) == {"input.tokens", "input.positions", "input.segments", "input.attention_mask"}
    expected = {"tokens": [16, 0, 4], "positions": [7, 7, 0],
                "segments": [1, 0, 1], "attention_mask": [True, False, True]}
    for name, row in expected.items():
        owned = result["input." + name]
        assert owned.shape == (1, 3)
        assert owned.dtype == np.dtype(np.bool_ if name == "attention_mask" else np.int64)
        assert owned.flags.owndata
        assert not owned.flags.writeable
        caller = getattr(inputs, name)
        if kind == "list":
            caller[0][:] = [False if name == "attention_mask" else 0] * 3
        else:
            assert not np.shares_memory(caller, owned)
            caller[:] = False if name == "attention_mask" else 0
        np.testing.assert_array_equal(owned, [row])


def test_snapshot_defaults_use_admitted_shape():
    result = snapshot_forward_inputs(ForwardInputs([[16, 0, 4]]), maximum=8,
                                     domains=DOMAINS, sequence=3)
    for name, row in {"positions": [0, 1, 2], "segments": [0, 0, 0],
                      "attention_mask": [True, True, True]}.items():
        np.testing.assert_array_equal(result["input." + name], [row])
        assert result["input." + name].shape == (1, 3)
        assert not result["input." + name].flags.writeable


def test_parse_and_read_nested_json_execute_through_genuine_forward_owner(tmp_path):
    from pllm import Deployment, Experiment, ForwardBudget, Model
    from pllm.kernels import Cpu
    from pllm.numeric import SourceFloat32
    from pllm.profiles import ClientOnlyCpu
    from pllm.sources.generated_encoder import tiny_checkpoint, write_checkpoint
    from encoder_forward_oracle import ATOL, RTOL, evaluate

    config, tensors = tiny_checkpoint()
    checkpoint = write_checkpoint(tmp_path / "encoder-json", config, tensors)
    document = {"schema_version": "pllm.forward_inputs.v1", "tokens": [[16, 0, 4]],
                "positions": [[7, 7, 0]], "segments": [[1, 0, 1]],
                "attention_mask": [[True, False, True]]}
    raw = json.dumps(document).encode()
    path = tmp_path / "forward-inputs.json"
    path.write_bytes(raw)
    expected = evaluate(config, tensors, document["tokens"],
                        position_ids=document["positions"], token_type_ids=document["segments"],
                        attention_mask=document["attention_mask"])
    inputs = (parse_forward_inputs(raw), read_forward_inputs(path))
    experiment = Experiment(
        "parsed-json-forward",
        pipeline=ClientOnlyCpu(Model.path(str(checkpoint)), kernels=Cpu(threads=2), numeric=SourceFloat32()),
        deployment=Deployment.local(root=str(tmp_path / "json-deployment")),
        budget=ForwardBudget(requests=2, max_input_tokens=6),
    )
    with experiment.compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        results = [model.forward(item) for item in inputs]
    shapes = {"last_hidden_state": (1, 3, 8), "pooler_output": (1, 8), "logits": (1, 3)}
    for result in results:
        assert set(result.outputs) == set(shapes)
        for name, shape in shapes.items():
            assert result.outputs[name].shape == shape
            assert result.outputs[name].dtype == np.dtype(np.float32)
            assert not result.outputs[name].flags.writeable
            np.testing.assert_allclose(result.outputs[name], expected[name], atol=ATOL, rtol=RTOL)

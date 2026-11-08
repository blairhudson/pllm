"""Whole-output source-FP32 encoder contracts through public forward APIs.

The three-output contract is intentionally richer than Hugging Face's classifier
return tuple. Expected values use original source fields and tensors only. No
runtime captures, plan edits, private owners, or historical binary parity claims.
"""

from collections.abc import Mapping

import numpy as np
import pytest

from pllm import Deployment, Experiment, ForwardBudget, ForwardInputs, Model
from pllm.kernels import Cpu
from pllm.numeric import SourceFloat32
from pllm.profiles import ClientOnlyCpu
from pllm.sources.generated_encoder import tiny_checkpoint, write_checkpoint

from encoder_forward_oracle import ATOL, RTOL, evaluate


MEMORY_BYTES = 128 << 20
DETECTABLE_MARGIN = 1e-6
OUTPUT_NAMES = {"last_hidden_state", "pooler_output", "logits"}
PUBLIC_ERRORS = (ValueError, RuntimeError)


def _experiment(root, checkpoint, *, requests):
    return Experiment(
        "encoder-forward-execution",
        pipeline=ClientOnlyCpu(
            Model.path(str(checkpoint)), kernels=Cpu(threads=2), numeric=SourceFloat32(),
        ),
        deployment=Deployment.local(root=str(root)),
        budget=ForwardBudget(requests=requests, max_input_tokens=6),
    )


@pytest.fixture
def source(tmp_path):
    config, tensors = tiny_checkpoint()
    checkpoint = write_checkpoint(tmp_path / "checkpoint", config, tensors)
    return config, tensors, checkpoint


def _expected(source, inputs):
    config, tensors, _ = source

    def row(value):
        return None if value is None else np.asarray(value).reshape(1, -1)

    return evaluate(
        config, tensors, row(inputs.tokens),
        position_ids=row(inputs.positions), token_type_ids=row(inputs.segments),
        attention_mask=row(inputs.attention_mask),
    )


def _assert_outputs(result, expected, sequence):
    outputs = result.outputs
    assert isinstance(outputs, Mapping)
    assert set(outputs) == OUTPUT_NAMES
    shapes = {
        "last_hidden_state": (1, sequence, 8),
        "pooler_output": (1, 8),
        "logits": (1, 3),
    }
    for name, shape in shapes.items():
        value = outputs[name]
        assert isinstance(value, np.ndarray)
        assert value.shape == shape
        assert value.dtype == np.dtype(np.float32)
        assert np.isfinite(value).all()
        assert not value.flags.writeable
        np.testing.assert_allclose(value, expected[name], atol=ATOL, rtol=RTOL)


def _strided(values, dtype):
    # Distinct storage with gaps, not a caller-owned runtime tensor factory.
    storage = np.zeros((1, len(values) * 2), dtype=dtype)
    storage[:, ::2] = values
    return storage[:, ::2]


def _observe_native_activations(patch, calls, *, fail_once=False):
    from pllm import _native as native

    original = native.source_float32_activation

    def observed(kind, input):
        assert type(kind) is str
        assert kind in {"gelu_erf", "tanh"}
        assert type(input) is bytes
        assert len(input) <= 1 << 20
        assert len(input) % 4 == 0
        assert np.isfinite(np.frombuffer(input, dtype="<f4")).all()
        calls.append(kind)
        if fail_once and len(calls) == 1:
            raise RuntimeError("injected native activation failure")
        output = original(kind, input)
        assert type(output) is bytes
        assert len(output) == len(input)
        return output

    patch.setattr(native, "source_float32_activation", observed)


def test_forward_dispatches_real_native_activations(source, tmp_path, monkeypatch):
    inputs = ForwardInputs([16, 0, 4, 2, 3, 1], positions=[7, 2, 7, 0, 5, 1],
                           segments=[1, 0, 1, 1, 0, 1],
                           attention_mask=[True, False, True, True, False, True])
    expected = _expected(source, inputs)
    experiment = _experiment(tmp_path / "deployment", source[2], requests=1)
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        calls = []
        with monkeypatch.context() as patch:
            _observe_native_activations(patch, calls)
            result = model.forward(inputs)
        _assert_outputs(result, expected, 6)
        # L2 requires exact GELU at least once per layer; chunking may add calls.
        assert calls.count("gelu_erf") >= 2
        assert calls.count("tanh") >= 1


def test_forward_uses_unoptimized_f32_einsum_without_blas(source, tmp_path, monkeypatch):
    inputs = ForwardInputs([16, 0, 4, 2, 3, 1], positions=[7, 2, 7, 0, 5, 1],
                           segments=[1, 0, 1, 1, 0, 1],
                           attention_mask=[True, False, True, True, False, True])
    # Independent oracle may use matmul; finish it before installing BLAS traps.
    expected = _expected(source, inputs)
    experiment = _experiment(tmp_path / "deployment", source[2], requests=1)
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        metadata = model.execution_metadata
        assert isinstance(metadata, Mapping)
        declared = {
            "backend": "numpy_einsum_unoptimized_f32_v1",
            "requested_compute_threads": 2,
            "effective_compute_threads": 1,
            "thread_policy": "serial_compute_within_requested_budget",
        }
        for key, value in declared.items():
            assert metadata[key] == value
            with pytest.raises(TypeError):
                metadata[key] = value
            with pytest.raises(TypeError):
                del metadata[key]
        # Compute-thread policy says nothing about total process thread count.
        original_einsum = np.einsum
        blas_calls, einsum_calls, projection_calls = [], [], []

        def forbidden_blas(*args, **kwargs):
            blas_calls.append(True)
            raise AssertionError("production dispatched a forbidden BLAS entry point")

        def observed_einsum(signature, *operands, **kwargs):
            einsum_calls.append((signature, kwargs.get("optimize")))
            assert kwargs.get("optimize") is False
            if signature == "...i,oi->...o":
                assert kwargs.get("dtype") is np.float32
                assert kwargs.get("casting") == "no"
                output = kwargs.get("out")
                assert isinstance(output, np.ndarray)
                assert output.dtype == np.dtype(np.float32)
                assert output.flags.writeable
                assert len(operands) == 2
                assert output.shape == operands[0].shape[:-1] + (operands[1].shape[0],)
                projection_calls.append(signature)
                result = original_einsum(signature, *operands, **kwargs)
                assert result is output
                return result
            return original_einsum(signature, *operands, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(np, "matmul", forbidden_blas)
            patch.setattr(np, "dot", forbidden_blas)
            patch.setattr(np, "tensordot", forbidden_blas)
            patch.setattr(np, "einsum", observed_einsum)
            result = model.forward(inputs)

        assert not blas_calls
        assert einsum_calls
        assert all(optimize is False for _, optimize in einsum_calls)
        assert projection_calls
        _assert_outputs(result, expected, 6)
        for key, value in declared.items():
            assert model.execution_metadata[key] == metadata[key] == value
        # BLAS exclusion does not establish absence of integer kernels; a spy on
        # their actual shared runtime interface remains separately deferred.


def test_native_failure_publishes_no_result_and_consumes_attempt(source, tmp_path, monkeypatch):
    inputs = ForwardInputs([16, 0, 4], positions=[7, 7, 2], segments=[1, 0, 1])
    expected = _expected(source, inputs)
    experiment = _experiment(tmp_path / "deployment", source[2], requests=2)
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        calls, published = [], []
        with monkeypatch.context() as patch:
            # Failure occurs at a real execution dispatch, after compilation and
            # admission, not by fabricating an owner, plan, or admission result.
            _observe_native_activations(patch, calls, fail_once=True)
            with pytest.raises(PUBLIC_ERRORS) as error:
                published.append(model.forward(inputs))
            assert str(error.value).strip()
            assert len(calls) == 1
            assert not published  # No result returned through the public boundary.

            successful_start = len(calls)
            result = model.forward(inputs)
            _assert_outputs(result, expected, 3)
            successful_calls = calls[successful_start:]
            assert successful_calls.count("gelu_erf") >= 2
            assert successful_calls.count("tanh") >= 1

            # Failed admitted attempt plus successful retry exhaust requests=2.
            before_exhausted = len(calls)
            with pytest.raises(PUBLIC_ERRORS) as error:
                model.forward(inputs)
            assert str(error.value).strip()
            assert len(calls) == before_exhausted
        _assert_outputs(result, expected, 3)


@pytest.mark.parametrize("sequence", range(1, 7))
def test_forward_whole_output_matrix(source, tmp_path, sequence):
    tokens = [0, 16, 5, 2, 11, 3][:sequence]
    positions = [7, 2, 7, 0, 5, 1][:sequence]
    segments = [1, 0, 1, 1, 0, 1][:sequence]
    cases = [
        ForwardInputs(tokens),
        ForwardInputs(tokens, positions=positions, segments=segments),
        ForwardInputs(
            _strided(tokens, np.int64),
            positions=_strided(positions, np.int64),
            segments=_strided(segments, np.int64),
            attention_mask=_strided([True] * sequence, np.bool_),
        ),
    ]
    # Right padding, interior hole, leading masked query; S=1 keeps one key.
    masks = [
        [index < max(1, sequence - 2) for index in range(sequence)],
        [index != sequence // 2 or sequence == 1 for index in range(sequence)],
        [index > 0 or sequence == 1 for index in range(sequence)],
    ]
    cases.extend(
        ForwardInputs(tokens, positions=positions, segments=segments, attention_mask=mask)
        for mask in masks
    )
    experiment = _experiment(tmp_path / "deployment", source[2], requests=len(cases))
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        for inputs in cases:
            expected = _expected(source, inputs)
            _assert_outputs(model.forward(inputs), expected, sequence)


def test_later_visible_token_changes_earlier_hidden_state(source, tmp_path):
    original = ForwardInputs([0, 4, 2, 3])
    changed = ForwardInputs([0, 4, 2, 16])
    expected_a, expected_b = (_expected(source, item) for item in (original, changed))
    # Fixture must demonstrate noncausality before checking production behavior.
    oracle_delta = np.max(np.abs(
        expected_a["last_hidden_state"][:, 0] - expected_b["last_hidden_state"][:, 0]
    ))
    assert oracle_delta > DETECTABLE_MARGIN, "investigate fixture; do not widen tolerance"
    experiment = _experiment(tmp_path / "deployment", source[2], requests=2)
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        first, second = model.forward(original), model.forward(changed)
        _assert_outputs(first, expected_a, 4)
        _assert_outputs(second, expected_b, 4)
        assert np.max(np.abs(
            first.outputs["last_hidden_state"][:, 0]
            - second.outputs["last_hidden_state"][:, 0]
        )) > DETECTABLE_MARGIN


@pytest.mark.parametrize("mask", [
    [True, True, False, False],
    [True, False, True, True],
    [False, True, True, True],
])
def test_masked_keys_do_not_affect_visible_queries(source, tmp_path, mask):
    tokens = [0, 4, 2, 3]
    changed = [token if visible else 16 for token, visible in zip(tokens, mask)]
    inputs = [ForwardInputs(item, attention_mask=mask) for item in (tokens, changed)]
    expected = [_expected(source, item) for item in inputs]
    visible = np.asarray(mask)
    np.testing.assert_allclose(
        expected[0]["last_hidden_state"][:, visible],
        expected[1]["last_hidden_state"][:, visible], atol=ATOL, rtol=RTOL,
    )
    # Masking removes keys, not query rows, including a masked first-token pooler.
    assert np.any(expected[0]["last_hidden_state"][:, ~visible] != 0)
    experiment = _experiment(tmp_path / "deployment", source[2], requests=2)
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        results = [model.forward(item) for item in inputs]
        for result, reference in zip(results, expected):
            _assert_outputs(result, reference, 4)
        np.testing.assert_allclose(
            results[0].outputs["last_hidden_state"][:, visible],
            results[1].outputs["last_hidden_state"][:, visible], atol=ATOL, rtol=RTOL,
        )


def test_zero_token_does_not_infer_padding_mask(source, tmp_path):
    cases = [
        ForwardInputs([4, 0, 2]),
        ForwardInputs([4, 0, 2], attention_mask=[True, True, True]),
        ForwardInputs([4, 0, 2], attention_mask=[True, False, True]),
    ]
    expected = [_expected(source, item) for item in cases]
    assert np.max(np.abs(
        expected[0]["last_hidden_state"][:, 0]
        - expected[2]["last_hidden_state"][:, 0]
    )) > DETECTABLE_MARGIN, "investigate fixture; do not widen tolerance"
    experiment = _experiment(tmp_path / "deployment", source[2], requests=3)
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        results = [model.forward(item) for item in cases]
        for result, reference in zip(results, expected):
            _assert_outputs(result, reference, 3)
        for name in OUTPUT_NAMES:
            np.testing.assert_array_equal(results[0].outputs[name], results[1].outputs[name])


def test_requests_are_stateless_and_results_outlive_owner(source, tmp_path):
    cases = [
        ForwardInputs([16, 0, 3, 4, 2, 1], positions=[7, 7, 1, 0, 4, 2],
                      segments=[1, 1, 0, 1, 0, 1]),
        ForwardInputs([2]),
        ForwardInputs([0, 16, 3]),
        ForwardInputs([0, 16, 3], positions=[0, 1, 2], segments=[0, 0, 0],
                      attention_mask=[True, True, True]),
        ForwardInputs([16, 0, 3, 4, 2, 1]),
        ForwardInputs([0, 16, 3]),
    ]
    expected = [_expected(source, item) for item in cases]
    experiment = _experiment(tmp_path / "deployment", source[2], requests=len(cases))
    results, saved = [], []
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        for item, reference in zip(cases, expected):
            result = model.forward(item)
            _assert_outputs(result, reference, len(item.tokens))
            results.append(result)
            saved.append({name: value.copy() for name, value in result.outputs.items()})
    for result, reference, inputs, snapshot in zip(results, expected, cases, saved):
        _assert_outputs(result, reference, len(inputs.tokens))
        for name in OUTPUT_NAMES:
            np.testing.assert_array_equal(result.outputs[name], snapshot[name])
            with pytest.raises(ValueError):
                result.outputs[name].flat[0] = 0
            with pytest.raises(ValueError):
                result.outputs[name].setflags(write=True)
        with pytest.raises(TypeError):
            result.outputs["logits"] = np.zeros((1, 3), dtype=np.float32)
        with pytest.raises(TypeError):
            del result.outputs["logits"]
    for name in OUTPUT_NAMES:
        np.testing.assert_array_equal(results[2].outputs[name], results[3].outputs[name])
        np.testing.assert_array_equal(results[2].outputs[name], results[5].outputs[name])


def test_convenience_forward_uses_same_public_contract(source, tmp_path):
    inputs = ForwardInputs([16, 0, 4], positions=[7, 7, 2], segments=[1, 0, 1],
                           attention_mask=[True, False, True])
    experiment = _experiment(tmp_path / "deployment", source[2], requests=1)
    result = experiment.forward(inputs, memory_bytes=MEMORY_BYTES)
    _assert_outputs(result, _expected(source, inputs), 3)


@pytest.mark.parametrize("field,bad", [
    ("tokens", [-1]), ("tokens", [17]),
    ("positions", [-1]), ("positions", [8]),
    ("segments", [-1]), ("segments", [2]),
])
def test_lookup_domains_not_classifier_dimension(source, tmp_path, field, bad):
    # Vocabulary 17, positions 8, segments 2, classifier 3 are independent domains.
    experiment = _experiment(tmp_path / "deployment", source[2], requests=2)
    good = ForwardInputs([16], positions=[7], segments=[1])
    values = {"tokens": [16], "positions": [7], "segments": [1]}
    values[field] = bad
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        _assert_outputs(model.forward(good), _expected(source, good), 1)
        with pytest.raises(PUBLIC_ERRORS) as error:
            model.forward(ForwardInputs(**values))
        assert str(error.value).strip()


@pytest.mark.parametrize("variant", [
    "model_type", "architecture", "decoder", "cross_attention", "relative",
    "activation", "wrong_heads", "wrong_classifier", "quantized_dtype",
    "missing_weight", "wrong_weight_shape",
])
def test_unsupported_source_is_refused(tmp_path, variant):
    config, tensors = tiny_checkpoint()
    changes = {
        "model_type": ("model_type", "roberta"),
        "architecture": ("architectures", ["BertForTokenClassification"]),
        "decoder": ("is_decoder", True),
        "cross_attention": ("add_cross_attention", True),
        "relative": ("position_embedding_type", "relative_key"),
        "activation": ("hidden_act", "gelu_new"),
        "wrong_heads": ("num_attention_heads", 3),
    }
    if variant in changes:
        key, value = changes[variant]
        config[key] = value
    elif variant == "wrong_classifier":
        tensors["classifier.weight"] = np.zeros((17, 8), dtype=np.float32)
        tensors["classifier.bias"] = np.zeros(17, dtype=np.float32)
    elif variant == "quantized_dtype":
        tensors["bert.encoder.layer.0.attention.self.query.weight"] = np.zeros(
            (8, 8), dtype=np.int8,
        )
    elif variant == "missing_weight":
        del tensors["bert.encoder.layer.0.attention.self.query.weight"]
    elif variant == "wrong_weight_shape":
        tensors["bert.encoder.layer.0.intermediate.dense.weight"] = np.zeros(
            (8, 13), dtype=np.float32,
        )
    checkpoint = write_checkpoint(tmp_path / "checkpoint", config, tensors)
    experiment = _experiment(tmp_path / "deployment", checkpoint, requests=1)
    with pytest.raises(PUBLIC_ERRORS) as error:
        with experiment.compile_forward(memory_bytes=MEMORY_BYTES):
            pytest.fail("unsupported source reached a compiled forward owner")
    assert str(error.value).strip()


class _CoercionBomb:
    def __array__(self, *args, **kwargs):
        raise AssertionError("input array coercion ran before refusal")

    def __int__(self):
        raise AssertionError("input integer coercion ran before refusal")

    def __index__(self):
        raise AssertionError("input index coercion ran before refusal")


class _ListBomb(list):
    def __iter__(self):
        raise AssertionError("custom list iteration ran before refusal")

    def __len__(self):
        raise AssertionError("custom list length ran before refusal")

    def __array__(self, *args, **kwargs):
        raise AssertionError("custom list conversion ran before refusal")


class _ArrayBomb(np.ndarray):
    def __array__(self, *args, **kwargs):
        raise AssertionError("custom array conversion ran before refusal")

    def __getitem__(self, key):
        raise AssertionError("custom array indexing ran before refusal")


def _invalid_inputs():
    return [
        ForwardInputs([]),
        ForwardInputs([[1], [2, 3]]),
        ForwardInputs([1] * 7),
        ForwardInputs(np.ones((2, 2), dtype=np.int64)),
        ForwardInputs(np.ones((1, 1, 1), dtype=np.int64)),
        ForwardInputs(np.array([1], dtype=np.int32)),
        ForwardInputs(np.array([1.0], dtype=np.float32)),
        ForwardInputs(np.array([True], dtype=np.bool_)),
        ForwardInputs(np.array([1], dtype=object)),
        ForwardInputs([True]),
        ForwardInputs([1.0]),
        ForwardInputs([1 << 64]),
        ForwardInputs([_CoercionBomb()]),
        ForwardInputs(_CoercionBomb()),
        ForwardInputs(_ListBomb([1])),
        ForwardInputs(np.array([1], dtype=np.int64).view(_ArrayBomb)),
        ForwardInputs([1, 2], positions=[0]),
        ForwardInputs([1, 2], segments=[0]),
        ForwardInputs([1], positions=_CoercionBomb()),
        ForwardInputs([1], segments=_ListBomb([0])),
        ForwardInputs([1], attention_mask=_CoercionBomb()),
        ForwardInputs([1], attention_mask=[False]),
        ForwardInputs([1], attention_mask=[2]),
        ForwardInputs([1], attention_mask=np.array([2], dtype=np.int64)),
        ForwardInputs([1], attention_mask=np.array([0.5], dtype=np.float32)),
    ]


@pytest.mark.parametrize("case", range(25))
def test_malformed_request_refused_before_numpy_conversion(source, tmp_path, monkeypatch, case):
    inputs = _invalid_inputs()[case]
    experiment = _experiment(tmp_path / "deployment", source[2], requests=1)
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        # Real compiled owner; patch conversion only after source compilation.
        calls = []
        activation_calls = []

        def forbidden_conversion(*args, **kwargs):
            calls.append(True)
            raise AssertionError("NumPy conversion ran before input refusal")

        with monkeypatch.context() as patch:
            _observe_native_activations(patch, activation_calls)
            patch.setattr(np, "array", forbidden_conversion)
            patch.setattr(np, "asarray", forbidden_conversion)
            with pytest.raises(PUBLIC_ERRORS) as error:
                model.forward(inputs)
            assert str(error.value).strip()
            assert not calls
            # Proves no activation dispatch, not absence of earlier lookup/linear
            # work. Conversion refusal remains a separate observed guarantee.
            assert not activation_calls


def test_request_budget_is_bounded(source, tmp_path):
    experiment = _experiment(tmp_path / "deployment", source[2], requests=1)
    inputs = ForwardInputs([16])
    with experiment.compile_forward(memory_bytes=MEMORY_BYTES) as model:
        _assert_outputs(model.forward(inputs), _expected(source, inputs), 1)
        with pytest.raises(PUBLIC_ERRORS) as error:
            model.forward(inputs)
        assert str(error.value).strip()

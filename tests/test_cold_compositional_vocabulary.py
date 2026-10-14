"""Independent tiny scalar oracles and negative controls."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/probe_cold_compositional_vocabulary.py"
SPEC = importlib.util.spec_from_file_location("cold_compositional", PATH)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def test_independent_tied_lookup_and_complete_head():
    books = np.array([[[1, 2], [3, 4]], [[-1, 1], [2, -2]]], np.float16)
    codes = np.array([[0, 1], [1, 0], [0, 0]], np.uint8)
    scales = np.array([0.5, 2, 1], np.float32)
    model = p.Vocabulary(books, codes, scales)
    expected = np.array([[1.5, 0], [4, 10], [0, 3]], np.float32)
    np.testing.assert_array_equal(model.lookup(np.array([2, 0, 1])), expected[[2, 0, 1]])
    x = np.array([[2, -1], [-2, 3]], np.float32)
    oracle = np.array([[sum(float(a) * float(b) for a, b in zip(row, token)) for token in expected] for row in x], np.float32)
    np.testing.assert_array_equal(model.logits(x), oracle)
    assert model.logits(x).shape == (2, 3)


def test_fit_improves_zero_baseline_and_is_reproducible():
    w = np.random.default_rng(17).normal(size=(32, 6)).astype(np.float32)
    a, b = p.fit_additive(w), p.fit_additive(w)
    np.testing.assert_array_equal(a.books, b.books)
    reconstructed = p.independent_dense(a)
    assert np.mean((reconstructed - w) ** 2) < np.mean(w ** 2)
    np.testing.assert_array_equal(a.lookup(np.arange(32)), reconstructed)


def test_float_order_is_not_dense_equivalence():
    report = p.run()
    assert report["structured"]["factor_bit_mismatches"] == 0
    assert report["random_float_order"]["factor_bit_mismatches"] > 0
    assert report["cancellation"]["factor_bit_mismatches"] == 1
    assert report["adversarial"]["source_argmax_mismatches"] == 1
    assert report["requantization"]["round_each_then_sum"] != report["requantization"]["sum_then_round"]


@pytest.mark.parametrize("kind", ["index", "nan", "scale", "geometry", "dtype"])
def test_bad_public_models_reject(kind):
    b = np.zeros((1, 2, 3), np.float16)
    i = np.zeros((2, 1), np.uint8)
    s = np.ones(2, np.float32)
    if kind == "index":
        i[0, 0] = 2
    elif kind == "nan":
        b[0, 0, 0] = np.nan
    elif kind == "scale":
        s[0] = 0
    elif kind == "geometry":
        b = np.zeros((1, 2, 129), np.float16)
    else:
        i = i.astype(np.int32)
    with pytest.raises(ValueError):
        p.Vocabulary(b, i, s)


def test_private_inputs_and_fit_admission():
    model = p.Vocabulary(np.ones((1, 1, 2), np.float16), np.zeros((2, 1), np.uint8), np.ones(2, np.float32))
    for ids in [np.array([-1]), np.array([2]), np.array([0.5])]:
        with pytest.raises(ValueError):
            model.lookup(ids)
    with pytest.raises(ValueError):
        model.logits(np.ones((33, 2), np.float32))
    with pytest.raises(ValueError):
        p.fit_additive(np.zeros((129, 2), np.float32))


def test_artifact_bill_and_qwen_arithmetic():
    report = p.run()
    a = report["structured"]["artifact"]
    assert a["raw_bytes"] == 4 * 8 * 12 * 2 + 96 * 4 + 96 * 4
    assert a["charged_artifact_bytes"] == a["frame_bytes"] + a["descriptor_bytes"]
    q = report["qwen_structural_not_wire"]
    assert q["raw_total_bytes"] == 4 * 256 * 896 * 2 + 151936 * 8
    assert q["dense_i8_entries_bytes"] == 151936 * 896
    assert q["dense_i8_plus_f32_scale_bytes"] == 151936 * 900


def test_rounding_and_clipping_do_not_distribute():
    assert np.rint(np.float32(0.5)) + np.rint(np.float32(0.5)) == 0
    assert np.rint(np.float32(0.5) + np.float32(0.5)) == 1
    assert np.clip(100 + 100, -127, 127) == 127
    assert np.clip(100, -127, 127) + np.clip(100, -127, 127) == 200

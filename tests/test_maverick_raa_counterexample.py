"""Ordinary exact witness tests; authored, not yet executed."""

from fractions import Fraction
import importlib.util
from itertools import combinations
from math import comb
from pathlib import Path

import pytest


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1] / "examples/maverick_raa_counterexample.py"
    spec = importlib.util.spec_from_file_location("maverick_raa_counterexample_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_primary_weight_two_witness(experiment):
    assert (experiment.FIELD, experiment.DIMENSION, experiment.REPEAT, experiment.LENGTH) == (
        2_013_265_921, 64, 8, 512
    )
    message, pi1, pi2, scales1, scales2 = experiment.primary_witness()
    assert message == [0] * 63 + [1]
    assert len(message) * experiment.REPEAT == 512
    assert pi1 == list(range(512))
    first_support = [504, 506, 508, 510]
    assert pi2 == list(range(504)) + [505, 507, 509, 511] + first_support
    for permutation in (pi1, pi2):
        assert all(type(index) is int for index in permutation)
        assert sorted(permutation) == list(range(512))
    for scales, negative_positions in (
        (scales1, {505, 507, 509, 511}), (scales2, {509, 511})
    ):
        assert len(scales) == 512
        assert all(type(value) is int and 0 < value < experiment.FIELD for value in scales)
        assert scales == [
            experiment.FIELD - 1 if index in negative_positions else 1
            for index in range(512)
        ]

    repeated = [value for value in message for _ in range(8)]
    assert repeated == [0] * 504 + [1] * 8
    first_input = [repeated[pi1[j]] * scales1[j] % experiment.FIELD for j in range(512)]
    first = experiment.prefix_sums(first_input, experiment.FIELD)
    assert [index for index, value in enumerate(first) if value] == first_support
    assert first == [int(index in first_support) for index in range(512)]
    second_input = [first[pi2[j]] * scales2[j] % experiment.FIELD for j in range(512)]
    assert second_input == [0] * 508 + [1, experiment.FIELD - 1, 1, experiment.FIELD - 1]
    expected = [0] * 512
    expected[508] = expected[510] = 1
    encoded = experiment.encode_literal(message, 8, experiment.FIELD, pi1, pi2, scales1, scales2)
    assert experiment.prefix_sums(second_input, experiment.FIELD) == expected
    assert encoded == expected
    assert [index for index, value in enumerate(encoded) if value] == [508, 510]
    assert [encoded[508], encoded[510]] == [1, 1]
    assert sum(value != 0 for value in encoded) == 2


def test_exact_conditional_support_avoidance(experiment):
    marked = {1, 6}
    supports = list(combinations(range(8), 3))
    avoiding = sum(marked.isdisjoint(support) for support in supports)
    enumerated = Fraction(avoiding, len(supports)) ** 2
    assert experiment.miss_bound(8, 2, 3, 2) == enumerated
    primary = experiment.miss_bound(512, 2, 31, 2)
    assert primary == Fraction(comb(510, 31), comb(512, 31)) ** 2
    assert primary == Fraction(481 * 480, 512 * 511) ** 2
    assert primary > Fraction(3, 4)
    assert experiment.miss_bound(8, 6, 3, 2) == Fraction(0)
    assert experiment.miss_bound(8, 0, 3, 2) == Fraction(1)
    for arguments in ((True, 0, 0, 1), (8, False, 3, 2), (8, 2, True, 2), (8, 2, 3, False)):
        with pytest.raises(TypeError):
            experiment.miss_bound(*arguments)
    for arguments in ((-1, 0, 0, 1), (8, 9, 3, 2), (8, 2, 9, 2), (8, 2, 3, 0)):
        with pytest.raises(ValueError):
            experiment.miss_bound(*arguments)
    with pytest.raises(TypeError):
        experiment.prefix_sums([True], 7)
    with pytest.raises(TypeError):
        experiment.encode_literal([1], True, 7, [0], [0], [1], [1])

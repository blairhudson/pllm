import pytest

from pllm.runtime.benchmark_context import context_bytes, validate_context


def test_message_context_is_copied_and_has_canonical_cohort_bytes():
    value = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    normalized = validate_context(value)
    assert normalized == value and normalized is not value
    assert normalized[0] is not value[0]
    assert context_bytes(value) == context_bytes([dict(reversed(list(row.items()))) for row in value])
    value[0]["content"] = "Changed"
    assert normalized[0]["content"] == "Hi"


@pytest.mark.parametrize("value", [[], " ", 42, ["x"], [{"role": [], "content": "x"}],
    [{"role": "tool", "content": "x"}], [{"role": "user", "content": ""}],
    [{"role": "user", "content": "x", "extra": True}], [{"role": "user", "content": "😀" * 5000}]])
def test_context_rejects_unsupported_or_unbounded_inputs(value):
    with pytest.raises(ValueError):
        validate_context(value)

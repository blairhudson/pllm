from __future__ import annotations

import numpy as np
import pytest

from pllm.runtime.greedy_verification_reference import verify_greedy


class Recurrence:
    def __init__(self):
        self.history = []
        self.attempts = 0

    @property
    def position(self):
        return len(self.history)

    def scores(self):
        selected = sum((index + 1) * token for index, token in enumerate(self.history)) % 7
        scores = np.zeros(7, np.float32)
        scores[selected] = 1
        return scores

    def prepare_ids(self, tokens):
        self.history = list(tokens)
        return list(tokens), self.scores(), []

    def forward_ids(self, tokens):
        assert len(tokens) == 1
        self.attempts += 1
        self.history.extend(tokens)
        return self.scores()[None]

    def snapshot(self):
        return list(self.history)

    def restore(self, state):
        self.history = list(state)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("wrong", [False, True])
def test_rejected_kv_rolls_back_without_refunding_attempts(width, wrong):
    prompt = [2, 3, 1]
    expected, history = [], list(prompt)
    for _ in range(16):
        token = sum((index + 1) * value for index, value in enumerate(history)) % 7
        expected.append(token)
        history.append(token)

    def propose(prefix, cap):
        prefix = list(prefix)
        result = []
        for _ in range(cap):
            token = sum((index + 1) * value for index, value in enumerate(prefix)) % 7
            token = (token + int(wrong)) % 7
            result.append(token)
            prefix.append(token)
        return result

    runtime = Recurrence()
    result = verify_greedy(runtime, prompt, propose, outputs=16, width=width)
    assert result["tokens"] == expected
    assert runtime.history == prompt + expected[:-1]
    assert runtime.attempts == result["target_decode_rows"]
    assert runtime.attempts == 15 + result["rejected_target_rows"]
    if wrong:
        assert result["draft_accepted"] == 0
        assert runtime.attempts > 15


@pytest.mark.parametrize("draft", [[], [True], [-1], [1, 2, 3], (1,)])
def test_malformed_draft_fails_before_target_attempt(draft):
    runtime = Recurrence()
    with pytest.raises(ValueError):
        verify_greedy(runtime, [1, 2], lambda _, __: draft, outputs=4, width=2)
    assert runtime.attempts == 0


def test_one_output_needs_no_draft_or_decode():
    runtime = Recurrence()
    result = verify_greedy(
        runtime, [2], lambda _, __: pytest.fail("draft not needed"), outputs=1, width=2
    )
    assert result["tokens"] == [2]
    assert runtime.position == 1 and runtime.attempts == 0


@pytest.mark.parametrize("prompt", [[], [True], [-1], [1] * 257])
def test_invalid_prompt_rejected_before_prefill(prompt):
    runtime = Recurrence()
    with pytest.raises(ValueError):
        verify_greedy(runtime, prompt, lambda _, __: [1], outputs=2, width=2)
    assert runtime.position == runtime.attempts == 0

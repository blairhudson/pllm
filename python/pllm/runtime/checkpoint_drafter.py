"""Client-local existing-checkpoint drafting for the exact greedy research verifier.

Different vocabularies bridge through private local text, with exact target-prefix
round trips. This is not distribution-preserving stochastic speculation or an SDK
provider protocol. Target verification alone determines every emitted token.
"""
from __future__ import annotations

from collections import Counter
import secrets
import time

import numpy as np
from tokenizers import Tokenizer

from .greedy_verification_reference import _truncate_response_prefix


def _decode(tokenizer, tokens):
    return (tokenizer.decode(tokens, skip_special_tokens=False)
            if isinstance(tokenizer, Tokenizer) else tokenizer.decode(tokens))


def _encode(tokenizer, text):
    return (tokenizer.encode(text, add_special_tokens=False).ids
            if isinstance(tokenizer, Tokenizer) else tokenizer.encode(text))


def target_suffix(tokenizer, history, suffix, cap):
    """A changed BPE boundary, lossy Unicode or special-token mapping is no draft."""
    prefix = _decode(tokenizer, history)
    if _encode(tokenizer, prefix) != history:
        return []
    encoded = _encode(tokenizer, prefix + suffix)
    return encoded[len(history):len(history) + cap] if encoded[:len(history)] == history else []


class CheckpointDrafter:
    def __init__(self, runtime, contract, target_tokenizer, prompt_ids, rendered_draft_prompt):
        from .semantic_executor import SemanticDecoderRuntime
        if (type(runtime) is not SemanticDecoderRuntime or runtime._causal_reduction != "prefix_f32"
                or runtime._retained is not None or runtime._tiered is not None
                or runtime.nonlinear_evaluator is not None or runtime._verification_bits
                or hasattr(runtime.remote, "inventory") or runtime.position):
            raise ValueError("checkpoint drafting requires fresh canonical clear full-KV execution")
        if (not isinstance(prompt_ids, list) or not 1 <= len(prompt_ids) <= 256
                or any(type(t) is not int or t < 0 for t in prompt_ids)
                or not isinstance(rendered_draft_prompt, str) or not 1 <= len(rendered_draft_prompt) <= 16384):
            raise ValueError("invalid checkpoint draft prompt boundary")
        runtime.install_continuation(contract)
        self.runtime, self.contract = runtime, contract
        self.target_tokenizer = getattr(target_tokenizer, "inner", target_tokenizer)
        self.draft_tokenizer = getattr(runtime.tokenizer, "inner", runtime.tokenizer)
        self.prompt_ids, self.rendered = list(prompt_ids), rendered_draft_prompt
        self.executed, self.logits = [], None
        self.owner = secrets.token_hex(16)
        self.counts = Counter()
        self.closed = False

    def _align(self, tokens):
        runtime = self.runtime
        if not 1 <= len(tokens) <= min(256, runtime._graphs["prefill"]["query_sequence"]):
            raise ValueError("drafter prefix exceeds admitted geometry")
        common = 0
        for left, right in zip(tokens, self.executed):
            if left != right:
                break
            common += 1
        if common == len(tokens) == len(self.executed):
            return
        # Recompute the last kept token so no stale pre-rollback logits survive.
        keep = min(common, len(tokens) - 1)
        if keep == 0:
            _, self.logits, _ = runtime.prepare_ids(tokens)
            runtime.restore_for_response(runtime.snapshot_for_response(self.owner), self.owner)
            self.counts["prefill_rows"] += len(tokens)
        else:
            self.counts["erased_rows"] += len(self.executed) - keep
            _truncate_response_prefix(runtime, keep, self.owner)
            self.logits = runtime.continue_ids(tokens[keep:])[-1]
            self.counts["teacher_forced_rows"] += len(tokens) - keep
            self.counts["reused_prefix_rows"] += keep
        self.executed = list(tokens)

    def __call__(self, history, cap):
        if self.closed:
            raise ValueError("checkpoint drafter is closed")
        start = time.process_time()
        try:
            if (type(cap) is not int or not 1 <= cap <= 16 or not isinstance(history, list)
                    or not len(self.prompt_ids) <= len(history) <= 288
                    or history[:len(self.prompt_ids)] != self.prompt_ids
                    or any(type(t) is not int or t < 0 for t in history)):
                raise ValueError("draft target history or width differs")
            self.counts["calls"] += 1
            suffix = _decode(self.target_tokenizer, history[len(self.prompt_ids):])
            self._align(self.runtime.encode_prompt(self.rendered + suffix))
            offered = []
            for _ in range(min(16, 2 * cap + 4)):
                token = int(np.argmax(self.logits))
                offered.append(token)
                self.logits = self.runtime.forward_ids([token])[-1]
                self.executed.append(token)
                self.counts["generated_rows"] += 1
                text = _decode(self.draft_tokenizer, offered)
                proposals = target_suffix(self.target_tokenizer, history, text, cap)
                if len(proposals) >= cap:
                    return proposals
            if proposals:
                return proposals
            self.counts["boundary_fallbacks"] += 1
            return [history[-1]]
        except BaseException:
            self.close()
            raise
        finally:
            self.counts["cpu_seconds"] += time.process_time() - start

    def close(self):
        if not self.closed:
            self.runtime.reset()
            self.executed.clear()
            if self.logits is not None:
                self.logits.fill(0)
                self.logits = None
            self.closed = True

"""Client-local greedy draft verification with exact KV rollback.

Research controller only. Callbacks consume attempts even when state rolls back;
this is not a distributed reservation ledger or a batched decode compiler plan.
"""

from __future__ import annotations

import numpy as np


def verify_greedy(runtime, prompt: list[int], propose, *, outputs: int, width: int) -> dict:
    if (
        type(outputs) is not int
        or not 1 <= outputs <= 32
        or type(width) is not int
        or not 1 <= width <= 8
    ):
        raise ValueError("bounded greedy verification requires 1..32 outputs and 1..8 draft width")
    if (
        not isinstance(prompt, list)
        or not 1 <= len(prompt) <= 256
        or any(type(token) is not int or token < 0 for token in prompt)
    ):
        raise ValueError("verification requires a nonempty bounded integer prompt")
    history, logits, _ = runtime.prepare_ids(prompt)
    history = list(history)
    generated = []
    pending = None
    offered = accepted_total = wasted = target_rows = windows = 0
    while len(generated) < outputs:
        if pending is not None:
            logits = runtime.forward_ids([pending])[-1]
            history.append(pending)
            target_rows += 1
            pending = None
        remaining = outputs - len(generated)
        if remaining == 1:
            generated.append(int(np.argmax(logits)))
            break
        proposals = propose(history, min(width, remaining - 1))
        if (
            not isinstance(proposals, list)
            or not 1 <= len(proposals) <= min(width, remaining - 1)
            or any(type(token) is not int or token < 0 for token in proposals)
        ):
            raise ValueError("draft must return a bounded nonempty token vector")
        windows += 1
        offered += len(proposals)
        snapshots, scores = [runtime.snapshot()], [logits.copy()]
        for token in proposals:
            scores.append(runtime.forward_ids([token])[-1].copy())
            snapshots.append(runtime.snapshot())
            target_rows += 1
        accepted = 0
        for index, token in enumerate(proposals):
            if token != int(np.argmax(scores[index])):
                break
            accepted += 1
        # Restore only KV; charged target attempts never roll back.
        runtime.restore(snapshots[accepted])
        accepted_total += accepted
        wasted += len(proposals) - accepted
        history.extend(proposals[:accepted])
        pending = int(np.argmax(scores[accepted]))
        generated.extend(proposals[:accepted] + [pending])
        logits = scores[accepted]
    if runtime.position != len(prompt) + outputs - 1:
        raise AssertionError("verification retained rejected or final-unconsumed KV")
    return {
        "tokens": generated,
        "draft_offered": offered,
        "draft_accepted": accepted_total,
        "rejected_target_rows": wasted,
        "target_decode_rows": target_rows,
        "verification_windows": windows,
        "retained_position": runtime.position,
    }

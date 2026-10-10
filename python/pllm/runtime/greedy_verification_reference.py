"""Client-local greedy draft verification with exact KV rollback.

Research controllers only. Callbacks consume attempts even when state rolls back;
neither controller provides a distributed reservation ledger. The batched screen
uses a checked continuation plus explicitly charged client-head observations.
"""

from __future__ import annotations

import numpy as np


def _truncate_response_prefix(runtime, kept, owner):
    """Erase rejected clear research KV without refunding any attempted work."""
    import hashlib
    from .transformer_client import _snapshot_state_basis
    basis = runtime._snapshot_basis
    if (basis is None or not basis.valid() or basis.position != runtime.position
            or basis.owner_response_id != owner or not 1 <= kept <= runtime.position):
        raise ValueError("batched response ownership changed during verification")
    for cache in runtime.caches:
        cache.key[kept:cache.length].fill(0)
        cache.value[kept:cache.length].fill(0)
        cache.length = kept
    runtime.position = kept
    execution = hashlib.sha256(b"pllm.research.greedy-prefix.v1\0"
        + basis.execution_digest.encode() + kept.to_bytes(8, "big")).hexdigest()
    runtime._snapshot_basis = _snapshot_state_basis("incremental", kept,
        basis.source_binding_digest, basis.native_phase_digest, execution, owner, basis.prefill_extent)


def verify_batched_greedy(runtime, contract, prompt, propose, *, outputs, width,
                          on_output=None):
    """Bounded clear-kernel screen for exact blocks under canonical prefix_f32.

    This research controller owns response-only KV, erases rejected rows and
    charges every target row. It neither reserves provider inventory nor enables
    speculative SDK serving. A live protocol must separately price/admit its
    worst-case attempt budget and provide a real, independently costed drafter.
    """
    import hashlib
    import secrets
    from .semantic_executor import SemanticDecoderRuntime

    if (type(outputs) is not int or not 1 <= outputs <= 32
            or type(width) is not int or not 1 <= width <= 16):
        raise ValueError("bounded greedy verification requires 1..32 outputs and 1..16 draft width")
    if (type(runtime) is not SemanticDecoderRuntime or runtime._causal_reduction != "prefix_f32"
            or runtime._verification_bits or runtime.nonlinear_evaluator is not None
             or runtime._retained is not None or runtime._tiered is not None
             or hasattr(runtime.remote, "inventory")):
        raise ValueError("batched research verification requires canonical clear full-KV execution")
    if (not isinstance(prompt, list) or not 1 <= len(prompt) <= 256
            or any(type(token) is not int or not 0 <= token < runtime.cfg["vocab_size"]
                   for token in prompt)):
        raise ValueError("verification requires a nonempty bounded integer prompt")
    runtime.install_continuation(contract)
    document = contract.to_dict()
    heads = [op for op in document["graph"]["operations"] if op["operator"] == "output_head"]
    if len(heads) != 1 or len(heads[0]["inputs"]) != 1 or "_local" in runtime.__dict__:
        raise ValueError("research blocks require one ordinary client-owned head")
    head = heads[0]
    last = next(op for op in document["graph"]["operations"] if op["id"] == head["inputs"][0])
    stage_id = runtime._stages[f"prefill:{head['id']}"]
    stage = runtime.bundle.stages[stage_id]
    if (last["operator"] != "last_token" or stage.client_weight is None
            or head["attributes"].get("output_dtype") not in {None, "model_native"}):
        raise ValueError("research blocks require an ordinary final-row client head")

    def target_block(block):
        # The ordinary continuation graph exposes only the final head row. This
        # explicit clear research observer evaluates the same committed local
        # head on earlier hidden rows; no native/provider tensor schedule is
        # silently changed. Price those extra heads and their allocation here.
        extra_bytes = 4 * len(block) * (3 * stage.out_features + stage.in_features)
        if extra_bytes > 32 << 20:
            raise ValueError("research block head workspace exceeds 32 MiB")
        contract.admit(runtime.position, len(block), memory_bytes=(2 << 30) - extra_bytes)
        local = runtime._local
        earlier = []

        def observe(operation, values, state_kinds, pending_keys):
            if operation["id"] == last["id"]:
                hidden = np.asarray(values[operation["inputs"][0]])
                if hidden.shape != (len(block), stage.in_features) or hidden.dtype != np.float32:
                    raise ValueError("research block hidden geometry mismatch")
                if len(block) > 1:
                    earlier.append(runtime.bundle.local_linear(stage_id, hidden[:-1]))
            return local(operation, values, state_kinds, pending_keys)

        runtime._local = observe
        try:
            final = runtime.continue_ids(block)
            return np.concatenate([*earlier, final], axis=0) if earlier else final
        finally:
            runtime.__dict__.pop("_local", None)
    if outputs > 1:
        contract.admit(len(prompt), outputs - 1)
    history, logits, _ = runtime.prepare_ids(prompt)
    history = list(history)
    owner = secrets.token_hex(16)
    runtime.restore_for_response(runtime.snapshot_for_response(owner), owner)
    generated, pending = [], []
    offered = accepted_total = wasted = target_rows = windows = batches = 0
    maximum_queried_position = runtime.position

    def emit(token, scores):
        generated.append(token)
        if on_output is not None:
            on_output(scores.copy())

    try:
        # The prefill already determines the first output. Emit it immediately;
        # do not evaluate a speculative block to rediscover an available score.
        first = int(np.argmax(logits))
        emit(first, logits)
        history.append(first)
        pending = [first]
        while len(generated) < outputs:
            remaining = outputs - len(generated)
            proposals = [] if remaining == 1 else propose(history, min(width, remaining - 1))
            if remaining > 1 and (
                not isinstance(proposals, list)
                or not 1 <= len(proposals) <= min(width, remaining - 1)
                or any(type(token) is not int or not 0 <= token < runtime.cfg["vocab_size"]
                       for token in proposals)
            ):
                raise ValueError("draft must return a bounded nonempty token vector")
            start = runtime.position
            block = pending + proposals
            scores = [logits]
            if block:
                rows = target_block(block)
                if rows.shape != (len(block), runtime.cfg["vocab_size"]):
                    raise ValueError("batched target did not return every row's logits")
                scores = list(rows) if pending else [logits, *rows]
                batches += 1
                target_rows += len(block)
                maximum_queried_position = max(maximum_queried_position, runtime.position)
            accepted = 0
            for index, token in enumerate(proposals):
                if token != int(np.argmax(scores[index])):
                    break
                accepted += 1
                emit(token, scores[index])
            correction = int(np.argmax(scores[accepted]))
            emit(correction, scores[accepted])
            kept = start + len(pending) + accepted
            # Causal prefix_f32 proves later proposal rows cannot alter earlier
            # rows. Keep this cut response-owned; never qualify it for a fresh
            # prefill cache. Material/callback attempt counters are not rewound.
            _truncate_response_prefix(runtime, kept, owner)
            windows += bool(proposals)
            offered += len(proposals)
            accepted_total += accepted
            wasted += len(proposals) - accepted
            history.extend(proposals[:accepted] + [correction])
            pending = [correction]
            logits = scores[accepted]
        if runtime.position != len(prompt) + outputs - 1:
            raise AssertionError("verification retained rejected or final-unconsumed KV")
        return {
            "tokens": generated, "draft_offered": offered, "draft_accepted": accepted_total,
            "rejected_target_rows": wasted, "target_decode_rows": target_rows,
            "verification_windows": windows, "target_batches": batches,
            "retained_position": runtime.position,
            "maximum_queried_position": maximum_queried_position,
            "extra_client_head_rows": target_rows - batches,
            "execution_contract": hashlib.sha256(
                b"pllm.research.batched-greedy.v1\0" + contract.canonical_bytes()
                + width.to_bytes(1, "big")
            ).hexdigest(),
        }
    except BaseException:
        runtime.reset()
        runtime._continuation_failed = True
        raise


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

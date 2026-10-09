# Bounded SDK conversations and native continuation admission

`OpenAI.conversation` and `AsyncOpenAI.conversation` own one response continuation
chain through an ordinary prepared Experiment with bounded `ClientPrefixReuse`.
They preserve the existing native full-KV contract, numeric choices and one-use
inventory. The pending sampled token remains part of the next continuation.

Each committed turn retires the prior response, history and KV snapshot. A failed
turn preserves the prior owner and retires its attempted response. Async
cancellation drains the worker and retires the chain, including a concurrently
committed turn. Closing the underlying client also releases its response and
model-state owners.

Before reservation, the compiled maximum state geometry plus one next-token
logit vector must fit `max_state_bytes`, selected from 1 byte to 256 MiB. This
bounds the retained numeric payload. It does not bound process RSS, history text,
active execution, old/new snapshot overlap, tokenizer storage or the separately
bounded prefix cache. Generated Qwen2/Qwen3 tests compare every retained logit
and KV word as uint32 bits against explicit response continuation; failure,
reserved-row burning and async cancellation have separate controls.

The installed-wheel example uses only public APIs and ordinary separate prepared
role children:

```sh
python examples/conversation.py --root .pllm/conversation-example
```

## Validate public geometry once

Previously each `DecoderContinuationSchedule.admit` reproduced and compared the
complete source/composition/schedule contract. The new immutable Rust handle
performs that validation once, then retains only three unsigned resource limits.
Every admission still checks overflow, token bounds and the same workspace limit.
It holds no weights, tokens, KV, provider acknowledgement or one-use material.
Pickle reconstruction revalidates the public metadata rather than serializing a
native handle. Changed source, numeric configuration or contract bytes reject.

Runtime creation also reuses the schedule produced during its ordinary binding
validation, avoiding a second reconstruction. Bundle mutation checks remain on
that path.

A metadata-only pinned Qwen2.5-0.5B probe alternates six pairs of 32 admissions
under a 512-input/32-output contract. All returned resource-count digests match.
Median total process CPU is **1.169766 seconds** for repeated validation versus
**0.034027 seconds** for construction plus handle admissions, about **34.4×**
less for this isolated operation. The initial common plan/contract construction
is outside both timings. No tensor values or providers are loaded; this is not
a response-latency, client-RSS or thin-native-client result.

```sh
python scripts/probe_continuation_admission.py --output continuation-admission.json
```

Evidence: `docs/evidence/research-sdk-continuation-admission.json`, including
source, implementation and environment commitments. Tests:
`tests/test_conversation.py` and `tests/test_decoder_continuation.py`.

## SDK frontier scope

This lands bounded conversation ownership and a small native-client foundation.
The complete Python-free numeric executor and load-ready client capsule remain
larger work items. The subsequent [shared native snapshot slice](native-snapshot-reference.md)
removes the second paged weight-file copy while retaining actual-byte validation.
Existing paging, indexed tokenization and provider artifact
caching retain their own measured costs; they do not become a new capsule result.
The separate [batched greedy screen](batched-greedy-reference.md) checks exact
rollback and charges failed drafts. Larger protected regions retain the numeric,
cost and role gates documented in `nonlinear-decoders.md`; two-online-worker
regions do not replace offline Preparation plus one online Inference provider.

# Four incremental optimizations through SDK Experiments

## Executable choices

All four choices now resolve through the normal native composition validator,
compiled prepared execution, SDK, local role supervisor and ordinary
`pllm benchmark run` driver:

1. `pllm.tokenization.IndexedTokenizer`: trusted offline byte-level BPE
   compilation, source-bound index admission and client-local private queries.
2. `pllm.preparation.PreparedInventory(allocation="demand", refill="on-demand")`:
   per-stage correction rows from the native schedule, terminal demand and
   qualified prefix reuse, with one-use reservation and cancellation burns.
3. `pllm.preparation.ModelAwareCorrections(storage="paged")`: authenticated
   private raw-page weights for CPU Preparation, retaining bounded buffers
   instead of all stage weights in memory.
4. `pllm.state.PublicPrefixCapsule`: trusted-publisher public full-KV state,
   authenticated before cache import under the exact source, numeric and
   compiled workload contracts. Requires `prefix_f32` and `ClientPrefixReuse`.

The indexed tokenizer bounds input to 4,096 UTF-8 bytes, BPE pieces to 1,024
symbols and decoding to 4,096 IDs. Unsupported tokenization contracts reject.
Its SQLite snapshot is request-owned and closed on scope exit or failure,
including async worker calls. Neither lookup text nor private suffix state
leaves the client. Artifact SHA-256 commitments must come from the trusted
publisher, independently of serving providers; self-asserted state is not proof
of correct computation. Private response state is never implicitly published.

Paged Preparation rejects Metal and Freivalds compositions. Public-prefix and
indexed-tokenizer selections currently admit ordinary prepared execution only.
Existing default component identities and uniform inventory behavior are
preserved. The all-stage contract retains one burn-only row for any zero-demand
stage. Request-dependent row counts and prefix hits remain visible shape/access
patterns under the existing non-collusion and Preparation-erasure assumptions.

## Matched real-checkpoint benchmark

Evidence: [`../evidence/incremental-sdk-qwen25.json`](../evidence/incremental-sdk-qwen25.json).
Public compilation: [`../evidence/incremental-sdk-publisher.json`](../evidence/incremental-sdk-publisher.json).

- Source: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Body: `5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
- W8A8, `prefix_f32`, fixed 256-input-token plan, 150 actual input tokens and
  eight generated tokens, deterministic greedy selection. Public capsule: first
  96 already-public tokens from the fixed reproduction prompt.
- One cold client-bundle-cache response per candidate, zero warmups, one CLI
  invocation and cohort salt, two fresh co-located role children per candidate.
  Checkpoint files were already cached. The client process spans all candidates.
- Control already uses terminal pruning, stage-packed requests, row-residue
  outputs and paged/compressed/batched client artifacts. Preparation window is
  one: larger concurrent windows reject this row count under their 16 MiB gate.

All six candidates pass the canonical runtime checks and have the same source
lock, body, workload and output-text digest. Each emits eight outputs, zero
plaintext prompt/token-ID bytes and zero observed new swap. Tiny generated-model
regressions additionally compare final logits and every retained KV value through
synchronous and asynchronous role-backed execution.

All MB below are decimal. Covered bytes include recorded startup/response
application bodies once, excluding public artifact distribution and checkpoint
download. Peak RSS is the separate Preparation child's lifetime maximum.

| Candidate | Covered MB | Online MB | Stage rows issued | Preparation peak MB | Full request s | Cold aggregate CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 481.915095 | 219.202689 | 15,072 | 581.21 | 16.3894 | 38.5492 |
| Demand | 476.952830 | 219.202689 | 14,625 | 584.99 | 16.9098 | 38.9670 |
| Paged Preparation | 481.915095 | 219.202689 | 15,072 | 217.86 | 16.6228 | 39.2505 |
| Indexed tokenizer | 481.915095 | 219.202689 | 15,072 | 587.73 | 16.3162 | 37.4698 |
| Capsule | 266.912708 | 88.391607 | 5,856 | 532.61 | 11.1112 | 25.5147 |
| Combined | 266.912810 | 88.391607 | 5,856 | 164.51 | 11.7807 | 26.8287 |

Demand alone removes 447 unused terminal stage rows and 4.96 MB covered bodies
(1.03%), while online bodies are unchanged. Paged Preparation reduces its own
peak 2.67-fold in this pair, with 1.82% more cold aggregate CPU. Private snapshot
disk remains separate; admission prices 493.96 MB of snapshot/staging headroom.

The capsule removes public prefill work; decode-only remains exactly 1.461942
MB per generated decode output in every candidate. Capsule plus demand has no
additional row savings here: the continuation already uses all suffix rows, so
the combined authorization adds 102 body bytes. Improvements are not multiplied.

The combined response uses 44.61% fewer covered bodies (60.24 to 33.36 MB per
generated output) and 59.68% fewer online bodies. Its fixed public prefix makes
this workload-specific, not a general fresh-private-prompt saving. Adding one
full index and capsule payload gives 279.165773 MB, or 42.07% fewer bytes. That
is an explicitly priced payload addition, not observed full-wire traffic.

Public artifacts were pre-positioned. Index database: 9,281,536 bytes; index
contract: 3,800 bytes; capsule: 2,967,627 bytes. The publisher record's original
`tokenizer.bytes` field counts the database alone; the SDK local-read counter
includes the contract. The reproduction builder now reports both separately.
Public compilation including checkpoint load took 7.47 wall seconds and 8.43 CPU
seconds. That one-time CPU is outside response counters; charging it once to
the combined candidate gives 35.26 CPU seconds versus control's 38.55, before
unmeasured distribution work. No population latency or compute-cap conclusion
follows from these single samples.

Independent per-candidate client peaks require fresh processes; this sequential
cohort cannot establish a new client-memory reduction. The historical 4B
indexed-tokenizer and paged-Preparation probes remain separate evidence, not
measurements of this newly promoted composition. Full wire, upstream checkpoint
distribution, representative generation quality, WAN timing and independent
operators remain unmeasured here.

## Reproduce

From a checkout with the pinned checkpoint cached, choose a new artifact directory:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/build_incremental_sdk_experiments.py /absolute/new/public-artifacts
HF_HUB_OFFLINE=1 uv run pllm benchmark run \
  --experiment /absolute/new/public-artifacts/control.json \
  --experiment /absolute/new/public-artifacts/demand.json \
  --experiment /absolute/new/public-artifacts/paged.json \
  --experiment /absolute/new/public-artifacts/indexed.json \
  --experiment /absolute/new/public-artifacts/capsule.json \
  --experiment /absolute/new/public-artifacts/combined.json \
  --prompt-file /absolute/new/public-artifacts/prompt.txt \
  --max-output-tokens 8 --warmups 0 --repetitions 1 \
  --temperature 0 --capture-output-digest --backend native \
  --timeout 900 --output incremental-sdk.json
```

Artifacts bind absolute local paths into Experiment identity. Rebuild into a new
directory rather than copying one candidate's report as another's evidence.

Focused checks live in `tests/test_incremental_sdk.py`,
`tests/test_public_client_artifacts.py`, `tests/test_sdk_setup_reuse.py` and the
existing inventory, benchmark-memory, prepared transport and prefix-cache suites.

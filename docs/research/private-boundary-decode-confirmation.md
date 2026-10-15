# Independent boundary-head confirmation and longer-output costs

This document defines the confirmation measurement contract. Completed cohorts,
including the independent 150+8 result and the separate swap-rejected attempt,
are recorded in `docs/research/private-boundary-decode-followup.md`. The original
five-choice screen retains its own observations.

## Contract

Each invocation measures exactly one output workload and one independent private
cohort salt. Three arms: maintained prepared comparator, coupled serial/u32
control, and coupled parallel/uniform_residues candidate. Each arm gets three
fresh, serial, balanced clients: nine clients per cohort. The 150+8 confirmation
and generated-checkpoint smoke have completed; 150+32 and 150+64 remain separate
workloads. Never pool their observations, the
old screen, or failed/restarted attempts. The same recipe factory is used across
caps; only output budget and cold scenario allowance change.

Each cohort binds one captured installed build, source lock, prompt commitment,
full numeric body fingerprint, public request budget, original MessagePack
serializer, native shared 100/40 Mbps WAN caps, and exact ordinary SDK target.
Request count is one; input bound and maintained fixed-prefix-cache envelope
stay 150 (128 in tiny smoke). No public-prefix reuse. Source/checkpoint and
indexed tokenizer are already present. Their distribution is **not measured**.
Two non-colluding boundary workers change the role contract. TCP stream payload
is not physical-wire traffic; Internet RTT, independent operators, quality, and
global optimality are not established.

`private_boundary_decode_confirmation.py` composes the frozen screen's strict
at-cap raw-body and directed TCP helpers and the tier preflight. A bounded
workload-aware EOS raw-body path retains the same byte/stage/owner/TCP checks
without rewriting or sanitizing raw evidence. It requires issued = claimed +
burned + discarded, reserved = claimed + burned, sealed remaining rows = zero,
and exact reconstruction of post-close material accounting from startup/request
counters. Unused material stays charged. At cap, the original strict rule still
requires issued = reserved = claimed and burned = discarded = zero.
It deliberately does not call the old tier audit: that audit hardcodes eight
output tokens/selections and cannot safely audit natural EOS or longer caps.
New workload-aware checks enforce exact full logits, active KV, positions,
selected-token history, sampler RNG before/after, output digest, and source/build
identity across all nine clients. Cryptographic randomness remains unfrozen.
Missing, changed, mixed, duplicate, failed, or partial cells fail closed. Result
schema, observed build attributes/digest, environment, admission, oracle, all
report checks, request/sampler/cap, zero warmups and explicit metric contracts
are checked. Priority metric values are reconstructed from raw durations, CPU,
memory and byte observations; throughput requires the actual N−1 denominator.
Numeric/state/workload/output
comparison commitments must match across clients; privacy and measurement
contracts must match across replicates within each arm. Only the explicitly
different prepared versus boundary body-coverage scopes may differ by topology.

Raw startup/request ledgers are reconstructed, including every owner/phase and
directed body edge; stage evidence reconciles against raw protocol counters;
matching TCP windows independently bound every directed body total. Prepared
row/material work must match between the two boundary arms. Variable-width
MessagePack `server_ns` remains charged: independent stage-byte evidence permits
only the existing conservative metadata-width bound, never an invented exact
timing-byte subtraction. Bootstrap/control/metadata residuals remain charged.

Conservative topology admission and full-state audit scratch pricing remain
unchanged. Full fresh worker owns SDK, supervisor, dashboard, native relay, and
audit. Largest finalized whole-worker lifetime RSS across **all three**
replicates determines observed Low (256 MiB), Medium, and High eligibility—not
median or SDK-child RSS. Observed Low does not relax conservative admission.
Swap growth, a tripped memory guard, or incomplete owner coverage rejects.

## Natural EOS and actual work

Output length is a cap, never forced. In the SDK, EOS is sampled but not emitted:
at cap, selections = emitted outputs and decode steps = selections minus one;
with early EOS, selections = emitted outputs plus one. Head calls equal
selections; boundary lookup rows equal input tokens plus executed decode steps.
The summary retains actual output tokens, selections, executed decode steps,
preparation rows, prefill/decode/padding rows, claimed/burned/discarded/reserved
rows, and stage calls per replicate. One-token output has no N-1 decode TPS:
its null value remains unknown, with null delta/improvement/win and paired
deltas, never a zero or a fabricated win. Zero-output EOS is explicitly
unsupported by current SDK generation attribution before first output; audit
rejects with `confirmation_zero_output_telemetry_unsupported` and retains raw
attempt rather than inventing missing counters or claiming complete EOS support.

`long_output_cap_reached: false` means this cohort does **not** demonstrate a full
32- or 64-token output, even when exact-state audits succeed. Report its actual
work, not nominal cap or an extrapolated throughput/byte cost. Do not silently
change prompt, suppress EOS, or transplant observations from another cap.

The completed short cohorts prove eight emitted tokens only; they cannot
establish whether the existing prompt naturally terminates before 32 or 64.
Inspect each new summary. If early EOS prevents the requested long-output measurement,
retain this attempt and report the limitation; a different public prompt needs
an explicitly separate frozen prompt/workload identity and fresh cohort, not an
in-place rerun or pooled observation.

## Replay commands

Run from repository root. Use the installed environment; never add `python/` to
`PYTHONPATH`. Every command needs a fresh output and companion paths. Existing
output/plan/progress/summary files are never overwritten. Tiny model directories
must also be fresh. Shared `~/.cache/pllm/frontier-probe.lock` fails fast if any
other serialized frontier job runs. Keep every original attempt, including
rejections, interruptions, raw results, plans, and progress records.

Lightweight tests (synthetic evidence only):

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. tests/test_private_boundary_decode_confirmation.py
```

Metadata/admission-only production preflights (no tensor loading or timing):

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_decode_confirmation.py --max-output-tokens 8 --preflight-only --output target/private-boundary-confirmation-150x8-preflight-v1.json
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_decode_confirmation.py --max-output-tokens 32 --preflight-only --output target/private-boundary-confirmation-150x32-preflight-v1.json
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_decode_confirmation.py --max-output-tokens 64 --preflight-only --output target/private-boundary-confirmation-150x64-preflight-v1.json
```

Tiny 128+2 full-worker smoke (generated checkpoint, nine clients,
not production evidence). Tiny preflight uses generated-model metadata without
creating checkpoint tensors; actual smoke below creates a small Qwen2 fixture:

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_decode_confirmation.py --tiny --preflight-only --output target/private-boundary-confirmation-tiny-preflight-v1.json
```

Start with a tiny smoke, then run each real workload independently. Inspect each `.summary.json` and
`.progress.json` before proceeding; do not execute concurrently:

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_decode_confirmation.py --tiny --output target/private-boundary-confirmation-tiny-v1.json
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_decode_confirmation.py --max-output-tokens 32 --output target/private-boundary-confirmation-150x32-v1.json
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_decode_confirmation.py --max-output-tokens 64 --output target/private-boundary-confirmation-150x64-v1.json
```

Separate 8-token confirmation:

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_decode_confirmation.py --max-output-tokens 8 --output target/private-boundary-confirmation-150x8-v1.json
```

Defaults use `target/prepared-frontier-artifacts`. Each run has at most 3,600
seconds per fresh worker and the unchanged 8 GiB process admission envelope.
Do not weaken an admission rejection to obtain a result. Successful audit does
not promise timing wins: compare medians and same-replicate deltas against both
fresh controls, retaining whole-worker maximum RSS and actual executed work.

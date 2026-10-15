# Private-boundary decode timing attribution (opt-in research)

Status: the separate 18-client 150+8 cohort completed with exact state parity
and zero observed swap growth. Measurements and hash-bound replay are in
`docs/research/private-boundary-decode-followup.md`. This document defines the
instrumentation contract; installed SDK execution and ordinary SOTA evidence
retain their existing identities. Comparator subtraction does not measure a
timing bin or isolate probe overhead.

## Commands

Run from repository root with existing installed native build. Parent owns all
model/build/timing runs and must run them serially after review. Every output
prefix must be fresh; an existing or interrupted prefix is not resumed.

```sh
# Model-free tests and CLI inspection; no model/build jobs.
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. tests/test_private_boundary_decode_profile.py
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_decode.py --help

# Real-checkpoint admission only; cached artifacts/tokenizer required.
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_decode.py \
  --artifacts target/prepared-frontier-artifacts \
  --output target/private-boundary-profile-preflight.json --preflight-only

# Parent-only generated Qwen2 128+2 smoke: still starts real native roles.
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_decode.py \
  --output target/private-boundary-profile-tiny.json --tiny

# Parent-only real source-present 150+8 cohort, no other model jobs running.
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_decode.py \
  --artifacts target/prepared-frontier-artifacts \
  --output target/private-boundary-profile-qwen25.json
```

Preflight validates public input counts, installed build identity and native
topology/full-state audit memory admission. It can load tokenizer/source metadata;
it is not an inference run. Tiny `--preflight-only` uses metadata-only `Model.tiny()`;
it does not create random tensors or checkpoint files. Tiny execution creates its
Qwen2 fixture only after obtaining lock and exclusive output reservation.
Ordinary conservative estimates remain; another 8 MiB profiling allowance is
priced for **both** control and probe, before full-state audit admission.
The real-run shared lock is `~/.cache/pllm/frontier-probe.lock`, nonblocking.
Ordinary native watchdog/no-swap rules and runner descendant cleanup remain.
No build/install command is performed by this driver.

## Cohort and separate identity

Three unchanged recipes: maintained, coupled serial/u32 and
coupled parallel/uniform-residues. Intermediate choices belong to independent
confirmation cohort, not this remaining-delay investigation. Each gets unprobed
`control--` and instrumented `probe--` arms, three fresh replicates each:
**18 serial workers**, one private cohort salt, one installed build, balanced
ordering, one thread, no warmup, one cold request per process. No early EOS or
incomplete 150+8/128+2 cohort is accepted as complete.

All arms pay existing full-logit/active-full-KV/position/selection/RNG audit.
Every audit binding must equal every other binding, including control/probe and
topology changes. Public greedy seed remains existing audit fixture; cryptographic
RNG is untouched. No masks, calls, numerical algorithms, returned arrays, HTTP
bodies or dispatch schedules are replaced. Wrapper timings perturb scheduling;
they cannot be production latency evidence.

Research identity `pllm.private_boundary_decode_profile.selected3.v1` appears in suite,
details and measurement digest, together with control/probe mode and frozen
research source digest. Ordinary target specification stays unchanged. Ordinary
runner launches a research-local worker script instead of its exact private
worker command; script wraps installed allowlisted target executor then invokes
ordinary worker protocol. All provider commands stay unchanged. This deliberately
private bootstrap is not an SDK extension or supported generic hook.

Raw benchmark JSON remains authoritative. Separate plan/progress/summary files
record identities, admission, state equality, profiles and paired observed
probe-minus-control metric deltas. Those deltas show perturbation plus run noise;
they are **not** an isolated causal overhead calibration. No subtraction repairs
probe throughput, no cross-cohort pooling, no untraced SOTA rewrite.

An exclusive, durable `.reservation` companion claims output prefix before
writing plan/progress or creating fixtures. Concurrent and interrupted prefix
reuse fails without replacing previous evidence. Always choose a fresh prefix.

Audit validates real `benchmark_result.v2`/run schemas, observed build and
environment digests, passed oracle, both memory admissions, no failures/warmups,
and complete fresh-process ownership. Environment, numeric/state/output identity
and final salted workload identity must match cohort-wide. Cross-topology privacy
commitments are **not** equated: each is independently reconstructed from its
own ordinary target. Ordinary source/numeric/state/output and pre-HMAC workload
digests are reconstructed, followed by mode/source-specific measurement digest.
Final workload HMAC cannot be recomputed without private salt; its checked
pre-HMAC value and cohort-wide final equality are separate checks. Required
latency/CPU/RSS/network metric metadata and N−1 denominator must match explicit
ordinary contracts; all metric contracts must also match control/probe and
replicates within each recipe. Covered-body metric scope is checked against
that topology's original coverage identity.

Use specified frontier interpreter, not `.venv`. For tests, `-o pythonpath=.` is
important: repository pytest default prepends `python/`, potentially loading a
different stale checkout native module instead of installed frontier SDK.

## Exact clocks and owner semantics

Spans use `monotonic_ns`; window bounds are copied from existing dashboard
`_RunCapture` when `_finish_run` is called, never inferred from token count or
wall-clock timestamps:

* **Request:** `started_monotonic_ns` through `finished_monotonic_ns`, exactly
  ordinary `full_seconds`; not process birth, build inspection, role lifetime or
  all target-finalization overhead.
* **After first output:** first through last output delta's monotonic timestamp,
  exactly ordinary `generation_seconds`. Denominator is **N−1**, not N. Work
  for first output is excluded; post-last-output lifecycle tails are excluded.
  Partial boundary spans crossing a window edge are clipped.

Both durations are checked against ordinary report within one microsecond.
Request relative origin must be exactly zero. Decode interval must be positive,
nonnegative and contained in request; its first-output offset must equal
`full_seconds - online_seconds + ttft_seconds` within one microsecond. Merely
shifting intervals while preserving duration is rejected. Audit also requires
N chronological, non-overlapping forward/audited-sample pairs, one nested lookup
and sampler per pair, and, for boundary recipes, one nested head with its
quantizer/exchange/dequantizer (plus residue codec only for residue recipe).
First sample must finish before first-output timestamp; all samples must finish
by last output. Nested-owner coverage is checked, not just arbitrary label totals.
Within each head, synchronous child intervals must be non-overlapping and follow
original order: quantizer → paired exchange → optional residue codec →
dequantizer. Recomputed partitions do not excuse reordered or overlapping child
owners. A/B HTTP diagnostics inside paired exchange may still overlap; gaps
between synchronous children remain unknown head residual, not inferred work.
Tiny N=2 gives one inter-output interval; real N=8 gives seven. No averaging
prefill/head totals over N and relabelling them decode cost.

Top-level elapsed partition uses nested-owner priority, assigning every
monotonic segment once, with an explicit remainder:

| Exclusive bin | Original owner wrapped | Interpretation |
| --- | --- | --- |
| `head_quantization` | transport module quantizer alias | Head quantizer call only. |
| `paired_exchange` | transport `_exchange("head", ...)` | One pair's enclosing wall time: executor creation/submission/drain, framing, both HTTP calls, reply validation. |
| `head_codec_decode` | `HeadOutputCodec.decode` | Uniform residue decode and checks. Legacy u32 path does not call codec. |
| `head_dequantization` | transport module dequantizer alias | Dequantizer call only. |
| `head_unclassified` | transport `linear`, minus above | Includes share generation, conversion, legacy u32 reconstruction, validation and zeroing. **Not share-generation timing.** |
| `token_lookup` | semantic `_token_lookup` | Complete local/private lookup owner, including issue/exchange/reconstruct/checks. Lookup exchange is never charged to head-pair bin. |
| `audit_and_sampler` | existing audited `sample` wrapper | Includes full-state hashing, commitments, sampler, record building. |
| `body_execution_excluding_lookup_and_head` | semantic `_forward`, minus higher owners | Remaining forward wall: local operations **and prepared remote stage waits**, not pure body CPU. Maintained local head remains here; no separate local-head seam is claimed. |
| `unclassified` | window minus covered owners | Tokenization, preparation/transition outside forward, stream delivery, other scheduling/lifecycle/probe work where present. No assumed cause. |

Raw spans are inclusive; the top-level partition is exclusive. Do not sum raw
inclusive spans, or add diagnostic children to parent bins. Reconciled exclusive
wall sums equal enclosing elapsed exactly.

HTTP `_post` spans retain only public A/B labels. Diagnostic `http_exchange`
reports union coverage of both parties; individual A/B spans are also retained.
`frame_pack`, `frame_unpack`, `lookup_exchange` and original `sampler` are
diagnostic children, not extra elapsed bins. They report clipped interval unions,
not duration sums. During parallel head dispatch A/B requests overlap; neither
sum nor `max(A,B)` is substituted for pair wall time. Pair time includes launch,
join and validation; interval unions account for staggered starts correctly.
Framing/HTTP unions are not an additive decomposition of pair time: processing
one role can overlap HTTP wait for another. Serial dispatch uses same semantics.

## CPU, safety and limits

Each raw span has calling-thread `thread_time_ns` delta: excludes other client
threads, native pool CPU and all role-process CPU; nested spans overlap. Never
sum these as client CPU. Separate `executor_client_process_cpu_ns` is whole
client-process CPU across response executor, including all its threads and report
work. It is not request/decode-window CPU. Ordinary process/role CPU accounting
remains in original report, with its original lifetime/window semantics.

HTTP wait includes transport, WAN pacing, remote compute, server codec, queueing
and client work. **Role CPU, role compute split and true share-generation timing
remain unknown** (`null`), not inferred by subtracting comparator timings.
Worker protocol exposes no per-role timing field and workers are separate
processes; client monkeypatches do not instrument them. No aligned role-clock
timeline is claimed. If needed later, minimal separate SDK proposal: extract a
non-inlinable head-share owner and add opt-in bounded worker compute/codec
timings under distinct research identity, preserving all current arithmetic,
fresh randomness, memory admission and trusted ownership. That change is not
implemented here.

At most 4096 spans per worker; overflow fails closed. Span records retain only
allowlisted labels, relative monotonic intervals and calling-thread CPU counts.
No prompts, tokens, activations, logits, raw replies, credentials, sessions or
RNG state enter new timing telemetry. Existing salted full-state commitments
and ordinary bounded benchmark report remain unchanged. Wrappers retain no
argument/return data and restore all patched functions on failure. Instrumentation
rejects concurrent use in one worker. Source hashes are rechecked after cohort.
Model-free tests cover interval overlap/clipping, lookup/head ownership, wrapper
return/exception behavior, bounds/restoration, worker-command routing, separate
cohort contract, exact state equality and rejection of corrupt timing evidence.

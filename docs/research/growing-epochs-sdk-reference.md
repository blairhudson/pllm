# Growing epochs: frozen installed-SDK measurement protocol

Frozen methodology for serial installed-SDK measurements. Generated SDK/gateway
controls and an isolated eight-cell smoke cohort preserve exact state. These
checks do not establish real-checkpoint performance or Low-tier qualification.
Completed checkpoint results are published separately from this protocol.

## Frozen cells and replay

`scripts/benchmark_growing_epochs_sdk.py` (also exposed by
`examples/benchmarks/growing_epochs_sdk.py`) declares two cold one-response
workloads: natural short `Hello world` instruction and detailed long answer.
Both allow 64 outputs. Each compares full-allowance inventory, fixed 4, fixed 8,
and immutable growing `(4,60)`. Same pinned model, indexed tokenizer, i8/i8
`prefix_f32`, paged preparation and bundle, demand/on-demand inventory, four-stage
64 MiB preparation window, 32 MiB prefix cache and 128-position continuation bound
apply throughout. Full allowance remains admitted even when EOS is expected early.
No generation override, output-dependent policy selection, cache priming or
generated-prefix reuse saving is claimed. Tiny mode uses eight outputs and one
small prompt; its lifecycle checks are explicitly non-confirmatory.

Default is three replicates, reversing the entire cell order on odd replicates
(24 fresh cells); real cohorts permit 3–12 replicates. Each invocation requires a
new output path and generates its own private salt. Do not combine invocations,
interrupted attempts or salt cohorts. Prompts, credentials and salt go to client
children on stdin; archives contain no token values, text or raw logits. Exact
logits/KV bits, actual structured message history, rendered/raw history, pending
selection and output are salted and matched across every policy and replicate.
History is not inferred from output equality. Greedy fixture explicitly makes no
RNG trajectory claim. Witness serialization has bounded byte/node/depth limits
and retains no additional decoder owners. Natural early EOS and full 64-output
completion must actually occur, or the real cohort rejects.

After building and installing the validated wheel:

```sh
target/frontier-sdk-env/bin/python -I -m pytest -o pythonpath= \
  tests/test_benchmark_growing_epochs_sdk.py -q
target/frontier-sdk-env/bin/python scripts/benchmark_growing_epochs_sdk.py \
  --tiny --preflight-only --output target/growing-epochs-tiny-admission.json
target/frontier-sdk-env/bin/python scripts/benchmark_growing_epochs_sdk.py \
  --tiny --replicates 1 --output target/growing-epochs-tiny.json
target/frontier-sdk-env/bin/python scripts/benchmark_growing_epochs_sdk.py \
  --preflight-only --output target/growing-epochs-admission.json
target/frontier-sdk-env/bin/python scripts/benchmark_growing_epochs_sdk.py \
  --replicates 3 --output target/growing-epochs-sdk.json
```

Default checkpoint declaration is
`target/prepared-frontier-artifacts/indexed.json`; `--artifacts` selects another
pre-positioned declaration directory. No driver builds, installs or downloads.
`--smoke` aliases `--tiny`; `--processes` aliases `--replicates`.

## Seven priorities and explicit windows

1. Client lifetime peak RSS: OS process high-water, including imports/setup,
   exact-state hashing and close. Largest value across **all** cells/replicates,
   not median, gates Low at **268435456 bytes**. Admission scope is reported
   separately and never treated as RSS.
2. Client CPU: cumulative process user+system through close and audit.
3. Preparation CPU: cumulative from role birth through client exit; excludes
   provider shutdown. Complete-cell aggregate CPU separately includes launcher,
   relays and reaped client/provider children through cleanup.
4. Preparation RSS: 50 ms sampled peak after provider startup through client exit.
   Startup high-water is unknown. Inference has the same separate measurements.
5. TTFT: request start through first public text delta; setup/admission/preparation
   waiting within the request stays charged. Earlier cold client setup is separate.
6. Charged decode TPS: `(N-1)/(last_delta-first_delta)`, including every epoch
   pause. Single-output throughput is null.
7. Traffic: complete directed all-link TCP streams from native relay startup
   through role/network shutdown, plus separate v3 covered application bodies
   from client construction through close. Native shared access caps are 100/40
   Mbps, no added RTT; TCP streams are not physical wire.

Role shutdown closes and joins the relay loop; final stream snapshot reads its
same counter owner after join, not the public live-loop sampler. Application
bodies charge cold bundle, issued unused material, setup, continuation admission,
retirement, lifecycle controls and correction ACKs exactly once where counters
cover them. Failed workers retain available actual audit/body/stream counters.

Retired lifetime requires integer nonnegative issued/reserved/claimed/burned/
discarded counters, `issued = reserved + discarded`, and
`reserved = claimed + burned`. These are not request-window equalities.
Cap truncation must evaluate exactly N-1 decode steps and leave one pending token;
EOS completion evaluates N steps and leaves none. Snapshot position is checked
against input count plus those exact evaluated steps.
Abrupt worker death leaves client counters null/unknown, not fabricated zeros.
Provider startup failure can lack traffic counters; bounded attempt and error
type remain archived. Error messages and provider logs are never archived.

Existing v3 decomposition is reused: total equals known base plus known
output-dependent plus unattributed. Epochs currently lack full phase attribution;
mixed envelopes, unclassified material, controls and ACKs stay residual. Unused
byte amounts stay null unless directly metered; row burn counts remain charged,
not converted to estimated bytes. No bytes-per-row extrapolation. Publisher
artifact/checkpoint/tokenizer distribution and disk are separate unknown fields,
not silently included in covered network totals.

## Admission and computation gate

Existing `benchmark_memory` and `group_memory_contract` price native roles,
checkpoint ownership, full-cap KV/scratch/snapshots, prefix cache, inventory and
caller metadata. Group helper adds 128 MiB; launcher/relays add another 256 MiB.
Thus **384 MiB fixed overhead** sits above the topology and slack-adjusted caller
owners. One caller's metadata includes 16 MiB plus boxed input/output allowances,
with 25% slack. Exact total is source-dependent, printed by preflight and retained
per cell; do not substitute historical short-cap estimates. Host OS/application
reserve and swap guard remain unchanged. Recheck admission before every cell,
under existing exclusive `frontier-probe.lock` and `fresh_workers` watchdog.

Public native semantic schedule supplies body projection dimensions, ownership
and terminal-only prefill pruning. Actual input count and audited executed decode
steps weight these dimensions; each cell checks remote MACs >=90% of scheduled
transformer-body dense linear MACs. This is **not** inferred from role labels,
whole process CPU, or all operator FLOPs. Head, attention, normalization, masking
and Preparation are outside that declared denominator; whole-body-computation
fraction remains null. No broader >=90% computation claim is licensed.

Build reference, source lock, experiment canonical hashes, native plan/schedule
digests and measurement-file hashes are frozen. Installed build is checked before
and after the locked cohort; measurement sources after every cell. Progress
records are bounded by at most 96 declared attempts. Started, rejected, failed
and interrupted attempts survive atomic saves; only fully validated cohorts set
`complete=true`. Low RSS gate may fail without suppressing actual measurements.

Local bootstrap runs launcher and every Python client/role/metadata child with
`-I -S`, explicit installed wheel paths and no startup, user-site or `.pth` hooks.
Actual SDK source/native file hashes are checked against BuildRef before each
measured child target runs. This does not modify historical runner helpers.
Bootstrap/identity-check CPU and RSS stay charged from process birth; full cell
wall time includes them. Generated smoke results remain functionality evidence;
real-checkpoint performance and memory require the complete matched cohort.

# In-flight prefix ownership: bounded local-clear screen

## Result and scope

`scripts/probe_inflight_prefix.py` implements a research-only, one-level prefix
DAG: one trusted client owns a shared prefix task; each suffix/decode branch owns
an independent full-KV snapshot through the existing semantic executor and
continuation contracts. Existing completed cache hits bypass the new owner.

Three fresh serial generated Qwen2/Qwen3/Llama workers now pass complete logit,
active-KV and pending-token parity under terminal pruning, packed inputs and
row-residue outputs. Every successful branch and the surviving branch of an
injected failure agrees. All owners retire, with zero observed swap growth.
The pinned eight-branch 150+8 checkpoint cohort was rejected twice **before
worker launch** by its unchanged 6 GiB post-reserve admission.

Current two-branch, 12+2-token Qwen2 results include prefix setup and fresh
correction demand:

| Case | Independent stage rows | In-flight stage rows | Independent projected tensor bytes | In-flight projected tensor bytes |
| --- | ---: | ---: | ---: | ---: |
| Eight-token shared prefix, divergent suffixes | 142 | 123 | 27,176 | 23,682 |
| Identical full prompts | 142 | 79 | 27,176 | 15,138 |
| No matching first token | 142 | 166 | 27,176 | 31,904 |

Qwen3 and Llama have the same row counts; their respective tensor projections
are 27,160 versus 23,638/15,126/31,856 bytes. These are approximately **13%/44%
lower** packed tensor projections for shared/identical prompts, and **17% higher**
for no match. Splitting unmatched requests loses terminal-pruning opportunities;
an eventual selector must price that regression. None is measured provider traffic.
The completed-cache controls have the same total arithmetic demand as in-flight
execution, including their original setup. Already-populated cache hits gain
**zero** additional rows and create zero prefix tasks.

The three workers' combined-process lifetime peaks are 246.09–246.42 MB. They
include clear weights, imports and retained parity controls; they are not
client-only peaks or a memory comparison. Evidence retains all three reports in
`docs/evidence/research-inflight-prefix.json`, factoring only identical public
records into `common`. Recombining each report reproduces its original JSON hash.
This generated screen does not validate the earlier eight-request 50% estimate.

### Historical unpruned fixture

Historical pre-review, unpruned tiny generated Qwen2 functional check: two requests, 12 input tokens, eight-token
common prefix, two sampled outputs each. Complete logits and every active KV bit
match fresh native execution. The last sampled token remains pending at position
13; it is not promoted to executed KV. Observed stage rows, including setup:

| Case | Independent cold | In-flight | Completed cache |
| --- | ---: | ---: | ---: |
| Divergent suffixes | 208 | 144 | 144 |
| Identical full prompts | 208 | 112 | 112 |
| No matching first token | 208 | 208 | 208 |

For divergent suffixes, shared prefix setup costs 64 stage rows; both branches
together cost another 80. Identical full prompts execute their entire common
prefill once (96 stage rows), then charge both independent decodes (16 rows).
Already-populated controls charge only 80/16/80 response rows respectively;
their historical setup remains visible separately. Adding in-flight ownership
to those hits saves **zero** rows and creates zero prefix tasks.

One injected branch-result failure executes four QKV rows, then retires that
private branch. Other branch remains bit-exact. Its 108-row failed workload is
not compared as a saving against two successful responses. Failed arithmetic
material-demand units remain charged and cannot be reused.

Current probe explicitly selects terminal pruning, stage_packed inputs and
row_residues outputs for **every** control and candidate. Historical table above
is retained, not attributed to that revised execution contract. The three revised
generated controls are reported separately above.

These historical tiny counts do not validate the earlier approximately 50% estimate
for eight 150+8-token requests sharing 96 tokens. That pinned cohort remains
memory-admission blocked. No checkpoint was substituted. Historical tiny
combined-process lifetime peak RSS: 243,875,840
bytes; watchdog observed zero swap growth. This includes clear executor weights,
source loading, retained parity controls and client state, **not client-only RSS**.
Tiny timings are diagnostic only, not a timing cohort or speed claim.

## Ownership and numeric gates

- Client boundary is an object capability, not a caller-chosen account string.
  Source, composition and state binding must match before admission. The native
  compiled binding and source/control locks are recorded separately.
- Prefix keys use a private per-owner HMAC key; no keys, tokens, logits, KV or
  state commitments are emitted. Source/build hashes identify public code only.
- Only untiled, unverified, native `prefix_f32`, completed-prefill, exclusive
  full-KV state is admitted. Incremental/generated response-owned snapshots and
  pending-token aliases reject. Execution-basis digest is recomputed from native
  phase identity, absolute positions and exact prefix tokens, blocking a valid
  snapshot substituted from a different prompt.
- Prefix result commitment detects mutation after publication. Waiters receive
  copies, not shared mutable arrays. This is trusted-process integrity, not
  remote attestation or an encrypted storage protocol.
- One waiter cancellation cannot cancel another waiter's producer. Last waiter
  cancellation cancels and drains cooperative owned work. Retiring producers
  continue counting against node and RAM bounds until cleanup actually finishes.
  Detached threads/non-cooperative executors are unsupported.
- Last lease retires and erases owned prefix and branch arrays. Replay after
  retirement requires fresh execution; this owner is not a completed cache.

Owner caps: at most 64 nodes, 64 waiters and 256 MiB reserved payload/overlap.
Cohort caps: at most eight nodes/waiters, 150 input tokens and eight outputs.
Reservations separately cover producer/runtime publication overlap and branch
snapshot/runtime overlap, including a second maximum-size KV allocation. These
are conservative state-payload allowances, not whole-process RAM limits. Python
metadata is count-bounded; model weights, nonlinear workspaces and parity copies
remain visible in process RSS and governed by shared host-pressure admission.

## Accounting boundaries

Stage callback counts come from actual native execution, including first-prefix
setup, suffix continuation, decode, terminal scheduling and failed work. No
`shared_tokens * requests` shortcut replaces this ledger. Phase-separated rows
and exact stage geometry are returned for each control.

Primary byte ledger is now imported **W8 stage_packed/row_residue tensor-payload
projection**, separate from observed execution rows. Each actual stage callback
records its row count and chunks of at most 16 rows, also capped by the imported
4,000,000-cell limit. For each chunk with `n` rows:

- Client→Inference input: `ceil(n * in_features * max(row_bits) / 8)`.
- Inference→Client output: `ceil(n * sum(row_bits) / 8)`.
- Fresh Preparation→Inference correction demand: same row-residue payload size.

Row bits come from actual imported stage `output_residue_bits`, checked against
its W8 weight commitment and seeded ring. Callback/chunk rounding is preserved,
not approximated by aggregating all rows first. Prefix/setup and branch costs,
fresh corrections and failed callback demand remain separate. Correction sizes
use matched 16-row chunk demand; different provider issuance batching may change
padding and must be measured. Public width metadata and bundle delivery remain
separate. This projection creates no masks or tickets and makes no framing claim.

Two secondary byte ledgers remain explicitly arithmetic:

1. Clear f32 callback input/output: `4 * rows * (in_features + out_features)`.
2. Unframed hypothetical i64 input/output/correction bodies:
   `8 * rows * (in_features + 2 * out_features)`.

The secondary i64 ledger is historical hypothetical arithmetic only; **do not use
it for goal prioritization**. None of these projections is measured packed/residue
traffic, live protocol framing, actual one-use
material issuance, or whole-response network traffic. Bundle delivery, control
authorization, seeds, setup framing, unused reservations, correction pushes,
TCP and provider teardown remain unmeasured (`null` where applicable). Public
bundle delivery must be charged once per trusted-client horizon in a future
all-link cohort; it is not an invented saving here.

Proposed placement keeps all body MACs remote (100%, hence >90%). Observed clear
executor is local (0% remote). Fresh provider material must be issued for every
genuinely different prefix/suffix/decode operation; sharing plaintext result/state
inside client boundary never licenses reusing masks, tickets or provider keys.
Material-demand counts are arithmetic stage-row units only. No private provider
key ownership or privacy-protocol claim is made.

Owner reports private snapshot-copy bytes, copy CPU, wait duration summed over
waiters, reserved overlap peak, execution count and retirement separately.
Summed waiter duration is not response latency and includes producer execution.
Worker reports source construction, runtime construction, setup, branch combined
CPU, clear callback CPU and residual runtime CPU separately. Residual CPU is not
a production thin-client CPU measurement. Parity copies/scans run after measured
branch CPU boundaries and still contribute to lifetime RSS.

## Tiny checks and parent serial commands

Already run: `27 passed, 3 skipped` lifecycle/accounting tests, including native
synthetic residue packing lengths, checkpoint/config/symlink mutation, actual W8
weight-hash substitution, existing-output rejection and unexpected-failure cleanup.
Prior tiny Qwen2 native screen passed under shared lock/512 MiB admission before
terminal-pruning revision. The parent subsequently completed all three revised
generated-family CLI workers serially. The three pytest skips avoid duplicating
those native workers in the lightweight test invocation.

Use existing environment; no installation or Rust/wheel rebuild:

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= tests/test_inflight_prefix_probe.py -q
PLLM_INFLIGHT_NATIVE=1 target/frontier-sdk-env/bin/python -m pytest -o pythonpath= tests/test_inflight_prefix_probe.py -q
target/frontier-sdk-env/bin/python scripts/probe_inflight_prefix.py --family qwen2 --output target/inflight-prefix-qwen2-pruned.json
target/frontier-sdk-env/bin/python scripts/probe_inflight_prefix.py --family qwen3 --output target/inflight-prefix-qwen3-pruned.json
target/frontier-sdk-env/bin/python scripts/probe_inflight_prefix.py --family llama --output target/inflight-prefix-llama-pruned.json
target/frontier-sdk-env/bin/python scripts/probe_inflight_prefix.py --family real --tokens 150 --prefix 96 --branches 8 --outputs 8 --output target/inflight-prefix-real-150-pruned.json
```

Run serially, without pytest-xdist. Native pytest checks run generated Qwen2,
Qwen3 and Llama workers serially; use that command **or** the three explicit
generated-family CLI commands, not both. Each worker uses existing
`scripts/frontier_probe_support.py::fresh_workers()`, exclusive
`~/.cache/pllm/frontier-probe.lock`, host headroom/swap guard, offline checkpoint
resolution and thread limits. Real cohort requires 6 GiB post-reserve headroom
and has a 1,200-second timeout; generated worker admission is 512 MiB/240 seconds.
Admission estimate is not a hard process RSS limit.

All CLI outputs require fresh paths (`open("x")` before admission); existing
files/symlinks reject without replacement or expensive work. Failed attempts remove
only their newly reserved output. Choose new names for repeats; never delete or
overwrite old evidence. Unit/native pytest temporary outputs are fresh by design.

**Keep 6 GiB admission.** Maximum configured Qwen state owner allowance is 202,686,464
bytes, conservatively allowing eight producer/eight lease slots with 150-token
prefixes, 158-token maximum capacity and 151,936 logits (identical prompts actually
need only one producer). The 96-token geometry allowance is
181,452,800 bytes. Three eight-branch cold reference cohorts retain roughly
209,289,216 bytes of active KV and eight logits per branch; cache owner is capped
at 64 MiB. These known state components alone approach 0.45 GiB. Streamed import
does not prove the peak of weight/native copies, largest-tensor quantization,
Torch/dependency startup, nonlinear workspace and compile/validation transients.
No measured/complete importer-wide bound justifies lowering admission to 4 GiB.
Per-mode cleanup prevents registration callbacks retaining snapshots across modes.

Real prerequisites: cached `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`; W8A8 body fingerprint
`5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`;
working installed native continuation contract and prefix_f32 compiler/runtime;
NumPy, Torch, safetensors, filelock and current installed `pllm`.

Before worker launch, freeze own probe/test, shared runner/decoder fixture, source
Python, installed package Python/native files and installed dependency binaries,
plus historical confirmation and stage geometry JSON bytes. For the real cohort,
under the shared measurement lock and before worker launch, resolve the pinned
cached snapshot offline and hash every actual config/checkpoint/tokenizer file.
Preserve logical snapshot paths so symlink substitution also rejects; verify the
complete file namespace and bytes before/after worker and again in parent,
including failed attempts. No values are loaded during this pinning scan.
Verify before and after worker and again in parent; record actual imported paths,
hashes and dependency versions. Shared checkout changes during a cohort fail
closed. Source freeze audits are outside serving-time claims. Source/checkpoint,
composition, compiled/native schedules and state bindings are recorded per run.
Actual loaded source/checkpoint/source-lock/body identities must match historical
geometry and confirmation. Every body stage shape and actual streamed W8 weight
SHA-256 must match the pinned geometry, not merely its stored descriptor. Repeat
that admission after execution. Encoding/pruning/composition changes are explicit
new controls; historical schedule digests are not incorrectly required to match.

An ExitStack registers runtime reset, cache clear, lease retirement and temporary
snapshot erasure as soon as ownership is acquired. Unexpected worker failures,
partial admission, restore/decoder errors and integrity-audit failures drain
cleanup; successful modes retire registrations before the next mode. One waiter's
cancellation and retiring-producer RAM/node invariants remain unchanged.

**Remaining blockers:** memory-admitted pinned-checkpoint control; production
single-client concurrency integration; live one-use material cancellation/burn
semantics; all directed bodies/bundle/setup; separate client/provider RSS/CPU;
WAN waiting/latency and provider failure/replay authentication. This screen does
not resolve those gates.

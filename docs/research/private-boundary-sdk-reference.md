# Coupled private lookup and complete output head

## Executable contract

`pllm.roles.PrivateLookupAndFullHead()` and
`pllm.roles.PreparedBoundaryRoles()` select an explicit five-role composition:

```text
                   offline Preparation ──corrections──▶ Inference
                            ▲                              ▲
                            │                              │
                            └────────── Client ────────────┘
                                       │  │
                              private lookup / shared full head
                                       │  │
                              Boundary A  Boundary B
```

The original prepared decoder body stays on Inference. Each boundary worker owns
the same public tied W8 matrix, but receives only its own private lookup query
and hidden-vector share. The client owns tokenization, activation quantization
and scales, nonlinear work, attention, full KV, complete reconstructed logits and
the original sampler. Both embedding and head move together; the thin client
bundle contains **no vocabulary matrix**. No vocabulary rows are excluded.

This is a bounded semi-honest, non-colluding-worker contract. Preparation retains
its existing erasure and non-collusion requirements. Co-located children validate
execution and lifecycle, not independent operators or cryptographic review.

## Compiler and source custody

The model-neutral native pass requires one main token lookup and one complete
output head using the same declared artifact, dense float32 decoder operations,
ordinary full KV and W8A8 CPU execution. The optional
`pllm.private_boundary.v1` contract on `pllm.decoder_runtime_schedule.v2` binds
source-plan configuration, composition, operation identities, weight geometry,
numeric boundaries, owner names, query/head bounds and allocation floors.
Omitting the components preserves ordinary schedule serialization.

Each worker independently resolves the locked source, validates checkpoint
headers and re-lowers the fixed plan before importing the tied head. Actual
matrix, scale and lookup-record digests must agree with the native contract and
the client bundle. A source-config alias is not checkpoint evidence. Generated
`Model.tiny()` sources retain their original kind across all four children so
source-lock equality does not depend on a launcher path rewrite.

Both lookup and head share one immutable native matrix owner per boundary worker.
The lookup record is the original i8 row plus its original float32 scale; no
second serialized full lookup table is retained. The client's ordinary bundle
validator checks the exact required local tensors and rejects hidden matrix
payloads, unmatched scales, substituted source locks and incomplete artifacts.

## Numeric and one-use execution

* Lookup uses packed128 DPF queries, at most 64 selections per public batch.
  A separate native context binds source, plan, paired session, epoch and admitted
  workload. Each selected row preserves its original quantized bytes and scale.
* The full head uses fresh independent OS-random ring32 input shares. Each worker
  evaluates the original immutable head. The client adds replies modulo 2³²,
  center-decodes and applies the original float32 scale expression. A public
  signed-output bound proves centered reconstruction is unambiguous.
* Every already-selected token needed by a later decode step receives a fresh
  private lookup, even when its token ID repeats. The last pending selection is
  not executed. The initial lookup and each head application remain causal.
* Separate credentials authenticate workers; each child receives only its own
  secret and the peer's credential hash. Public descriptor and credential pairing
  must agree before any online operation. Remote URLs require HTTPS; redirects and
  compressed boundary replies reject. Native loopback rate relays remain local.
* Monotonic frames and independently derived operation budgets consume ownership
  before native work. Malformed frames, replay, expiry, partial admission,
  cancellation and completion retire the session. Client cancellation drains
  outstanding calls and stops subsequent decode. Partial-stream close permits a
  fresh response, never reuse of retired material.

The worker admits at most eight live sessions, 4,096 contexts over its lifetime,
4,096 lookup rows per response, 16 MiB data frames and 16 KiB control frames.
Lookup replay ledgers are per-context. Idle sessions expire after 90 seconds;
retired contexts never become reusable within that worker incarnation.

## Composition and measurement

The ordinary SDK and gateway use the shared role supervisor. Public artifact
delivery, zlib, client/Preparation paging, demand issuance, bounded stage windows,
terminal pruning, duplex prefill and packed/residue prepared-body frames compose.
Boundary stages are excluded from prepared inventory and residue commitments;
their own full-width protocols remain separately bound and metered.

This initial composition rejects untied/auxiliary boundaries, Metal, verification,
state/cache reuse, epochs, placement, MLP/attention regions, Docker and network
planning. Native shared 100/40 Mbps access limits cover the client, Preparation,
Inference and both boundary workers. Application-body coverage uses a new
versioned identity; historical totals retain their old meaning.

`scripts/benchmark_private_boundary.py` runs alternating matched arms in fresh
SDK clients. Controls retain their explicit original components. Full logits,
active KV, token history, pending selections, output and sampler state are hashed
equally in both arms under one private cohort salt. A complete report requires
every configured replicate, body/source identity, prepared-row conservation,
boundary operation counts, cleanup and directed-byte reconciliation.

Client process peak RSS, provider RSS samples, client request CPU, complete-case
aggregate CPU, first text and N−1 decode throughput retain distinct scopes.
The parent owns rate-relay threads; its CPU is included once alongside reaped
children in complete-case CPU. Client-subprocess RSS excludes that parent/relay
owner; a simultaneous complete client-process-group peak is unmeasured. Provider
startup memory peaks, OS-cache state, physical wire and checkpoint distribution
are not supplied by those samples.
Incomplete base/output attribution remains null, not a ratio-based estimate.

Source-present and missing-worker cases are separate. The current SDK driver
requires cached sources; missing-worker delivery is explicitly blocked. Moving a
client artifact does not make either worker's source delivery free. The older
[boundary operator gate](tied-private-boundary-gate.md) has different owners and
framing; its synthetic or projected bytes cannot substitute for SDK evidence.

## Pinned 48+8 cold-client cohort

Raw evidence: [`private-boundary-qwen-48-sdk.json`](../evidence/private-boundary-qwen-48-sdk.json).
The exact control is retained in
[`private-boundary-qwen-control.json`](../evidence/private-boundary-qwen-control.json).
It pins `Qwen/Qwen2.5-0.5B-Instruct` at
`7ae557604adf67be50417f59c2c2f167def9a775`, W8A8 `prefix_f32`, one native thread,
paged client and Preparation weights, compressed batched artifacts, compiler-demand
issuance, a four-stage/64 MiB window, terminal pruning and sixteen-row duplex
packed/residue frames. There is no prefix or decode reuse. Only coupled boundary
placement and its explicit role graph differ.

Three alternating pairs run serially, with fresh client interpreters, empty client
artifact caches and new provider processes for every case. Checkpoint source and
the 9,285,336-byte indexed-tokenizer artifact are already present. Their delivery
is not measured. Shared native 100/40 Mbps caps cover every role; these are local
TCP-stream measurements without Internet RTT or physical-wire accounting.

All six responses select eight outputs after 48 input tokens. Complete sampled
logits, active KV, executed history, pending selections, output and RNG commitments
agree. Both arms charge the same state audits. The fixture is greedy with sampler
seed 42; cryptographic randomness stays fresh. Each coupled response performs 55
private lookups and eight complete-head calls, retains no client vocabulary
matrix and reconciles every boundary body. All children retire; observed swap
growth is zero.

Medians, with decimal MB:

| Metric | Local boundary | Coupled boundary | Change |
| --- | ---: | ---: | ---: |
| All-link TCP streams | 249.87 MB | 138.32 MB | −44.64% |
| Covered all-link application bodies | 248.89 MB | 137.33 MB | −44.82% |
| Request latency | 52.922 s | 36.009 s | −31.96% |
| Complete case, including startup and cleanup | 73.815 s | 64.252 s | −12.95% |
| Request-to-first-text | 51.125 s | 32.392 s | −36.64% |
| Charged N−1 decode throughput | 3.923 tokens/s | 1.949 tokens/s | −50.32% |
| Client-subprocess lifetime peak RSS | 148.14 MB | 151.96 MB | +2.58% |
| Client request CPU | 6.570 s | 9.190 s | +39.89% |
| Complete-case aggregate CPU | 41.161 s | 47.551 s | +15.53% |

Cold client bundle bodies fall from 124.91 to 3.41 MB. The extra two-worker
boundary bodies total 9.93 MB, including setup and retirement; complete-head
replies alone cost 9.73 MB. Ordinary prepared-body traffic and its 5,139 issued
rows remain identical. Thus online all-link bodies **increase** from 77.18 to
87.11 MB even while cold totals fall. The measured benefit comes from avoiding
cold vocabulary delivery, with a substantial decode-throughput tradeoff.

Median sampled Preparation RSS is 162.51/159.81 MB and its sampled birth-through-
client-exit CPU is 11.35/11.64 s. Additional boundary workers each reach roughly
391 MB in post-startup RSS samples and roughly 3.89 s of sampled birth-through-
client-exit CPU. These samples do not include startup memory peaks and are not
summed into a topology peak. Complete-case aggregate CPU instead uses parent all-thread CPU plus reaped
child CPU exactly once.

The largest client subprocess peak is 145.91 MiB, but parent/relay ownership and
conservative runtime admission prevent interpreting that sample as whole-client
Low-tier qualification. This cohort establishes a source-present short-request
traffic/latency tradeoff, not a client-memory improvement or an independently
operated deployment. Complete base/output headline rates remain null because
mixed envelopes and controls retain unattributed bytes.

Earlier 150-token attempts remain separate:
[`initial admission rejection`](../evidence/private-boundary-qwen-150-initial-blocked.json)
used the subsequently corrected duplicate-table estimate;
[`incomplete attempt`](../evidence/private-boundary-qwen-150-incomplete.json)
retains one control and a typed failure before a coupled result. Neither supplies
a paired result or missing replicate to this cohort.

## Independent 150+8 cold-client cohort

Raw evidence: [`private-boundary-qwen-150-sdk.json`](../evidence/private-boundary-qwen-150-sdk.json).
After headroom admitted the complete topology, a separate three-pair cohort ran
under the same installed build and control composition. It has its own private
cohort salt and is not pooled with the shorter workload or failed attempts.
All six complete trajectories agree, every child retires and observed swap growth
is zero. Each coupled response executes 157 private lookups and eight full heads.

| Metric | Local boundary | Coupled boundary | Change |
| --- | ---: | ---: | ---: |
| All-link TCP streams | 478.50 MB | 367.22 MB | −23.26% |
| Covered all-link application bodies | 477.10 MB | 365.80 MB | −23.33% |
| Request latency | 98.532 s | 82.436 s | −16.34% |
| Complete case, including startup and cleanup | 119.189 s | 110.287 s | −7.47% |
| Request-to-first-text | 96.650 s | 78.769 s | −18.50% |
| Charged N−1 decode throughput | 3.758 tokens/s | 1.923 tokens/s | −48.82% |
| Client-subprocess lifetime peak RSS | 190.09 MB | 194.56 MB | +2.35% |
| Client request CPU | 9.128 s | 11.679 s | +27.95% |
| Complete-case aggregate CPU | 56.773 s | 63.641 s | +12.10% |

The 121.49 MB cold-bundle reduction pays for 10.19 MB of additional boundary
bodies. The complete application-body difference is exactly 111,306,739 bytes.
Prepared arithmetic bodies and all 14,625 issued rows agree across all six cases.
Online all-link bodies rise from 219.33 to 229.52 MB (+4.64%). The smaller
percentage cold saving at longer input follows from unchanged body prefill work.

Median sampled Preparation RSS is 196.77/205.39 MB; its sampled birth-through-
client-exit CPU is 15.19/15.21 s. Each added boundary worker reaches about 399 MB
in post-startup RSS samples and 4.34–4.39 s of sampled CPU. Largest client-subprocess
peak is 186.72 MiB; parent/relay ownership and conservative admission retain the
same whole-client qualification gap. Full-head downloads lengthen decode despite
shorter cold requests. No whole-client memory improvement follows.

The two completed cohorts contain twelve fresh client responses. Both preserve
full-vocabulary behavior and show a cached-source cold-delivery tradeoff; neither
establishes missing-worker savings, independent operators or a warm-request win.
The exact installed modules, extension and dependency commitments are retained in
[`private-boundary-sdk-build.json`](../evidence/private-boundary-sdk-build.json).

## Validation

Both capabilities also run through ordinary `BenchmarkTarget.response` and
`benchmark_suite`; `examples/benchmarks/private_boundary.py` executes its local
and coupled choices in fresh processes. Its generated-checkpoint 100/40 smoke
completes both cases. The benchmark's whole-topology memory budget remains
distinct from a client memory tier.

`scripts/summarize_private_boundary_sdk.py` revalidates the two original cohorts
without model execution. Hash-bound summaries retain their frozen control recipe,
native admission identities, exact parity, directed counters, complete replicates
and historical build. Repository capability coverage recognizes those records as
validated historical evidence, or stale when a requested build differs.
The research registry adds separate 48+8 and 150+8 screen entries with an explicit
five-role topology. First-text/setup, N−1 text-event decode, lifetime child CPU
and parent-excluding child RAM have their own labels. Missing whole-client RAM
prevents all memory-tier qualification; the original three-role control and new
five-role placement keep separate topology leaders.

The later [complete-worker tier confirmation](private-boundary-tiers-reference.md)
uses a new 24-client 150+8 cohort with SDK, dashboard, relay and audit in one
measured process. Its coupled maximum is 210.22 MiB, qualifying observed Low;
traffic and full request latency beat the best matched prepared incumbents by
19.93% and 20.26%. That independent result leaves these two historical child-only
cohorts' memory status unchanged and retains the additional boundary-worker trust
contract.

The native compiler suite and all eight private-page tests pass. Installed-wheel
validation includes 18 SDK lifecycle tests across generated Qwen2/Qwen3/Llama,
319 lower-level regression checks, 101 existing SDK checks and a final 142-test
accounting/history/CLI/WAN selection. These overlapping selections are not summed.
The composed SDK example, the replay Python blocks below, and an ordinary
five-role `pllm benchmark run --backend native --wan` generated-source request
complete. The latter passes runtime checks, retires its children and observes
zero swap growth. Coverage-contract, generated-reference, documentation-content,
Ruff, Rust formatting and diff checks also pass. Generated-source timing is not
checkpoint performance evidence.

## Replay

The generated-source example executes the installed SDK:

```bash
python examples/private_boundary.py --root target/private-boundary-example
```

Ordinary SDK benchmarking and read-only historical summary replay:

```bash
python examples/benchmarks/private_boundary.py --output target/private-boundary-suite.json
python scripts/summarize_private_boundary_sdk.py docs/evidence/private-boundary-qwen-48-sdk.json \
  --control docs/evidence/private-boundary-qwen-control.json \
  --output target/private-boundary-48-summary.json
```

The first command produces fresh `BenchmarkResult` records through the standard
runner. It is not a substitute for the original diagnostic driver or its complete
state-audit window. The second reproduces the archived summary byte-for-byte.

For a cached checkpoint, `examples/private_boundary.py --model-dir PATH` builds
the offline indexed tokenizer and selects the same paging/codec composition.
`--local-boundary-control --write-experiment PATH` exports its matching local-head
control. Offline tokenizer creation and distribution are additional costs.

The matched driver accepts that control with `--baseline-spec`, a private
`--prompt-file`, authoritative `--input-tokens`, `--replicates 3`, `--wan`, and a
new `--output` path. `--preflight-only` checks installed composition and memory
admission before roles start. Prompts, tokens, credentials and cohort salts are
not persisted in reports.

To reconstruct the pinned control with a new local artifact path, use the public
example factory. This requires the pinned checkpoint already in the shared Hub
cache; it does not download weights or run model inference:

```python
from pathlib import Path
from huggingface_hub import hf_hub_download
import pllm
from pllm.tokenization import IndexedTokenizer
from examples.private_boundary import build_experiment

repo = "Qwen/Qwen2.5-0.5B-Instruct"
revision = "7ae557604adf67be50417f59c2c2f167def9a775"
root = Path("target/private-boundary-replay")
root.mkdir(parents=True, exist_ok=True)
tokenizer_source = Path(hf_hub_download(
    repo, "tokenizer.json", revision=revision, local_files_only=True,
))
tokenizer = IndexedTokenizer.compile(tokenizer_source, root / "tokenizer")
control = build_experiment(
    pllm.Model.hf(repo, revision=revision), tokenizer=tokenizer,
    private_boundary=False, root=root, max_input_tokens=150,
)
control.resolve()
(root / "control.json").write_bytes(control.canonical_bytes())
```

Supply private prompt files whose rendered input counts match the selected
workload. Each invocation creates its own cohort; do not pool their replicates:

```bash
python scripts/benchmark_private_boundary.py \
  --baseline-spec target/private-boundary-replay/control.json \
  --prompt-file "$PRIVATE_48_ROW_PROMPT" --input-tokens 48 \
  --replicates 3 --wan --source-case source-present \
  --output target/private-boundary-replay/48-sdk.json

python scripts/benchmark_private_boundary.py \
  --baseline-spec target/private-boundary-replay/control.json \
  --prompt-file "$PRIVATE_150_ROW_PROMPT" --input-tokens 150 \
  --replicates 3 --wan --source-case source-present \
  --output target/private-boundary-replay/150-sdk.json
```

The stored results can also be revalidated without checkpoint execution:

```python
import json
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from benchmark_private_boundary import verify_results

for rows in (48, 150):
    path = Path(f"docs/evidence/private-boundary-qwen-{rows}-sdk.json")
    report = json.loads(path.read_bytes())
    assert report["complete"] is True
    assert report["verification"] == verify_results(
        report["results"], replicates=3, input_tokens=(rows,), wan=True,
        source_lock_digest=report["source_lock_digest"],
    )
```

Installed-build and measurement-script commitments remain in each report.
Changing local artifact paths changes composition identity; it does not authorize
cross-cohort ranking. Generated functionality tests and the preceding lookup-only
acceleration result retain their separate evidence scopes.

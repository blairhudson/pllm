# Ordinary-SDK decode-cache matched benchmark

Frozen methodology for serial installed-SDK measurements. Generated SDK and
encrypted-restart controls pass; an isolated six-client, eighteen-request smoke
cohort preserves state/history/RNG and verifies zero material on full hits.
These checks do not establish real-checkpoint savings or Low-tier qualification.
Completed checkpoint results are published separately from this protocol.
Local adapters share growing benchmark gates.

## Frozen cells

`scripts/benchmark_decode_cache_sdk.py` and
`examples/benchmarks/decode_cache_sdk.py` declare two three-request scenarios,
each with one repeated private prompt, cap eight, temperature `.7`, top-p `.9`:

* Same seeds `[7,7,7]`: exact repeated path should become fully warm.
* Distinct seeds `[7,11,19]`: report observed branch variation without forcing
  divergence or selecting policies from outputs.

Three policies share generated-prefix canonical execution, i8/i8 `prefix_f32`,
same source/indexed tokenizer, paged bundle/preparation, demand/on-demand
inventory, existing 32 MiB prefix cache and 128-position full-cap state bound:

1. Ordinary full-allowance inventory.
2. Fixed one-selection epochs, no decode disk owner.
3. Fixed one-selection epochs plus caller-owned encrypted `DecodeCache`.

Ordinary and fixed-one controls remain separate: candidate cold epoch regression
must not disappear into comparisons with only a slow epoch control. Default
three alternating replicates yield **18 fresh clients and 54 requests**. Each
client has fresh role children, empty bundle cache and fresh disk cache/key;
three requests reuse only that client's owners. Odd replicates reverse all six
cells. Real cohorts require 3–12 replicates; tiny permits one and uses a generated
source/small prompt but retains cap eight and all policies/scenarios.

Semantic runtime owns `default_rng(42)`; NumPy-global seeding has no effect.
Each serial request uses a **harness-only fixture**: temporarily wrap
`SemanticDecoderRuntime.sample`, seed the actual sampling runtime once with
`numpy.random.default_rng(seed)`, then call the original sampler unchanged.
Constructor bypass cannot silently defeat seeding. Restore the original sampler
in `finally`, including cancellation and failure. `harness_numpy_seed` plus the
explicit `sampling_fixture` field identify this fixture, not an SDK seed API.
Unit tests exercise the actual semantic sampler without checkpoint loading,
including constructor-bypassed runtimes, same/distinct seeds, and restoration.
Only weak runtime references and fixed-size salted witnesses are retained.
Initial/final RNG states and a chained per-selection RNG/token trajectory are
committed with the private salt; missing sampling witnesses reject a cell.
No caller concurrency or global NumPy RNG mutation. Cached logits are
sampled afresh by ordinary runtime; benchmark never substitutes cached token IDs.
Final logits/KV bit patterns, actual structured message history, rendered/raw
history, pending token, output and actual request RNG witnesses are
salted commitments matched separately for every scenario/request across policies
and replicates. Same-seed outputs must match within each client. Distinct seeds
need not force different outputs; actual variation is recorded.

## Installed API and evidence gates

The small `owner_factory` follows current public owner signature:

```python
from pllm.state.decode import DecodeCache
owner = DecodeCache(directory, key, **OWNER_LIMITS)
client = pllm.OpenAI(experiment=experiment, decode_cache=owner, ...)
```

Fresh owner uses no trusted head. Directory must not preexist at first bind;
cache parent is a private temporary cell directory. Key goes to client on stdin,
as do prompts and cohort salt. Head/index/page names, tokens, keys, logits and
private text are never published. This cohort does not test fresh-interpreter
restoration; parent's generated integration tests cover trusted-head restart.

The owner exposes `storage_metrics`; SDK constructor and audit counters
may be absent in an older installed build. `api_boundary()` archives missing interfaces and
blocks measurement, while `--preflight-only` preserves estimates. Structural
presence is **not integration validation**: generated controls must pass before running.

The installed smoke cohort verifies these audit contracts:

* `WORK_COUNTERS`: cumulative per-response semantic counts for equivalent
  requested prefill tokens/evaluated decode paths, actually remote-executed prefill
  tokens, terminal-only prefill rows and decode steps. Values are derived by
  runtime execution, not by lookup labels. Counter names currently are
  `generation_requested_prefill_tokens`, `generation_requested_decode_steps`,
  `generation_remote_prefill_tokens`, `generation_remote_terminal_prefill_rows`,
  `generation_remote_decode_steps`.
  Requested decode steps include cache-hit evaluated paths but exclude the final
  pending/unexecuted cap selection: cap eight means seven decode paths. Natural
   EOS after N visible tokens evaluates N decode paths. Terminal prefill rows are
   one for fresh prefill and zero for complete-root reuse. Ordinary resident
   prefix reuse can restore a proper prefix even for an identical request; its
   continuation executes every remaining suffix row at all stages, including the
   terminal stages. The ledger charges those actual suffix rows rather than
   assuming zero or fresh-prefill pruning. Claimed prepared rows must equal semantic executed remote
  stage rows, not authorized/burned minimum rows.
* `CACHE_COUNTERS`: `decode_cache_root_hits`, `decode_cache_branch_hits`,
  `decode_cache_misses`, `decode_cache_skipped_remote_rows`,
  `decode_cache_skipped_prefill_tokens`, `decode_cache_skipped_decode_steps`,
  `decode_cache_fully_warm_responses`. All integer, monotonic, actual counters.
  Skipped rows mean scheduled remote rows eliminated before inventory minting,
  not lookup counts multiplied by guessed rows. Fully-warm response marker must
  agree with zero executed work and zero newly issued/reserved/preparation bodies.

Mapping changes create a new frozen cohort; no old evidence is relabeled. Missing
actual work is null, never zero. Fully warm same-seed requests two and three must
exercise root and executed-child hits, execute no body work and mint no material.
Partial branch hits may still require fresh material; actual issued/burned rows
and correction bytes remain charged. Avoided correction bytes stay unknown:
matched measured totals can be compared, but labels do not synthesize byte savings.

## Memory, disk, computation and traffic

Owner limits: 128 nodes, **eight staged states**, 8192 pages, 16384 files, **256 MiB disk**,
2 MiB index, **8 MiB snapshot**, **32 MiB transient envelope**, 128-token paths. All
staged root/children publish only at successful response ACK, so the staged bound
covers the entire eight-selection response, not a single epoch. Key is
32 bytes and caller-owned. Plain indexes/decryption buffers/native snapshots can
overlap and are priced separately from ordinary full-cap roots/prefix-cache
owners. All encrypted file reads/writes, index publication, cold state staging,
and retained orphan files are charged using actual owner counters. Request disk
deltas are subordinate to lifecycle totals; shutdown/setup residual is retained,
not discarded. Logical encrypted bytes are not physical filesystem block writes
or energy. No key/head/index content is serialized to evidence.

Writer-lock/filesystem metadata and journal I/O are outside those logical byte
counters, not measured zeros. Cache disk capacity is added to existing compiled
cache/bundle disk requirements and reserve before admitting a cell.

Decode composition alone sets `decode_max_bytes=32 << 20`; controls omit it.
Native/compiler admission already charges this **additional 32 MiB owner** above
ordinary prefix-cache/full-cap owners. The source-derived Python native memory
report confirms `estimate.components.client_decode_owner_bytes`, included in
client allocation before 25% slack. Driver requires exact agreement with the
composition bound and owner transient maximum; **no second RAM addition or
fictitious helper/fallback**. Missing priced field blocks admission. Index,
decryption/page frames and snapshot overlap must fit this envelope.
Base group admission already includes topology,
full-cap native states, three retained responses/metadata and 128 MiB group helper;
launcher/relay allowance is another 256 MiB. Exact source-dependent total comes
from preflight. Existing host reserve/swap guard and exclusive `fresh_workers`
frontier lock remain unchanged. Each cell rechecks composition-priced admission
plus the explicitly additional disk capacity.

Largest whole-client OS lifetime RSS across every replicate is reported **per
policy** against **268435456 bytes**, never replaced with median, owner estimates,
or sum of independent peaks. Client CPU/RSS include imports, hashing and client/
disk-owner close. Preparation/Inference CPU run from process birth through client
exit; RSS sampled every 50 ms after startup. Their startup high-water and separate
shutdown CPU are unknown. Aggregate cell CPU includes launcher, relays and all
reaped children through cleanup. TTFT, N−1 decode TPS (epoch pauses included),
request CPU/time, covered body ledger and disk I/O are retained individually for
cold and later requests; actual warmth is measured, not presumed from index.

Semantic schedule dimensions, terminal pruning and actual execution counts give
both requested/equivalent body linear MACs and **actually executed** body linear
MACs. >=90% remote gate uses the latter denominator, excluding eliminated work;
remote/equivalent ratio is separately reported. Fully eliminated requests have
zero work and null remote fraction—not an invented 100%. Schedule rejects added
client dense work if its remaining remote share falls below threshold. This gate
does not describe whole CPU or all body operator FLOPs: head, attention,
normalization, masking and Preparation are excluded; broader fraction stays null.

Native shared 100/40 Mbps, no added RTT. Complete all-link TCP streams include
relay startup through role/network shutdown. Covered v3 application bodies include
cold bundle, actual correction material (used or unused/burned), admission,
ACKs and cleanup exactly once. Per-request body windows plus lifetime total retain
setup/shutdown costs. Audit deltas preserve the cumulative supported counter namespace even when all ACK/preparation deltas
are zero, so an all-hit request retains v3 coverage rather than reverting to v2.
Source-only accounting tests verify zero ACK bodies are measured zero with
complete namespace, not unknown missing counters.
Existing partial base/output/residual decomposition is reused;
unknown classification stays null/residual, never extrapolated bytes-per-row.

Retired lifetime row counters must be nonnegative integers, with
`issued = reserved + discarded` and `reserved = claimed + burned`.
Request-window deltas remain separate: prepaid setup material is not relabeled
as request-issued material. Evaluated positions/pending counts must agree with
exactly N-1 decode paths at cap truncation and N paths at natural EOS.
Publisher/checkpoint/tokenizer artifact distribution/disk and physical wire are
separate unknowns. Source lock, canonical configuration, native semantic geometry,
installed build and all benchmark/dependency source hashes are frozen.

Local script bootstrap uses `-I -S` and explicitly adds only installed wheel
library paths plus the trusted measurement script directory. Python startup,
user-site and `.pth` injection is disabled for launcher, metadata inspection,
fresh clients and roles. Every measured Python child checks actual installed
SDK source/native file hashes against the captured BuildRef before its target
runs. Identity hashing/bootstrap CPU and RSS remain inside process-birth windows;
complete cell wall time includes them. Historical runner/helper files are unchanged.

Failure/interruption records are bounded by at most 72 planned client attempts.
Available actual disk/body/stream counters survive failed work; abrupt child death
can leave client counters unknown. Exception messages/logs are excluded. Each
invocation owns a fresh private salt and immutable progress report; do not pool
separate, rejected or interrupted cohorts. Only fully checked cells and complete
cell coverage produce `complete=true`; failing Low RSS does not erase results.

## Commands: after installed-wheel integration validation

```sh
target/frontier-sdk-env/bin/python -I -m pytest -o pythonpath= \
  tests/test_benchmark_decode_cache_sdk.py -q
target/frontier-sdk-env/bin/python -m ruff check \
  scripts/benchmark_decode_cache_sdk.py tests/test_benchmark_decode_cache_sdk.py \
  examples/benchmarks/decode_cache_sdk.py
target/frontier-sdk-env/bin/python scripts/benchmark_decode_cache_sdk.py \
  --tiny --preflight-only --output target/decode-cache-tiny-admission.json
target/frontier-sdk-env/bin/python scripts/benchmark_decode_cache_sdk.py \
  --tiny --replicates 1 --output target/decode-cache-tiny.json
target/frontier-sdk-env/bin/python scripts/benchmark_decode_cache_sdk.py \
  --preflight-only --output target/decode-cache-admission.json
target/frontier-sdk-env/bin/python scripts/benchmark_decode_cache_sdk.py \
  --replicates 3 --output target/decode-cache-sdk.json
```

`--smoke` aliases `--tiny`; `--artifacts` selects a pre-positioned directory
containing `indexed.json`. Every output path must be new. Missing interfaces or
failed parity/accounting gates reject rather than being worked around.

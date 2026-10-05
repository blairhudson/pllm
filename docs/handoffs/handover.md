# PLLM handover for OpenCode v2

> Historical handoff. The later [completed 4B diagnostic](../evidence/qwen3-4b-client-paged-2026-10-05.md)
> records the successful run and the user-requested guard change. The status and
> memory policy below describe the earlier handoff point.

- **Prepared:** 2026-10-05
- **Repository:** `/Users/blairhudson/workspace/pllm`
- **Branch:** `main`
- **Verified implementation HEAD:** `c4f6a7c` — `Reuse request tokenization and defer unused decoder vocabularies`
- **Repository state before writing this handover:** clean; the implementation and evidence described below are committed.

This file replaces the lost conversation context. The user is moving from the
current OpenCode process to OpenCode v2 because the coding harness itself has
been consuming enough RAM to prevent the requested model benchmark. Lower memory
use in v2 is the user's expectation, not something we have measured yet.

## 1. Read this first

The latest immediate request is: **run a complete Qwen 4B benchmark now that the
coding-agent memory problem may be addressed**. The selected model is specifically
**`Qwen/Qwen3-4B`**, not Qwen3.5-4B, using the pinned revision below.

The overriding optimization priority is **10× lower whole-client memory** while
preserving the admitted numeric behavior, input/privacy boundary, and predominantly
remote body computation. The user explicitly made client RAM the top priority
over the preceding network/TPS research.

Current status:

- Several substantial memory reductions are implemented through the normal SDK,
  gateway, native compiler/runtime contracts, and benchmark path.
- **10× whole-client memory is still unmet. Do not report it as achieved.**
- A real Qwen3-4B **storage/import/masks/token-head-only** probe measured **9.44×**
  lower process peak RSS. It did **not** execute decoder body, attention, KV, or a
  complete provider-backed response.
- A matched **complete Qwen2.5-0.5B response** measured **665.49 → 306.43 MB peak
  client/dashboard RSS**, or **2.17×**.
- A separate complete-response tokenizer ablation measured **308.003 → 290.963 MB**,
  another **5.53%** in its own matched pair. These cohorts have different salts;
  **do not multiply their reductions or invent a new matched combined factor**.
- The conservative Qwen3-4B client allocation estimate is now **1.07 GiB**, versus
  the historical **8.25 GiB** estimate. The initial estimate target is **0.83 GiB**.
  Some of the estimate improvement is accounting correction, not new allocation
  savings.
- **A complete Qwen3-4B benchmark has not succeeded in this work.** Recent attempts
  were rejected by whole-topology memory admission before provider launch.
- The latest implementation, tests, docs, and retained evidence are committed.
  This handover itself was created after that clean implementation baseline.

### Immediate next actions

1. Read this file, inspect `git status`, and confirm which revision is running.
2. Sample current host RAM. Do not assume the v2 restart has solved headroom.
3. Run the **paged-client native CPU preflight** in section 3.
4. If admitted, run one complete 4B cold request with the same configuration.
5. Inspect the report's completion/privacy/numeric checks, measured client RSS,
   token counts, CPU/timing, and memory guard. Confirm owned providers shut down.
6. Preserve the exact report and a scoped explanation. Only then broaden to
   resident/paged or historical-allocation comparisons if each candidate fits.
7. Keep the 10× whole-client target open until matched end-to-end evidence earns it.

Do not start with the three-way historical-control probe: its eager/resident arm
requires more headroom than the optimized paged candidate.

## 2. Why the 4B benchmark was blocked

The machine is an **Apple Silicon macOS host with 32 GiB physical RAM**. The normal
admission policy reserves **8 GiB** for the OS and other applications, then compares
the remaining available physical memory with a conservative whole-topology estimate.
Swap is not capacity.

For `examples/benchmarks/client_memory.py:qwen3_paged`, at a **64-input / 8-output
token bound**, the last preflight reported:

| Process / budget | Estimated peak including allocation margin |
| --- | ---: |
| Inference | 5.98 GiB |
| Preparation | 5.52 GiB |
| Client | 1.07 GiB |
| Native topology total | **12.57 GiB** |
| Host reserve | **8.00 GiB** |
| Available host RAM needed to admit this estimate | **about 20.57 GiB** |

The exact reported native total was `13,498,754,870` bytes. This is an estimate,
not an observed 4B peak and not a guarantee against OOM.

The original BF16 checkpoint is approximately **8.04 decimal GB / 7.49 GiB**.
It is not the total application-memory requirement. There are two provider
processes, retained int8 matrices, correction inventory, loading temporaries,
client runtime state, and allocation margins. Provider allocations dominate the
current total even though client weight residency is much smaller.

### Last two retry attempts

- Before restarting the old harness, OpenCode reported roughly **18.14 GiB RSS**.
  A fresh preflight rejected for insufficient headroom.
- The user restarted OpenCode and asked to try again. The new process was already
  reporting roughly **11.27 GiB RSS** after reopening the session. The latest
  preflight had **8.64 GiB safe headroom**, short of **12.57 GiB** by **3.93 GiB**.
- At other moments available RAM and safe headroom fluctuated substantially.
  Those figures are historical snapshots, not facts about the new v2 session.
- The previous suggestion was to run from a separate terminal with OpenCode
  completely closed. The user instead chose to migrate to v2 and requested this
  handover. We have not verified a v2 memory improvement.

An earlier 4B attempt exhausted this 32 GiB host after Docker Desktop was raised
to **20 GiB**. That attempt produced no successful benchmark. **Do not repeat that
approach, enlarge Docker automatically, disable admission, count swap as RAM, or
weaken reserves merely to force a run.**

Relevant implementation: `python/pllm/runtime/benchmark_memory.py`.

- Per-allocation estimate slack: **25%**.
- Preflight reserve: `max(2 GiB, total physical RAM / 4)`.
- Runtime watchdog samples every **250 ms**.
- Runtime available-memory floor: `max(1 GiB, preflight reserve / 2)`.
- Abort on new swap growth over **256 MiB**, including growth after recovery.
- Native, Docker, and auto selection preserve the chosen Pipeline.
- Docker uses existing VM capacity and hard provider limits; potentially older
  images retain conservative legacy allocation bounds.
- Enforced WAN/link shaping requires Docker and cannot silently fall back to
  an unthrottled native measurement.

## 3. Exact commands to resume

Run these **from `/Users/blairhudson/workspace/pllm`**. The local `.venv` and cached
checkpoints were already present. Use a fresh output filename; do not overwrite
evidence with `--force` casually.

### Current headroom

```bash
.venv/bin/python -c 'from pllm.runtime.benchmark_memory import host_memory, GiB; h=host_memory(); print({"total_gib":h.total/GiB,"available_gib":h.available/GiB,"reserve_gib":h.reserve/GiB,"safe_headroom_gib":max(0,h.available-h.reserve)/GiB,"swap_used_gib":h.swap_used/GiB})'
```

### 4B optimized native preflight

This exact configuration was checked and rejected for RAM in the previous session:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pllm benchmark run \
  --experiment examples/benchmarks/client_memory.py:qwen3_paged \
  --trust-python --backend native \
  --prompt "Explain private inference in one sentence." \
  --max-output-tokens 8 --warmups 0 --repetitions 1 \
  --temperature 0 --capture-output-digest \
  --preflight-only --format json
```

`--preflight-only` can exit successfully while returning `admitted: false`.
Inspect the JSON, not just the shell exit status. The CLI JSON envelope places
the memory cohort under `data`; each candidate has its own `admitted`, `estimate`,
`host`, and backend reasons. The Python `pllm.metrics.benchmark_memory(...)` API
returns the memory report directly.

### Full 4B benchmark, only if admitted

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pllm benchmark run \
  --experiment examples/benchmarks/client_memory.py:qwen3_paged \
  --trust-python --backend native \
  --prompt "Explain private inference in one sentence." \
  --max-output-tokens 8 --warmups 0 --repetitions 1 \
  --temperature 0 --capture-output-digest \
  --output qwen3-4b-paged.json
```

The full command still performs admission and runtime pressure monitoring.
The **64 input tokens are a configured maximum**, not a claim that this prompt
actually tokenizes to 64. Record the authoritative input/output counts. The
64+8 estimate allows 71 executed preparation rows; actual demand can be smaller.

The selected `qwen3_paged` Experiment currently declares:

- `Cpu(threads=1)` and W8A8 `SymmetricPerRow(causal_reduction="prefix_f32")`.
- `MaskedLinear(output_encoding="row_residues")`.
- Request-sized, on-demand preparation with `rows=1` and `stage_window=4`.
- `ClientPrefixReuse(max_bytes=128 << 20, fixed_input_tokens=64, generated_prefixes=True)`.
- `ClientBundleTransport("artifacts", compression="zlib", batch_objects=64, storage="paged")`.
- One request, a 64-token input bound, and at most eight generated tokens.

Conflicting CLI overrides of immutable Experiment choices reject. Changing input,
cache, placement or inventory parameters creates a different candidate; record
that change rather than calling it the same matched experiment.

No `qwen3-4b-paged.json` was found when preparing this handover. Check for a new
report before launching a duplicate run in case the user runs it separately.

### Memory-specific probes

```bash
# Complete Qwen2.5-0.5B response, fresh processes and one cohort salt.
.venv/bin/python scripts/probe_client_runtime_memory.py \
  --output client-memory-new.json

# Same harness for 4B. This is a three-arm comparison, including a larger control.
.venv/bin/python scripts/probe_client_runtime_memory.py --four-b \
  --output client-memory-4b-new.json

# Single paged candidate with phase diagnostics; not a matched comparison.
.venv/bin/python scripts/probe_client_runtime_memory.py --four-b --profile-only \
  --output client-memory-4b-profile-new.json

# Matched eager/lazy decoder-tokenizer allocation ablation.
.venv/bin/python scripts/probe_client_runtime_memory.py --tokenizer-ablation \
  --output tokenizer-memory-new.json

# Small transport/CLI smoke test, not model-quality or large-memory evidence.
.venv/bin/python scripts/probe_client_runtime_memory.py --tiny \
  --output client-memory-tiny-new.json
```

The parent prices coordinator overhead and extra eager-control mask allocations;
it does not bypass the normal topology check. It gives each child a fresh public
artifact cache, preserves the shared Hub cache, and shares one private comparison
salt. Reports are checkpointed after completed candidates. A file left with
`status: "running"` and incomplete records is **not** a successful result.
Nested phase diagnostics are non-additive. Their tokenizer call-site traces
contain code locations, not prompts, tokens, or payloads.

The isolated **4B storage-only** probe already works at lower headroom:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_client_storage.py \
  --source "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
  --output client-storage-4b-new.json
```

It needs approximately 3.92 GiB extra headroom beyond the reserve, but it is not a
substitute for the requested complete response. Its bounded publisher discards
body weights after validating them; that is a probe fixture, not a serving role.

## 4. Models and environment

| Purpose | Model | Pinned revision |
| --- | --- | --- |
| Main existing real-model controls | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Immediate requested larger benchmark | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |

- Both use the standard shared Hugging Face Hub cache, normally under
  `~/.cache/huggingface/hub/`. Preserve cache/source-lock identities.
- There is older **Qwen3.5-4B text-reference evidence** in the repository. It is
  a different model and is not evidence that this Qwen3-4B benchmark succeeded.
- Runtime package/distribution: `pllm.run`; import and CLI: `pllm`.
- Current project version: Rust `0.1.0-alpha.2` / Python `0.1.0a2`.
- Last benchmark interpreter: Python **3.13.15**. Supported Python range is
  **3.11–3.13** (`>=3.11,<3.14`).
- Rust toolchain: **1.85.1**, from `rust-toolchain.toml`.
- Build backend: Maturin; extension: `pllm._native`; PyO3 abi3-py311.
- Source code: `python/pllm/` and `crates/`; do not create a second root Python
  package named `pllm`.
- Prefer the checkout's `.venv/bin/python` or `uv run --no-sync` when validating
  this code, so an unrelated globally installed CLI is not measured.

If rebuilding the native extension is needed, the session used:

```bash
VIRTUAL_ENV="$PWD/.venv" CARGO_BUILD_JOBS=1 .venv/bin/maturin develop --release
```

`uv sync --all-extras --dev` is the repository setup command when dependencies are
needed. Consult `CONTRIBUTING.md` and `pyproject.toml` rather than changing global
tool installs during a benchmark. Keep build concurrency bounded on this host.
`PLLM_KERNEL_BACKEND=python` is a correctness reference, not eligible native
performance evidence or a valid way around native allocation admission.

## 5. Runtime and architectural boundaries

PLLM is both a private-inference runtime and a typed, reproducible experimentation
system. It has immutable `Model`, component, `Pipeline`, `Experiment`, deployment,
budget, plan, and evidence records. New methods should compose through these
existing contracts and the ordinary SDK/gateway/benchmark path.

### Public prepared inference

The main benchmark composition is Client + Preparation + Inference:

1. The client retains private prompts, token IDs, activations, nonlinear/attention
   work, KV state, and outputs.
2. Preparation expands a client-provided stage-bound seed and computes `W*r-s`.
3. It pushes corrections to Inference's authenticated fixed endpoint.
4. After every stage is acknowledged and the inventory is sealed, the client
   reserves one-use rows.
5. Online execution sends a ticket plus `x-r`; Inference consumes the correction
   once and returns `W*x-s`; the client adds `s` and center-decodes.
6. Reservations are burn boundaries. Cancellation, malformed replies, failure,
   and unused reserved rows cannot turn material into reusable inventory.

The public prepared threat model is honest/semi-honest, non-colluding Preparation
and Inference. Co-located processes and containers do not demonstrate independent
operators or non-collusion. Freivalds verification adds its explicit correctness
contract; it is not a blanket malicious-security proof. Public weight delivery
does not provide model confidentiality. Complete protected proprietary-model
execution remains a separate, incomplete research track.

### Other supported or researched paths

- `TwoOnlineOffsetCpu`: two independently masked online workers; exact additive
  reconstruction. Seeded ingress and packed output residues are explicit options.
- `ClientOnlyCpu`: a useful numeric/local baseline, not a way to satisfy remote
  private-inference goals by silently moving the model onto the client.
- Existing local roles can use native CPU or an explicitly chosen Apple Metal
  component where admitted. Metal is not automatically faster and adds separately
  priced unified-memory snapshots.
- Python owns orchestration, SDK/configuration, HTTP/WebSocket lifecycle,
  source locks, diagnostics, and independent references. Rust owns hot integer
  arithmetic, codecs, secret expansion, one-use opaque handles, and native checks.
- Preserve model-neutral semantic lowering. Implement operator/state contracts,
  not family-name switches or benchmark-only model exceptions in the live runtime.

The current memory-oriented profile keeps body linear computation remote; client
token lookup/head weights are public boundary assets. Paged lookups are **local
disk reads**. They never become remote requests selected by a private token ID.

## 6. Memory work completed, in chronological order

### A. Whole-topology admission and one immutable native weight snapshot

Commit `eafb8d8`:

- Added conservative native/Docker/auto memory admission and host-pressure guards.
- Streamed floating-point import/quantization and validation in bounded chunks.
- Reused read-only NumPy aliases of native immutable matrix storage via
  `CompiledMatrix.weight_view()` instead of retaining duplicate stage arrays.
- Preserved actual source/weight validation and native-owner lifetime.
- Historical four-stage loader/kernel control: **522.08 → 213.06 MB peak RSS**,
  with equal hashes and integer outputs. This is not whole-decoder evidence.

The shared Hub cache identity fix in `0faa785` also matters: legitimate snapshot
blobs are verified through repository content identities. Do not reintroduce
unsafe arbitrary symlink traversal or reject normal content-addressed Hub blobs.

### B. Segmented bundle serialization and direct artifact reconstruction

Commit `298d369`:

- Providers stream the existing canonical MessagePack bundle from immutable
  native-backed binary segments rather than materializing/copying the whole blob.
- Artifact export uses the same segments instead of pack/unpack/repack.
- Clients authenticate each object and the exact original ordered raw-bundle
  commitment without making another contiguous bundle copy.
- Mutable caller-owned storage is snapshotted before commitment; readonly flags
  alone are not an immutability proof.
- Wire identities and existing global zlib frame boundaries are preserved.
- Bounded serialization control: **441.94 → 244.83 MB peak RSS**, 44.6% lower.

Evidence: [bundle memory](../evidence/bundle-memory-2026-10-05.md).

### C. Role-specific provider weight retention

Commit `0f8b616`:

- Preparation retains provider-executed matrices only.
- Inference/offset workers retain required delivery matrices, but can drop a tied
  token-table orientation already supplied by the head. Auxiliary token slices
  retain independent ownership.
- Omitted executable weights are still imported/validated; checked digests,
  scales, bounds, biases and source metadata remain. Omitted matrices reject
  execution rather than becoming executable placeholders.
- Direct all-stage engines and externally supplied engines keep their ownership.
- Synthetic retained-i8 storage fell 40% for preparation and 20% for delivery;
  peak reductions were smaller because import temporaries still mattered.

Evidence: [role weight memory](../evidence/role-weight-memory-2026-10-05.md).

### D. Seed-backed, one-use lazy client masks

Commit `f3a917f`:

- Native SHAKE256 cursors and a row-burn ledger replace fully expanded client
  masks for every stage/row.
- Only claimed active-stage rows expand. Ordered claims advance the stream;
  disjoint out-of-order leases can replay/skip a bounded XOF prefix.
- Exact original domains, mask bytes, tickets, rings and wire protocol are kept.
- Closing a lease burns unused ranges without expanding them. Returned owned
  buffers remain valid for in-flight work. Python retains the eager oracle for
  explicit correctness-reference mode and the test-local historical control.
- 144-stage, 71-row 4B-geometry probe: retained masks **507,764,736 → 224,496 B**;
  isolated process peak **628.16 → 89.60 MB**; combined mask CPU **1.033 → 0.924 s**.
- Expansion moves from inventory construction into row claims; do not claim an
  end-to-end latency win from that microbenchmark.

Evidence: [client masks](../evidence/client-mask-memory-2026-10-05.md).

### E. Authenticated paged weights through the ordinary SDK

Commit `f695058`:

```python
from pllm.protocols import ClientBundleTransport

delivery = ClientBundleTransport(
    "artifacts", compression="zlib", batch_objects=64, storage="paged"
)
```

- `storage="paged"` is an explicit `delivery` option. The resident default keeps
  its old composition identity. Paging requires artifact delivery.
- Integer weight objects stream into digest-verified temporary files, then Rust
  makes independently authenticated private file snapshots.
- No full client weight array or contiguous reconstructed raw bundle is required.
- All public matrix objects are fetched in full, independently of input tokens.
- Native page kernels preserve exact i8, modular and wrap32 behavior. Local row
  and column gathers support tied/untied and auxiliary token orientations.
- One executor is shared across matrices, avoiding a thread pool per matrix.
- SDK row tiling preserves native per-call workspace limits for larger admitted
  linear/gather requests.
- Failed imports close provisional snapshots; temporary raw files close on
  success/failure. Cache corruption is repaired; eviction/source mutation cannot
  alter an admitted snapshot.
- Prepared, verified, offset, prefix reuse, CPU client placements, raw/zlib and
  batching compose. Client-owned paged body matrices reject Metal snapshots.
- Public cache, temporary import files, and private snapshots are distinct disk
  owners. OS filesystem cache is outside process RSS and may use host memory.
- Paged storage and matrix-sized serialization-copy removal are source/shape/
  scale/weight-digest bound in the compiled runtime, not a benchmark-only path.

Evidence: [4B client storage](../evidence/client-storage-qwen3-4b-2026-10-05.md).

### F. Live workspace accounting, response-state cleanup, isolated measurement

Commit `7d8500a`:

- **The semantic decoder already released last-use intermediates.** We discovered
  the old estimator was summing all prefill/decode graph outputs. Do not present
  a corrected estimate as a new liveness optimization.
- `decoder_memory.py` follows whole allocation roots through views and grouped
  projection slices, includes numeric/FFI scratch and loop-local references, and
  charges geometric full-KV capacity plus snapshot/growth overlap.
- Unknown operators/state layouts retain the historical unreleased-output bound.
- The benchmark driver was retaining response-owned KV/history across runs.
  It now retires that state after results are collected. Independently sealed
  prefix-cache reuse still works. General SDK application history semantics were
  not changed.
- Ordinary reports now contain `client_process_memory`: current RSS and OS
  **process-lifetime** high-water RSS, including the in-process dashboard.
- Fresh-process comparisons are implemented in
  `scripts/probe_client_runtime_memory.py` with one cohort salt and normal guards.

Evidence: [complete client memory](../evidence/client-runtime-memory-2026-10-05.md).

### G. Remove redundant decoder tokenization/vocabulary allocation

Commit `c4f6a7c`:

- The SDK had already encoded the prompt, yet the decoder constructed another
  tokenizer and the SDK re-encoded via `runtime.encode_prompt()`.
- Decoder tokenizer construction is now lazy for direct string helpers.
- SDK execution uses the already-admitted `full_prefill_ids`, the request
  tokenizer for continuation suffixes, and the same tokenizer for output decoding.
- No new persistent tokenizer cache or private-input retention was introduced.
- Direct string-helper behavior and explicit tokenizer assignment remain usable.
- An initial lazy-property change alone failed to reduce memory because later SDK
  accesses recreated the tokenizer. Call-site diagnostics found those accesses;
  the final change removes them too.
- Final matched paged-client pair: **308.003 → 290.963 MB**, **5.53%** lower peak;
  tokenizer constructions **4 → 3**; outputs/privacy checks match and swap growth
  is zero. CPU/latency values are single-sample observations.

Evidence: [tokenizer memory](../evidence/tokenizer-memory-2026-10-05.md).

## 7. Results and claim boundaries

All MB figures below are decimal; GiB admission figures are binary.

| Scope | Control | Candidate | Valid conclusion |
| --- | ---: | ---: | --- |
| Mask-only process, 4B geometry | 628.16 MB | 89.60 MB | 7.01× mask-process peak reduction |
| Real 4B import + masks + token/head operations | 1,471.61 MB | 155.98 MB | 9.44× in this restricted scope |
| Complete 0.5B response, eager/resident vs lazy/paged | 665.49 MB | 306.43 MB | 2.17× complete client/dashboard peak reduction |
| Separate complete 0.5B tokenizer pair | 308.003 MB | 290.963 MB | 5.53% in this pair |
| Historical vs current 4B client allocation estimate | 8.25 GiB | 1.07 GiB | Accounting/implementation estimate change; not measured RSS |

The 4B storage-only pair used the actual pinned checkpoint's public bundle and
compiled semantic binding, 71 mask rows over 144 body stages, 64+7 token-lookup
rows and eight head operations. It did not run those body stages. Its combined
client CPU increased **6.062 → 6.985 s (15.2%)**. Paging is a RAM/disk/CPU trade.
Those head calls consumed lookup vectors, not a complete decoder's hidden state.

The complete 0.5B response cohort used **37+8 tokens**, W8A8 `prefix_f32`, greedy
sampling, cold request/no warmup, and all body linear work remote:

| Allocation path | Peak RSS MB | Cold client CPU s | Request time s |
| --- | ---: | ---: | ---: |
| Eager masks + resident boundary | 665.49 | 14.23 | 24.54 |
| Lazy masks + resident boundary | 582.84 | 12.42 | 20.07 |
| Lazy masks + paged boundary | 306.43 | 10.97 | 19.33 |

Important qualifications:

- Checked tiny prefill/decode **logits and KV arrays** agree, not just final text.
  Real measured cohorts have matching generated-output digests and runtime/privacy
  checks. This is not broad generation-quality validation against float32.
- W8A8 lossless transport/storage preserves the existing W8A8 arithmetic; it does
  not make quantization identical to the original BF16/float32 checkpoint.
- These memory measurements are single matched samples, not distributions.
- Resident head execution uses its existing native default thread pool, whereas
  paged execution follows the selected CPU thread setting. Do not overclaim CPU/
  latency ranking without controlling the effective kernel resources.
- Different salts, prompts, numeric policies, token counts, cache states, role
  graphs, or kernel scopes cannot be merged into a matched comparison.
- The last attempted final three-way rerun after tokenizer work was RAM-rejected
  before model inspection. There is no final combined matched factor beyond the
  independently reported cohorts.
- No successful full 4B response means no measured 4B TTFT, full latency, decoder
  throughput or whole-client peak RSS from this work.

## 8. Earlier network/TPS work to retain

The conversation previously prioritized network reduction, then WAN-constrained
TPS, then making larger-model benchmarks fit, and finally urgent client RAM.
Useful earlier work is implemented; do not start from scratch or re-promote
rejected methods without a materially new hypothesis and evidence.

### Implemented useful components

- **Seeded additive ingress:**
  `TwoOnlineOffsetLinear(input_encoding="seeded")`. A fresh domain-bound seed
  replaces one worker's input upload; the other sees the complementary share.
  Never give a worker both shares or a reconstructable complete mask.
- **Exact row output residues:** `output_encoding="row_residues"`. Widths depend
  on public weight-derived signed bounds, not private activation observations.
- **Seed-first dispatch:** `dispatch="seed_first"` overlaps one worker with
  complement construction and the other exchange.
- **Coalesced public artifacts:** `batch_objects=64`, at most 1 MiB raw groups,
  verified ordered object identities; large objects stream separately.
- **Windowed preparation:** `stage_window=4`, bounded outstanding work, all
  acknowledgements required before sealing, no preparation during online chat.
- **Duplex prefill:** `MaskedLinear(..., prefill_chunk_rows=4)` is explicit and
  exact, but its incremental end-to-end benefit was inconsistent. It is not an
  automatic winner and cannot remove one-row decode dependencies.
- **Canonical prefix reuse:** `SymmetricPerRow(causal_reduction="prefix_f32")`
  enables checked valid-prefix numeric equivalence. `generated_prefixes=True`
  qualifies only executed, independently admissible state; pending tokens and
  response-owned state cannot be silently relabeled as fresh-cache state.
- **State equivalence:** checked `CompiledRuntimeModel.transfer_snapshot` exists;
  automatic planner/cache migration and broad transition-cost evidence are separate.
- **Bounded network planning:** typed network snapshots, authenticated offers,
  capacity reservation/arm/release, epochs and admission exist for a bounded set
  of public topologies. Unknown required costs reject; compatibility is not benefit.

Representative retained results:

- Seeded ingress + row residues reduced matched 39+32 offset online bodies
  **227.08 → 172.73 MB (23.9%)**, with equal outputs. Not a 10× network result.
- A matched WAN cohort used one **100 Mbps down / 40 Mbps up budget per party**
  shared across peers, plus **20 ms egress delay per party** (40 ms added
  request/reply delay). It is not a free 100/40 link for every pair of peers.
- Prepared batching/window overlap improved request TPS **17.3%**.
- Seed-first offset improved request TPS **47.2%** and decode TPS **94.1%**.
- Online body sizes did not shrink in those scheduling pairs. WAN results are
  single samples and exclude provider startup/checkpoint distribution from their
  request-time denominator.

Primary WAN evidence:
[wan-tps-qwen25-2026-10-04.md](../evidence/wan-tps-qwen25-2026-10-04.md).

### Vetoed or research-only directions

- Public native rANS artifact coding lost to zlib on bytes and client decoding
  CPU. It remains a probe, not a live delivery choice.
- DPF/XOR-PIR private vocabulary pages have exact bounded native probes, but scan
  both worker tables and do not remove a still-local tied head. Not a complete
  selectable decoder boundary.
- Compact private head indices did not preserve tested winner coverage or earn
  exact certificates at useful fixed budgets. Do not replace the full head with
  approximate candidate selection.
- Compressed online weight pages saved disk but caused excessive repeated head
  CPU. Raw local pages are the memory-oriented execution path; transport zlib is
  a separate one-time delivery decision.
- Masked output aggregation moved client download bytes onto worker links and
  increased client CPU; smaller client traffic was not an all-link win.
- Token-local projection memoization had a small whole-body saving, not a 10× route.
- Progressive-head certificates had promising bounded arithmetic but expensive
  private refinement; not an admitted whole-decoder replacement.
- Polynomial/exact-correction, resident-share, HE projection and related methods
  have substantial research records. Simple public shifted coefficients leaked;
  sparse residual tails/material costs and numeric rollout failures vetoed tested
  variants. Short ordinary PRG seeds are not a nonlinear correlation generator.
- Fresh-prompt network 10×/100× and full protected whole-model composition remain
  open research goals. Lower byte counts cannot excuse failed numerics, privacy,
  or unbounded client/dealer/aggregate compute.

Read the outcome sections and linked evidence in:

- [network-io-10x.md](../research/network-io-10x.md)
- [network-io-100x.md](../research/network-io-100x.md)
- [next-five-network-methods.md](../research/next-five-network-methods.md)
- [low-client-network.md](../research/low-client-network.md)
- [wan-tps.md](../research/wan-tps.md)
- [network-inference.md](../plans/network-inference.md)

**Historical-text warning:** some older sections still describe work as pending.
For example, `docs/research/low-client-network.md` predates completed SDK paged import,
and portions of the network design predate generated-prefix qualification.
Prefer the latest code, recent commits, current SDK pages and dated evidence over
an old forward-looking paragraph.

## 9. File map for efficient continuation

| Area | Main files |
| --- | --- |
| User-facing memory SDK option | `python/pllm/protocols/bundle_transport.py` |
| Composition resolution/admission | `python/pllm/profiles/__init__.py`, `crates/pllm-compiler/src/lib.rs` |
| Native mask cursors | `crates/pllm-core/src/mask_stream.rs`, `crates/pllm-python/src/mask_stream.rs` |
| Mask domains/oracle | `python/pllm/runtime/preparation_protocol.py` |
| Inventory leases, bundles, local boundaries, lazy decoder tokenizer | `python/pllm/runtime/transformer_client.py` |
| SDK orchestration, streamed import, token reuse | `python/pllm/runtime/client.py` |
| Provider import and role retention | `python/pllm/runtime/transformer_engine.py`, `servers.py`, `offset_worker.py`, `cli.py` in the same runtime directory |
| Segmented wire document | `python/pllm/runtime/bundle_document.py` |
| Artifact/cache identity and batching | `python/pllm/runtime/bundle_artifacts.py` |
| Bounded frame decoding | `python/pllm/runtime/bundle_compression.py` |
| Verified temporary files | `python/pllm/runtime/bundle_storage.py` |
| Native paged matrices / shared executor binding | `crates/pllm-core/src/paged.rs`, `crates/pllm-python/src/paged.rs`, `crates/pllm-python/src/lib.rs` |
| Python paged facade | `python/pllm/runtime/paged.py` |
| Source/shape/scale/tokenizer/runtime binding | `python/pllm/runtime/model_binding.py` |
| Actual semantic execution and existing last-use release | `python/pllm/runtime/semantic_executor.py` |
| Live-root workspace estimates | `python/pllm/runtime/decoder_memory.py` |
| Host/backend admission and pressure guard | `python/pllm/runtime/benchmark_memory.py` |
| Canonical loopback benchmark/reporting | `python/pllm/runtime/benchmark_cli.py` |
| Client process driver, response-state retirement | `python/pllm/runtime/dashboard.py` |
| Matched resident/paged targets | `examples/benchmarks/client_memory.py` |
| Earlier lean/placement/offset 4B targets | `examples/benchmarks/qwen3_4b.py` |
| Mask / storage / whole-client probes | `scripts/probe_mask_memory.py`, `scripts/probe_client_storage.py`, `scripts/probe_client_runtime_memory.py` |
| Provider mechanism controls | `scripts/probe_weight_memory.py`, `scripts/probe_bundle_memory.py` |

Current memory guide:
[`docs/content/docs/sdk/evaluate/client-memory.mdx`](../content/docs/sdk/evaluate/client-memory.mdx).
The larger architecture reference is [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

## 10. Commit trail

The important recent commits are local history on `main`; inspect each with Git
when changing its contracts. Do not squash or rewrite the research history merely
to resume this work.

| Commit | Slice |
| --- | --- |
| `c4f6a7c` | Reuse request tokenization; defer unused decoder vocabularies |
| `7d8500a` | Live decoder-state pricing; benchmark KV retirement; isolated client RSS |
| `f695058` | Stream authenticated client weights into native paged storage |
| `f3a917f` | Expand prepared masks through bounded one-use native cursors |
| `0f8b616` | Role-specific executable/deliverable weight retention |
| `298d369` | Immutable bundle segments and direct artifact import |
| `eafb8d8` | Topology memory guards and shared native snapshots |
| `0faa785` | Shared Hub blob verification through repository identities |
| `69fd2e3` | WAN overlap, seed-first dispatch, batching/window/duplex integration |
| `e88d918` | Native entropy and duplex feasibility screens |
| `429f39b` | WAN evidence bound to verified delay/loss queues |
| `7ead799` | Exact reuse/transport composition and enforced WAN benchmarking |
| `9a3e791` | Native network/client-cost research primitives |
| `6b4d515` | Direct exact paged native kernels |
| `228a24e` | Private-head residual certificate gate |
| `dfda8b8` | Private vocabulary DPF/page scan measurements |
| `1cec75d` | Exact output residue packing and batched native expansion |
| `2313eab` | Context-bound seeded two-worker input shares |

The user previously asked to commit validated major slices. Continue that habit
when authorized, stage intended files only, and do not push or create a PR unless
requested. At handover preparation, all runtime changes above were committed and
the working tree was clean before adding this document.

## 11. Validation performed and commands to reuse

Validation was intentionally scoped and sometimes sharded for RAM:

- Native mask vectors, one-use ledger/replay/cancellation/concurrent leases,
  raw-page digests, source mutation, exact kernels and row/column gathers passed.
- Rust formatting and focused Clippy checks passed.
- The paged integration slice passed a 195-test focused Python set plus an
  additional provisional-snapshot cleanup test.
- A broad later attempt passed 97 checks but had 11 benchmark cases rejected for
  low host headroom. **All 11 subsequently passed in smaller fresh test processes.**
  Do not weaken admission to fix such an environment failure.
- The tokenizer changes passed 69 runtime/model/engine/E2E tests, four paged
  SDK/gateway variants, the token-ID no-tokenizer regression, and the real matched
  tokenizer ablation.
- The documentation suite passed **64 tests**. Production build succeeded with
  **404 publication records / 406 generated static pages**, including rendered
  navigation, import links, API backlinks and 84 research-paper links.
- Older research ledgers record larger historical suite counts. Those are not a
  claim that every optional/slow suite was rerun on the newest change.

Useful focused commands:

```bash
CARGO_BUILD_JOBS=1 cargo test -p pllm-core mask_stream --lib
CARGO_BUILD_JOBS=1 cargo test -p pllm-core paged --lib
cargo fmt --all --check
CARGO_BUILD_JOBS=1 cargo clippy -p pllm-core -p pllm-compiler -p pllm-python --all-targets -- -D warnings

.venv/bin/pytest -q tests/test_prepared_mask_stream.py tests/test_inventory_accounting.py
.venv/bin/pytest -q tests/test_paged_native.py tests/test_paged_bundle.py \
  tests/test_bundle_artifacts.py tests/test_artifact_delivery.py tests/test_bundle_compression.py
.venv/bin/pytest -q tests/test_decoder_memory.py tests/test_benchmark_memory.py \
  -m 'not integration' --tb=short
.venv/bin/pytest -q tests/test_benchmark_memory.py -m integration --tb=short
.venv/bin/pytest -q tests/test_runtime_model_binding.py tests/test_transformer_engine.py \
  tests/test_transformer_e2e.py -m 'not slow and not he' --tb=short

.venv/bin/ruff check python/pllm scripts tests
npm --prefix docs test
npm --prefix docs run build
```

Run only checks warranted by changes/failures. Some suites import Torch or start
multiple roles; one giant pytest process can reduce available headroom enough to
cause valid admission failures. Separate heavy integration and metadata checks
when necessary. A preflight rejection is not numerical test failure or completed
performance evidence.

Docs are authored under `docs/content/`, with Next.js/Fumadocs generation owned by
`docs/package.json`, `docs/navigation.json`, and the site scripts. Generated
reference/Markdown/publication surfaces should be regenerated, not manually
edited. Follow [docs/README.md](../README.md) and
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## 12. Known open issues and pitfalls

1. **Whole-client 10× remains open.** The next immediate gate is the complete
   optimized 4B response, then a genuinely matched comparison if controls fit.
   Further work must target measured remaining overhead rather than relabeling
   estimates or excluding costs from the denominator.
2. **OpenCode v1 memory was a real benchmark blocker.** Measure v2's footprint
   and current headroom. Do not diagnose insufficient benchmark RAM as an SDK
   functional failure or keep repeating unsafe launch attempts.
3. **Default tiny identity bug:** an `Experiment` using `Model.tiny()` without
   an explicit `model_id` can fail with `composition model differs from the
   imported checkpoint`. Benchmark/client paths infer `qwen2`, whereas other
   identity/import paths infer `pllm-tiny-qwen2`. A proper fix must canonicalize
   all paths together; changing only the benchmark hits a Client default-model
   conflict. Current working tests use, for example,
   `Model.tiny(model_id="bounded-client-history")`.
4. **Resident vs paged head thread scope:** effective resident head defaults and
   paged selected-thread behavior differ. Account for that before CPU rankings.
5. **Full physical-wire accounting remains incomplete.** Linux directed service
   IPv4/TCP hooks and application bodies exist, with separately scoped helpers;
   TLS/headers, offload/post-hook behavior, excluded telemetry/DNS/IPv6/control,
   and checkpoint distribution are not a complete all-link physical-wire result.
   macOS BPF was denied and nettop did not reconcile; that route was not promoted.
6. **Independent operators, broad quality and production/adversarial security**
   remain separate evidence requirements. Local role processes do not establish
   those properties.
7. **Older method-guide tradeoffs need an audit:** SDK guides around papers
   R01–R04, R07–R08, R18, R23–R24 may still need explicit RAM/CPU/disk/network
   tradeoffs and reciprocal method links. This is lower priority than client RAM
   and the requested 4B benchmark.
8. **Do not confuse API/plan completeness with executable support.** Research
   probes and planned components must remain explicitly scoped; absent costs or
   unsupported protected compositions should fail closed.
9. **Preserve negative evidence.** The two preliminary tokenizer reports show why
   lazy construction alone was insufficient; the final shared-request report is
   the valid improvement. Other failed research screens likewise matter.

## 13. Operating conventions and evidence discipline

- Read current harness/repository instructions in the new session. The previous
  global instruction requested the `caveman` skill: concise user updates, normal
  technical English in code, commits and documentation.
- Prefer source and dated evidence over recollections. Treat unfamiliar local
  changes as user work; inspect before modifying or deleting them.
- Keep native hot paths in Rust and orchestration/reference work in Python.
  New candidate methods should use generic semantic/component contracts and
  normal SDK/gateway/benchmark integration rather than alternate products.
- Preserve pretrained source identity and numeric quality gates. Do not retrain,
  loosen privacy, move most body compute onto the client, or send plaintext
  token-dependent lookups remotely to obtain a resource win.
- Preserve one-use material and failure/cancellation semantics. Never reuse
  seeds/tickets/corrections between distinct uses as an optimization.
- Distinguish retained bytes, temporary workspace, peak process RSS, filesystem
  cache, whole-machine memory, serialized storage and disk duplication.
- Distinguish cold startup/distribution, request preparation, online bodies,
  decode-only work and full-response costs. Count all relevant parties/links.
- Decode throughput uses generated outputs after the first (`N-1`) over the
  appropriate measured interval; prefill-inclusive bytes/output use a different
  denominator. Cache/repeat results are not fresh-prompt savings.
- Use bounded fresh-process cohorts for peak memory. `ru_maxrss` is a lifetime
  high-water mark, so sequential candidates in one PID cannot be compared as
  though that counter reset.
- Keep provenance, exact source/plan/composition identities and cohort salts.
  Do not merge independent percentages or cherry-pick incomplete samples.
- No credentials, private prompts/tokens, private caches, KV, masks, material
  identities, or model checkpoint blobs belong in committed evidence. Opt-in
  public-task output hashes and sanitized numeric diagnostics are available.
- Maintain practical memory safeguards and clean up only processes owned by the
  run. Avoid increasing VM memory or killing unrelated applications to force tests.
- At least the latest source, tests, evidence and docs are already committed;
  resume from them rather than rebuilding the previous agent's conversation.

## 14. Suggested opening prompt for the new session

> Read `docs/handoffs/handover.md` and resume PLLM from the checked repository state. Client RAM
> is the highest-priority optimization target; 10× whole-client reduction is not
> achieved. First recheck current host headroom under OpenCode v2, then run the
> pinned `qwen3_paged` native preflight. If admitted, run the complete Qwen3-4B
> benchmark command in section 3, inspect and preserve its measurements and
> correctness/privacy checks, and confirm provider cleanup. Keep the existing
> admission/pressure guards, numeric contract, one-use material and remote-heavy
> role graph. Use fresh processes for memory comparisons, distinguish estimates
> from measured peaks, and commit each validated substantive slice.

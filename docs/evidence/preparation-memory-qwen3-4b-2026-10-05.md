# Paged Preparation weights: Qwen3-4B memory probe

## Result

Authenticated native raw-page snapshots reduced median Preparation-process peak
RSS **6.24×**, from **4,613.77 MB to 739.09 MB**. All four samples produced
identical canonical correction frames. Issuance CPU changed by +0.14%; loading
plus issuance used 12.41% more CPU. Each paged process owned 3.63 GB of private
snapshot files. No new host swap was observed.

This is an isolated loader/issuance experiment, not an executable Pipeline option
or a whole-topology benchmark. It does not establish the separate 10× whole-client
memory target.

Exact samples: [preparation-memory-qwen3-4b-2026-10-05.json](preparation-memory-qwen3-4b-2026-10-05.json).
Probe: [`scripts/probe_preparation_memory.py`](../../scripts/probe_preparation_memory.py).

| Median of two fresh processes per mode | Resident | Paged |
| --- | ---: | ---: |
| Process-lifetime peak RSS | 4,613,767,168 B | 739,090,432 B |
| Post-load RSS | 4,236,967,936 B | 317,235,200 B |
| Post-issuance RSS | 4,290,224,128 B | 216,170,496 B |
| Loading CPU | 34.842 s | 49.292 s |
| Issuance CPU | 82.541 s | 82.656 s |
| Loading plus issuance CPU | 117.387 s | 131.953 s |
| Loading wall time | 35.065 s | 49.748 s |
| Issuance wall time | 40.234 s | 40.272 s |
| Loading plus issuance wall time | 75.303 s | 90.024 s |
| Logical body-weight bytes | 3,633,315,840 B | 3,633,315,840 B |
| Retained resident weight bytes | 3,633,315,840 B | 0 B |
| Private snapshot disk bytes | 0 B | 3,633,315,840 B |
| Serialized correction bodies | 242,159,493 B | 242,159,493 B |

RSS excludes filesystem cache and is not whole-device memory. Private snapshots
are additional to source and compiled-cache disk storage. Raw staging files are
removed after import; transient disk peak is not measured. The `after_unload`
sample still has probe-held stage references and is not a weight-release
measurement. Every worker process exits after its sample.

## Method and parity

- Pinned `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`, W8A8 and row-residue outputs.
- Both modes load the real provider-only `MaskedTransformerEngine` and call its
  ordinary `prepare_seeded_stage`. The probe-only loader substitutes `PagedGEMM`
  after stage validation, preserving metadata and scale commitments while
  releasing resident weights. It shares the native executor and uses 1 MiB raw
  pages; runtime component selection is unchanged.
- 144 stages, 71 rows per stage, four concurrent stages: 10,224 stage rows. This
  tests the 64-input/eight-output bound, not the smaller actual 16+8 workload of
  the separate complete-response diagnostic.
- Fresh child-process order: resident, paged, paged, resident. Source files were
  cached; a fresh process does not imply a cold filesystem cache.
- A fresh parent-only root provides identical per-stage seeds to both arms via
  worker environments. Reports retain no seeds, masks or correction payloads.
  Checkpoint, source lock, body, stage metadata/commitment, correction digest and
  counts must agree across all samples.
- Only nondeterministic `server_ns` is zeroed before hashing serialized frames.
  Correction payloads and all other encoded fields match exactly.
- The shared CPU executor has one worker; the four-stage window includes other
  asynchronous preparation/conversion work. CPU is complete process CPU, not a
  single-thread kernel timer. Source resolution precedes the measured window.
- Admission prices one resident Preparation process plus a 512 MiB controller
  allowance, checks disk, and retains host-pressure/swap monitoring. It does not
  reduce live benchmark admission estimates.

The reported 8 MiB paged buffer bound covers four concurrent native encoded and
decoded page buffers, not loading temporaries, masks, corrections, serialization
or all runtime allocations. Retained page metadata is 112,896 bytes.

## Provenance and reproduction

- Baseline: `c4f6a7c`, with the host-reserve change and new probe/tests.
- Probe SHA-256: `ac032f471a1c71d42179bc4ea154a14b9d458a78b7acbc695250a1c6fd4e6cc2`.
- Checkpoint: `75a83dc4d370f01ae8e62e060c02d2a16316ef7638f9dfbc3b920be9882bc586`.
- Source lock: `308249a17bc5540401ce8ee373e24232036de08601b13e215fbb312943b8fdf1`.
- Body: `8dd9e09f7e920487870d67d30ad38162d9372ac4cad852092ff87fbdc936fefc`.
- Stage commitment: `39299516c51025a743f41e9ebdb4457fbe586f8bb4131ffbc21e2a3514388a3b`.
- Correction digest: `ca52cf5cb1d98aa0d1a10816e396fbb2d11a5bb9ca4e9061d51721e6161b9cf1`.
- macOS 26.5.2 arm64, Python 3.13.15, 32 GiB physical RAM.

Executed from the repository root:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_preparation_memory.py \
  --four-b --rows 71 --stage-window 4 --repetitions 2 \
  --scratch-dir /private/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode \
  --output docs/evidence/preparation-memory-qwen3-4b-2026-10-05.json
```

The scratch path is host-specific. Reruns require sufficient disk and a fresh
output filename. `.venv/bin/pytest -q tests/test_preparation_memory_probe.py
tests/test_paged_native.py` passed **19 tests**, covering all-stage tiny parity,
cohort and weight-commitment rejection, no retained arrays, forbidden bundle
export, failure cleanup, authenticated snapshot validation and native arithmetic.

Promotion still needs component/plan selection, role resource admission,
compatibility gates for array-dependent verifier/Metal paths, and complete
SDK/gateway/provider lifecycle tests. This probe does not exercise inventory
reservation, push acknowledgement, inter-role cancellation, HTTP/TLS or
full-response memory/CPU.

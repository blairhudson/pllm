# Ten optimizations inside fixed client-RSS tiers

The preceding attention-tiling slice is recorded in
[`attention-tiling-reference.md`](../research/attention-tiling-reference.md).
Its small, recipe-dependent complete-client results do not establish a new SOTA.
This follow-up investigates CPU, TTFT, decode throughput and all-link traffic
under fixed whole-client memory constraints, rather than optimizing isolated RSS.

## Shared investigation contract

- Low: at most 268,435,456 bytes. Medium: at most 2,147,483,648 bytes.
  Qualification uses the largest whole-client peak across matched fresh-process
  replicates; conservative runtime admission and host headroom remain separate.
- Preserve exact numeric identity, existing trust boundaries and at least 90%
  remote body MACs independently for prefill and decode. New client weights,
  buffers, caches, thread stacks and overlapping ownership must be charged.
- Keep client CPU/RSS, Preparation CPU/RSS, TTFT, N−1 decode throughput and
  all-link traffic visible. Cold/base and output-dependent costs remain separate.
- Each worker owns only its specifically assigned probe, tests, research note
  and evidence files. Do not edit shared SDK, native registration, configuration,
  architecture, coverage, this plan or another worker's files. No commits,
  dependency installation, whole-wheel builds or additional delegation.
- Implement bounded executable references with independent numeric/lifecycle
  controls. Negative results are useful. Do not substitute a simulation or an
  operation-count estimate for a measured native or complete-response result.
- Workers may run small correctness checks. Model-value loading, checkpoint
  scans, sustained timing cohorts and resource-heavy native compilation are
  serialized by the parent under existing admission and the frontier lock.
  Provide a runnable bounded command and its resource estimate for that gate.
- Keep source/probe/evidence hashes, exact replay and missing gates explicit.
  Do not weaken admission, alter a numeric contract or promote an SDK component
  merely to obtain a result. No traffic, tier or SOTA qualification follows from
  local algebra or synthetic correctness alone.

## Independent investigations

| ID | Assigned file stem | Mechanism | Main objective |
| --- | --- | --- | --- |
| 1 | `certified_digit_blocks` | Source-weight absolute-sum certificates for larger exact float32 digit-GEMM reduction blocks | Preparation CPU |
| 2 | `phase_split_placement` | Different public stage placement for prefill and decode, charging the union of client weights | Decode throughput and traffic |
| 3 | `mask_runahead` | One-stage-ahead expansion of already-reserved fresh masks with bounded one-use ownership | TTFT and decode latency |
| 4 | `preparation_flowshop` | Bounded compute/upload-aware ordering of independent correction stages | Preparation completion and TTFT |
| 5 | `exact_f64_corrections` | Checked exact-integer FP64 GEMM as an alternative to digit/block float32 correction GEMM | Preparation CPU |
| 6 | `limb_stationary_corrections` | Pack each fresh input-mask digit slab once and reuse its representation across output tiles | Preparation CPU |
| 7 | `resident_head_shards` | Explicitly budgeted immutable resident head shards alongside authenticated paged execution | Client CPU and decode throughput |
| 8 | `fused_decode_attention` | Single-query GQA execution with fused lifetimes and original reduction/softmax semantics | Client CPU and decode throughput |
| 9 | `column_streaming_linear` | Exact within-stage column partial products while authenticated masked-input blocks arrive | Prefill latency |
| 10 | `masked_decode_microbatch` | Deadline-bounded batching of ready masked rows across separately owned sessions | Loaded-serving decode throughput |

Expected per-worker files are `scripts/probe_<stem>.py`,
`tests/test_<stem>_probe.py`, `docs/research/<hyphenated-stem>-reference.md`
and optionally `docs/evidence/<hyphenated-stem>-synthetic.json`. If a small native
fixture is necessary, use a unique test-only file under
`scripts/research_native/<stem>/`, with explicit standalone invocation and no
shared workspace edits. The installed validation environment is
`target/frontier-sdk-env/bin/python`; avoid checkout shadowing for installed SDK
checks. Parent consolidation will record individual outcomes and next gates.

## Consolidation

The [ten-objective research reference](../research/tier-objectives-reference.md)
links all bounded implementations, negative controls, source-weight scans and
serial parent operator measurements. Whole-client tier qualification and serving
comparisons remain separate from these local screens.

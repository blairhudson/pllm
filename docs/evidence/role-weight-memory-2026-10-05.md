# Role-local weight retention and admission

Fresh public role launchers now select storage by their existing execution and
delivery responsibilities. Preparation retains provider-executed stages only.
Inference and offset workers retain client delivery weights but drop a tied
main-token orientation already supplied by the head. Auxiliary token slices
remain separately retained. Direct all-stage engines and externally supplied
engines preserve their ownership.

All stages are still imported and validated. Retired stages retain their original
digests, signed bounds, scales, biases and source identity, but no i8 array, native
snapshot or compiled-cache residency claim. Attempting to execute one fails.
This changes storage, not composition identity, numeric behavior or wire bytes.

## Bounded loader measurement

`role-weight-memory-2026-10-05.json` uses four synthetic 4096-square source matrices
and five stage views: a tied token/head pair plus three body stages. Separate
processes load each policy, validate all stage hashes/bounds and execute the same
three remote matrix operations. This is a mechanism probe, not a whole decoder.

| Retention | i8 storage, MB | End RSS, MB | Peak RSS, MB |
| --- | ---: | ---: | ---: |
| All stages | 83.89 | 176.77 | 227.08 |
| Preparation | 50.33 | 141.74 | 208.85 |
| Inference/delivery | 67.11 | 158.68 | 208.99 |

Weights, signed bounds and remote integer outputs matched. Observed swap growth
was zero. Loading temporaries still dominate part of the peak: 40% and 20% less
retained i8 storage do not imply equally large peak-RSS reductions.

```bash
uv run --no-sync python scripts/probe_weight_memory.py --role-residency \
  --output /tmp/pllm-role-memory.json
```

Use a fresh output path. The probe applies its existing 2 GiB host-headroom gate
and pressure monitor before creating bounded synthetic source artifacts.

## Live native CPU/Metal functionality

`role-residency-native-tiny-2026-10-05.json` records one shared-cohort invocation
with generated tiny weights, 22 input and two output tokens. Both configurations
passed runtime/privacy checks, generated identical output digests and observed
zero swap growth. Owned provider children shut down. The mixed CPU/GPU comparison
is deliberately unranked: its aggregate `checks.passed` is false solely because
`matched_kernel_backend` is false; each candidate's own checks passed.

```bash
uv run --no-sync python -m pllm benchmark run \
  --experiment examples/benchmarks/memory_safety.py:cpu \
  --experiment examples/benchmarks/memory_safety.py:metal \
  --trust-python --backend native --prompt Hi --max-output-tokens 2 \
  --warmups 0 --repetitions 1 --temperature 0 --capture-output-digest \
  --output /tmp/pllm-role-native.json
```

An earlier attempt rejected when safe headroom temporarily fell to 0.90 GiB,
below the 1.34 GiB tiny estimate. A separate Qwen2.5-0.5B attempt rejected before
provider launch at 3.85 GiB headroom against a 4.36 GiB estimate. Neither rejected
attempt is a completed benchmark. A later host sample permitted the tiny cohort;
the guard and reserve were unchanged.

## Why the 4B estimate remains large

`qwen3-4b-role-residency-admission-2026-10-05.json` prices the pinned lean prepared
composition at a 64-input/eight-output bound, with 71 inventory rows. Its CPU-only
native estimate fell from 23.60 GiB after single-snapshot loading, through 21.12
GiB after segmented bundle delivery, to **19.75 GiB** after role pruning.

| Process | Estimated peak including 25% margin, GiB |
| --- | ---: |
| Inference | 5.98 |
| Preparation | 5.52 |
| Client | 8.25 |
| Total | 19.75 |

The source checkpoint is **8.04 GB / 7.49 GiB**, stored in BF16. The two providers
retain **7.13 GiB** of int8 weights combined, including Inference's head. The
client's pre-margin allowances include **1.89 GiB** of masks, **1.78 GiB** of
decoder tensor workspace, and **2.55 GiB** for copies/working storage of a **0.43
GiB** bundle. That bundle multiplier remains conservative after delivery changes.
The 25% allocation margin adds about **3.95 GiB** to the summed base estimate.
Summing process peaks assumes they can coincide. The separate host reserve is
additional; swap is not capacity.

The 4B run still rejected against **3.08 GiB** safe headroom. No 4B tensor values
or provider processes were loaded by this preflight. The number is not a measured
4B peak or an inherent 4B-model requirement. Client mask allocation, tensor
lifetimes and measured import-copy bounds remain the next memory targets.

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python -m pllm benchmark run \
  --experiment examples/benchmarks/qwen3_4b.py:lean --trust-python \
  --backend native --preflight-only --max-output-tokens 8 \
  --output /tmp/pllm-qwen3-role-admission.json
```

Regression coverage includes tied/untied heads, provider-owned heads, client
prefix/projection ownership, auxiliary token tables, exact bundle bytes, native
integer results, verified reuse, offset reuse and public equalization. Import
immutability and forged replacement rejection are checked separately. Whole-model
quality, independent operators and large-checkpoint measured peaks remain separate
claims.

# Single-snapshot stage loading

## Implemented change

- Rust matrices export a read-only buffer. `CompiledMatrix.weight_view()` retains
  the owner and cannot be made writable through NumPy's supported interface.
- Provider stage arrays alias that immutable snapshot. Original quantized arrays
  and mappings retire after compilation; active compiled-cache entries remain
  protected. Initial import still copies into Rust.
- Float32/F16/BF16/F64 loading and weight metadata use row chunks, capped at one
  Mi elements (1,048,576) except when a single row is wider. Per-row quantization and
  stage identities are preserved. Packed-source allocation models remain gated.
- Client bundle matrices retain immutable unpacked bytes without another NumPy
  copy. Their later kernel snapshots and GPU storage remain separately counted.
- Native preflight prices the changed allocations. Stale/reference storage
  backends reject; potentially older Docker images retain legacy upper bounds.
  Host reserve, slack, swap exclusion and runtime pressure monitoring stay active.

## Isolated measurement

Four 4096×4096 public synthetic stages, W8 weights, one native CPU thread, four
source artifacts totaling 268,435,456 bytes. Each candidate runs in its own child.
The legacy control recreates whole-float quantization, full-matrix metadata
validation and duplicate retained i8 arrays. It is a mechanism control, not a
historical full-engine benchmark. Values below use decimal MB.

| Measurement | Legacy control | Shared snapshot |
| --- | ---: | ---: |
| Distinct quantized weight storage | 134.22 MB | 67.11 MB |
| Process end RSS | 505.35 MB | 162.74 MB |
| Process lifetime peak RSS | 522.08 MB | 213.06 MB |
| Loading plus one-row-per-stage CPU | 0.537 s | 0.371 s |

Peak RSS fell **59.2%** in this single pair. RSS includes allocator retention and
temporaries; the difference is not just the retired weight arrays. All four
weight hashes, signed output bounds and integer output hashes match. The shared
snapshot also checks actual NumPy storage aliasing. The host monitor measured
zero additional swap. Filesystem cache, whole-topology/client peak and GPU memory
are outside this isolated measurement.

Raw evidence: `weight-snapshot-memory-2026-10-04.json`.

## Admission and native execution

The cached Qwen3-4B lean prepared native estimate fell from **34.78 to 23.60 GiB**.
It still rejects before weight allocation on the current 32 GiB host. The raw
header-only follow-up is `qwen3-4b-snapshot-admission-2026-10-04.json`.

A pinned Qwen2.5-0.5B 64+8-cap CPU/Metal preflight also rejected under its sampled
headroom (`qwen-snapshot-memory-admission-2026-10-04.json`); these are allocation
estimates, not measured requests. Smaller bounded requests remain subject to
the same live admission.

An admitted ordinary native tiny **22+2** CPU/Metal pair completed through both
provider children. Each candidate passed runtime/privacy and memory checks, had
the same output digest and zero sampled swap growth. Cleanup completed. The
cross-kernel comparison remains intentionally unranked (`matched_kernel_backend`
is false); it establishes functionality, not a CPU/GPU performance winner or
real-checkpoint quality. Report: `weight-snapshot-native-tiny-2026-10-04.json`.

## Reproduce

From the repository root with the current native extension installed; choose
fresh output filenames. The isolated probe is bounded to four stages and width
4096, requires 2 GiB of RAM above the host reserve, and owns its child cleanup.

```sh
uv run --no-sync python scripts/probe_weight_memory.py --output /tmp/pllm-weight-memory.json

HF_HUB_OFFLINE=1 uv run --no-sync python -m pllm benchmark run \
  --experiment examples/benchmarks/qwen3_4b.py:lean --trust-python \
  --backend auto --preflight-only --max-output-tokens 8

uv run --no-sync python -m pllm benchmark run \
  --experiment examples/benchmarks/memory_safety.py:cpu \
  --experiment examples/benchmarks/memory_safety.py:metal --trust-python \
  --backend native --prompt Hi --max-output-tokens 2 --warmups 0 --repetitions 1 \
  --temperature 0 --capture-output-digest --output /tmp/pllm-snapshot-tiny.json
```

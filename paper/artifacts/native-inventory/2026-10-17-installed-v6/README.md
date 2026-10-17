# Installed inventory and source scan

This append-only archive retains 45 original files (1,951,008 bytes), including
the fifth rejected build, the sixth cached build, both failed fixture-test
attempts, the corrected 67-test result, and the third source scan. `index.json`
records their original paths, byte lengths and SHA-256 hashes. Original archives
for attempts one through four remain unchanged.

## What passed

- Cached offline Maturin build: 94.854 seconds, 1,233,289,216 bytes maximum sampled
  owned RSS, unchanged sources/observers, successful cleanup, zero observed swap
  usage or actual Darwin compressor-swapout growth. This is not a clean build.
- Wheel SHA-256:
  `275716e3376c87ce2c062f935027ee13375f411126d689ae16f917e2daec905c`.
  The wheel remains at the `retained_wheel` path in the index; automatic target
  cleanup excludes wheels. Environment dependencies and install logs are retained.
- The final installed-wheel suite passes all **67** cases with no skips, including
  the two inventory binding tests and three generated-family native checks.
  Earlier failures lacked required Qwen3 configuration fields in a test fixture;
  the fixture was repaired without changing the built runtime.
- The pinned Qwen source scan checks all 290 tensors and 494,032,768 source
  elements. All 169 linear/head tensors fit signed i16 after nearest-even/2048
  encoding. No source value is nonfinite. There are 5,858 independent scalar-oracle
  checks. The seven official source hashes and selected executed module identities
  pass before/after checks.
- `scan/finalization.json` binds the original report bytes and qualifies the final
  report-publication observation. Its `source_numeric_admission` is `pass`;
  `complete_gkm_admission` remains `blocked`. The main report deliberately retains
  `pending_receipt_finalization` and cannot be accepted without its final receipt.

The observed scan interval is 12.243 seconds; cumulative scanner CPU is 12.170
seconds and its high-water RSS at the final observation is 110,460,928 bytes.
Actual swap use and compressor swapouts remain constant. Final receipt formatting
and interpreter exit remain outside the observation. These are source-publication
costs, not serving measurements.

## Counter correction

macOS psutil 7.2.2 maps `sout` to file pageouts, not compressor swapouts. See
[`mem.c` at the pinned release commit](https://github.com/giampaolo/psutil/blob/9eea97dd6f1d16ea33f5144c8925f1ce7a0688e1/psutil/arch/osx/mem.c).
The repaired helper reads revision-one `HOST_VM_INFO64.swapouts` directly, keeps
pageouts as diagnostics, and rejects missing, changed or regressing telemetry.
Page-scaled counters are page-equivalent activity, not physical disk-I/O bytes.
During the accepted scan, pageouts rose by 4,014,080 bytes while actual swapouts
stayed constant. Historical failed receipts retain their status; their `sout`
increments cannot establish actual compressor swap activity.

Verify archive integrity and bindings without loading the checkpoint:

```sh
.venv/bin/python paper/artifacts/native-inventory/verify.py
```

The verifier also checks the retained wheel when present. Actual stage input
bounds, the local numerical graph, full-width publication ownership, verification
windows, decoder state, quality and worker transport remain separate gates.

# Streamed GKM source import

`source-stream-v1/` freezes the cached `gkm-source-stream-wheel-v1` build and
its checks. This is a separate successor to `local-v1/`; it does not change that
archive or extend its measurements. The new archive contains 109 hash-bound
files and 2,796,018 payload bytes.

## Change and checked contracts

The ordinary SDK, gateway and benchmark source path feeds ordered float32 frames
of at most 64 KiB to a one-use native importer. It encodes matrices directly into
their final i16 buffers and preserves original lookup and coefficient bytes.
Workers take those buffers without another full source snapshot. A separate
16 MiB process allowance bounds live import payload through ownership transfer.
Source identities, numerical operations and existing checkpoint limits remain.

The checks include exact source and compiler digests, signed zero, F32/F16/BF16
import, grouped matrices, tied and untied heads, original lookup bytes, integer
products and the final verification barrier. Allocator failures check payload
ownership transfer. Ordered-frame, signature, range, source-mutation and
cancellation failures retire partial importers. The installed SDK controls compare
complete logits and active KV through actual worker processes; gateway, CLI and
ordinary response benchmarks exercise the same importer.

## Recorded validation

- **571 Python tests pass**, without skips, in a fresh Python 3.13 environment.
  All 362 installed package files match the wheel before and after testing.
  Test sources remain unchanged; isolated imports disable the checkout package.
- **120 native checks pass**: 50 compiler/inventory/schedule cases, one source-import
  admission case, 42 core GKM cases, 19 allocation/wire/process cases and eight
  ownership doctests. Clippy passes with the existing `needless_range_loop` and
  `nonminimal_bool` exclusions. Native input hashes match the wheel build.
- The cached build reports no source or observer changes and zero observed swap
  use or compressor-swapout growth. It took 42.605 seconds; largest sampled
  build-process-tree RSS was 1,526,349,824 bytes. Observer and final receipt tail
  remain outside that sampling scope.

Wheel SHA-256:
`c52c633ac56497dd5bf3ff6a0edfb73cc78b0729511131e60a714696947c25ec`.

`rejected-native-v1/` retains the earlier failing allocation regression. That
attempt had changed the legacy float64 import workspace. The accepted source
keeps the legacy workspace contract and limits ownership transfer to the new
streaming path. The failed receipt and input hashes remain distinct.

## Read-only replay

From the repository root:

```sh
python -I -B paper/artifacts/gkm-sdk/verify.py --archive source-stream-v1
```

The verifier checks the original receipts, test outcomes, source hashes and
installed identities. It checks wheel bytes when retained locally and reports
their absence otherwise. Exact commands remain in `source-stream-v1/tests/command.json`
and `source-stream-v1/native/commands.json`; re-execution needs matching installed
dependencies and relocated paths.

This establishes bounded import and SDK correctness. It is not a clean build,
real-checkpoint execution, serving measurement or process-memory improvement.
Full-Qwen publication capacity, activation admission, quality and matched
measurements remain open. The ten-method study still has zero qualified
complete-system comparators.

# Bounded GKM SDK conformance

`local-v1/` freezes the final cached wheel build (`gkm-sdk-wheel-v4`) and its
installed tests. The archive retains 106 hash-bound files: build and observer
receipts, native and Python checks, original test inputs, selected implementation
sources and the rejected v3 test attempt. Total payload is 2,861,674 bytes.

## Accepted checks

- **547 Python tests pass**, with no skips, in a fresh Python 3.13 environment.
  All 362 installed `pllm/` files match the wheel before and after the suite.
  Test source hashes are unchanged. Imports use isolated Python with the checkout
  package path disabled.
- **117 native checks pass**: 48 compiler/inventory/schedule cases, 42 core GKM
  cases, 19 allocation/wire/process cases and eight ownership doctests. Clippy
  passes with the existing `needless_range_loop` and `nonminimal_bool` exclusions.
  Native inputs match the wheel build.
- Generated Qwen2/Qwen3/Llama fixtures exercise complete lookup, final logits and
  active KV, histories and pending selections. Real worker processes exercise
  acceptance, final corruption, replay, partial admission and cancellation.
- The ordinary SDK, buffered streaming, HTTP gateway, CLI benchmark and
  `BenchmarkTarget.response` use the same bounded Experiment. The public
  `examples/delegated_sdk.py` executes as part of the interface suite.
- Allocation, raw gateway ingress, concurrent stream closure, stale-session
  isolation, absolute transport deadlines, failed metrics and ContextVar
  lifetimes have dedicated regressions. A separate read-only recheck passed
  56 installed-v4 lifecycle tests and closed all reported findings within their
  stated scope. See [the audit record](audit.md).

Wheel SHA-256:
`42ca820389b2504840d5d3110df472ccf188acdb612762f9e065c6a4fb053b32`.

The guarded cached build took 74.778 seconds. Its largest sampled owned-process
RSS was 1,168,392,192 bytes, with zero observed swap-use or compressor-swapout
growth. Source and observer identities stayed unchanged. This measures the build
process tree, excluding the observer and final receipt tail. It is not an empty-
target build, provider-child attestation, serving measurement or release/CI proof.

The rejected v3 suite retains its original two failures: sidecar context lifetime
and an obsolete test call to the removed single-process shutdown method. The
final suite covers both repairs. Its archived result does not replace the failed
attempt's source or measurement identity.

## Replay

From the repository root, check the archive without executing models:

```sh
python -I -B paper/artifacts/gkm-sdk/verify.py
```

The verifier checks every member, complete input identities, test results and
installed-file receipts. It also checks the retained wheel when available; its
absence is reported explicitly. `local-v1/tests/command.json` and
`local-v1/native/commands.json` retain the exact execution commands. Re-execution
requires a matching installed wheel and dependencies; recorded absolute paths
must be relocated. The generated checkpoint fixture is constructed by the retained
test code, not downloaded from a model repository.

The source-mapped numeric contracts remain separate from ordinary W8A8.
Vocabulary at most 256, fixed-count greedy generation, explicit full shares and
local source directories bound this slice. IP-literal origins avoid unbounded
DNS executor retirement; HTTPS verifies certificates, and HTTP is loopback-only.
Co-located processes do not establish independent operators. Official Qwen and
original-paper model admission, activation bounds, task quality, seeded transport,
matched performance and memory-tier eligibility remain open. The ten-method
publication count remains zero qualified complete-system comparators.

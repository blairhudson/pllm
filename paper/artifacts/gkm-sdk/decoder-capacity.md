# Explicit Decoder-capacity conformance

`decoder-capacity-v1/` retains 135 hash-bound files (7,090,103 bytes) from the
final cached `gkm-capacity-wheel-v3` build, installed checks, native checks and
the source-present Qwen metadata preflight. The original Capsule and source-stream
archives retain their own identities.

## Accepted slice

- **640 installed Python tests pass**, without skips. All 362 installed package
  files match the wheel before and after execution; test sources do not change.
- **134 native checks and Clippy pass** on the same native source inputs. The
  existing Clippy exclusions remain in `native/commands.json`.
- Six generated Qwen2/Qwen3/Llama two-worker cases cover tied and untied heads
  with 1,100 vocabulary entries. Their complete logits and active KV preserve
  **20,376 float32 values** against the independent decoder oracle.
- The same explicit capacity choice runs through ordinary Pipeline/Experiment,
  SDK, gateway, CLI and response benchmarks. Default Capsule serialization and
  digests remain stable, including reconstruction and `Experiment.with_params`.
- Geometry-derived result admission covers the complete logits/KV envelope.
  Failure regressions check tentative arrays, nested exception frames, cancelled
  HTTP bodies and host reservations retained by live worker replies. Final native
  acceptance still precedes public output.

Wheel SHA-256:
`135fef0c7502e21f82669e1faacdc7a3c1174848f7111d06da4928c0200c7989`.

This is a guarded **cached local build**, not an empty-target build or external
attestation. It observes zero new swap growth. Its process measurements concern
the build, not serving. Recorded native and Python inputs match this wheel.

## Original full-Qwen preflight

The probe checks actual source-file identities for Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Native metadata admits a 64-input,
two-output bound: 97 matrices, 290 source tensors, 121 local tensors and
1,537,852,928 imported source bytes. These are resource bounds, not execution.

The ordinary whole-topology preflight rejects before tensor-value decoding or
worker launch:

| Conservative allocation | Bytes |
| --- | ---: |
| Client, including margin | 3,962,826,300 |
| Worker A, including margin | 3,834,438,400 |
| Worker B, including margin | 3,834,438,400 |
| Complete required host allocation | 11,631,703,100 |
| Safe host headroom after reserve | 8,709,210,112 |

Thus the attempt needs **10.83 GiB** against **8.11 GiB** of safe headroom. Existing
swap is not additional capacity. No full-Qwen response, activation-bound result,
quality score, measured client RSS or performance result was produced.

A later, separately retained [Qwen response](qwen-response.md) passes fresh host
admission and completes four input tokens plus two verified outputs through both
workers. Its measured resources and functionality scope do not change this
original rejection or supply independent numeric/quality comparison evidence.

`qwen-smoke-v1/` retains an earlier incomplete metadata capture and its original
runner. `rejected-wheel-v1/` and `rejected-wheel-v2/` preserve their failed installed
suite receipts. The final tests do not rewrite those attempts.

## Read-only replay

From the repository root:

```sh
python -I -B paper/artifacts/gkm-sdk/verify.py --archive decoder-capacity-v1
```

The verifier checks all retained bytes, build/native/test input identities,
installed-wheel equality, six complete-result counts and the source-bound memory
rejection. It checks the original wheel when present and reports its absence.
The negative controls in `tests/test_gkm_capacity_archive.py` reject omitted worker
memory, promoted rejection status, changed source or runner and missing parity
counts.

Exact test commands, dependency versions and preflight inputs are retained.
Absolute paths must be relocated for a new execution. A retry requires fresh
whole-topology host admission before source import; the saved rejection is never
an execution permit. Full-checkpoint numerical and quality qualification,
independent operators and matched measurements remain open.

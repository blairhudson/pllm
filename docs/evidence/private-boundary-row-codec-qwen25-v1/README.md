# Qwen2.5 certified head-row codec: separate initial SDK screens

These files preserve original executions, not regenerated measurements.

- `150x8/`: accepted six-client original report, frozen plan, receipt, summary,
  progress, launcher log and separate independent read-only audit.
- `150x64/`: accepted fresh six-client v2 report and the same companion records,
  under its own private cohort; no sample from the rejected attempt is reused.
- `150x64-rejected-v1/`: original attempt with one completed control and five
  topology-memory rejections. No receipt or accepted summary exists.
- `publication-v1/`: original pre-tensor memory-admission rejection.
- `publication-v2/`: successful offline certificate, independent source/native
  commitments, exact frozen Experiments and caller-declared delivery accounting.
- `preflight/`: metadata-only workload admissions, separate from execution.
- `source-snapshots/` and `publisher-inputs/`: byte-identical provenance copies,
  never automatically imported or executed by replay.
- `validation/`: original installed-wheel correctness log.

Both source-present screens preserve complete recorded state/output
commitments and reduce exact head bytes. Each passes independent raw audit and
receipt-backed replay. They do not establish confirmation,
an overall performance win, causal memory improvement, complete cold distribution,
physical-wire or general model-quality results. The certificate remains a
trusted-publisher assertion. The unsigned receipt binds public consistency, not
worker authentication or private-input preimages.

`index.json` binds every preserved original by path, exact size and SHA-256.
Derived descriptions do not replace the original reports. Short and long screens
retain 52,561,920 and 486,555,648 checked values respectively, with zero observed
swap growth. See the benchmark reference for all seven serving metrics, complete
request and lifecycle costs, mixed paired RSS changes and publisher accounting.

The native binary, full interpreter/dependency environment, checkpoint and indexed
tokenizer remain external. Original reports retain their complete source/build
commitments. The pinned independent historical row scan remains separately stored
at `docs/evidence/compact-private-head-qwen25-v1.json`; its original implementation
hashes differ from this SDK build, while source, widths and bounds match.

Use the trusted matching checkout and installed environment recorded by the plan:

```sh
target/frontier-sdk-env/bin/python scripts/summarize_private_boundary_row_codec.py \
  docs/evidence/private-boundary-row-codec-qwen25-v1/150x8/certified-head-qwen150x8-v1.json \
  --plan docs/evidence/private-boundary-row-codec-qwen25-v1/150x8/certified-head-qwen150x8-v1.plan.json \
  --receipt docs/evidence/private-boundary-row-codec-qwen25-v1/150x8/certified-head-qwen150x8-v1.receipt.json
```

Replay prints a recomputed summary without loading tensors or providers. Verifier
source drift rejects; selecting an older trusted checkout is an explicit action,
not automatic execution of archived Python. Full costs, measurement windows and
limitations: `docs/research/private-boundary-row-codec-benchmark.md`.

# Explicit public-weight PUMA conformance

This slice adds the approved public-checkpoint adaptation:
`PumaThreePartyDecoder(weights="public")` in `PumaThreePartyCpu`.
The private-weight default retains its existing composition and execution
identities. Public mode binds a separate numeric identity and source contract.

The client publishes bounded commitments and public ranges. Each worker imports
the locked source into immutable native Q18 files and scans public weight ranges
for both replicated activation components. A complete dot product is accumulated
before its single Q18 truncation. Private one-hot lookup, batched equality and a
stable protected argmax tournament retain private prompts, activations and token
feedback. Only final fixed-count token IDs reach the client. Model weights are
public in this mode; the three-party corruption assumption remains explicit.

The [contract](../../../../docs/research/puma-truncation-reference.md#explicit-public-weight-adaptation)
defines source, file-buffer, disk, compiler, primitive, work and range limits.
These generated-model checks do not establish full-checkpoint quality, whole-client
RSS, serving speed, independent operators or cryptographic review.

## Evidence and limits

`summary.json` records accepted checks and their exact build identity. The
`build-v1/` and `validation-v2/` directories retain original build and validation
receipts, installed package hashes, dependency versions, test reports, native logs
and the metadata gate. Earlier failed fixture/admission attempts remain separate
diagnostics. The first broad suite encountered macOS memory pressure: 23 admission
failures and one watchdog-cancelled request. It supplies no accepted broad-suite
result; a fresh-process retry preserves the original memory guards.

| Accepted check | Result |
| --- | --- |
| Installed-wheel regression suite | 243 passed; seven optional skips |
| Replicated compiler checks | 15 passed |
| Installed package files | All 367 match before and after the suite |
| Clippy and coverage-surface review | Passed; two recorded Clippy allowances |
| Companion specification/guard tests | 60 passed |
| Maverick native checks and ownership doctests | 15 and six passed, respectively |

The source-only pinned Qwen2.5 check uses four input tokens and two outputs. Public
mode passes the former source-geometry gate, then rejects with
`replicated compiler view payload limit`. Private mode still rejects source
expansion. All-zero artifact pins are metadata placeholders: this check reads no
tensor values, authenticates no checkpoint bytes and starts no workers. Practical
full-checkpoint use still needs compact activation views and reusable tensor
loops, complete-model range admission, quality and matched resource measurements.
`qwen-metadata-replay.json` retains the exact public configuration and Experiment
arguments for an independent repeat of both blocked metadata calls.

The cached local wheel build reports zero sampled build swap growth. Test and
serving windows have separate scope. This is conformance evidence, not clean-room
release provenance or revision-bound CI. Suite counts overlap earlier slices.

`companion-checks/` separately retains Maverick, FuseFSS and Poulpy conformance
checks for the other changes committed with this slice. Maverick's bounded native
batch and allocation checks remain distinct from PUMA integration. The original
unoptimized allocation-failure run was stopped and replaced with the documented
optimized test build; it supplies no accepted exhaustive-sweep result.

## Replay

From this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

`inputs.tar.gz` and `validation-inputs.json` retain tested source/test bytes.
Build/install the reconstructed wheel, then run the exact commands from
`validation-v2/validation-commands.json`. In particular:

```sh
python -I -m pytest -q -o pythonpath=. -o junit_family=xunit1 \
  tests/test_replicated_*.py tests/test_reference_agreement.py
cargo test --locked --offline -p pllm-compiler --lib replicated_decoder
```

The repository root supplies test helpers, not a second `pllm` package. File
receipts verify installed-wheel imports before and after the suite. Companion
tests additionally require their source-locked specification artifacts; they do
not execute or link upstream implementations.

The metadata gate can be repeated from this evidence directory without a
checkpoint download or tensor import:

```sh
python -I - <<'PY'
import json
from pathlib import Path
from pllm import _native
r = json.loads(Path("qwen-metadata-replay.json").read_text())
for mode, case in r["modes"].items():
    try:
        _native.replicated_source_contract(
            case["experiment"].encode(), r["normalized_config"].encode(),
            r["rows"], r["outputs"], r["source_lock"], r["checkpoint"],
        )
    except ValueError as error:
        assert str(error) == case["reason"], (mode, str(error))
    else:
        raise AssertionError((mode, "historical gate unexpectedly admitted"))
PY
```

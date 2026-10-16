# Paper 1 Listing 1 replay

`qwen-v1/` is the original 16 October 2026 cached-checkpoint execution of the
Python listing extracted from `paper/manuscript.md`. `preflight-v1/` is the
separate admission-only attempt. Neither is a performance or memory-tier cohort.

## Frozen v1 evidence

The installed SDK and in-process ASGI gateway each completed 40 input and eight
output tokens through separate local TCP provider children. Output text matched;
the SDK reported positive masked-stage activity and zero plaintext prompt-byte/
token-ID counters. The existing host watchdog observed zero swap growth. The public
fixture, pinned configuration, **model source/checkpoint digests only**, parent
installed build record, admission, source driver and exact executed listing are
retained. The full model source lock was not retained in v1. The parent's isolated
Python invocation did not propagate isolation to provider children, and their
actual import identities were not recorded: **v1 child imports are unknown**.
The parent build record must not be attributed to those children retrospectively.

`index.json` and both v1 directories remain byte-for-byte original. Their drivers
are verified against their own recorded hashes, never against the evolving v2
driver. Current Listing 1 is checked separately against the frozen listing. The
surrounding manuscript subsequently changed; its original hash remains execution
provenance and is not rewritten to the current draft.

## Accepted fresh-environment replay: `qwen-fresh-v2/`

This separate 16 October 2026 execution used the freshly installed release wheel.
The 12 files in `qwen-fresh-v2/` are byte-for-byte copies of `result.json` and its
11 enumerated top-level files from `target/paper-listing-fresh-release-v1`.
Operational provider logs and bundle caches were excluded. `index-v2.json` freezes
their hashes, the original `index.json` hash, and four linked release metadata files.

Accepted invocation (historical; the output directory already exists):

```sh
paper/artifacts/release/.work/fresh-replay-44b5fdd8174a4a3493396125f0983da3/venv/bin/python \
  -I -B scripts/replay_paper_listing.py --output target/paper-listing-fresh-release-v1
```

Recorded results:

- SDK and in-process ASGI gateway each completed **40 input / 8 output tokens**;
  output hashes match. SDK audit records **4,512 online steps** and zero plaintext
  prompt-byte/token-ID counters.
- Native admission estimated 4,234,084,690 host bytes against a 7,238,090,752-byte
  budget. Watchdog recorded **zero observed swap growth**, with 4,710,989,824 bytes
  of swap already in use at admission. This is not a peak-RSS or zero-swap claim.
- Actual child PIDs **13408 / 13651** identify their imported package/native files
  before provider CLI entry and match parent **13265**. Both children were reaped
  with return code **-15** during normal topology cleanup.
- All **seven model-source files** are retained as full before/after hash inventories;
  locks match at revision `7ae557604adf67be50417f59c2c2f167def9a775`. Parent SDK/native/
  Python identities match before/after; both child receipts match the initial parent.

### Release linkage and observation limits

`index-v2.json` directly hashes these read-only links:

- `paper/artifacts/release/2026-10-16-fresh-target-v2/{receipt,manifest}.json`
- `paper/artifacts/release/2026-10-16-fresh-environment-v1/receipt.json`
- `paper/artifacts/release/2026-10-16-fresh-environment-v1/replay/provenance.json`

The original build receipt remains **blocked at its offline install**. Its compilation
passed; the separate fresh-environment continuation subsequently installed the exact
wheel and passed its own validation. Neither receipt is relabelled by this Qwen run.

| Identity | SHA-256 |
| --- | --- |
| Release wheel | `3b63ae30631f0b54773d708c0921b75dbf951142f14dfb80dd9eedac145043b6` |
| Release package inventory | `af581eddf511344e2cdeff49ce295153662350bf58bc9b84f287025d9a430963` |
| Qwen-observed native | `cb607aa0b3776b21882a6363a3cedf055caf6e41ae71ed913127fb467ba3cba9` |
| Qwen-observed Python executable | `7710b0490e6af648676d7ad163fa1d54bffb75d8505162609a1c3c9b76e1929d` |

The verifier hashes the linked wheel and its **355 package files**, compares that
inventory with the earlier installation provenance, and projects its `.py`/`.json`
entries onto the **313 SDK sources actually recorded by Qwen's parent and children**.
Native/Python hashes and recorded installation paths also match. The full 355-file
installed-package inventory and full dependency checks belong to the **earlier
fresh-environment validation**. Qwen did not record those complete inventories
before/after execution. Its selected dependency metadata in `build.json` does not
establish full dependency stability. Verification uses archived observations; the
ephemeral environment need not still exist.

## V2 driver evidence

V2 adds the following evidence without changing production runtime code:

- A scoped launch shim runs each actual provider PID with `-I -B`, removes inherited
  Python path/home/user-base settings, disables user-site imports, and loads the
  installed `pllm` and `pllm._native` before `runpy` enters the provider CLI.
  Each child writes its PID/parent PID, actual module paths, SDK/native/Python
  hashes and isolation flags. Receipts must match the parent's observed identity
  and the process returned by the launcher. Final records include reaped exit codes.
- Complete `source-lock-before.json` and `source-lock-after.json` retain the public
  model file inventory. Both checkpoint and source-lock digests, repository and
  pinned revision/commit are checked. A fresh resolution rehashes model sources
  after execution/cleanup; before/after locks must match. Parent package/native
  identities are likewise checked before/after. These are boundary observations,
  not a claim that transient between-sample changes can be detected.
- Failure phase, stable reason and exception type survive rejection. Gateway HTTP
  status and available counters are saved before acceptance checks. Cleanup and
  final source-check failures are retained alongside the first failure. Exception
  text, response bodies, credentials and private prompt/output text are not saved
  in failure diagnostics. Provider logs and bundle caches remain local operational
  files, excluded from the publication file inventory.

## Read-only verification

The standard-library verifier defaults to **both immutable archives, the current
published listing, and the fresh-release linkage**:

```sh
python3 -I -B paper/artifacts/listing1/verify.py
```

It verifies frozen driver identities against each version's own records, including
the new archive's source locks, child PID/import bindings, admission and outcome.
Current publication binding is separate; subsequent manuscript edits never rewrite
historical manuscript hashes. `--run PATH` additionally verifies another run;
`--expect-status rejected` accepts only a retained rejection's hashes/diagnostics,
and `--expect-status admitted_not_executed` checks an admission-only attempt.

Full build/source/tool/dependency internals remain the release verifier's scope:

```sh
python3 -I -B paper/artifacts/release/verify.py \
  paper/artifacts/release/2026-10-16-fresh-target-v2 \
  --replay paper/artifacts/release/2026-10-16-fresh-environment-v1
```

These commands read retained evidence only; they do not execute models or providers.
For later runs, use a fresh output path and retain a new immutable index. Archive
`result.json` plus only its enumerated publication files.

Archive-linkage and mock-request regressions (no providers or checkpoint):

```sh
python3 -I -B tests/test_paper_listing_replay.py ArchiveLinkageTests ReplayTests
```

The result establishes functionality, not exact whole-state parity, representative
quality, an independent operator deployment, timing security or a measured speedup.
Gateway HTTP is exercised through ASGI in process; no claim of a separately
hosted network gateway is made. The original parent used the installed identity
recorded in `build.json`; later conformance/release evidence does not establish v1
child imports or relabel that run. These are local, unsigned integrity records,
not independent-operator or CI attestations.

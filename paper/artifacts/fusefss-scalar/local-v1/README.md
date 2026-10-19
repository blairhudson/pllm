# FuseFSS bounded native scalar qualification

This closes the first native scalar tranche of the
[completion plan](../../../methods/fusefss-completion-plan.md). It preserves the
approved `fusefss.source-derived-ring50-q12.v1` GELU descriptor, external
nearest-even encoding and internal floor arithmetic. This is local native
execution with test-only reconstruction. Complete gates, semantic decoder
lowering, authenticated roles, SDK selection, source-model quality and independent
security qualification remain open.

## Implemented composition

- `crates/pllm-garble/src/fusefss/`: typed descriptors, canonical identities,
  Protocol-3 padding and predicate rewrites, dealer-only compiled handoff. The
  frozen GELU descriptor remains 32,900 bytes with identity
  `5f1e888e7bdabe3cbad4b6bbf1abd6749c8dff0918c6f243916db071398e79d3`.
- `fusefss_backend/`: consuming party owners, original-input shared Horner
  product, source/backend/session/nonce binding and reservation-carrying exports.
  The ordinary frozen adapter accepts no caller JSON or supplied partition.
- Existing `compact_dcf`, `interval_fss`, `shared_affine_lookup_reference` and
  `shared_wide_product` families supply common scalar/vector traversal, ring-aware
  universal comparisons, public-table lookup and consuming Beaver products.
  Legacy scalar identities and wide-product draw order remain regression checked.

The private-vector control implements arbitrary admitted private affine payloads
with one vector DCF per padded interval. The public-boundary specialization uses
the original frozen public table and one universal DCF per lane. Public duplicate
padding preserves the complete ring function; hidden rotation corrections enter
freshly shared bases. Evaluators receive no sampled-mask-dependent table order,
clear corrections or selected clear coefficient. Their original public table
copies remain separately charged. This specialization does not implement arbitrary
private-partition Gen or reproduce the paper's optimized backend costs.

## Separate correctness and ownership evidence

| Evidence | Checked scope |
| --- | --- |
| Clear compiler output | 184 descriptors, 1,552 runs, 58,400 gate evaluations |
| Protected public-boundary GELU | 43,544 outputs in 10,888 fresh issuances, eight fixture runs |
| Exact-rational external encoding | All 38 cases in each replay |
| Original public-boundary engineering review | 45 protected public outputs, eight private controls, 1,024 compiler endpoints |
| Review ideal algebra | 8,328 partition/mask/input cases and 39,204 comparison cases |
| Native allocation review | Sixteen injected allocation failures, exported frame/output leases, zero final requested-heap/Budget residual |

Each protected scalar lane issues fresh OS-random mask/key/product material;
input sharing is independently randomized. The exporter callback never receives
the fixed public-fixture mask. Fixed-mask compiler diagnostics and fresh protected
outputs therefore retain different meanings. The independent checker calculates
expected values from the original equations and rejects substituted outputs,
geometry, ordering and encoding. It cannot itself prove crypto execution.

`run-receipt.json` and `logs/` record final release-library, frozen-reference,
independent-fixture, formatting and export/checker runs with exact commands and
exit codes. Logs retain their original bytes in gzip files. The library passes
208 tests with five opt-in tests ignored; both
conformance exporters run explicitly in separate successful steps. The original
reference and independent-fixture suites pass 20 and 22 Python tests respectively.
`clippy.json` records the separate successful all-target garble/PyO3
check with warnings denied except the three explicitly listed existing lint
classes. No wheel was built or installed. These records are not a performance
cohort or revision-bound release CI.

Both engineering reviews found no blocking scalar algebra, provenance, lifecycle
or resource defect. Their P3 documentation and stale-receipt findings were
addressed. A deliberate missing-opt-in attempt removes **both** prior success
receipts before rejecting, then a new complete run republishes them. A complete
claim requires the adapter receipt, linked runtime receipt, successful process
exit and independent checker result together.

The public-boundary review has its own source snapshot under `initial/`.
`reviews/review-to-final.patch` records subsequent documentation, lint and
test-export changes. No reviewed lookup or product equation changed. Final
compiler/protected replays use the final `inputs.tar.gz` source overlay.

## Complete bounded scalar costs

For one ring50 GELU lane, 511 source / 512 padded intervals:

| Quantity, per party unless stated | Private vector control | Public boundary |
| --- | ---: | ---: |
| DCF keys | 512 | 1 |
| Logical private material bytes | 886,832 | 1,372 |
| Original public table copy bytes | 0 | 12,296 |
| DCF evaluations | 512 | 512 |
| DCF path levels | 25,600 | 25,600 |
| Framed peer bytes, both parties/phases | 368 | 368 |

One key still traverses every boundary. Public table copies are neither private
key bytes nor measured distribution. Inputs/outputs remain ring50 shares; the
Q0 slope multiplies the original Q12 input without another rescale.

The complete public-boundary replay charges **119,484,736 logical private material
bytes**, **1,070,834,048 cumulative local public-table-copy bytes** and **5,574,272
actual native peer-frame bytes**. The original source table is another 12,296
bytes. Four-lane calls use a 2,764,902-byte combined reservation, including the
separate descriptor owner, under the unchanged 64 MiB ceiling. Final native
Budget use is zero. Frame totals exclude keys, tables, authentication, transport
and client boundaries; they are not whole-wire costs.

The reviewed allocator counts requested live heap, not process RSS. One/four-lane
issuance/online peaks are 102,236/187,004 bytes; construction peaks at 557,300
bytes. Stack, allocator overhead, caller copies and external oracle storage are
excluded. Generic JSON descriptor parsing is outside the frozen adapter's bounded
path. No whole-client memory tier, checkpoint admission or serving speed follows.

## Read-only replay and native reproduction

From repository root:

```sh
python3 -B paper/artifacts/fusefss-scalar/local-v1/replay.py
```

This verifies artifact and source hashes, original review/final source overlays,
receipt linkage, native exits, material/frame conservation and both independent
checker passes. It reads original compressed outputs; it issues no keys, compiles
no code and accesses no model or network. Preserve this repository's frozen
fixture/checker files, or restore their verified overlay first.

Native reproduction on the recorded source/build inputs:

```sh
CARGO_BUILD_JOBS=1 PLLM_FUSEFSS_EXPORT_COMPILER_CONFORMANCE=1 cargo test --release --locked --offline -p pllm-garble --lib fusefss::conformance::export_compiler_conformance_all -- --ignored --exact --test-threads=1 --nocapture
python3 -B paper/reproduction-contracts/fusefss/native-conformance/check_native.py --suite all target/fusefss-native-conformance-v1/compiler.jsonl
CARGO_BUILD_JOBS=1 PLLM_FUSEFSS_EXPORT_PROTECTED_GELU=1 cargo test --release --locked --offline -p pllm-garble --lib fusefss_backend::conformance::export_public_boundary_frozen_gelu -- --ignored --exact --test-threads=1 --nocapture
python3 -B paper/reproduction-contracts/fusefss/native-conformance/check_native.py --suite gelu target/fusefss-native-protected-v1/protected.jsonl
```

The archived `run-native.py` additionally documents the validation sequence and
stale-receipt negative control; that control requires the prior successful export
receipts in `target/`. Test-only reconstruction never becomes a production clear
interpreter or an SDK capability. Signed38 frontend provenance and complete-model
material/work feasibility remain separate gates in the completion plan.

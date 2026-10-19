# Checked acceptance, compact lookup and authenticated peer streams

This slice exposes an explicit public-weight composition through the ordinary
Pipeline, SDK, gateway and response benchmarks. It preserves the existing default
identities and resource ceilings. Pinned Qwen source publication reaches
layer-eight attention, then rejects its product bound before worker execution.

## Implemented contract

- Private full-word RMSNorm membership checks select a safe zero operand before
  range-dependent arithmetic. Shared validity remains sticky across selections;
  final masking suppresses every token if any check fails. Only aggregate final
  acceptance is disclosed to the client, under the approved D12 contract.
- Tiled full-vocabulary lookup shares immutable predicate templates and public
  weight scans. Proven-domain equality uses distinct compact Boolean frames while
  retaining full-width arithmetic OT values and fresh material for every use.
- Tiled stable selection preserves the lowest original index on ties and prices
  its dedicated full-head gather separately from the original logits.
- Persistent authenticated peer streams bind source, request, policy and worker
  epochs. Ordered one-use sessions, bounded queues, completion seals and draining
  cancellation retain the original phase barriers. Admission charges an additional
  8 MiB per worker exactly once.

The independent native/Python oracle compares complete generated hidden tensors,
K/V, vocabulary logits, private feedback, guard validity and final acceptance.
Its AES reconstruction consumes every actual truncation draw exactly once. The
final audit covers **64 generated cases and 824 negative controls**, including
416 wrong-KV-value-group mutations and 26 sticky-validity mutations. A dense
151,936-row head uses a narrow synthetic decoder; it is not a Qwen checkpoint run.
Test-only captures and secret material are not SDK exports.

## Validation

The installed wheel passes **462 Python tests**, with seven optional skips:
16 new native SDK gates, 115 focused controls and 331 additional compatibility
checks. Another **89 independent Python oracle tests** pass. Native validation
covers **218 distinct checks**: 98 core, 69 compiler, 46 integration and five
compile-fail doctests. The final three compiler oracle tests replace the earlier
weaker fixtures; their cases are not added to the earlier compiler run's count.
Clippy passes with the existing `needless_range_loop` and `nonminimal_bool`
allowances. Benchmark surface validation passes and retains its explicit missing
and unsupported evidence cells.

All **368 installed `pllm/` files** match the tested wheel before and after the
compatibility run. The separate SDK audit additionally verifies 373 wheel payload
files and the installed RECORD. Pytest explicitly overrides the checkout's
`pythonpath` setting; Python `-I` alone is insufficient.

The original tested wheel is
`c599f0781f1c7fdc0697b67118c3f0b897ae504d2f771c0a506ce8af350de73f`.
The subsequent cached build is
`232db832b9b9f42a5e0f83dbde95838f8b8b846cf1fe5092a0c07cb48479abbe`.
Their installed package files are byte-identical; only wheel RECORD and CycloneDX
SBOM payloads differ. The original test receipts retain their original wheel
identity. These are guarded cached-local builds, not clean release attestations.

## Pinned checkpoint and feasibility

The ordinary SDK source publisher uses four fixed public zero IDs and two output
selections with `range_policy="checked"`, `max_abs_activation=256`,
`lookup="tiled"`, `selection="tiled"` and `peer_transport="streams"`.
Fresh memory admission and the ordinary pressure watchdog precede publication.
It rejects at selection zero, step 238, layer-eight `AttentionScores`, with Q/K
intervals `[-519693187,520545155]` and `[-447354360,442887319]`.
Observed swap growth is zero. This attempt reads checkpoint values but launches
no workers and produces no online response, quality result or client-memory tier.

The separately retained [feasibility account](../../../methods/puma-feasibility.md)
also identifies complete-body metadata, live-owner, protected-work and directed
traffic limits. Full-Qwen execution and the common 64+8 workload remain blocked.
The measured 128-lane equality reduction from 69,768 to 19,920 encoded bytes is
an isolated primitive result; it does not establish whole-response improvement.

## Retained evidence and replay

`summary.json` derives counts from retained logs, explicit session excerpts and
JUnit records. `sdk/`
retains the installed gate receipts; `compat/` retains additional installed checks.
`native/` and `oracle/` preserve native results and the stronger independent audit.
The older core/integration and Clippy stdout cache had expired before collection;
their files are explicitly labeled verbatim session excerpts, not full raw logs.
`native/commands.json` distinguishes these from retained original stdout.
`checkpoint/` retains the original source-publication attempt. `build-v1/` and
`build-v2/` preserve both build receipts. Earlier contaminated and fixture-failing
SDK attempts remain under `sdk-attempts/` and contribute no passing cases.

`validation-inputs.json` and `inputs.tar.gz` bind the final source and test inputs.
The exact SDK and native-oracle test revisions are retained separately where a
later command-documentation or formatting-only edit changed their bytes.
`wheel-equivalence.json` binds the two wheel payload inventories. No wheel binary
or checkpoint is embedded in this evidence directory.

Verify the retained files:

```sh
shasum -a 256 -c SHA256SUMS
```

Rebuild/install the preserved inputs and replay the commands in the original
receipts, adapting interpreter/output paths. Installed tests must use
`python -I -m pytest -o pythonpath=.`. Checkpoint publication additionally requires
the pinned source and fresh live resource admission; old receipts authorize no
new launch. Full contracts: [checked execution](../../../../docs/research/puma-truncation-reference.md#explicit-checked-range-and-compact-execution-policies)
and [completion plan](../../../methods/puma-completion-plan.md).

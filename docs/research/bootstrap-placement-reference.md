# One-time public gate/up bootstrap: bounded cost screen

**Outcome:** half-channel arithmetic survives source-derived geometry, but not
the original ≥90%-remote constraint. Uniform half ownership retains **70.86%
prefill / 70.77% decode body MACs remote**, and conditionally reaches 20% after
five distinct-prefix cold/exact-repeat pairs sharing the same public cache.
No Low qualification, production subset compatibility, full logits/KV parity,
or measured whole-response saving is established. Existing whole-group selective
placement wins the same-budget raw cost comparison in every tested cell.

## Numeric contract and scope

`scripts/probe_bootstrap_placement.py` inspects the installed native compiler
using an existing public config and pipeline specification only. It reads no
checkpoint tensors, downloads nothing, starts no providers, and measures no
serving performance. Locked network-algebra shapes and quantized row commitments
must match all 96 compiled body groups and the historical control fingerprint.

Preparation would publish selected **paired output rows** of gate/up matrices
once, with source/numeric/placement commitments. Quantize each original input
exactly as before. Compute owned rows locally, recover remaining remote integer
rows with fresh one-use masks, and stitch in original gate-then-up order **before
the original dequantization, SiLU and multiplication**. Down projections,
attention and nonowned expansion rows remain remote. Any split stage retains
its entire original masked ingress; only fully owned groups eliminate ingress.

Tiny synthetic 3×17 activations and 22×17 i8 weights preserve exact integers,
dequantized bits and SiLU/multiply bits across five partitions, checked against
an independent Python scalar-integer oracle. Separate fresh-mask wrap16/24/32
tests check remote recovery. These are algebra tests, not a security proof or a
complete generated decoder. No real checkpoint values or logits/KV were tested.

Native prefill executes last-layer attention-output, gate/up and down groups on
one terminal row. Other groups execute 150 rows. Body denominators are
51,606,274,048 prefill MACs and 357,826,560 MACs per ordinary decode row. They
exclude output head, attention kernels, nonlinear work and all non-MAC CPU.
Budgets independently bound each phase, never only an averaged pair.

Historical scope: 150 input + eight generated tokens in each cold/exact-repeat
pair, **14 decode executions across the pair**, not 16. Exact repeat reuses RAM
prefix state. Each additional pair assumes a distinct prefix then its exact
repeat, while both arms retain the same public artifacts. Costs are **per pair**,
not per token. Dividing by 16 generated tokens would be a workload average,
not a marginal decode-token claim. The historical ledger does not independently
attribute every manifest, frame, prefill and decode body; that remains unknown.

## Half-channel reconciliation

| Cost / computation | Value |
| --- | ---: |
| Owned i8 weights | 104,595,456 bytes |
| Original float32 row scales | 466,944 bytes |
| Explicit uint32 channel maps | 233,472 bytes |
| Assumed page authentication/header and descriptors | 10,752 bytes |
| Incremental bootstrap, uncompressed | **105,306,624 bytes** |
| Gross correction/reply bodies avoided per pair | **105,713,709 bytes** |
| Original ingress retained per pair | **90,443,518 bytes** |
| Assumed 1% body-framing loss | 1,057,137 bytes/pair |
| Assumed extra split frames | 242,368 bytes/pair |
| Net recurring saving after those assumptions | **104,414,204 bytes/pair** |

Gross saving is 21.42% of the historical 493,575,247-byte cold pair and 28.68%
of its input/correction/reply arithmetic classes. It is not a measured exact
lane-payload count: historical stage bodies include framing. The sensitivity
charges 1% of prorated correction/reply bodies as unrecoverable framing and 64
new bytes per executed row of each split group. No compressor was tested.

## Warm both controls

Compare **C0 + R·T0** against **C0 + B + R·(T0 − S + H)**. Default C0 is the
124,905,091-byte historical public bundle; T0 is 368,670,156 bytes per pair.
B includes weights, scales, maps and artifact descriptors/authenticated pages.
S excludes a 1% framing allowance; H charges new split frames. Costs relocated
to Preparation→Client remain charged once, not treated as free setup.

For a target fraction t, break-even is
`ceil((B + t*C0) / (S - H - t*T0))`; no finite horizon exists when denominator
is nonpositive. Evidence also exposes hypothetical bootstrap multipliers
0.8/1.0/1.1 and recurring-common-body allowances 0/390,000 bytes per pair.
These are assumptions, not measured compression ratios or attributed manifests.

| Uniform paired-channel ownership | Remote prefill / decode | 20% pairs | 35% pairs | 50% pairs |
| --- | ---: | ---: | ---: | ---: |
| 17% | 90.10% / 90.07% | impossible | impossible | impossible |
| 34% | 80.19% / 80.13% | impossible | impossible | impossible |
| 50% | 70.86% / 70.77% | **5** | impossible | impossible |
| 75% | 56.28% / 56.15% | 3 | 8 | impossible |
| 100% | 41.71% / 41.54% | 2 | 3 | 8 |

Last two rows explicitly violate even a 70%-remote minimum. Full gate/up
ownership also removes its ingress; it is whole-group ownership, not partial
ownership. Its 50% result cannot justify an H1 promotion under original budgets.

## Separate remote-compute frontier and comparator

Public prefix channel counts vary by layer. Search exactly optimizes the
**linear correction/reply-minus-raw-weight objective** within this checkpoint's
equal-input-width, one-terminal-expansion geometry. It enumerates terminal
channel count and fills other equal-cost channels by value. This is not a global
optimum after whole-group ingress bonuses, integer maps or discrete page/frame
overheads, and not a joint H1+whole-group optimizer. Full owned gate/up groups are
priced honestly afterward. Evidence freezes all selected channel counts.

At selection horizon 16, default uncompressed sensitivity:

| Required remote minimum | Actual prefill / decode | Bootstrap MB | Recurring asymptote | 20% pairs | 35% / 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 90% | 90.00% / 90.00% | 36.025 | 10.11% | impossible | impossible |
| 80% | 80.00% / 80.00% | 72.050 | 20.33% | **79** | impossible |
| 70% | 70.00% / 70.00% | 108.076 | 30.56% | **4** | impossible |

20% sensitivity ranges: 80%-remote **67–91 pairs**; 70%-remote **3–4 pairs**;
90%-remote remains impossible. The narrow 80% margin depends strongly on
unattributed recurring control/framing. These are planning projections.

Whole-group comparator uses all 96 groups, identical independent phase budgets,
one-time raw weights/scales and the same historical stage bodies. Exact dynamic
programming plus enumeration of three terminal groups solves its raw objective.
At 16 pairs, comparable raw net savings are **566.26 / 1,139.41 / 1,712.27 MB**
for gate/up channel candidates versus **803.90 / 1,423.50 / 1,983.96 MB** for
whole-group placement at 90/80/70% remote. Whole groups win every tested horizon
(1/4/16/64), even before added split overheads. This comparator is a projection;
only the separately documented ≥90% SDK cohort has serving measurements.
The present restricted mechanism has not demonstrated an additional H1 win.

## Memory and next gate

Half ownership needs ~105.31 MB additional disk artifact storage, plus common
cache and storage/authentication metadata. Paging is a proposal, not proof of
268,435,456-byte whole-client peak: two 1 MiB page buffers plus a 622,592-byte
16-row integer stitch workspace exclude base runtime, KV, full activations,
dequantized output buffers and native snapshots. Disk reads, CPU, OS file cache
and transient copies retain separate owners. OS cache is neither zero nor the
same quantity as process RSS. Native weight snapshot/quantization behavior is
unmeasured and may defeat bounded paging. No RSS sampling or swap claim follows.

**Next implementation gate:** only pursue a joint row/whole-group frontier that
beats the matched whole-group control. Then add model-neutral paired-row
ownership commitments to compiler schedules, provider inventory, fresh remote
material, authenticated bounded cache import and explicit client admission.
Resolve MLP tiling and integer-stitch ownership before production compatibility
claims. Require tiny generated full logits/KV parity first; only afterward run
an exclusive source-frozen matched serving cohort measuring every fresh-client
peak, all directed artifacts/bodies, page I/O, CPU and ordinary response metrics.
Changing ≥90% remote to ≥80% or ≥70% requires explicit user opt-in.

## Reproduce

Use an existing local public config; no checkpoint download or tensor read:

```sh
target/frontier-sdk-env/bin/python scripts/probe_bootstrap_placement.py \
  --config /path/to/existing/config.json \
  --output target/bootstrap-placement-replay.json
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= -q \
  tests/test_bootstrap_placement_probe.py
```

`docs/evidence/research-bootstrap-placement.json` is exclusive-create immutable
evidence. Rerun to a new path. It freezes script/tests, prior evidence, config,
pipeline, semantic/quantizer/placement/compiler sources and native extension
hashes, and records the compiled schedule digest and original matrix commitments.
29 focused tests pass; focused Ruff and whitespace checks pass.

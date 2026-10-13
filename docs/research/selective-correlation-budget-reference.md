# Selective exact-ring correlation budget

This is a conditional cost gate, not a cryptographic implementation or runtime
selector. The probe uses frozen real-checkpoint geometry and header-only semantic
lowering; it never resolves or reads checkpoint tensors.

## Decision

**Reject gate/up-prefill as a universal 20% cold-pair saving.** On the matched
150-token unplaced control, even granting every correction residual byte to this
subset reaches only **19.5412%**. The historical 493,575,247-byte control reaches
the same rejection. Decode-only is far too small. This does not reject a
fixed-weight generator or warm/longer-prefix deployment.

The first construction-cost target should be semantic **`mlp_gate_up` prefill
with more than one executed row**, layers 0–22 in this frozen model. Leave
terminal one-row gate/up, all decode and other stages on the prepared protocol.
That avoids a separate setup/key owner for a 27,939-byte single-row opportunity.
If setup is shared at negligible marginal cost, include that terminal row again;
the whole gate/up-prefill group below is a slightly more generous bound.

Within this bulk subset, **layer 18** has the highest exact payload density:
`layers.18.mlp.gate_up_proj`, shape **9,728 × 896**, original ring `Z/(2^24)`,
24-bit packed inputs, and output-width histogram
`{21:29, 22:250, 23:9437, 24:12}`. Its weight SHA-256 is
`0e861ffd8b10c26774a2e0e5b44253253e2aae2ee682a6bc80b930fe6f47a9d7`.
It executes 150/288 prefill rows, consumes **4,189,650 / 8,044,128** mandatory
correction bytes and generates **1,459,200 / 2,801,664** output lanes. This is a
semantic compiler-derived choice, not a family-name parser or runtime selector.
One stage alone cannot deliver the whole-boundary 20% target. No native generator
implementation is recommended until a proved fixed-W specialization is priced.

### Free-replacement maxima before any implementation

Percentages below use the sample with the median **total** within each of the
four separate three-replicate cohorts; they do not combine category medians.
Each cell is **cold-plus-repeat / warm-all-public-reused**, optimistically
granting the subset all correction residual bytes. The report retains exact
rationals, all sample totals and representative replicate IDs.

| Subset | 150 unplaced | 150 placement | 288 unplaced | 288 placement |
| --- | ---: | ---: | ---: | ---: |
| All remote groups, both phases | 28.172 / 37.715% | 25.667 / 38.280% | 31.843 / 37.714% | 30.200 / 38.278% |
| Gate/up, both phases | 21.440 / 28.703% | 22.419 / 33.436% | 24.225 / 28.691% | 26.378 / 33.434% |
| All prefill | 25.684 / 34.385% | 23.389 / 34.882% | 30.312 / 35.901% | 28.741 / 36.429% |
| Decode only | 2.527 / 3.383% | 2.309 / 3.444% | 1.555 / 1.841% | 1.479 / 1.875% |
| Gate/up prefill, including terminal | 19.541 / 26.161% | 20.433 / 30.473% | 23.057 / 27.308% | 25.106 / 31.822% |
| Bulk gate/up prefill, recommended cost target | 19.536 / 26.153% | 20.427 / 30.465% | 23.053 / 27.304% | 25.102 / 31.817% |

### Exact bulk-subset allowances and optimistic ceilings

Bulk mandatory correction volume is **96,234,199 bytes** at 150 tokens and
**184,769,676 bytes** at 288. It uses 3,450/6,624 row tickets, 33,561,600/64,438,272
output lanes and 3,091,200/5,935,104 input-mask lanes. Client expansion must produce
both input masks and output shares; these lane counts are not measured CPU costs.

At 20%, the table separates **conservative payload-only allowances** from
**optimistic all-correction-residual ceilings** for all new traffic. These are
not seed-size targets. The payload-only allowance is not a necessary upper bound:
proving that additional correction overhead disappears can increase it. Residual
remains unallocated; granting all of it is only a generous necessary upper bound.

| Control | First-pair headroom: mandatory / optimistic bytes | First-pair mandatory bytes/output lane | Warm mandatory bytes/output lane |
| --- | ---: | ---: | ---: |
| 150 unplaced | −2,489,454.6 / −2,292,734.6 | −0.074175683 | 0.670157668 |
| 150 placement | 1,868,210.4 / 2,013,184.4 | 0.055665117 | 0.982109470 |
| 288 unplaced | 24,299,500.4 / 24,496,282.4 | 0.377097331 | 0.764770952 |
| 288 placement | 37,439,195.2 / 37,584,207.2 | 0.581008678 | 1.063531778 |

At any listed horizon, the optimistic lane ceiling exceeds the mandatory lane
allowance by exactly `correction_residual / output_lanes`: residuals are
196,720 / 144,974 / 196,782 / 145,012 bytes respectively. The evidence retains
both exact rational series. Only a negative **optimistic** allowance proves
impossibility from this screen; the 150-unplaced cold-pair rejection survives.

For example, 150-placement bulk prefill gets **541.5103 bytes per row correlation**
in the first pair **when crediting mandatory payload alone**, before any setup,
role-key delivery, cartridge, control or renewal deduction. Proven overhead
removal may raise this allowance, up to the optimistic ceiling above.
The terminal-inclusive subset gets a mandatory-only allowance of 1,896,149.4 bytes,
0.056481215 bytes/output lane. The narrower target preserves almost all savings
but makes the marginal terminal setup explicit. Its optimistic allowance is
2,041,123.4 bytes. At finite horizons 1,2,4,16,64,4,096 both exact series are in the
evidence; any concrete E(R) debits the applicable credited-removal budget.
No candidate has measured costs establishing this gate.

30% warm is impossible for gate/up-prefill without placement (maxima 26.16/27.31%).
Placement leaves only a small positive 30% warm budget. 35% is impossible for this
subset even with zero replacement cost in every listed cohort. Broader all-prefill
or whole-group screens have more headroom but change the deployment surface.

The 128-square bounded-column control is decisively unsuitable here. It explicitly
includes the terminal gate/up row, unlike the recommended bulk-only subset:
terminal-inclusive gate/up prefill openings alone are
**2,457,993,216 / 3,660,840,960 bytes** in the first
150/288 pair, versus about 96/185 MB removed. At four pairs they cost
6,066,536,448 / 10,877,927,424 bytes. Extra client dense adaptation is
30,079,909,888 / 57,745,408,000 MACs per pair before PCG expansion. Finite fresh
columns are charged, but this particular consumer is neither client-light nor
within the 20% byte gate; cross-triple mask reuse is not assumed.

## Accounting contract

The saved 150/288-token Low controls each execute one new-prefix, eight-output
request and its exact repeat. Each sample and workload stays separate. The full
compiled model plan digest, stage shapes, semantic roles and phase demand must
match the saved preflight. Historical weight fingerprints, rings, residue-width
histograms and serializer payloads are joined across independent reports.
Duplicate JSON keys, duplicate/missing stages, shape/width/ring/weight/source-lock
drift and runtime issuance disagreement reject the screen.

Current v3 reports have no full per-stage envelope or job-padding attribution.
The Rust serializer's exact payload size is `ceil(rows*sum(widths)/8)`, continuous
across rows. Historical saved stage aggregates reconcile to **138,864,059 bytes**.
For current phase/stage subsets, the probe counts complete bytes in exact bit
volume, rounding **down**; unlocated job/frame padding stays in the residual.
This conservative mandatory payload is therefore a strict subset, not a claim
of fully measured candidate attribution. The optimistic subset bound
grants **all remaining correction bytes** to each selected subset, including bytes
that likely belong to unselected stages. No ratio allocates unattributed bodies.
Acknowledgements, seeds, lifecycle controls, inputs and results remain charged.
Known-base, known-output and residual categories conserve the measured total;
complete base and per-generated-token rates remain unknown. Application bodies,
TCP streams and physical wire are not interchangeable. Original checkpoint and
tokenizer publication/distribution are outside these recorded response bodies.

The candidate retains identical original placement: at least 90% of original
body MACs stay remote independently in prefill and decode. Additional client
cryptographic work is unknown and additional, not hidden inside that ratio.
Neither Low <=256 MiB nor Medium <=2 GiB receives candidate qualification from
payload arithmetic or the historical control peaks. New generator, model,
numeric and verifier identities must be established; saved identities apply only
to their original controls.

## Budget and finite ownership

For R cold/repeat pairs, common public bytes C, recurring control T, explicitly
credited removal S, generated output lanes L, and all additional candidate
bytes E(R), the exact conditional lane allowance at target g is:

```text
b_max = (R*S - g*(C + R*T) - E(R)) / (R*L)
```

Use mandatory payload for the conservative allowance; use mandatory payload
plus all correction residual for the optimistic necessary upper bound. The
former must not be presented as a hard candidate-cost ceiling independent of
what additional overhead the construction proves removable.

The no-reuse endpoint uses `R*(C+T)` for both candidates. Public reuse is symmetric;
the all-reused aggregate is a sensitivity endpoint, not certification that every
manifest byte is reusable. E includes initial setup and both role keys, controls,
public cartridge creation and every distribution owner, failures/burns, framing,
helpers and renewals. Missing required costs propagate as unknown, never zero.
Creation CPU is separate from transferred creation inputs/outputs and must also
be measured. Cost zero below means an explicit optimistic bound, not a claim
that generator expansion or setup is free.

The callable finite row-ticket model reserves one ticket per stage invocation.
For capacity Q, rows N and burned rows B, pools are `ceil((N+B)/Q)`; initial
delivery and each subsequent renewal are charged, and unused slots are reported.
Per-stage capacities require independent pools. Cancellation, rollback/restore,
cloning, retries and expiry must burn committed indices; this arithmetic is not
an ownership implementation. Renewal boundaries can reduce savings abruptly.

The bounded threshold optimizer enumerates every distinct exact payload/lane
ratio. With known uniform nonnegative replacement cost b and no coupled setup,
selecting precisely groups with `payload > b*lanes` maximizes net payload saving.
Unknown b returns unknown selection. This separable optimum is not an optimizer
for arbitrary shared setup, CPU, RAM or construction-specific constraints.

## Paper contracts

The source-pinned [full-text review](compact-setup-fulltext-review.md) records
1-based PDF pages and hashes. `2025/1223`, pp. 7 and 33–36, supports exact-ring
correlation constructions, but does not supply the required client-light fixed-W
Gen. PDF SHA-256:
`130178519ec48cbbe64b201e42bb581b513f77503f671e1a56e4b0909b4aa375`.

`2026/2298`, pp. 70–72, supports padded rectangles and at most n fresh sequential
columns of one triple; pp. 74–75 do not supply cross-instance private-mask reuse.
Page 72 leaves power-of-two-ring extension to future work; authentication is
unpriced on p. 75. PDF SHA-256:
`744e311e462746192951d27046242561db6b085bac64664076b331cfebc148cb`.
The probe separately costs 128-square **hypothetical exact-ring** blocks, fresh
left masks at each column-pool renewal and two dense opening shares per block.
It does not import the field construction's security or claim 31.55 GB
fresh-per-use openings reject every bounded-column schedule. Dense client
adaptation, expansion, keys and authentication remain additional.

## Replay

```sh
target/frontier-sdk-env/bin/python scripts/probe_selective_correlation_budget.py \
  --output target/selective-correlation-budget-replay-new.json
target/frontier-sdk-env/bin/python scripts/probe_selective_correlation_budget.py \
  --summary --output target/selective-correlation-budget-summary-new.json
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath= -q \
  tests/test_selective_correlation_budget_probe.py
```

Output must be new. Probe verifies nine immutable evidence hashes and both PDF
hashes before reporting. No build, install, checkpoint download, heavy benchmark,
production selection or crypto execution occurs.

The [compact evidence](../evidence/research-selective-correlation-budget.json)
is the deterministic `--summary` result; it pins a canonical hash of the richer
full replay, source/implementation hashes, all twelve category-conserving samples,
exact subset ceilings and paper/text hashes. All public reuse and zero-cost
endpoints remain conditional rather than new serving identities or measurements.

# One-time bootstrap traffic frontier

This investigation asks whether reusable, model-specific client artifacts can
remove 20–50% of recurring all-link traffic under the **256 MiB whole-client
ceiling**. Bootstrap delivery, recurring material, client computation and storage
have separate owners. A public artifact can survive many requests; masks,
correlations, tickets and protocol attempts retain their one-use contracts.

The starting hypotheses extend the
[Low-tier traffic plan](../plans/low-tier-all-link-hypotheses.md). There are five
independent investigations: reusable correlation cartridges, mask-aware reply
coding, partial expansion-stage ownership, private narrow ingress, and an
explicitly different attested-executor trust contract. Their percentages cannot
be added or ranked as measured serving results.

## Five completed investigations

Five independent reviews produced bounded replayable probes. The most promising
unchanged-trust lead is **checkpointed public correlations plus mask-aware reply
coding**; it still lacks a certified private consumer and a lightweight client
implementation. An **attested full linear executor** has a larger projected
budget, under a different hardware-isolation trust contract. None qualifies Low
or supplies a measured serving improvement.

| Direction | Result and conditional impact | Decisive next gate |
| --- | --- | --- |
| [Correlation cartridges](bootstrap-cartridge-reference.md) | Sparse fresh-query algebra and checkpoint reconstruction pass. Stride 64 reduces public bootstrap to 713.23 MB from 23.26 GB for the complete two-checkpoint layout. The separate P-only layout is 11.45 GB and omits verification. | Reconstruction still requires client work; the native BabyBear reference cannot inherit the source's `q > 2^31` distance bound. Parameter certification, issuance lifetime, paging and complete client cost are missing. |
| [Cyclic masked replies](cyclic-mask-codec-reference.md) | Deterministic zero-error coding passes exhaustive and large-prime checks. Public-bound reply widths match existing row residues, so this is an enabler for field conversion, not an additional residue-codec saving. | Range promises, malformed-reply behavior, authenticated layouts and any interactive verification require a composed security contract. |
| [Bootstrapped channel placement](bootstrap-placement-reference.md) | Half gate/up ownership projects 104.41 MB removed per pair after assumed framing, for 105.31 MB bootstrap. It leaves only 70.86%/70.77% of prefill/decode MACs remote. | Fails the required 90% remote-compute floor. Whole-group placement beats the tested restricted row candidates at matched budgets and horizons. |
| [Private byte ingress](bootstrap-ingress-reference.md) | Exact carry algebra passes, but the strongest tested biased vector-OT layout needs 1.24 TB of selected vectors per pair. Even a free byte/correction helper tops out at 40.61% on the first pair. | A circuit-specific exact-ring PCF and its complete private carry consumer remain unspecified and unpriced. Generic PCF literature is not an implementation or impossibility result. |
| [Attested full linear execution](attested-linear-bootstrap-reference.md) | Software AEAD controls pass. Public-bound packed replies project 39.49% first-pair and 52.87% recurring all-link-body savings before extra setup/renewal/distribution; zero-extra-cost 50% begins at six pairs. | Actual attestation, checkpoint execution, hardware trust, complete distribution, IPC/device traffic, latency and fresh client/provider peaks remain unmeasured. |

These are screens with different execution and trust assumptions. Hardware-isolated
execution accepts clear quantized activations inside the enclave; it does not
inherit the ordinary prepared protocol's provider-activation privacy contract.
The correlation-only enclave design also retains `W*r` work and internal transfer;
its traffic cannot be treated as eliminated by relabeling a network edge as IPC.

## Matched lifecycle accounting

The historical [reconciled body ledger](all-link-frontier-reference.md) records
493,575,247 bytes for one new 150-token prefix with eight generated outputs and
its exact repeat. Arithmetic accounts for 368,627,357 bytes; other recorded
controls/seeds add 42,799 bytes. Public delivery contributes 124,905,091 bytes.
This is a sensitivity reference, not the latest Low-tier configuration or
physical wire. Complete per-request base and per-generated-token rates cannot
be inferred from those category totals.

For `R` such workload pairs, use the same artifact-reuse assumption on both sides:

```text
control   = C0 + R * T0
candidate = C0 + B + R * (T0 - S + H) + ceil(R / K) * E
```

`C0` is common one-time delivery; `T0` is all recurring control traffic; `B` is
additional reusable bootstrap across every link; `S` is removed recurring
traffic; `H` is new recurring helper traffic. Optional renewal costs `E` every
`K` pairs include initial issuance. Renewals may lower the saving at an epoch
boundary, so reaching a target once does not certify every later horizon.
Provider/checkpoint distribution, failed attempts and protocol framing must be
added where the historical ledger has no measurement.

Only bytes proven immutable and reused belong in `C0`. The existing public-bundle
aggregate does not establish its entire recurrence split. The separate
[budget evidence](../evidence/research-bootstrap-budget.json) therefore checks
both endpoints: all recorded public delivery becomes reusable, or all recurs.
Both reconstruct the same first pair. They are conditional accounting cases,
not measured cache traces.

Even eliminating corrections for free removes only 28.17% of the first pair.
With full reuse of the original public-delivery aggregate, the asymptotic
ceiling is 37.72% of recurring bodies. A hypothetical 100 MB bootstrap reaches
20% at two pairs; a 1 GB bootstrap takes sixteen. Those ideal cases price zero
replacement traffic, client expansion cost or renewal. Correction elimination
alone cannot reach 50% in this ledger.

Preloading a finite tape of `R` corrections pays `B = R * S`; its matched
network saving is zero before additional framing. A successful reusable
cartridge must change that cost equation, not only the delivery date.

## Joining cartridge and reply-codec budgets

The joint accounting probe uses the same 96 checkpoint stage identities and
weight digests, the independently reconciled **15,297 executed stage rows**, and
saved per-output public residue widths. It does not execute a composed protocol.

A source condition `q > 2^31` requires **32-bit**, not 31-bit, field elements.
For prime `q = 2^32 - 5`, each saved 23/24/26-bit output row admits the cyclic
codec's same-width alphabet. The check widens each public row interval to the
largest symmetric interval fitting that width, so it uses no private values or
unrecorded row norms. Tests separately exercise the actual codec at those bounds
and at wrap endpoints. This prices payloads; new layout/frame/verification costs
remain additional.

| Historical pair payload | Original | Hypothetical prime-field/codec |
| --- | ---: | ---: |
| Masked inputs | 90.104 MB | 114.932 MB |
| Masked output payload | 138.864 MB | 138.864 MB |
| Correction bodies, including their original metadata | 139.061 MB | Optimistically zero |
| Net removed before new costs | — | **114.233 MB** |

With both controls reusing the same original public delivery, this gives an
optimistic **30.99% recurring all-link-body ceiling**, not the original
37.72% free-correction ceiling. It permits at most **40.499 MB of additional
recurring helper traffic per pair** to retain a 20% asymptotic reduction. Any
finite bootstrap reduces that allowance further at a finite horizon.

| Additional cartridge bootstrap assumption | First 20% horizon | First 25% horizon | First 30% horizon |
| --- | ---: | ---: | ---: |
| 713.229 MB, checkpoint stride 64 | 19 pairs | 34 pairs | 207 pairs |

Each pair contains one new 150-token prefix/eight-output request and its exact
repeat, not two fresh prefills or an endless identical-prompt cache-hit stream.
These horizons assume zero other new costs, no renewal and complete reuse of
the original public-delivery aggregate. The separate cartridge report preserves
its historical 24-to-31/32-bit sensitivities; this joined screen instead uses
actual saved stage widths and output-codec algebra. Their projections must not
be pooled. None certifies the prime-field cartridge's security or reuse capacity.

### Follow-up priority

1. Specify a complete exact-ring matrix-PCF consumer and a small prime-field
   cartridge/codec composition. Price every fresh correlation, private carry,
   output verification, source delivery and renewal before implementation at
   checkpoint scale. Existing ring/circuit PCF papers leave this possibility open.
2. Measure checkpoint reconstruction, paging and client ownership for one actual
   stage against the existing whole-group-placement control. Dense client
   regeneration must not pass as lightweight remote execution; enforce the
   remote-compute floor and 256 MiB whole-client maximum separately.
3. Pursue actual CPU-enclave execution only as an explicit hardware-trust option.
   Direct client attestation avoids introducing a new Preparation key-release
   authority. Charge both internal and external traffic and preserve source/plan
   commitments before comparing complete responses.

Masked reply coding merits further work as a component of the first route. The
tested row-placement and vector-OT constructions have no surviving benefit over
their matched controls under the original constraints.

## Admission and interpretation

- **Exactness:** standalone integer/codec parity, complete decoder logits/KV,
  checkpoint fidelity and protected execution are separate checks.
- **Remote work:** the original frontier requires at least 90% body MACs remote
  in prefill and decode independently. Relaxed 80%/70% candidates must disclose
  the changed budget and compare with whole-group placement at that same budget.
- **Memory:** artifact size and bounded scratch do not establish a 256 MiB
  process peak. Admission, every replicate's peak, overlap, SDK/tokenizer/state,
  disk, page cache and device owners remain visible.
- **Privacy:** mask reuse or a provider-reconstructible mask seed cannot replace
  missing correlation generation. Algebra does not certify LPN parameters,
  malicious-output security, oblivious transfer or hardware attestation.
- **Serving costs:** client CPU/RSS, Preparation CPU/RSS, TTFT, decode throughput
  and all-link traffic remain separate objectives. A traffic projection earns
  no latency or client-memory result.

## Budget replay

```sh
python3 scripts/probe_bootstrap_budget.py \
  --output target/bootstrap-budget-replay.json
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= --noconftest -q \
  tests/test_bootstrap_budget_probe.py
```

The bounded exact-integer accounting checks cover both candidates warming their
public delivery, finite-tape non-savings, renewal discontinuities, exact target
boundaries, body conservation, source-bound prime-codec payloads and unknown
memory/per-token metrics. The report binds its source script and linked evidence
by SHA-256. Each investigation above supplies its own replay and exact scope.
The accounting report's `complete` field means the screen completed, not that a
candidate protocol exists.

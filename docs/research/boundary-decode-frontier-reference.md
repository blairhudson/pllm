# Coupled-boundary decode frontier

This five-direction investigation builds on the completed
[private-boundary tier confirmation](private-boundary-tiers-reference.md).
Its objective is faster full-vocabulary decode while retaining exact logits,
state and sampling, the measured cold-request traffic advantage, and bounded
client ownership. The individual references below are research probes; their
resource scopes do not supply new SDK measurements.

The completed parent [serial measurement and decision report](private-boundary-decode-frontier.md)
retains all five probes, original raw reports and read-only summary replay.
Parallel full replies are the first integration candidate; uniform residues and
an allocation-matched byte-dot follow-up retain separate gates.

## Fixed starting evidence

The original Qwen2.5 W8A8 150+8 confirmation has three fresh clients per recipe,
one private cohort, shared native 100/40 Mbps party caps, and complete-client
ownership. Coupled decode is 1.923513 tokens/s against the maintained recipe's
3.793981 tokens/s. The best selective-placement incumbent reaches 4.393 tokens/s
under its own configuration. The coupled maximum client peak is 210.22 MiB;
all original state/output checks pass with zero observed swap growth.

A read-only audit of all three coupled records in
`docs/evidence/private-boundary-tiers-qwen25-2026-10-06.json` finds identical
boundary counters:

| Request-window item | Count or application bytes |
| --- | ---: |
| Complete full-head calls | 8 |
| Lookup rows, including generated-token lookups | 157 |
| Head uploads | 59,584 |
| Head downloads | 9,726,176 |
| Lookup uploads | 92,910 |
| Lookup downloads | 306,744 |
| Total boundary-operation bodies | 10,185,414 |

Head replies are 95.49% of these boundary-operation bodies. The underlying
eight pairs of `151936 * 4`-byte vectors occupy 9,723,904 bytes; the remaining
2,272 bytes are their recorded reply framing. This partition concerns boundary
operations, not all request traffic or physical wire.

At each worker's 40 Mbps upload cap, two serial raw replies imply 243.098 ms of
steady-state serialization; independent concurrent uploads imply 121.549 ms.
Their combined 80 Mbps fits the client's shared 100 Mbps download capacity.
These calculations omit burst credit, computation, framing and scheduling.
The difference between reciprocals of the original maintained/coupled decode
medians is 256.307 ms/output. Similar magnitudes motivate the experiments;
they do not attribute the observed delay to a measured phase.

## Five directions

1. [Duplex and striped head delivery](boundary-head-duplex-probe.md): concurrent
   full replies, public output stripes, and bounded compute/upload overlap.
2. [Random-head-share runahead](boundary-head-runahead-probe.md): issue fresh
   `Wr` during the current token's body execution, then join `W(q-r)`.
3. [Exact smaller head residues](boundary-head-residues-probe.md): public,
   source/geometry-bound output rings with unchanged integer reconstruction.
4. [Lookup/head scan fusion](boundary-scan-fusion-probe.md): reuse a tied-weight
   traversal for XOR lookup and independent additive head work.
5. [Byte-dot head execution](boundary-head-dotprod-probe.md): carry-correct
   signed-byte limbs and runtime-gated ARM SDOT for exact ring32 outputs.

## Comparison rules and interactions

Concurrent replies are the control for judging any additional runahead benefit.
With independent equal A/B service times, ready-parallel execution already hides
B behind A; issuing B earlier does not remove A's dependency on the final hidden
state. Asymmetric workers, contention and shared-link scheduling need their own
measurements. Normal EOS and a known output cap need not issue a future head;
cancelled work after current-step issuance remains charged.

Stripes must retain complete output order and final-logit reconstruction. Sending
each stripe before computing the next is a different schedule from one bounded
compute-ahead slot. Extra framing, queued outputs and drain time belong in both
comparisons. Neither changes the client's shared access capacity.

The uniform W8A8 geometry bound can be verified from an already admitted dtype,
quantizer and hidden width. A tighter per-row L1 certificate has an additional
source-verification requirement. A thin client cannot authenticate arbitrary
worker-declared widths merely by hashing them.

Fusion and byte-dot kernels target worker work, not head-output serialization.
Fusion must execute a fused traversal to earn that label. Reusing one immutable
matrix in two existing calls is only storage sharing. Any arithmetic-kernel
improvement must include limb packing and the original API's allocation scope.

Prototype timings require fixed source/binary identities and counterbalanced
orders. Whole-client RAM qualification still requires fresh complete SDK workers
and their largest replicate peak; isolated fixture RSS cannot inherit the
210.22 MiB baseline. A serving comparison must charge setup, cancellation,
complete directed traffic and N-1 decode, and recheck every logit/KV/token/RNG
commitment under the same source, workload, build and access conditions.

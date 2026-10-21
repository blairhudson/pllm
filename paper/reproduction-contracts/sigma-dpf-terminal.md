# SIGMA CPU DPF terminal audit

Status: **source-profile admission blocked**. This is a primitive-level finding
in one inspected source snapshot, not a demonstrated attack on every SIGMA
implementation or an executed model. No upstream code was built or executed.
The leaking CPU encoding remains prohibited. Decision D16 in
`paper/methods/DECISIONS.md` separately approves a distinct CPU port of the GPU
primitive, with full AES-expanded terminal words and actual key/PRG accounting.
That approval does not qualify an implementation or provide security review.

## Inspected identity and equations

The CPU source is pinned to EzPC commit
`aa0f2f860266fb79901405ef5e9965b8c9fe2e88`. In
[`keyGenDPFET`](https://github.com/mpc-msri/EzPC/blob/aa0f2f860266fb79901405ef5e9965b8c9fe2e88/GPU-MPC/ext/sytorch/ext/llama/dpf.cpp#L312-L400):

- Lines 327–328 clear the least significant bit of each root seed.
- Lines 349 and 361–380 preserve that zero bit in every corrected child seed.
- The target path retains `t0 XOR t1 = 1`.
- Lines 387–400 insert each terminal control in the seed's low bit and XOR both
  seeds with a 128-bit one-hot target suffix. Both keys receive the same terminal
  correction.

[`OneBlock`](https://github.com/mpc-msri/EzPC/blob/aa0f2f860266fb79901405ef5e9965b8c9fe2e88/GPU-MPC/ext/sytorch/ext/cryptoTools/cryptoTools/Common/Defines.cpp#L10-L12)
has value one in the low word. The target suffix `127` selects that low bit.
Consequently the correction `L` satisfies

```text
LSB(L) = 1 XOR [point mod 128 = 127].
```

Either key alone therefore reveals that predicate. The observation needs no
evaluation, peer key, root seed or clear input. The complete terminal block is
sent to each party by
[`send_dpfet_keypack`](https://github.com/mpc-msri/EzPC/blob/aa0f2f860266fb79901405ef5e9965b8c9fe2e88/GPU-MPC/ext/sytorch/ext/llama/src/llama/comms.cpp#L526-L535),
so the low bit is not omitted by that serializer.

## Accounting is a separate discrepancy

SIGMA's Theorem 1 states `129(n-7)+254` logical bits for `n > 7` at
`lambda=127`. The inspected generator uses 127-bit roots/correction seeds but a
128-bit terminal correction, yielding `129(n-7)+255` logical bits before framing
or physical block/control padding. Dropping the extra bit loses target-suffix
information; retaining the inspected construction exposes it. Merely changing
the reported key size cannot repair privacy.

This audit does not infer an undocumented terminal compression from the theorem.
It also makes no finding about the distinct GPU terminal construction, or about
end-to-end exploitability of every wrapper using this primitive.

## PLLM disposition and regression

Do not adopt the inspected terminal construction. Existing PLLM profiles and
their 128-secret-bit seed/control encoding remain unchanged. Any alternative
SIGMA backend must bind its distinct profile and honestly charge its keys and
PRG work; substantial source/security adaptations need explicit approval.

`tests/test_sigma_source_terminal.py` independently checks the terminal algebra
for every seven-bit suffix, both target-path control assignments, several seed
pairs and larger tree widths. It is a negative algebraic regression, not an
upstream execution, security proof or complete decoder result.

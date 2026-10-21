# SIGMA CPU arithmetic-output mask audit

Status: **full-output-ring mask adaptation approved in D17**. This is a bounded
output-view finding in EzPC commit `aa0f2f860266fb79901405ef5e9965b8c9fe2e88`,
not an executed upstream attack, a finding about every SIGMA implementation,
or authorization for unrelated protocol changes.

## Exact source trace

Paths below are relative to the pinned
[`GPU-MPC/ext/sytorch/ext/llama/`](https://github.com/mpc-msri/EzPC/tree/aa0f2f860266fb79901405ef5e9965b8c9fe2e88/GPU-MPC/ext/sytorch/ext/llama)
directory.

- `include/llama/group_element.h:61–67`: `random_ge(bw)` draws a uint64 and
  applies `mod(a, bw)`. The parameter is a bit width, not a stream selector.
- `api.cpp:4033–4044`: `SlothLRSfromWrap` chooses `rout = random_ge(1)`, passes
  it to `keyGenSlothLRS`, and retains it as the dealer's output mask.
- `api.cpp:4068–4076`: online holders evaluate and reconstruct `bitlength`-bit
  outputs, rather than retaining independently hidden additive shares.
- `api.cpp:4167–4178,4202–4210`: `SlothTRfromWrap` makes the same one-bit choice
  and reconstructs a `bin-scale`-bit output.
- `pubdiv.cpp:218–247`: `rout` enters the arithmetic correction. Splitting that
  correction into full-width random shares does not enlarge the final output
  mask once both output shares are reconstructed.

## Independent boundary witness

Under the intended masked-output equation
`y_hat = floor(x / 2^s) + rout mod 2^m`, masks confined to `{0,1}` do not hide a
multi-bit result. Private inputs `0` and `2*2^s` yield disjoint observable sets
`{0,1}` and `{2,3}` whenever the two inputs are admitted and `m >= 2`. Numerous
positive inputs satisfying the quarter-ring promise already exhibit this issue;
neither signed wrap nor a broken DPF is needed.

`tests/test_sigma_source_masks.py` checks these distinct observation sets for
reduced- and same-ring outputs, including ring50/Q12 geometry. It separately
exhausts small full-ring masks to show that they remove this specific boundary
witness. The tests do not execute or establish correctness of the source
truncation routine and are not a composed security proof.

## Approved disposition (D17)

Do not reproduce literal one-bit arithmetic-output masks. The approved distinct
profile samples fresh uniform masks over the complete output ring: `n-s`
bits for TR and `n` bits for same-ring arithmetic shifts. Boolean outputs still
use fresh Boolean masks. Preserve the original integer floor function and
explicitly charge sampling, correction material, physical storage and delivery.

This changes the inspected source's randomness/view contract. The user's explicit
D17 approval authorizes this mask change separately from D16's DPF primitive.
It does not establish an implemented decoder, source-CPU byte equivalence or a
composed security proof. The next source arithmetic executor must not inherit
the unsafe one-bit choice; existing legacy profiles remain unchanged.

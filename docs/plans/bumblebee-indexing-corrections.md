# Approved paper-derived coefficient indexing

Scope: pure exact modular algebra only. These are approved interpretation
corrections, **not author-confirmed errata**, an evaluated-source identity, or a
claim that BumbleBee inference is implemented. B-N1's final-paper numerical/HE
manifest remains unresolved; the artifact's Q16 profile is not substituted.

Specification: NDSS 2025 final PDF, SHA-256
`184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b`,
locked in `paper/methods/source-lock.v1.json`. The shared-computation contract and
`paper/reproduction-contracts/shared-profile-resolution.md` section 4.4 permit
pure packing/interleaving/conversion algebra independently of B-N1.

## Corrections approved for this implementation

1. Equation (1), p.5: lhs row stride is `nw*mw`, not the printed `kw*mw`.
2. Algorithm 1, p.7: the multiplicand paired with `ct[alpha,beta]` is
   `V[beta,gamma]`, not `V[beta,alpha]`.
3. Figure 8, p.17: for group start `i` and local lane `j`, use global block
   `t=i+j`, then `(alpha,gamma)=divmod(t,n_prime)`. Ignore `t>=k_prime*n_prime`.
   The coefficient lane is `j`, not `(j+pidx)%mw`. Crop edge rows/columns.
4. Algorithm 1 pads the last group iff `mw` does not divide `k_prime*n_prime`.

For one window, with `mw` a power of two and `kw*mw*nw<=N`, define in
`Z_q[X]/(X^N+1)`:

```text
Qhat = sum_{r,h} Q[r,h] X^(r*nw*mw-h)
Vhat = sum_{h,c} V[h,c] X^(c*mw+h)
```

Negative powers carry negacyclic signs: `X^-h=-X^(N-h)` for `0<h<N`.
Product exponent for entries `(r,h)` and `(h',c)` is
`(r*nw+c)*mw+(h'-h)`. Since `|h'-h|<mw`, only `h'=h` contributes to
coefficients divisible by `mw`; distinct output cells have distinct such
coefficients below `N`. Off-target terms cannot wrap onto these coefficients
because `N` is divisible by `mw`. Thus coefficient `(r*nw+c)*mw` is the dense
modular dot product. This also proves rectangular windows, not just square ones.

Example: `(kw,mw,nw,N)=(2,2,3,16)`. Lhs row starts are `0,6`, not `0,4`;
target positions are `0,2,4,6,8,10`. For Q `[[1,2],[3,4]]`, its polynomial
is `1-2*X^15+4*X^5+3*X^6`. Figure 3's `(4,2,2)` row starts `0,4,8,12`
also corroborate the corrected row stride.

For output shape `(k,m,n)` and windows `(kw,mw,nw)`, set
`k_prime=ceil(k/kw)`, `n_prime=ceil(n/nw)`. Accumulate inner block products
before interleaving. Then:

```text
t = alpha*n_prime+gamma
(g,j) = divmod(t,mw)
U[alpha*kw+r, gamma*nw+c] = packed[g][(r*nw+c)*mw+j]
```

For shape `(3,2,7)` and windows `(2,2,3)`, six output blocks form three
groups. Block coordinates are `(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)`;
group 1 contains `(0,2)` in lane 0 and `(1,0)` in lane 1. Group-local `j`
cannot recover those coordinates without the global offset. No group rotation
appears in Figure 6's InterLeave invariant.

## Native boundary and reuse

`crates/pllm-core/src/coefficient_layout.rs` supplies checked immutable geometry,
window packing, exact odd-modulus trace/FFT-style interleaving, monomial shifts,
and parsing. The inverse-of-two and trace/merge laws reuse the algebra of
`coefficient_repacking.rs` and `python/pllm/runtime/coefficient_repacking.py`;
the old fixed serialization and public APIs are unchanged. No backend is forked.
Even a single input must be traced when its stride exceeds one.

All operations use caller buffers and return nonallocating typed errors. Safe
Rust disjoint borrows prohibit input/output/scratch overlap. Admission bills
simultaneously live input matrices, output matrix, two encoding polynomials,
product group, interleave scratch (padded inputs plus a temporary polynomial),
and packed outputs. The destination polynomial is already inside packed outputs;
it is not charged twice. Descriptor bytes are reported separately and own no
heap data. No polynomial multiplication implementation is supplied: caller's
existing arithmetic/backend owns product accumulation. Tests use an independent
dense negacyclic convolution only as an oracle.

Degree remains at most 4096, each odd coefficient limb below 2^61, interleaving
input count at most 256; matrix windows also bound `mw<=256`. These are existing
repacking bounds, not adoption of final-paper HE parameters. Ring interleaving
can have stride above 256 with at most 256 actual inputs.

The two-party term plan records local `Q0*V0`, `Q1*V1` and cross `Q1*V0`,
`Q0*V1`, independently derived from section III, p.5. It executes no OLT.
No HE, encryption, H2A privacy, OT/PRG, noise, rounding, numerical profile,
runtime registration, or complete secure execution capability is asserted.

Isolated child tests cover dense rectangular and partial-window products,
global group offsets, transpose directions, two cross terms, negative monomials,
odd/composite-modulus inverses, direct lane oracles, error atomicity, overflow,
resource bounds, and disjoint-buffer accounting. Execution is reserved for the
coordinator's serialized validation.

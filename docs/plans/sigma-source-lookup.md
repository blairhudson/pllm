# SIGMA source lookup: audited next gate

Status: **APPROVED FOR BOUNDED NATIVE IMPLEMENTATION**, both reconstructed and
deferred boundaries, including consuming deferred Select, the proposed physical
framing and aggregate pre-reserved accounting. User explicitly selected both
boundaries after review. This is separate from the approved TR6 → DReLU → Select
owning region. Full GELU/SiLU and decoder integration remain separate gates.
Approval is not executed protocol evidence or security qualification. No upstream
code is built, linked, vendored, or executed.

Bounded implementation validation now passes 513 garbling unit tests (eight
opt-in skips), 25 external/oracle tests and 17 documentation tests. Library
Clippy, the compiler library check, formatting and 280 source-leakage controls
also pass. Commands and scope are in
[sigma-native-reference.md](../research/sigma-native-reference.md). These checks
do not authorize the separate
[full activation proposal](sigma-full-activation.md).

## Source and independent implementation boundary

Pin: `mpc-msri/EzPC@aa0f2f860266fb79901405ef5e9965b8c9fe2e88`, beneath
`GPU-MPC/ext/sytorch/ext/llama/`:

- `api.cpp:2048–2052`: `LUT_dpf` selects `LUT_dfpet`.
- `api.cpp:2121–2208`: `LUT_dfpet`, masked result/sign reconstruction, Select,
  and final reconstruction-dependent subtraction.
- `lut.cpp:172–224`: DPF target, mask shares, modulo-four sign correction.
- `select.cpp:24–70`: five-word Select correlation and evaluator equation.
- `dpf.cpp:481–537`: signed complete-table reduction.
- `api.cpp:335–403`: `reconstructRT` opens arithmetic words and correction bits.
- `api.cpp:2741–2758,2818–2835`: full GELU/SiLU use deferred lookup and Select
  reconstruction, subtracting before the final activation reconstruction.

Parent independently fetched the first three files in memory and checked the
selected source ranges and SHA-256 values. These are source pins, not runtime
dependencies or author-libm table identity:

| File | SHA-256 |
| --- | --- |
| `api.cpp` | `a8233617b21952396e3bb4bd0c27d024b982eb8a9c5d27910cc6ba3a063c1650` |
| `lut.cpp` | `3f66d280712d63214bb7d0e286676a269c70c6ad5bd73201230f561139fe0b70` |
| `select.cpp` | `099b03680e82177cb123a837b8b0e119a8aba89bc45bd7a092fefa0a23e18c97` |

Reuse D16's existing GPU127 DPF, not the rejected CPU terminal. GPU127 `all()`
enumerates ascending indices with LSB-first terminal offsets. Consume those
indexed outputs directly: do not transplant CPU MSB-first bit extraction or
reverse table indices. Both possible target signs must work.

## Approved algebra

Let `N=2^n`, `M=2^m`, common masked index `Xhat=x+rin mod N`, DPF target
`alpha=-rin mod N`, and Boolean DPF outputs `b0 XOR b1=[i=alpha]`.
For party `p`:

```text
vp = (-1)^p * sum_i bp[i] * T[(i+Xhat) mod N] mod M
cp = (-1)^p * sum_i bp[i]
sigma = c0+c1 = b0[alpha]-b1[alpha] in {-1,+1}
v0+v1 = sigma*T[x] mod M
h0 = floor(((c0+1) mod 4)/2)
h1 = (2-floor((4-(c1 mod 4))/2)) mod 2
h0 XOR h1 = B = (sigma+1)/2
```

Individual counts need not be ±1. Add fresh shares of arithmetic mask `r` and
Boolean mask `q`, then reconstruct **both** `R=sigma*T[x]+r mod M` and
`C=B XOR q`. Source comments do not eliminate this masked-sign opening.

Existing five-word Select with masks `q,r,u` reconstructs
`S=B*(sigma*T[x])+u`. Final common masked value is:

```text
Yhat = 2*S-R = T[x]+rho mod M
rho  = 2*u-r mod M
```

Sample `r,u` independently and uniformly over the complete output ring and `q`
uniformly over one bit. D17 prohibits one-bit arithmetic masks. Derived `rho` is
uniform marginally and fresh across independently issued lanes, but
`rho mod 2 = r mod 2`: it is not independent of internal mask `r`.
This dependency is not by itself a demonstrated privacy attack or a security
proof. It must be explicit in the approved contract. Division by two cannot
derive `u` from arbitrarily chosen independent `rho,r` in a power-of-two ring.
Remasking, widening, added corrections or exchanges require separate approval.

## Reconstruction is an explicit contract

- **Reconstructed:** open masked `R,C`, reconstruct Select's `S`, then both
  parties compute `2*S-R`. Two exchange phases, not one.
- **Deferred:** open the same masked `R,C`, retain Select output shares `z0,z1`,
  and emit `2*z0-R, 2*z1`. Only party zero subtracts common `R`. Enclosing
  reconstruction remains a separately charged operation.

These require distinct enforceable output capabilities. Deferred shares cannot
be called a common masked value. Reconstructed standalone lookup cannot silently
replace full activation's deferred schedule. Lookup approval authorizes both
bounded native boundaries, not full activation composition.

Existing native Select exposes `begin → Frame → finish → MaskedOutput`, with
only the reconstructed output capability. Deferred lookup therefore additionally
needs the approved consuming Select extension with its own output type, lifecycle
and accounting. Transport payload bytes cannot become an output capability, and
`MaskedOutput` must not be relabeled as a deferred share. Both boundaries are
approved implementation scope, not already validated support.

## Bounded candidate and costs

Proposed geometry: 1–256 lanes, index width exactly 8 or 10, table exactly 256 or
1024 canonical output-ring entries, output width 3–64, and aggregate logical live
admission at most 64 MiB. All indices and canonical table residues are valid;
standalone lookup neither clips nor infers ranges from private samples.

For `d=n-7`, GPU127 has `K=129d+255` logical key bits per party/lane. At n=8/10,
this is 384/642 bits, or 66/99 encoded bytes including the existing 18-byte DPF
header. Paired issuance uses 4/12 AES blocks; each party's complete expansion
uses 2/14 AES blocks and scans 256/1024 entries. LUT plus Select logical key
material is `K+6m+1` bits per party/lane, excluding metadata and physical padding.

Reconstructed mode sends `L*(m+1)` logical bits per direction in the first phase
and `L*m` in the second. Deferred mode sends the first phase inside lookup but
must charge later reconstruction. The approved physical codec below uses u64
correction words, not packed bits. No upstream wire-parity claim follows.

Admission must price actual inline keys, masks, Select material, immutable table
owners, retained inputs/frames/outputs, scan workspace, metadata, and transition
overlap before randomness/allocation. Encoded-key length is not owner size.

### Approved physical framing and ownership

The approved follow-up uses u64 correction words rather than packed bits.
This is a PLLM codec contract, not validated implementation or upstream wire parity:

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 8 | `PLLMLU01` |
| 8 | 32 | Full invocation binding |
| 40 | 32 | Common masked-input digest |
| 72 | 1 | Sender role |
| 73 | 1 | Phase, exactly zero |
| 74 | 2 | Lane count, little-endian |
| 76 | 1 | Index width n |
| 77 | 1 | Output width m |
| 78 | 1 | Reconstructed/deferred boundary |
| 79 | 1 | Reserved zero |
| 80 | 16L | Lane-major arithmetic share and canonical 0/1 correction, u64 LE |

Bindings would include profile/codec, table/complete numeric identity, geometry,
boundary, source/plan/session/operation, every input/output/internal wire, first
tensor index and fresh issuance. The input digest additionally binds all common
masked indices. Reject wrong lengths, roles, phases, bindings, reserved bits,
noncanonical residues/corrections and trailing bytes before state mutation.
These bindings do not authenticate a peer.

Under this candidate, each first-phase frame is `80+16L` bytes; the unchanged
Select frame is `76+8L`. Reconstructed both-direction traffic would be
`312+48L`; deferred lookup-local traffic `160+32L`, with later reconstruction
separately charged. These are conditional codec formulas, not measured bytes.

Initial ownership would charge two immutable table copies, actual inline GPU127
lane owners (not encoded keys), all material/mask vectors, input/opened-value
backing, both send/receive owners, outputs, metadata and allocator allowances.
Lane-serial full expansion needs one complete N-entry scratch owner rather than
L copies, plus bounded hash/AES/stack workspace. Reserve all lookup and nested
Select owners together before randomness; split original reservations rather
than independently reacquiring inside an already reserved parent. Retained
first-phase frames, Select frames, Select output and final lookup output can
coexist and must all remain charged. Borrowed caller/transport copies retain
explicit allowances; logical reservations are not process-RSS bounds.

Deferred Select needs a consuming fresh-only execution path returning a distinct
non-cloneable share owner, with no frame emission or fake reconstructed finish.
It must bind role/context/ring/lanes, burn on failure, retain its lease through
consuming lookup transformation, and reject use after ordinary begin. Concrete
layout totals, admission boundaries and failure sweeps remain implementation
gates. Full activation ownership is not authorized by this plan.

## Independent acceptance gates

- Direct oracle `T[(Xhat-rin) mod N]`, independent of scan/Select code.
- Exhaust compatible count pairs summing to ±1 and modulo-four correction;
  both signs, Boolean masks, and arithmetic masks 0/1/half/max.
- Index/target endpoints around 0/63/64/127/128/255/256/511/512/1023;
  zero, constant, identity, alternating and ring-maximum tables.
- Output widths 3/64; no shift by 64 or overflowing signed arithmetic.
- Separate reconstructed/deferred checks, including one-sided subtraction.
- Complete bindings, consuming failures, retained-owner accounting, allocation
  and entropy failures, cancellation, and fixed-table/context substitution.

Legacy masked-wire Lookup uses different arithmetic sign correction. FuseFSS
lookup starts from additive shares. Neither supplies this source contract by
relabeling or unapproved conversion. Full GELU/SiLU still needs separately
approved original-X retention, absolute-value/clipping bounds and final deferred
subtraction. Numeric tables remain those in
`paper/reproduction-contracts/shared-profile-resolution.md` §2.1.

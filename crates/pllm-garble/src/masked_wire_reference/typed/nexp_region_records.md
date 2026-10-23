# D29 local region: proposed v1 records / owner inventory

**Review before any hash/issuance test. Not a reviewed token or public registration.**
`nexp_region_tests.rs` supplies independent edge/use/spec/coordinate/stage byte
encoders. Fixtures are serialization data, never capabilities/certificates.
No new packet format: eleven exchanges use the original native frames.

## Byte convention and fixed record order

`H(p0,...,pn)` means SHA-256 of successive `u64LE(len(pi)) || pi`. UTF-8
domains have NO trailing NUL except existing typed Binding/core edge/use domains.
Integers below have the named exact little-endian width, no native enum/layout.

Wire order (14, implicit fixed v1 count):
`X,X40,A40,H,B,W16,V16,C16,J0,J1,U0,U1,P,E`.
Widths/scales/interpretation:
`50/12 signed,40/12 signed,40/12 signed,1/0 predicate,1/0 predicate,
16/12 unsigned,16/12 unsigned,16/12 unsigned,8/0 unsigned,8/0 unsigned,
50/12 signed,50/12 signed,50/24 signed,50/12 signed`.
H/B use CommonXorMasked, others CommonAdditiveMasked. No public values.

Declaration preimage, exactly this order:

1. `pllm.sigma.nexp-region.v1`
2. `declaration.v1`
3. `floor(E0[min(X,65535)%256]*E1[min(X,65535)/256]/4096)`
4. ASCII paper SHA-256 `f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`
5. ASCII source commit `aa0f2f860266fb79901405ef5e9965b8c9fe2e88`
6. checkpoint, numeric profile, implementation, plan, session, operation, next:
   seven distinct 32-byte parts
7. i128LE input lower 0, input upper 2^38, output lower 0, output upper 4097
8. u64LE payload scale 4096
9. actual observed E0 then E1 u64LE-byte SHA-256 (32 bytes each), AFTER all floors
10. bytes `0c 04 32 0c 10 08 00 32 18` (denominators, payload, clipping,
    coordinate and product geometry)
11. u64LE lanes; u64LE first tensor index
12. fourteen existing core EdgeDescriptor digests in wire order
13. eighteen existing core ConsumerUse digests in slot order below

Frozen serialization fixture: checkpoint `01*32`, profile `02*32`,
implementation `03*32`, plan `04*32`, producer `05*32`, wire `(10+i)*32`,
session `5c*32`, operation `5a*32`, next `5b*32`; rank 1, dimension/count 2,
first index 7, no weight visibility. Synthetic observed hashes `a0*32,b0*32`
are ONLY serializer inputs, never accepted by construction. Input upper bytes:
`00 00 00 00 40 00 00 00 00 00 00 00 00 00 00 00`; output upper:
`01 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00`; scale:
`00 10 00 00 00 00 00 00`.

Coordinate declaration:
`H("pllm.sigma.nexp-coordinate.v1","declaration.v1",mapping,
edge(C),edge(Ji),actual_table_hash,u64LE(first),u64LE(lanes),[8,0],
u16LE(0),u16LE(256))`.
Mapping is exactly `low8.v1` or `high8-tr16-8.v1`.
Runtime provenance:
`H(coordinate_domain,"runtime.v1",declaration,binding(C),binding(Ji),[role])`.
Role is 0,1,2 (dealer), not derived from a pointer or native layout.

Existing typed Binding digest is unchanged (including exact descriptor/session/
namespace/issuance/mask-id/consumer/optional interval). Existing core edge/use
encodings are unchanged. Each ledger slot is
`H("pllm.sigma.nexp-use.v1",core_use_digest,binding_digest,[role])`.
Core ledger is H over `"pllm.sigma.nexp-ledger.v1"` then all eighteen slot hashes.
Complete role ledger = `H(core_ledger, runtime_low, runtime_high)`.
Region = `H(region_domain,declaration,binding(X),binding(E),ledger0,ledger1,
ledger2,fresh_32_byte_region_nonce)`.
Native child operation = `H(region,"stage.v1",u16LE(first_exchange))`.
Lookup subwire IDs = `H(region,"lookup-arithmetic.v1"/"lookup-correction.v1"/
"lookup-select.v1",[table_index])`.
Lookup numeric contract = `H(coordinate_domain,declaration,coordinate_declaration,
binding(C),binding(Ji))`; the certified helper preserves this EXACT context.
Arithmetic and Select use existing full Context encodings. Product uses the
neutral ordered two-input Context plus its own native fresh nonce.

No cycle: descriptor/use/table declarations -> actual independent binding IDs
-> role provenance/ledgers -> region nonce/hash -> child contexts -> native
material. Child output binding IDs are precommitted, not hashes of future data.

### Literal serialization vectors (not issued material)

The fixture above has these independently computed SHA-256 answers. These are
serialization checks only: synthetic hashes/identities do not grant table-floor,
source, range, Budget, randomness, transcript-authentication or privacy authority.
Python stdlib generated these answers only after checking all 512 selected floors
with the existing exact rational oracle. Rust checks remain unrun.

| Record | Hex digest |
| --- | --- |
| Declaration | `61459e90b393f3755e0d8afb590dfd0621a412deadb6a9333d35bf2c19a8bd7e` |
| Coordinate low | `bf2026ab18e825e6c9328a367108180badca53827aaf3563338d45a4e0a32351` |
| Coordinate high | `264f698ee6201c556e0cc78ea9f33c2367fd6c6df1c98d517048d7ce1d0d1099` |
| Stage 9, synthetic region `55*32` | `f0093f9dc41d3b1e9c46ed437e1255908eda4fcc29ef96ba8af374c524f828bf` |

For complete runtime-record fixtures, use the same fourteen descriptors; every
Binding has session `5c*32`, namespace `06*32`, issuance u64LE wire ordinal 0..13,
mask identity `(70+i)*32` (hex base 70), consumer `5a*32` except E's `5b*32`, and
interval presence tag 1. Intervals in wire order are
`[0,2^38), [0,2^38), [-65536,2^38-65536), [0,2), [0,2),
[0,65536), [0,65536), [0,65536), [0,256), [0,256), [0,4097),
[0,4097), [0,2^24+1), [0,4097)`.
These are actual typed Binding serialization fixtures, not imported capabilities.

| Record | Hex digest |
| --- | --- |
| X Binding | `9736e978578ea3c53714acc457d3e83b2aed138f184ee3d8325b2791f6525537` |
| P Binding | `c6fb9ec229f645b934ff89e16b310142245f6f09ee78ebb2f9724979219f6540` |
| E Binding | `68286f03e46021cf493abfa15e882a23bcfd80613c0a18b65b2f10a2426cb957` |
| Role 0 low provenance | `4a407febd4def87d82da557e2e08aa89e9f5bc955640dd86a7edb5d45302eee7` |
| Role 0 high provenance | `6dcae6879f14588883cb8df36c5e06d38b77ea00fd58c66d97ac23468c8bda3f` |
| Role 1 low provenance | `cd319acf42ffbf21bae74a2b92a0690742cbf0884ccfade4a823a1eb02b67d8f` |
| Role 1 high provenance | `43fbd5e391d1cfeee5cd7892f0a6ca9adbeb2b9186c2ac8b7206b788ae655c2d` |
| Dealer low provenance | `aa42da7a6aef89e65ec8d7e688b69c61b0ad216f3aac6bf57a31e9f295ada557` |
| Dealer high provenance | `0b937d6ae391c0ee266904c58a3e96f80ff3d2d2333261f0eebb0200bdb39a28` |
| Role 0 complete ledger | `1826b75f44eddb21c2b742fecb024c9d812c7140b2722a175f7fedb49c6ad5af` |
| Role 1 complete ledger | `a84564d8c58dc9e0b61038bf55163f18b95abcb7b3aa6e088421547d10db1e5d` |
| Dealer complete ledger | `0d1b6b1d4f6bb8277ba7f3431c1fc26e81ea99ddfec1d680a9612891758463b5` |
| Region, nonce `c0*32` | `1314063b70d4ac3ea2f54f13502642218add6556533b6058b30bc9ae62762b1a` |
| That region's stage 9 | `e4d35339159868955c2ed494b9af9e98c1d7c6a1b5ff601eda490b1fe3bc960b` |

## Uses and transcript

Uses: `X(entry),X(low40),X40(-65536),A40(DReLU),H(XOR1),X40(W16),B(Select),
W16(Select),V16(+65535),C16(low8),C16(TR),J1(bind),J0(E0),J1(E1),U0(product),
U1(product),P(GapARS),E(handoff)`.
Consumers are region operation except final next. Operand position is 1 only
slots 7 and 15, otherwise 0; ordinal is the global slot 0..17. Dealer and both
roles burn before the respective protected read; no retry/fanout/import API.

Exchanges: DReLU40; Select16; TR16,8 carry/output; E0 R/C/Select; E1 R/C/Select;
product50; GapARS50,12 carry/output. Exactly 11 per role, 22 frames total,
1688+208L peer bytes. No intermediate CommonMasked capability is returned.

## Complete reservation spine (logical, not physical peak)

Original input is consumed `OwnedCharged<Zeroizing<Vec<i64>>>`; actual full
capacity and wrapper are recounted, guard adequacy/original Budget identity
checked. Input creation is upstream. All original integers are validated before
any operation proof/hash/private construction/entropy. No inherited entry.

Caller holds actual bootstrap charge BEFORE fixed quote records. Same-counter
grow/split retains it continuously; constructor receives exact original native
lease. No credit from a numeric limit. Whole quote checked before first factory.

Total is the checked sum (NO retirement credits):
- original root owner's exact native quote;
- original native DReLU40, Select16, TR16,8, two reconstructed D19(8->50),
  neutral product50 and GapARS50,12 complete quotes;
- existing consuming typed product declaration/party/input/output wrappers,
  including actual per-role D19 value leases and temporary mask-wrapper leases;
- checked literal 4096-byte original public backing, certified preparation and
  pin/call/hash metadata, private custody Arc including original input owner;
- dealer exact 3L u64 buffer backings; each role exact 5L u64 buffers;
- two concrete State/Mutex Arc layouts and shared coordination Arc layout;
- retained wrapper maxima: eleven frames and steps PER role, final outputs,
  callbacks, original/partial/result construction and runtime slots, native
  query phases, concrete mask wrapper and all named fixed hash/edge workspace;
- original input backing remains separately charged in its non-detachable owner;
  the query reads that owner and counts its ACTUAL charge, including excess,
  not a caller-provided capacity baseline. Original dealer/caller bootstrap
  owners remain independently live on the same authoritative 64 MiB counter.

Native D19 resource quotes ALREADY contain four immutable table copies across
the two lookups; informational 8192-byte subset is NOT added twice. Native
reservations (including existing bounded hash allowances) are live already
during public preparation; there is no refund/reacquire between phases. Proof
workspace ends before native construction; no scratch retirement credit taken.
Caller proof workspace, argument and Verified/result overlap are named in
ConstructionSlots; the stage gate's actual validation slots are precharged at
bootstrap. It repeats whole-input validation, checks all512 certified rows and18
declared reads, then writes19 interval entries (including unchanged U0).
These deterministic categories are not CPU-instruction or physical/RSS claims;
no numeric quote supplies Budget authority.

Native masks remain original through their final correlation reads. Runtime
U0/U1 remain original through product finish, P through GapARS finish. Shared
custody keeps original input/counter through last frame/output. State backing
erases before guards, both Ready required, coordinator drop/error/reentry/unwind
revokes siblings; callbacks execute outside BOTH metadata/state locks.
The returned opaque mask also retains shared source custody; dropping it or
either final output revokes the region. No additional typed backing enum,
new root import, or raw-output/mask extraction API is introduced. The product
binding is minted by its existing consuming declaration before region hashing;
it consumes both actual D19 masks, and each role moves both actual D19 values
into the same private consuming product owner. GapARS borrows the resulting
original product output through its final reconstruction.

The verified tables are explicitly checked `0 <= Ei <= 4096` before operation
entropy. Thus genuine coordinates imply `0 <= P <= 4096^2 = 2^24`; this is
strictly inside the signed ring50 quarter-gap `[-2^48,2^48)`. GapARS50,12 is
therefore licensed by the actual selected table/consuming product chain, not a
caller interval assertion. All fourteen binding slots are checked for counter
exhaustion before floor/hash/factories/operation entropy.
The selected double-floor map is not globally nonincreasing or zero-persistent:
E(1279)=2995, E(1280)=2996; E(31487)=0, E(31488)=1. These public integer facts
are not accuracy, quality, privacy or adoption evidence; no smoothing/repair is
authorized. Actual owner-output regressions retain these exact quantized values.

All loops bounded: at most 256 actual input lanes, 512 fixed proof entries,
14 descriptors, 18 slots per dealer/role; finite native AES/work/entropy quotes
compose in WorkQuote. Input destruction also bounded by actual admitted capacity.
Caller callback/RNG/dependency machine work is not an instruction/latency bound.
Dependency-private SHA/native stacks, allocator overhead, OOM and RSS remain
unknown/excluded. No physical-peak, source authentication, RNG independence,
joint D19/product/GapARS privacy, general Multiply, softmax, normalization,
decoder, transport or SDK qualification.
